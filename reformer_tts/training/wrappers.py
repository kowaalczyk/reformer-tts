import time
from dataclasses import asdict
from tempfile import NamedTemporaryFile
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiplicativeLR
from torch.utils.data import DataLoader, random_split
from transformers.optimization import AdamW

from reformer_tts.config import Config, as_shallow_dict
from reformer_tts.dataset.interface import TextToSpectrogramDataset, SpectrogramToSpeechDataset
from reformer_tts.dataset.utils import custom_sequence_padder, get_subset_lengths, AddGaussianNoise
from reformer_tts.dataset.visualize import plot_spectrogram, plot_attention_matrix
from reformer_tts.model.loss import TTSLoss
from reformer_tts.model.reformer_tts import ReformerTTS
from reformer_tts.squeeze_wave.loss import SqueezeWaveLoss
from reformer_tts.squeeze_wave.modules import SqueezeWave


class LitReformerTTS(pl.LightningModule):

    def __init__(self, config: Config, on_gpu=False):
        super().__init__()
        self.config = config
        self.val_batch_size = self.config.experiment.tts_training.batch_size
        self.train_batch_size = self.config.experiment.tts_training.batch_size
        self.val_num_workers = self.config.experiment.val_workers
        self.train_num_workers = self.config.experiment.train_workers

        self.model = ReformerTTS(**asdict(self.config.model))
        self.loss = TTSLoss(
            torch.tensor(self.config.experiment.tts_training.positive_stop_weight),
            raw_pred_loss_weight=config.experiment.tts_training.raw_pred_loss_weight,
            post_pred_loss_weight=config.experiment.tts_training.post_pred_loss_weight,
            stop_loss_weight=config.experiment.tts_training.stop_loss_weight,
            spectrogram_loss=config.experiment.tts_training.spectrogram_loss,
        )
        if noise_std := config.experiment.tts_training.noise_std is not None:
            self.transform = AddGaussianNoise(mean=0, std=noise_std)
        else:
            self.transform = None
        self.on_gpu = on_gpu

        assert self.config.experiment.tts_training.num_visualizations <= self.val_batch_size

    def forward(self, phonemes, spectrogram, stop_tokens, loss_mask, use_transform: bool = True):
        spectrogram_input = spectrogram
        if use_transform and self.transform:
            spectrogram_input[:, 1:, :] = self.transform(spectrogram_input[:, 1:, :])
        if self.on_gpu:
            phonemes, spectrogram, spectrogram_input = phonemes.cuda(), spectrogram.cuda(), spectrogram_input.cuda()
            stop_tokens, loss_mask = stop_tokens.cuda(), loss_mask.cuda()

        raw_mel_out, post_mel_out, stop_out, attention_matrices = self.model.forward(
            phonemes,
            spectrogram_input[:, :-1, :],
            spectrogram_mask=loss_mask.mean(dim=-1),
        )
        loss, raw_mel_loss, post_mel_loss, stop_loss = self.loss.forward(
            raw_mel_out,
            post_mel_out,
            stop_out.view(stop_out.shape[0], -1),
            true_mel=spectrogram[:, 1:, :],
            true_stop=stop_tokens,
            true_mask=loss_mask
        )
        # differentiator is needed, because argamax may not return first maximal value when multiple values are maximal
        differentiator = torch.arange(stop_tokens.shape[1]) / 10000
        differentiator = differentiator.cuda() if self.on_gpu else differentiator
        differentiator = differentiator.unsqueeze(0).repeat(stop_tokens.shape[0], 1)
        stop_idx = torch.argmax((stop_out.view(stop_out.shape[0], -1) > 0).float() - differentiator, dim=1)
        stop_err = stop_idx - torch.argmax(stop_tokens, dim=1)
        stop_mae = stop_err.abs().float().mean()

        return loss, raw_mel_loss, post_mel_loss, stop_loss, stop_mae, attention_matrices

    def prepare_data(self):
        dataset = TextToSpectrogramDataset(
            self.config.merged_transcript_csv_path,
            self.config.mel_directory
        )
        lengths = get_subset_lengths(len(dataset), self.config.dataset.split_percentages)
        self.train_set, self.val_set, self.test_set = random_split(dataset, lengths)

    def training_step(self, batch, batch_idx):
        loss, raw_mel_loss, post_mel_loss, stop_loss, _, _ = self.forward(
            batch['phonemes'],
            batch['spectrogram'],
            batch['stop_tokens'],
            batch["loss_mask"],
        )
        logs = {
            'train_stop_loss': stop_loss.cpu(),
            'train_raw_pred_loss': raw_mel_loss.cpu(),
            'train_post_pred_loss': post_mel_loss.cpu(),
            'train_loss': loss.cpu(),
        }
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        loss, raw_mel_loss, post_mel_loss, stop_loss, stop_mae, attention_matrices = self.forward(
            batch['phonemes'],
            batch['spectrogram'],
            batch['stop_tokens'],
            batch["loss_mask"],
            use_transform=False,
        )
        attention_matrices = self.trim_attention_matrices(
            attention_matrices,
            batch["stop_tokens"],
        )
        return {
            'stop_loss': stop_loss.cpu(),
            'raw_pred_loss': raw_mel_loss.cpu(),
            'post_pred_loss': post_mel_loss.cpu(),
            'loss': loss.cpu(),
            'stop_mae': stop_mae.cpu(),
            'attention_matrices': [
                [matrix.cpu() for matrix in matrices] for matrices in attention_matrices
            ]
        }

    def validation_epoch_end(self, outputs):
        def mean(x: List[torch.Tensor]) -> torch.Tensor:
            mean_value = sum(x) / len(x)
            return mean_value

        concat_inference_outputs = self.validate_inference("concat")

        attention_matrices = [
            matrices for output in outputs for matrices in output['attention_matrices']
        ]
        num_visualizations = self.config.experiment.tts_training.num_visualizations
        attention_matrices = attention_matrices[:num_visualizations]
        for i, matrices in enumerate(attention_matrices):
            for j, matrix in enumerate(matrices):
                with NamedTemporaryFile(suffix=".png") as f:
                    fig = plot_attention_matrix(matrix)
                    self.logger.log_image(f"attention_layer_{j}", fig)
                    fig.savefig(f.name)
                    artifact_name = f"attention_layer_{j}/viz{i}_e{self.current_epoch}.png"
                    self.logger.log_artifact(f.name, artifact_name)
                    plt.close()

        val_loss = mean([o['loss'] for o in outputs])
        logs = {
            'val_stop_loss': mean([o['stop_loss'] for o in outputs]),
            'val_raw_pred_loss': mean([o['raw_pred_loss'] for o in outputs]),
            'val_post_pred_loss': mean([o['post_pred_loss'] for o in outputs]),
            'val_stop_mae': mean([o['stop_mae'] for o in outputs]),
            'val_loss': val_loss,
            **concat_inference_outputs,
        }

        torch.cuda.empty_cache()
        return {'val_loss': val_loss, 'log': logs}

    def validate_inference(self, inference_combine_strategy: str) -> Dict[str, Any]:
        batch = custom_sequence_padder([
            self.val_set[i] for i in range(self.val_batch_size)
        ])
        phonemes = batch['phonemes']
        true_mel = batch['spectrogram'][:, 1:, :].transpose(1, 2)
        true_mask = batch["loss_mask"].transpose(1, 2)
        true_stop = batch["stop_tokens"].argmax(dim=1)
        if self.on_gpu:
            phonemes = phonemes.cuda()
            true_mel = true_mel.cuda()
            true_mask = true_mask.cuda()
            true_stop = true_stop.cuda()

        start = time.time()
        stop_at_stop_token = self.config.experiment.tts_training.stop_loss_weight != 0.
        mel_out, stop_out = self.model.infer(
            phonemes,
            combine_strategy=inference_combine_strategy,
            stop_at_stop_token=stop_at_stop_token,
        )
        inference_time = time.time() - start

        padded_mel_out = torch.zeros_like(true_mel)
        common_mel_len = min(padded_mel_out.shape[-1], mel_out.shape[-1])
        padded_mel_out[:, :, :common_mel_len] = mel_out[:, :, :common_mel_len]
        padded_mel_out *= true_mask
        inference_pred_mse = mse_loss(padded_mel_out, true_mel).cpu().item()

        inference_stop_mae = torch.abs(stop_out - true_stop) \
            .to(dtype=torch.float).mean().cpu().item()

        for i in range(self.config.experiment.tts_training.num_visualizations):
            with NamedTemporaryFile(suffix=".png") as f:
                clipped_mel_pred = mel_out[i, :, :stop_out[i].item()].unsqueeze(0)
                plot_spectrogram(clipped_mel_pred.cpu())
                plt.savefig(f.name)
                self.logger.log_image(f"sample-image-{inference_combine_strategy}", f.name)
                self.logger.log_artifact(f.name, f'inference-{inference_combine_strategy}/viz{i}_e{self.current_epoch}.png')
                plt.close()

        output = {
            f"{inference_combine_strategy}_inference_time": inference_time,
            f"{inference_combine_strategy}_inference_pred_mse": inference_pred_mse,
            f"{inference_combine_strategy}_inference_stop_mae": inference_stop_mae
        }
        return output

    @pl.data_loader
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            drop_last=True,
            collate_fn=custom_sequence_padder,
            num_workers=self.train_num_workers,
            shuffle=True,
        )

    @pl.data_loader
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.val_batch_size,
            drop_last=True,
            collate_fn=custom_sequence_padder,
            num_workers=self.val_num_workers,
        )

    def configure_optimizers(self):
        # log config here because it's the only place where we always have the logger (it's never called during inference)
        with NamedTemporaryFile(suffix=".yml") as f:
            self.config.to_yaml_file(f.name)
            self.logger.log_artifact(f.name, "config.yml")

        no_decay = {"bias", "norm.weight"}  # norm.weight only applies to nn.LayerNorm
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.experiment.tts_training.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        def get_optimizer(lr):
            return AdamW(
                optimizer_grouped_parameters,
                lr=lr,
                weight_decay=self.config.experiment.tts_training.weight_decay,
            )

        if self.config.experiment.tts_training.lr_scheduler is not None:
            schedule_config = self.config.experiment.tts_training.lr_scheduler
            assert schedule_config.start_schedule_epoch >= 1, "start_schedule_epoch has to be >= 1"
            start = schedule_config.start_schedule_epoch
            end = schedule_config.end_schedule_epoch
            end = self.config.experiment.max_epochs if end is None else end
            gamma = np.log(schedule_config.initial_lr) - np.log(schedule_config.final_lr)
            gamma /= end - start

            def exp_dec(current):
                if start <= current <= end:
                    a = np.exp(-1 * gamma * float(current)) * schedule_config.initial_lr
                    b = np.exp(-1 * gamma * float(current-1)) * schedule_config.initial_lr
                    decay = a / b
                else:
                    decay = 1
                self.logger.log_metric("learning_rate_decay", decay)
                return decay

            optimizer = get_optimizer(schedule_config.initial_lr)
            scheduler = MultiplicativeLR(optimizer, exp_dec)
            return [optimizer], [scheduler]
        else:
            optimizer = get_optimizer(self.config.experiment.tts_training.learning_rate)
            return optimizer

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        # warm up lr
        warmup_steps = self.config.experiment.tts_training.warmup_steps
        if warmup_steps is not None and self.trainer.global_step < warmup_steps:
            base_lr = self.config.experiment.tts_training.learning_rate
            if self.config.experiment.tts_training.lr_scheduler is not None:
                base_lr = self.config.experiment.tts_training.lr_scheduler.initial_lr
            lr_scale = min(1., float(self.trainer.global_step + 1) / warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * base_lr

        # update params
        optimizer.step()
        optimizer.zero_grad()

    def get_phoneme_encoder(self):
        # todo: there has to be a nicer way of extracting this !!!
        if not hasattr(self, "train_set"):
            self.prepare_data()

        assert isinstance(self.train_set.dataset, TextToSpectrogramDataset)

        def encoder(phonemes: str) -> torch.Tensor:
            encoding = torch.LongTensor([
                self.train_set.dataset.phoneme_to_idx[phoneme]
                for phoneme in phonemes.split()]
            )
            return encoding

        return encoder

    @staticmethod
    def trim_attention_matrices(
            attention_matrices: List[torch.Tensor],
            stop_tokens: torch.Tensor,
    ) -> List[List[torch.Tensor]]:
        stop_indexes = stop_tokens.argmax(dim=1)
        result = []
        for i, stop_index in enumerate(stop_indexes):
            result.append([matrix[i, :stop_index, :] for matrix in attention_matrices])
        return result


class LitSqueezeWave(pl.LightningModule):
    def __init__(self, config: Config, on_gpu: bool):
        super().__init__()
        self.config = config
        self.model = SqueezeWave(**as_shallow_dict(self.config.squeeze_wave))
        self.loss = SqueezeWaveLoss(self.config.experiment.vocoder_training.loss_sigma)
        self.train_batch_size = self.config.experiment.vocoder_training.batch_size
        self.train_num_workers = self.config.experiment.train_workers
        self.val_batch_size = self.config.experiment.vocoder_training.batch_size
        self.val_num_workers = self.config.experiment.val_workers
        self.sample_idx = 0
        self.on_gpu = on_gpu

    def prepare_data(self):
        dataset = SpectrogramToSpeechDataset(
            self.config.merged_transcript_csv_path,
            self.config.mel_directory,
            audio_segment_length=self.config.experiment.vocoder_training.audio_segment_length,
            mel_hop_length=self.config.dataset.mel_format.hop_length,
        )
        lengths = get_subset_lengths(len(dataset), self.config.dataset.split_percentages)
        self.train_set, self.val_set, self.test_set = random_split(dataset, lengths)

    def forward(self, _input):
        if self.on_gpu:
            _input = tuple(tensor.cuda() for tensor in _input)
        return self.model.forward(_input)

    def training_step(self, batch, batch_idx):
        output = self.forward((batch['spectrogram'], batch['audio']))
        loss = self.loss(output)
        logs = {'loss': loss.cpu()}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        output = self.forward((batch['spectrogram'], batch['audio']))
        loss = self.loss(output)
        result = {'loss': loss.cpu()}
        return result

    def validation_epoch_end(self, outputs):
        logs = {'val_loss': sum([o['loss'] for o in outputs]) / len(outputs)}

        # generate samples of audio on full spectrograms (not clipped to 1s)
        for audio_sample in self.prepare_samples():
            audio_sample = audio_sample.to("cpu")
            with NamedTemporaryFile(suffix=".wav") as f:
                torchaudio.save(f.name, audio_sample, self.config.dataset.audio_format.sampling_rate)
                self.logger.log_artifact(f.name, f"sample_audio/sample-{self.sample_idx}.wav")
            self.sample_idx += 1

        return {'log': logs}

    def prepare_samples(self):
        validation_model = SqueezeWave(**as_shallow_dict(self.config.squeeze_wave))
        validation_model.load_state_dict(self.model.state_dict())
        validation_model = SqueezeWave.remove_norms(validation_model)
        validation_model.eval()

        for i in range(self.config.experiment.vocoder_training.num_visualizations):
            # hack to use spectrogram from raw_sample on torch Subset (result of random_split)
            _, mel = self.val_set.dataset.raw_sample(self.val_set.indices[i])
            mel = mel.unsqueeze(0)
            if self.on_gpu:
                mel = mel.cuda()
            audio = validation_model.infer(mel)
            yield audio

    @pl.data_loader
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            drop_last=True,
            num_workers=self.train_num_workers,
            shuffle=True,
        )

    @pl.data_loader
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.val_batch_size,
            drop_last=True,
            num_workers=self.val_num_workers,
        )

    def configure_optimizers(self):
        return Adam(
            self.model.parameters(),
            self.config.experiment.vocoder_training.learning_rate,
        )
