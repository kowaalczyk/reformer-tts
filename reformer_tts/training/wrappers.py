import time
from dataclasses import asdict
from tempfile import NamedTemporaryFile
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torchaudio
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from reformer_tts.config import Config, as_shallow_dict
from reformer_tts.dataset.interface import TextToSpectrogramDataset, SpectrogramToSpeechDataset
from reformer_tts.dataset.utils import custom_sequence_padder, get_subset_lengths, AddGaussianNoise
from reformer_tts.dataset.visualize import plot_spectrogram
from reformer_tts.model.loss import TTSLoss
from reformer_tts.model.reformer_tts import ReformerTTS
from reformer_tts.squeeze_wave.loss import SqueezeWaveLoss
from reformer_tts.squeeze_wave.modules import SqueezeWave


class LitReformerTTS(pl.LightningModule):

    def __init__(self, config: Config, on_gpu=False):
        super().__init__()
        self.config = config
        self.model = ReformerTTS(**asdict(self.config.model))
        self.loss = TTSLoss(
            torch.tensor(self.config.experiment.tts_training.positive_stop_weight)
        )
        self.val_batch_size = self.config.experiment.tts_training.batch_size
        self.train_batch_size = self.config.experiment.tts_training.batch_size
        self.val_num_workers = self.config.experiment.val_workers
        self.train_num_workers = self.config.experiment.train_workers
        self.on_gpu = on_gpu
        noise_std = config.experiment.tts_training.noise_std 
        self.transform = AddGaussianNoise(mean=0, std=noise_std) if noise_std is not None else None

    def forward(self, phonemes, spectrogram, stop_tokens, loss_mask):
        spectrogram_input = spectrogram
        if self.transform:
            spectrogram_input[:, 1:, :] = self.transform(spectrogram_input[:, 1:, :]) 
        if self.on_gpu:
            phonemes, spectrogram, spectrogram_input = phonemes.cuda(), spectrogram.cuda(), spectrogram_input.cuda()
            stop_tokens, loss_mask = stop_tokens.cuda(), loss_mask.cuda()

        raw_mel_out, post_mel_out, stop_out = self.model.forward(
            phonemes,
            spectrogram_input[:, :-1, :]
        )
        loss, raw_mel_loss, post_mel_loss, stop_loss = self.loss.forward(
            raw_mel_out,
            post_mel_out,
            stop_out.view(stop_out.shape[0], -1),
            true_mel=spectrogram[:, 1:, :],
            true_stop=stop_tokens,
            true_mask=loss_mask
        )
        return loss, raw_mel_loss, post_mel_loss, stop_loss

    def prepare_data(self):
        dataset = TextToSpectrogramDataset(
            self.config.merged_transcript_csv_path,
            self.config.mel_directory
        )
        lengths = get_subset_lengths(len(dataset), self.config.dataset.split_percentages)
        self.train_set, self.val_set, self.test_set = random_split(dataset, lengths)

    def training_step(self, batch, batch_idx):
        loss, raw_mel_loss, post_mel_loss, stop_loss = self.forward(
            batch['phonemes'],
            batch['spectrogram'],
            batch['stop_tokens'],
            batch["loss_mask"],
        )
        logs = {
            'train_stop_loss': stop_loss,
            'train_raw_pred_loss': raw_mel_loss,
            'train_post_pred_loss': post_mel_loss,
            'train_loss': loss,
        }
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        loss, raw_mel_loss, post_mel_loss, stop_loss = self.forward(
            batch['phonemes'],
            batch['spectrogram'],
            batch['stop_tokens'],
            batch["loss_mask"],
        )
        return {
            'stop_loss': stop_loss,
            'raw_pred_loss': raw_mel_loss,
            'post_pred_loss': post_mel_loss,
            'loss': loss,
        }

    def validation_epoch_end(self, outputs):
        def mean(x: List[torch.Tensor]) -> float:
            mean_value = sum(x) / len(x)
            if hasattr(mean_value, "item"):
                mean_value = mean_value.item()
            return mean_value

        def aggregate(outs: List[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
            """ Take min, max and mean for every key over entire list. """
            # assuming all dicts have same keys as the first one
            means = {
                f"{prefix}_mean_{k}": mean([o[k] for o in outs])
                for k in outs[0]
            }
            mins = {
                f"{prefix}_min_{k}": min([o[k] for o in outs])
                for k in outs[0]
            }
            maxes = {
                f"{prefix}_max_{k}": max([o[k] for o in outs])
                for k in outs[0]
            }
            return {**means, **mins, **maxes}

        concat_inference_outputs = self.validate_inference("concat")
        replace_inference_outputs = self.validate_inference("replace")

        val_loss = mean([o['loss'] for o in outputs])
        logs = {
            'val_stop_loss': mean([o['stop_loss'] for o in outputs]),
            'val_raw_pred_loss': mean([o['raw_pred_loss'] for o in outputs]),
            'val_post_pred_loss': mean([o['post_pred_loss'] for o in outputs]),
            'val_loss': val_loss,
            **aggregate(concat_inference_outputs, prefix="concat"),
            **aggregate(replace_inference_outputs, prefix="replace"),
        }

        return {'val_loss': val_loss, 'log': logs}

    def validate_inference(self, inference_combine_strategy: str):
        outputs = list()
        for i in range(self.config.experiment.tts_training.num_visualizations):
            sample = self.val_set[i]
            phonemes = sample['phonemes'].unsqueeze(0)
            if self.on_gpu:
                phonemes = phonemes.cuda()

            start = time.time()
            mel_out = self.model.infer(phonemes, combine_strategy=inference_combine_strategy)
            inference_time = time.time() - start

            true_mel = sample['spectrogram'].unsqueeze(0)[:, 1:, :].transpose(1, 2)
            if self.on_gpu:
                true_mel = true_mel.cuda()
            
            padded_mel_out = torch.zeros_like(true_mel)
            common_mel_len = min(padded_mel_out.shape[-1], mel_out.shape[-1])
            padded_mel_out[:, :, :common_mel_len] = mel_out[:, :, :common_mel_len]
            inference_mse = mse_loss(padded_mel_out, true_mel)
            outputs.append({
                'inference_time': inference_time,
                'inference_mse': inference_mse.cpu().item(),
                'true_minus_pred_len': max(true_mel.shape) - max(mel_out.shape)
            })

            with NamedTemporaryFile(suffix=".png") as f:
                plot_spectrogram(mel_out.cpu())
                plt.savefig(f.name)
                self.logger.log_image(f"sample-image-{inference_combine_strategy}", f.name)

        return outputs

    @pl.data_loader
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            drop_last=True,
            collate_fn=custom_sequence_padder,
            num_workers=self.train_num_workers,
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
        return Adam(
            self.model.parameters(),
            self.config.experiment.tts_training.learning_rate,
        )

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
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        output = self.forward((batch['spectrogram'], batch['audio']))
        loss = self.loss(output)
        result = {'loss': loss}
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

        for i in range(self.config.vocoder_training.num_visualizations):
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
