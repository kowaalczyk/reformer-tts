from dataclasses import asdict
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torchaudio
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from reformer_tts.config import Config, as_shallow_dict
from reformer_tts.dataset.interface import TextToSpectrogramDataset, SpectrogramToSpeechDataset
from reformer_tts.dataset.utils import custom_sequence_padder, get_subset_lengths
from reformer_tts.dataset.visualize import plot_spectrogram
from reformer_tts.model.reformer_tts import ReformerTTS
from reformer_tts.squeeze_wave.loss import SqueezeWaveLoss
from reformer_tts.squeeze_wave.modules import SqueezeWave


class LitReformerTTS(pl.LightningModule):

    def __init__(self, config: Config, on_gpu=False):
        super().__init__()
        self.config = config
        self.model = ReformerTTS(**asdict(self.config.model))
        self.pos_weight = torch.tensor(self.config.experiment.tts_training.positive_stop_weight)
        self.val_batch_size = self.config.experiment.tts_training.batch_size
        self.train_batch_size = self.config.experiment.tts_training.batch_size
        self.sample_idx = 0
        self.val_num_workers = self.config.experiment.val_workers
        self.train_num_workers = self.config.experiment.train_workers
        self.on_gpu = on_gpu

    def forward(self, text, spec):
        if self.on_gpu:
            text, spec = text.cuda(), spec.cuda()
        return self.model.forward(text, spec)

    def prepare_data(self):
        dataset = TextToSpectrogramDataset(
            self.config.merged_transcript_csv_path,
            self.config.mel_directory
        )
        lengths = get_subset_lengths(len(dataset), self.config.dataset.split_percentages)
        self.train_set, self.val_set, self.test_set = random_split(dataset, lengths)

    def training_step(self, batch, batch_idx):
        loss, pred_loss, stop_loss = self.calculate_loss(batch)
        logs = {
            'train_stop_loss': stop_loss,
            'train_pred_loss': pred_loss,
            'train_loss': loss,
        }
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        loss, pred_loss, stop_loss = self.calculate_loss(batch)
        return {
            'stop_loss': stop_loss,
            'pred_loss': pred_loss,
            'loss': loss,
        }

    def calculate_loss(self, batch):
        pred, stop = self.forward(batch['phonemes'], batch['spectrogram'][:, :-1, :])
        assert pred.shape == batch["loss_mask"].shape
        masked_pred = pred * batch["loss_mask"]
        pred_loss = mse_loss(masked_pred, batch['spectrogram'][:, 1:, :])
        stop_loss = binary_cross_entropy_with_logits(
            stop.view(stop.shape[0], -1),
            batch['stop_tokens'],
            pos_weight=self.pos_weight
        )
        loss = pred_loss + stop_loss
        return loss, pred_loss, stop_loss

    def validation_epoch_end(self, outputs):
        val_loss = sum([o['loss'] for o in outputs]) / len(outputs)
        logs = {'val_stop_loss': sum([o['stop_loss'] for o in outputs]) / len(outputs),
                'val_pred_loss': sum([o['pred_loss'] for o in outputs]) / len(outputs),
                'val_loss': val_loss,
                }
        logs = {k: v.item() for k, v in logs.items()}
        samples = self.prepare_samples()
        for sample in samples:
            with NamedTemporaryFile(suffix=".png") as f:
                plot_spectrogram(sample.transpose(1, 0).unsqueeze(0).cpu(), scale=False)
                plt.savefig(f.name)
                self.logger.log_image("sample-image", f.name)
                self.sample_idx += 1
        return {'val_loss': val_loss, 'log': logs}

    def prepare_samples(self):
        batch = custom_sequence_padder([self.val_set[i]
                                        for i in range(self.config.experiment.tts_training.num_visualizations)])
        pred, _ = self.forward(batch['phonemes'], batch['spectrogram'][:, :-1, :])
        return pred

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
