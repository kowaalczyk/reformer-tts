from dataclasses import asdict
from tempfile import NamedTemporaryFile

import pytorch_lightning as pl
import torch
import torchaudio
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from reformer_tts.config import Config, as_shallow_dict
from reformer_tts.dataset.interface import TextToSpectrogramDataset, SpectrogramToSpeechDataset
from reformer_tts.dataset.utils import custom_sequence_padder, get_subset_lengths
from reformer_tts.model.reformer_tts import ReformerTTS
from reformer_tts.squeeze_wave.loss import SqueezeWaveLoss
from reformer_tts.squeeze_wave.modules import SqueezeWave


class LitReformerTTS(pl.LightningModule):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model = ReformerTTS(**asdict(self.config.model))
        self.pos_weight = torch.tensor([1., self.config.tts_training.positive_stop_weight])
        self.val_batch_size = self.config.tts_training.batch_size
        self.train_batch_size = self.config.tts_training.batch_size
        self.sample_idx = 0

    def forward(self, text, spec):
        return self.model.forward(text, spec)

    def prepare_data(self):
        dataset = TextToSpectrogramDataset(
            self.config.merged_transcript_csv_path,
            self.config.mel_directory
        )
        lengths = get_subset_lengths(len(dataset), self.config.dataset.split_percentages)
        self.train_set, self.val_set, self.test_set = random_split(dataset, lengths)

    def training_step(self, batch, batch_idx):
        pred, stop = self.model.forward(batch['phonemes'], batch['spectrogram'][:, :-1, :])
        pred_loss = cross_entropy(pred, batch['spectrogram'][:, 1:, :])
        stop_loss = binary_cross_entropy_with_logits(stop, batch['stop_tokens'], pos_weight=self.pos_weight)
        loss = pred_loss + stop_loss
        logs = {'train_stop_loss': stop_loss,
                'train_pred_loss': pred_loss,
                'train_loss': loss,
                }
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        pred, stop = self.model.forward(batch['phonemes'], batch['spectrogram'][:, :-1, :])
        pred_loss = cross_entropy(pred, batch['spectrogram'][:, 1:, :])
        stop_loss = binary_cross_entropy_with_logits(stop, batch['stop_tokens'], pos_weight=self.pos_weight)
        loss = pred_loss + stop_loss
        result = {
            'stop_loss': stop_loss,
            'pred_loss': pred_loss,
            'loss': loss,
        }
        return result

    def validation_epoch_end(self, outputs):
        logs = {
            'val_stop_loss': sum([o['stop_loss'] for o in outputs]) / len(outputs),
            'val_pred_loss': sum([o['pred_loss'] for o in outputs]) / len(outputs),
            'val_loss': sum([o['loss'] for o in outputs]) / len(outputs),
        }
        samples = self.prepare_samples()
        for sample in samples:
            self.logger.log_image("sample_spectrograms", self.sample_idx, sample)
            self.sample_idx += 1
        return {'log': logs}

    def prepare_samples(self):
        batch = custom_sequence_padder([self.val_set[i]
                                        for i in range(self.config.tts_training.num_visualizations)])
        pred, _ = self.model.forward(batch['phonemes'], batch['spectrogram'][:, :-1, :])
        return pred

    @pl.data_loader
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            drop_last=True,
            collate_fn=custom_sequence_padder,
        )

    @pl.data_loader
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.val_batch_size,
            drop_last=True,
            collate_fn=custom_sequence_padder,
        )

    def configure_optimizers(self):
        return Adam(
            self.model.parameters(),
            self.config.tts_training.learning_rate,
        )


class LitSqueezeWave(pl.LightningModule):
    def __init__(self, config: Config, on_gpu: bool):
        super().__init__()
        self.config = config
        self.model = SqueezeWave(**as_shallow_dict(self.config.squeeze_wave))
        self.loss = SqueezeWaveLoss(self.config.vocoder_training.loss_sigma)
        self.train_batch_size = self.config.vocoder_training.batch_size
        self.train_num_workers = self.config.vocoder_training.train_workers
        self.val_batch_size = self.config.vocoder_training.batch_size
        self.val_num_workers = self.config.vocoder_training.val_workers
        self.sample_idx = 0
        self.on_gpu = on_gpu

    def prepare_data(self):
        dataset = SpectrogramToSpeechDataset(
            self.config.merged_transcript_csv_path,
            self.config.mel_directory,
            audio_segment_length=self.config.vocoder_training.audio_segment_length,
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
        samples = self.prepare_samples()
        for sample in samples:
            sample = sample.to("cpu").unsqueeze(0)
            with NamedTemporaryFile(suffix=".wav") as f:
                torchaudio.save(f.name, sample, self.config.dataset.audio_format.sampling_rate)
                self.logger.log_artifact(f.name, f"sample_audio/sample-{self.sample_idx}.wav")
            self.sample_idx += 1
        return {'log': logs}

    def prepare_samples(self):
        batch_spec = torch.stack([
            self.val_set[i]['spectrogram']
            for i in range(self.config.vocoder_training.num_visualizations)
        ])
        if self.on_gpu:
            batch_spec = batch_spec.cuda()
        audio = self.model.infer(batch_spec)
        return audio

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
            self.config.vocoder_training.learning_rate,
        )
