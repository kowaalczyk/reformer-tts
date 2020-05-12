import random
from pathlib import Path
from typing import Union, Dict, Any

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


class TextToSpectrogramDataset(Dataset):
    """ Text-to-Spectrogram interface for the Trump Speech Dataset """

    def __init__(self, merged_transcript_csv_path: Path, mel_directory: Path):
        self.transcripts_df = pd.read_csv(merged_transcript_csv_path)
        self.mel_directory = mel_directory
        phonemes = {p for s in self.transcripts_df['phonemes'] for p in s.split()}
        self.phoneme_to_idx = {p: i + 1 for i, p in enumerate(sorted(list(phonemes)))}

    def __len__(self):
        return len(self.transcripts_df)

    def __getitem__(self, idx: Union[torch.Tensor, np.array, int]) -> Dict[str, Any]:
        """
        :param idx: index or single-element tensor / numpy array
        :return: { phonemes: Tensor(len), spectrogram: Tensor(len x n_mels) }
        """
        if hasattr(idx, '__getitem__'):
            idx = int(idx[0])

        phonemes = self.transcripts_df.loc[idx, "phonemes"]
        phoneme_indices = torch.tensor([self.phoneme_to_idx[p] for p in phonemes.split()], dtype=torch.long)

        audio_path = Path(self.transcripts_df.loc[idx, "audio_path"])
        spectrogram_path = Path(self.mel_directory / audio_path.name) \
            .with_suffix(".pt")
        spectrogram = torch.load(spectrogram_path)
        spectrogram = spectrogram.view(spectrogram.shape[1:]).transpose(1, 0)

        sample = {"phonemes": phoneme_indices, "spectrogram": spectrogram}
        return sample


class SpectrogramToSpeechDataset(Dataset):
    """ Spectrogram-to-Speech interface for the Trump Speech Dataset """

    def __init__(
            self,
            merged_transcript_csv_path: Path,
            mel_directory: Path,
            audio_segment_length: int,
            mel_hop_length: int,
    ):
        self.transcripts_df = pd.read_csv(merged_transcript_csv_path)
        self.mel_directory = mel_directory
        self.audio_segment_length = audio_segment_length
        self.mel_hop_length = mel_hop_length

    def __len__(self):
        return len(self.transcripts_df)

    def __getitem__(self, idx: Union[torch.Tensor, np.array, int]) -> Dict[str, Any]:
        """
        :param idx: index or single-element tensor / numpy array
        :return: { audio: Tensor(audio_segment_length), spectrogram: Tensor(n_mels x mel_len) }
        """
        if hasattr(idx, '__getitem__'):
            idx = int(idx[0])

        audio_path = Path(self.transcripts_df.loc[idx, "audio_path"])
        audio, _ = torchaudio.load(audio_path)  # ignore sampling rate here
        audio = torch.squeeze(audio, dim=0)

        spectrogram_path = Path(self.mel_directory / audio_path.name) \
            .with_suffix(".pt")
        spectrogram = torch.load(spectrogram_path)
        spectrogram = torch.squeeze(spectrogram, dim=0)

        # based on https://github.com/tianrengao/SqueezeWave/blob/177aebb5a4c53bd70100a6e8676c338582fbcd55/mel2samp.py#L90
        audio_size = audio.size(0)
        n_hops = self.audio_segment_length // self.mel_hop_length
        if audio_size >= self.audio_segment_length:
            max_audio_start = audio_size - self.audio_segment_length
            random_hop = random.randint(0, max_audio_start // self.mel_hop_length)

            audio_start = random_hop * self.mel_hop_length
            audio_end = audio_start + n_hops * self.mel_hop_length
            audio = audio[audio_start:audio_end]

            spectrogram_start = random_hop
            spectrogram_end = spectrogram_start + n_hops
            spectrogram = spectrogram[:, spectrogram_start:spectrogram_end]
        else:
            audio_pad_size = self.audio_segment_length - audio_size
            audio = torch.nn.functional.pad(audio, [0, audio_pad_size], 'constant')

            spectrogram_pad_size = n_hops - spectrogram.shape[-1]
            spectrogram = torch.nn.functional.pad(
                spectrogram,
                [0, spectrogram_pad_size],
                'constant'
            )

        sample = {"audio": audio, "spectrogram": spectrogram}
        return sample
