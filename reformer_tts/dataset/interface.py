from pathlib import Path
from typing import Union, Dict, Any

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


class TTSTrumpDataset(Dataset):
    """ Text-to-Spectrogram interface for the Trump Speech Dataset """

    def __init__(self, merged_transcript_csv_path: Path, mel_directory: Path):
        self.transcripts_df = pd.read_csv(merged_transcript_csv_path)
        self.mel_directory = mel_directory

    def __len__(self):
        return len(self.transcripts_df)

    def __getitem__(self, idx: Union[torch.Tensor, np.array, int]) -> Dict[str, Any]:
        """
        :param idx: index or single-element tensor / numpy array
        :return: { phonemes: str, spectrogram: Tensor(1 x n_mels x len) }
        """
        if hasattr(idx, '__getitem__'):
            idx = int(idx[0])

        phonemes = self.transcripts_df.loc[idx, "phonemes"]

        audio_path = Path(self.transcripts_df.loc[idx, "audio_path"])
        spectrogram_path = Path(self.mel_directory / audio_path.name) \
            .with_suffix(".pt")
        spectrogram = torch.load(spectrogram_path)

        sample = {"phonemes": phonemes, "spectrogram": spectrogram}
        return sample


class VocoderTrumpDataset(Dataset):
    """ Spectrogram-to-Speech interface for the Trump Speech Dataset """

    def __init__(self, merged_transcript_csv_path: Path, mel_directory: Path):
        self.transcripts_df = pd.read_csv(merged_transcript_csv_path)
        self.mel_directory = mel_directory

    def __len__(self):
        return len(self.transcripts_df)

    def __getitem__(self, idx: Union[torch.Tensor, np.array, int]) -> Dict[str, Any]:
        """
        :param idx: index or single-element tensor / numpy array
        :return: { audio: Tensor(1 x audio_len), spectrogram: Tensor(1 x n_mels x mel_len) }
        """
        if hasattr(idx, '__getitem__'):
            idx = int(idx[0])

        audio_path = Path(self.transcripts_df.loc[idx, "audio_path"])
        audio, _ = torchaudio.load(audio_path)  # ignore sampling rate here

        spectrogram_path = Path(self.mel_directory / audio_path.name) \
            .with_suffix(".pt")
        spectrogram = torch.load(spectrogram_path)

        sample = {"audio": audio, "spectrogram": spectrogram}
        return sample
