from dataclasses import dataclass
from typing import Container


@dataclass
class AudioFormat:
    sampling_rate: int = 22050
    mono: bool = True
    min_duration_ms: int = 1000
    max_duration_ms: int = 768000


@dataclass
class MelFormat:
    n_fft: int = 1024  # waveglow: 1024, wavenet: 800 (config -> filter_length)
    win_length: int = 1024  # waveglow: 1024, wavenet: 800
    hop_length: int = 256  # waveglow: 256, wavenet: 200
    n_mels: int = 80  # TacotronSTFT.n_mel_channels=80, default PyTorch = 128


@dataclass
class DatasetConfig:
    source_url: str = "https://www.rev.com/blog/transcript-tag/donald-trump-speech-transcripts"
    """ URL to list of videos with transcripts - original source of the data """

    trump_speaker_names: Container[str] = ("Donald Trump", "President Trump")

    dict_size: int = 75  # number of unique phonemes
    split_percentages: (float, float, float) = (0.75, 0.25, 0.)  # (train, val, test)

    audio_format: AudioFormat = AudioFormat()

    mel_format: MelFormat = MelFormat()
    use_tacotron2_spectrograms: bool = False
    """ if True, use tacotron2-compatible spectrograms instead of mel spectrograms """
