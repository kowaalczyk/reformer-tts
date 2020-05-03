from abc import ABC
from pathlib import Path
from typing import List

import ffmpeg
import librosa
import nltk
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample, Spectrogram


class PhonemeSequenceCreator:
    def __init__(self, nltk_data_directory: Path):
        # workaround for https://github.com/Kyubyong/g2p/issues/12
        nltk_data_directory.mkdir(exist_ok=True, parents=True)
        nltk.download("averaged_perceptron_tagger", download_dir=nltk_data_directory)
        nltk.download("cmudict", download_dir=nltk_data_directory)
        nltk.download("punkt", download_dir=nltk_data_directory)
        nltk.data.path.append(nltk_data_directory.resolve())
        from g2p_en import G2p
        self._g2p = G2p()

    def phonemize(self, text: str) -> List[str]:
        return self._g2p(text)


class SpectrogramCreator(ABC):
    def audio_to_mel_spectrogram(self, input_path: Path, output_path: Path):
        raise NotImplementedError()


class MelSpectrogramCreator(SpectrogramCreator):
    """
    Converts audio to log-scaled mel spectrogram format.

    Use Tacotron2SpectrogramCreator implementation instead of this one
    if you need compatibility with Tacotron2, Waveglow, Wavenet and SqueezeWave.
    """

    def __init__(self, sample_rate, n_fft, win_length, hop_length, n_mels):
        self._factory = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=0.0,  # TacotronSTFT.mel_fmin == PyTorch default
            f_max=8000.0,  # TacotronSTFT.mel_fmax, default PyTorch is None
        )

    def audio_to_mel_spectrogram(self, input_path: Path, output_path: Path):
        """
        Converts audio at input_path to mel spectrogram.
        Output shape: (1, n_mels, audio_length//hop_length + 1)
        """
        waveform, sampling_rate = torchaudio.load(input_path)
        assert waveform.shape[0] == 1

        mel_spectrogram = self._factory(waveform)  # (channel, n_mels, time)
        mel_spectrogram = spectrogram_to_log_scale(mel_spectrogram)

        torch.save(mel_spectrogram, output_path)


class Tacotron2SpectrogramCreator(SpectrogramCreator):
    """
    Converts audio to log-scaled magnitudes in the nvidia/tacotron2 format.

    This was implemented from scratch using torchaudio, but is compatible with:
        - NVIDIA tacotron2 implementation (mel2samp - librosa, stft and scipy)
        - NVIDIA WaveNet implementation (tweaked mel2samp)
        - NVIDIA WaveGlow implementation (tweaked mel2samp)
        - SqueezeWave implementation (tweaked mel2samp)

    While called a spectrogram in implementations listed above, this is NOT
    actually a spectrogram: magnitudes (result of STFT) are not squared, which
    makes this just a log-scaled mel-basis magnitudes.

    There is also a difference in mel filters used in librosa implementations,
    details of which are discussed in this github issue under torchaudio:
    https://github.com/pytorch/audio/issues/287
    """

    def __init__(self, sample_rate, n_fft, win_length, hop_length, n_mels):
        self._spectrogram = Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad=0,
            power=2,
            normalized=False,
        )
        # librosa mel basis is different than the one used in torchaudio, we need it for compatibility
        self._mel_scale = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=0., fmax=8000.
        )

    def audio_to_mel_spectrogram(self, input_path: Path, output_path: Path):
        waveform, sampling_rate = torchaudio.load(input_path)
        assert waveform.shape[0] == 1

        # shape is always (channel, n_mels, time)
        spectrogram = self._spectrogram(waveform)
        spectrogram = spectrogram.sqrt()  # this was most likely a bug in original tacotron implementation
        spectrogram = torch.tensor(
            np.dot(self._mel_scale, spectrogram.squeeze().numpy())
        ).unsqueeze(0)
        spectrogram = spectrogram_to_log_scale(spectrogram)

        torch.save(spectrogram, output_path)


def spectrogram_to_log_scale(x, clip_val=1e-5):
    """
    Based on tacotron2.audio_processing.dynamic_range_compression -
    this has to be used instead of torchaudio.transforms.AmplitudeToDB
    as tacotron expects log_e - scaled audio (as opposed
    to DBs which are log_10 - scaled).
    """
    scaled = torch.log(torch.clamp(x, min=clip_val))
    return scaled


def mp4_to_wav(input_path: Path, output_path: Path):
    audio = ffmpeg.input(str(input_path)).audio
    audio.output(str(output_path)).run(quiet=True, overwrite_output=True)


def resample_wav(
        input_path: Path,
        output_path: Path,
        stereo_to_mono: bool = True,
        sampling_rate: int = 22050
):
    waveform, original_sampling_rate = torchaudio.load(input_path)

    waveform = Resample(original_sampling_rate, sampling_rate)(waveform)
    if stereo_to_mono and len(waveform.shape) == 2:
        # waveform.shape==(channels, time) - we have to trim to 1 channel
        # note: we could also take mean from 2 channels but this is not guaranteed to work:
        # https://dsp.stackexchange.com/questions/2484/converting-from-stereo-to-mono-by-averaging
        waveform = waveform[0].unsqueeze(0)

    torchaudio.save(str(output_path), waveform, sampling_rate)
