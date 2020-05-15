from typing import Union
from dataclasses import dataclass


@dataclass
class TTSTrainingConfig:
    batch_size: int = 8
    learning_rate: float = 1e-4
    positive_stop_weight: float = 5.
    num_visualizations: int = 3
    early_stopping_epochs: int = None


@dataclass
class VocoderTrainingConfig:
    audio_segment_length: int = 16384
    batch_size: int = 96  # 96 in original implementation (on V100 GPU)
    learning_rate: float = 4e-4
    loss_sigma: float = 1.
    num_visualizations: int = 10


@dataclass
class ExperimentConfig:
    tts_training: TTSTrainingConfig = TTSTrainingConfig()
    vocoder_training: VocoderTrainingConfig = VocoderTrainingConfig()
    experiment_name: str = "default"
    checkpoints_dir: str = "checkpoints"
    max_epochs: int = 2
    train_workers: int = 4
    val_workers: int = 4
