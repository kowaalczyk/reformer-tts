from dataclasses import dataclass
from typing import Optional


@dataclass
class TTSTrainingConfig:
    batch_size: int = 8
    learning_rate: float = 1e-4
    positive_stop_weight: float = 5.
    num_visualizations: int = 3
    early_stopping_epochs: Optional[int] = None
    weight_decay: int = 1e-4
    noise_std: Optional[float] = None
    gpus: Optional[int] = None


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
    early_stopping_epochs: Optional[int] = None  # None to disable early stopping
    n_saved_models: int = 50  # set between 10 and 100 depending on training length
    tags: str = ""
    save_top_k_checkpoints: int = 0  # set to zero to save all checkpoints
    # todo: need to refactor before final training (same keys in multiple places) !!!
