from dataclasses import dataclass

@dataclass
class TTSTrainingConfig:
    batch_size: int = 16
    learning_rate: float = 1e-4
    positive_stop_weight: float = 5.
    num_visualizations: int = 3


@dataclass
class VocoderTrainingConfig:
    batch_size: int = 8
    learning_rate: float = 1e-4
    loss_sigma: float = 1.
    num_visualizations: int = 3
