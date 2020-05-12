from dataclasses import dataclass


@dataclass
class WNConfig:
    # in_audio_channels and in_mel_channels have to be specified explicitly
    n_layers: int = 8
    n_channels: int = 256
    conv_kernel_size: int = 3
    mel_upsample_scale: int = 2  # set to match audio length


@dataclass
class SqueezeWaveConfig:
    # n_mel_channels depends on dataset config
    n_mel_channels: int
    n_flows: int = 12
    n_audio_channels: int = 128
    early_return_interval: int = 2  # original name: n_early_every
    early_return_size: int = 16  # original name: n_early_size
    wn_config: WNConfig = WNConfig()
