import torch

from reformer_tts.squeeze_wave.config import WNConfig
from reformer_tts.squeeze_wave.modules import SqueezeWave


def test_squeezewave_infer():
    wn_config = WNConfig(8, 256, 3, 2)
    sw = SqueezeWave(12, 128, 80, 2, 16, wn_config)
    sw = SqueezeWave.remove_norms(sw)
    sw = sw.eval()

    # batch, n_mels, mel_len
    zero_mel = torch.zeros(8, 80, 1024)
    with torch.no_grad():
        audio = sw.infer(zero_mel)

    # batch, audio_len
    assert audio.shape == (8, 256 * 1024)
