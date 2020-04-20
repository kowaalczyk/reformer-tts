import torch

from .reformer_tts import ReformerTTS


class AudioGenerator:

    def __init__(self, model):
        assert isinstance(model, ReformerTTS), "model must be instance of ReformerTTS"
        self.model = model

    def generate(self, text):
        spectrogram = torch.zeros((1, 1, self.model.num_mel_coeffs))
        stop = False

        while not stop:
            generated, stop_layer = self.model(text, spectrogram)
            spectrogram = torch.cat([spectrogram, generated[:, -1, :].view(1, 1, self.model.num_mel_coeffs)], dim=1)
            if stop_layer[1] > 0.5:
                stop = True

        return spectrogram[1, 1:, :]

