import torch

from .reformer_tts import ReformerTTS


class SpectrogramGenerator:
    """ Wrapper for using ReformerTTS for spectrogram generation. """

    def __init__(self, model):
        assert isinstance(model, ReformerTTS), "model must be instance of ReformerTTS"
        self.model = model

    def generate(self, encoded_phonemes: torch.LongTensor, max_len: int = 1024):
        """ Encoded phonemes shape: (batch, sequence length) """

        zeros = torch.zeros((1, 1, self.model.num_mel_coeffs))
        spectrogram = zeros.clone()
        stop = False

        while not stop:
            generated, stop_pred = self.model(encoded_phonemes, spectrogram)

            # alternatively:
            # spectrogram = torch.cat([spectrogram, generated[:, -1, :].view(1, 1, self.model.num_mel_coeffs)], dim=1)
            spectrogram = torch.cat([zeros, generated], dim=1)
            stop = torch.any(torch.sigmoid(stop_pred) > 0.25).item()

            # generated_tail = generated[:,-min(20, generated.shape[1]):,:]
            # print(f"{generated_tail.mean(dim=[0,2])=}, {stop_pred.max()=}")

            iteration = spectrogram.shape[1]

            if iteration % 100 == 0:
                print(f"reached {iteration=}...")
            
            if stop:
                stop_idx = torch.argmax(stop_pred) + 1
                spectrogram = spectrogram[:, :stop_idx, :]
                print(f"stopped at {stop_idx=} on {iteration=}")
            
            if max(spectrogram.shape) > max_len:
                print(f"stopped at {max_len=}")
                break

        spectrogram: torch.Tensor = spectrogram.transpose(1, 2)
        return spectrogram[:, :, 1:]
