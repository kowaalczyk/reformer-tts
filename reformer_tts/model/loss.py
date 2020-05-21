from typing import Tuple

from torch import nn, Tensor
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits


class TTSLoss(nn.Module):
    def __init__(self, pos_weight: Tensor):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
            self, raw_mel_out, postnet_mel_out, stop_out, true_mel, true_stop, true_mask
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        MSE on raw_mel_out, postnet_mel_out and BinaryCE on stop_out.
        :return: tuple (total_loss, raw_mel_loss, postnet_mel_loss, stop_loss)
        """
        assert raw_mel_out.shape == postnet_mel_out.shape == true_mask.shape == true_mel.shape
        assert stop_out.shape == true_stop.shape

        raw_mel_out *= true_mask
        raw_mel_loss = mse_loss(raw_mel_out, true_mel)

        postnet_mel_out *= true_mask
        postnet_mel_loss = mse_loss(postnet_mel_out, true_mel)

        stop_loss = binary_cross_entropy_with_logits(
            stop_out,
            true_stop,
            pos_weight=self.pos_weight
        )
        total_loss = raw_mel_loss + postnet_mel_loss + stop_loss
        return total_loss, raw_mel_loss, postnet_mel_loss, stop_loss
