from typing import Dict

import torch
from torch import nn

from .config import ReformerDecConfig, ReformerEncConfig, EncoderPreNetConfig, DecoderPreNetConfig, PostConvNetConfig
from .modules import EncoderPreNet, ScaledPositionalEncoding, DecoderPreNet, PostConvNet
from .reformer import ReformerEnc, ReformerDec


class Encoder(nn.Module):
    def __init__(
            self,
            dict_size: int,
            embedding_dim: int,
            scp_encoding_dropout: float,
            reformer_kwargs: Dict,
            prenet_kwargs: Dict,
    ):
        super().__init__()
        self.prenet = EncoderPreNet(
            num_embeddings=dict_size + 1,  # plus zero -- empty index
            embedding_dim=embedding_dim,
            **prenet_kwargs
        )
        self.positional_encoding = ScaledPositionalEncoding(embedding_dim, scp_encoding_dropout)
        self.reformer = ReformerEnc(embedding_dim, **reformer_kwargs)

    def forward(self, input_):
        input_ = self.prenet(input_)
        input_ = self.positional_encoding(input_)
        input_ = self.reformer(input_)

        return input_


class Decoder(nn.Module):
    def __init__(
            self,
            num_mel_coeffs: int,
            embedding_dim: int,
            scp_encoding_dropout: float,
            prenet_kwargs: Dict,
            reformer_kwargs: Dict,
            postnet_kwargs: Dict,
    ):
        super().__init__()
        self.prenet = DecoderPreNet(
            input_size=num_mel_coeffs,
            output_size=embedding_dim,
            **prenet_kwargs
        )
        self.positional_encoding = ScaledPositionalEncoding(embedding_dim, scp_encoding_dropout)
        self.reformer = ReformerDec(embedding_dim, **reformer_kwargs)
        self.mel_linear = nn.Linear(embedding_dim, num_mel_coeffs)
        self.stop_linear = nn.Linear(embedding_dim, 1)
        self.postnet = PostConvNet(num_mel_coeffs, num_hidden=embedding_dim, **postnet_kwargs)

    def forward(self, input_, keys):
        input_ = self.prenet(input_)
        input_ = self.positional_encoding(input_)
        input_ = self.reformer(input_, keys=keys)
        stop = self.stop_linear(input_)
        mel = self.mel_linear(input_)
        residual = self.postnet(mel)
        mel += residual
        return mel, stop


class ReformerTTS(nn.Module):
    def __init__(
            self,
            num_mel_coeffs: int,
            dict_size: int,
            pad_base: int,
            embedding_dim: int,
            scp_encoding_dropout: float,
            enc_reformer_kwargs: Dict,
            enc_prenet_kwargs: Dict,
            dec_prenet_kwargs: Dict,
            dec_reformer_kwargs: Dict,
            postnet_kwargs: Dict,
    ):
        super().__init__()
        self.num_mel_coeffs = num_mel_coeffs
        self.enc = Encoder(
            dict_size=dict_size,
            embedding_dim=embedding_dim,
            scp_encoding_dropout=scp_encoding_dropout,
            reformer_kwargs=enc_reformer_kwargs,
            prenet_kwargs=enc_prenet_kwargs,
        )
        self.dec = Decoder(
            num_mel_coeffs=num_mel_coeffs,
            embedding_dim=embedding_dim,
            scp_encoding_dropout=scp_encoding_dropout,
            prenet_kwargs=dec_prenet_kwargs,
            reformer_kwargs=dec_reformer_kwargs,
            postnet_kwargs=postnet_kwargs,
        )
        self.pad_base = pad_base

    def forward(self, text, dec_input):
        pad_text = pad_to_multiple(text.view((*text.shape, 1)), self.pad_base).view((text.shape[0], -1))
        pad_spec = pad_to_multiple(dec_input, self.pad_base)
        keys = self.enc(pad_text)
        mel, stop = self.dec(pad_spec, keys=keys)
        return mel[:, :dec_input.shape[1]], stop[:, :dec_input.shape[1]]


def pad_to_multiple(tensor, pad_base):
    """
    This function pads the sequence to be divisible by pad_base.
    Assumed tensor is of shape (batch, seq_len, channels)
    """
    new_len = ((tensor.shape[1] - 1) // pad_base + 1) * pad_base
    # one is subtracted for the case when tensor.shape[1] % pad_base == 0
    padding = torch.zeros(tensor.shape[0], new_len - tensor.shape[1], tensor.shape[2], dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=1)
