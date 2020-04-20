from dataclasses import asdict

import torch
from torch import nn

from .modules import EncoderPreNet, ScaledPositionalEncoding, DecoderPreNet, PostConvNet
from .reformer import ReformerEnc, ReformerDec
from .config import ReformerDecConfig, ReformerEncConfig, EncoderPreNetConfig, DecoderPreNetConfig, PostConvNetConfig


class Encoder(nn.Module):
    def __init__(
            self,
            dict_size: int,
            embedding_dim: int,
            reformer_kwargs: ReformerEncConfig,
            prenet_kwargs: EncoderPreNetConfig,
    ):
        super().__init__()
        self.prenet = EncoderPreNet(
            num_embeddings=dict_size,
            embedding_dim=embedding_dim,
            **asdict(prenet_kwargs)
        )
        self.positional_encoding = ScaledPositionalEncoding(embedding_dim)
        self.reformer = ReformerEnc(embedding_dim, **asdict(reformer_kwargs))

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
            prenet_kwargs: DecoderPreNetConfig,
            reformer_kwargs: ReformerDecConfig,
            postnet_kwargs: PostConvNetConfig,
    ):
        super().__init__()
        self.prenet = DecoderPreNet(
            input_size=num_mel_coeffs,
            output_size=num_mel_coeffs,
            **asdict(prenet_kwargs)
        )
        self.positional_encoding = ScaledPositionalEncoding(embedding_dim)
        self.reformer = ReformerDec(embedding_dim, **asdict(reformer_kwargs))
        self.mel_linear = nn.Linear(embedding_dim, num_mel_coeffs)
        self.stop_linear = nn.Linear(embedding_dim, 2)  # Note there are two outputs, use BinaryCE loss
        self.postnet = PostConvNet(num_mel_coeffs, num_hidden=embedding_dim, **asdict(postnet_kwargs))

    def forward(self, input_, keys):
        input_ = self.prenet(input_)
        input_ = self.positional_encoding(input_)
        input_ = self.reformer(input_, keys=keys)
        stop = torch.softmax(self.stop_linear(input_))
        mel = self.mel_linear(input_)
        residual = self.postnet(mel)
        mel += residual
        return mel, stop


class ReformerTTS(nn.Module):
    def __init__(
            self,
            num_mel_coeffs: int,
            dict_size: int,
            embedding_dim: int,
            enc_reformer_kwargs: ReformerEncConfig,
            enc_prenet_kwargs: EncoderPreNetConfig,
            dec_prenet_kwargs: DecoderPreNetConfig,
            dec_reformer_kwargs: ReformerDecConfig,
            postnet_kwargs: PostConvNetConfig,
    ):
        super().__init__()
        self.num_mel_coeffs = num_mel_coeffs
        self.enc = Encoder(
            dict_size=dict_size,
            embedding_dim=embedding_dim,
            reformer_kwargs=enc_reformer_kwargs,
            prenet_kwargs=enc_prenet_kwargs,
        )
        self.dec = Decoder(
            num_mel_coeffs=num_mel_coeffs,
            embedding_dim=embedding_dim,
            prenet_kwargs=dec_prenet_kwargs,
            reformer_kwargs=dec_reformer_kwargs,
            postnet_kwargs=postnet_kwargs,
        )

    def forward(self, text, dec_input):
        keys = self.enc(text)
        return self.dec(dec_input, keys=keys)
