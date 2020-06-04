from typing import Dict, Tuple, Optional

import torch
from torch import nn

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

    def forward(self, input_, input_mask=None):
        input_ = self.prenet(input_)
        input_ = self.positional_encoding(input_)
        input_ = self.reformer(input_, input_mask=input_mask)

        return input_


class Decoder(nn.Module):
    def __init__(
            self,
            num_mel_coeffs: int,
            embedding_dim: int,
            scp_encoding_dropout: float,
            prenet_kwargs: Dict,
            reformer_kwargs: Dict,
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

    def forward(self, input_, keys, key_padding_mask=None, input_mask=None):
        input_ = self.prenet(input_)
        input_ = self.positional_encoding(input_)
        input_, attention_matrices = self.reformer(
            input_,
            keys=keys,
            key_padding_mask=key_padding_mask,
            input_mask=input_mask,
        )
        stop = self.stop_linear(input_)
        mel = self.mel_linear(input_)
        return mel, stop, attention_matrices


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
        )
        self.pad_base = pad_base
        self.postnet = PostConvNet(num_mel_coeffs, num_hidden=embedding_dim, **postnet_kwargs)

    def forward(
            self, phonemes: torch.LongTensor, spectrogram: torch.Tensor,
            spectrogram_mask: Optional[torch.ByteTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param phonemes: of shape (batch size, phonemes length)
        :param spectrogram: of shape (batch size, spectrogram length, num_mel_coeffs)
        :return: tuple: (raw mel spectrogram, postnet mel spectrogram, stop vector) where
            raw mel spectrogram shape = postnet mel spectrogram shape = spectrogram shape,
            stop vector shape = (batch size, spectrogram length, 1)
        """
        pad_phonemes = pad_to_multiple(
            phonemes.view((*phonemes.shape, 1)),
            self.pad_base,
        ).view((phonemes.shape[0], -1))

        phoneme_mask = (pad_phonemes != 0).to(dtype=torch.bool, device=spectrogram.device)
        if spectrogram_mask is None:
            spectrogram_mask = torch.ones(spectrogram.shape[:2], device=spectrogram.device)
        spectrogram_mask = pad_to_multiple(
            spectrogram_mask.view((*spectrogram_mask.shape, 1)),
            self.pad_base,
        ).view((spectrogram_mask.shape[0], -1)).to(dtype=torch.bool, device=spectrogram.device)

        pad_spec = pad_to_multiple(spectrogram, self.pad_base)
        keys = self.enc(
            pad_phonemes,
            input_mask=phoneme_mask,
        )
        mel, stop, attention_matrices = self.dec(
            pad_spec,
            keys=keys,
            key_padding_mask=~phoneme_mask,
            input_mask=spectrogram_mask,
        )

        residual = self.postnet(mel)
        mel_postnet = mel + residual

        cutoff = spectrogram.shape[1]
        return mel[:, :cutoff], mel_postnet[:, :cutoff], stop[:, :cutoff], attention_matrices

    def infer(
            self, phonemes: torch.LongTensor, combine_strategy: str = "concat",
            max_len: int = 1024, stop_threshold: float = 0.25, verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Run inference loop to generate a new spectrogram

        :param phonemes: of shape (batch size, phonemes length)
        :param combine_strategy: "concat" to concatenate outputs from consecutive iterations,
            "replace" to replace previous output using output from current iteration
        :param max_len: maximum length of generated spectrogram
        :param stop_threshold: value in (-1 , 1) above which sigmoid(stop_pred)
            is consirered to indicate end of predicted sequence
        :param verbose: if true, prints progress info every 10 steps (useful for cpu)
        :return: tuple (spectrogram, stop_idx) where
            spectrogram shape = (batch size, num_mel_coeffs, spectrogram length)
            stop_idx shape = (batch size), contains end index for each spectrogram in batch
        """
        assert combine_strategy in {"concat", "replace"}
        assert -1. < stop_threshold < 1.

        batch_size = phonemes.shape[0]
        zeros = torch.zeros((batch_size, 1, self.num_mel_coeffs))
        spectrogram = zeros.clone()
        stop = torch.zeros(batch_size, dtype=torch.long)

        while not torch.all(stop > 0):
            iteration = spectrogram.shape[1]
            still_running = stop == 0
            if verbose and iteration % 10 == 0:
                number_of_running_samples = sum(still_running)
                print(f"reached {iteration=}, {number_of_running_samples=}...")

            _, generated, stop_pred, _ = self.forward(phonemes, spectrogram)
            stop_pred = stop_pred.view(-1, generated.shape[1])  # view as (batch, len)

            if combine_strategy == "concat":
                generated_slice = generated[:, -1, :].view(batch_size, 1, self.num_mel_coeffs)
                spectrogram = torch.cat([spectrogram, generated_slice], dim=1)

                stops_now = torch.sigmoid(stop_pred[:, -1]) > stop_threshold
                still_running_stops_now = still_running * stops_now
                stop = torch.where(
                    still_running_stops_now,
                    (stops_now.to(dtype=torch.long) * iteration) + 1,
                    stop
                )
            elif combine_strategy == "replace":
                spectrogram = torch.cat([zeros, generated], dim=1)

                stops_now = torch.any(
                    torch.sigmoid(stop_pred) > stop_threshold, dim=1
                )
                still_running_stops_now = stops_now * still_running
                stop = torch.where(
                    still_running_stops_now,
                    torch.argmax(stop_pred, dim=1) + 1,
                    stop
                )

            if max(spectrogram.shape) > max_len:
                if verbose:
                    print(f"stopped at {max_len=}")
                break

        stop_at_end = torch.ones_like(stop, dtype=torch.long) * max_len
        stop: torch.LongTensor = torch.where(stop == 0, stop_at_end, stop)
        spectrogram: torch.Tensor = spectrogram.transpose(1, 2)
        return spectrogram[:, :, 1:], stop


def pad_to_multiple(tensor, pad_base):
    """
    This function pads the sequence with zeros to be divisible by pad_base.
    Assumed tensor is of shape (batch, seq_len, channels)
    """
    new_len = ((tensor.shape[1] - 1) // pad_base + 1) * pad_base
    # one is subtracted for the case when tensor.shape[1] % pad_base == 0
    padding = torch.zeros(tensor.shape[0], new_len - tensor.shape[1], tensor.shape[2], dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=1)
