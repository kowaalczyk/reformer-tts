from collections import OrderedDict

import numpy as np
import torch
from torch import nn


class EncoderPreNet(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int = 512,
            dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.projection = nn.Linear(embedding_dim, embedding_dim)

        def _get_conv():
            return nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=embedding_dim,
                kernel_size=5,
                padding=5 // 2
            )

        def _get_batchnorm():
            return nn.BatchNorm1d(embedding_dim)

        def _get_dropout():
            return nn.Dropout(p=dropout)

        self.convolutions = nn.Sequential(OrderedDict([
            ("dropout0", _get_dropout()),
            ("conv1", _get_conv()),
            ("bn1", _get_batchnorm()),
            ("relu1", nn.ReLU()),
            ("dropout1", _get_dropout()),
            ("conv2", _get_conv()),
            ("bn2", _get_batchnorm()),
            ("relu2", nn.ReLU()),
            ("dropout2", _get_dropout()),
            ("conv3", _get_conv()),
            ("bn3", _get_batchnorm()),
            ("relu3", nn.ReLU()),
            ("dropout3", _get_dropout()),
        ]))

    def forward(self, input_):
        """
        :param input_: of shape (batch_size, sequence_length)
        :return: of shape (batch_size, sequence_length, self.embedding_dim)
        """
        input_ = self.embed(input_)
        input_ = input_.transpose(1, 2)
        input_ = self.convolutions(input_)
        input_ = input_.transpose(1, 2)
        input_ = self.projection(input_)

        return input_


class DecoderPreNet(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            hidden_size: int = 256,
            dropout: float = 0.5,
    ):
        """
        :param input_size: dimension of input
        :param hidden_size: dimension of hidden unit
        :param output_size: dimension of output
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        # Droput layers are not described in Transformer-TTS paper, but in Tacotron 2
        self.layer = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.input_size, self.hidden_size)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(dropout)),
            ('fc2', nn.Linear(self.hidden_size, self.output_size)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(dropout)),
            ('projection', nn.Linear(self.output_size, self.output_size)),
        ]))

    def forward(self, input_):
        """
        :param input_: of shape (batch_size, sequence_length, input_size) where input_size is num_of_mel_features
        :return: (batch_size, sequence_length, output_size) where output_size should be size of scaled positional embedding
        """

        out = self.layer(input_)

        return out


class PostConvNet(nn.Module):
    def __init__(
            self,
            mel_size: int,
            num_hidden: int,
            dropout: float,
            depth: int,
    ):
        """
        :param mel_size: number of mel spectrogram coefficients
        :param num_hidden: dimension of hidden
        """
        super().__init__()
        self.mel_size = mel_size

        def _get_conv():
            return nn.Conv1d(
                in_channels=num_hidden,
                out_channels=num_hidden,
                kernel_size=5,
                padding=int(np.floor(5 / 2)),
            )

        def _get_batch_norm():
            return nn.BatchNorm1d(num_hidden)

        def _get_conv0():
            return nn.Conv1d(
                in_channels=mel_size,
                out_channels=num_hidden,
                kernel_size=5,
                padding=int(np.floor(5 / 2)),
            )

        def _get_conv_end():
            return nn.Conv1d(
                in_channels=num_hidden,
                out_channels=mel_size,
                kernel_size=5,
                padding=int(np.floor(5 / 2)),
            )

        # Dropout is not described in paper, but it was implemented in found Transformer-TTS repo. To consider
        def _get_layer(i: int):
            conv = _get_conv0() if i==0 else _get_conv()
            return [
                (f'conv{i}', conv),
                (f'bn{i}', _get_batch_norm()),
                (f'tanh{i}', nn.Tanh()),
                (f'dropout{i}', nn.Dropout(dropout)),
            ]

        layers = list()
        for i in range(depth):
            layers += _get_layer(i)
        layers += [('convend', _get_conv_end())]
        self.layers = nn.Sequential(OrderedDict(layers))

    def forward(self, input_):
        """
        :param input_: of shape (batch_size, sequence_length, self.mel_size)
        :return: of shape (batch_size, sequence_length, self.mel_size)
        """
        input_ = input_.transpose(1, 2)
        input_ = self.layers(input_)
        input_ = input_.transpose(1, 2)
        return input_


class ScaledPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        self.alpha = nn.Parameter(torch.empty(1).normal_(0, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_):
        """
        :param input_: of shape (batch_size, sequence_length, d_model)
        :return: of shape (batch_size, sequence_length, d_model)
        """
        pos = torch.arange(input_.shape[1], device=input_.device).type(self.inv_freq.type())
        sinusoid_inp = torch.einsum("i,j->ij", pos, self.inv_freq)
        s = sinusoid_inp.sin().view(sinusoid_inp.shape[0], sinusoid_inp.shape[1], 1)
        c = sinusoid_inp.cos().view(sinusoid_inp.shape[0], sinusoid_inp.shape[1], 1)
        sc = torch.cat([s, c], dim=2).view(sinusoid_inp.shape[0], 2 * sinusoid_inp.shape[1])
        sc_drop = self.dropout(sc)
        input_ = input_ + self.alpha * sc_drop
        return input_


class FeedForward(nn.Module):
    def __init__(self, dim=512, hidden=2048, dropout=0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)
