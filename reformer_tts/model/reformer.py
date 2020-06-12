from typing import Dict, List, Optional

import torch
from reformer_pytorch import LSHSelfAttention
from torch import nn

from .modules import FeedForward
from .reversible import ReversibleSequence, ReversibleBlock, ReversibleHalfResidual, ReversibleSwap


# This code fragment comes from https://github.com/lucidrains/reformer-pytorch/tree/master repository
################################################################################
def cache_fn(f):
    cache = None

    def cached_fn(*args, **kwargs):
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


class WithNorm(nn.Module):
    def __init__(self, norm_class, dim, fn):
        super().__init__()
        self.norm = norm_class(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim=-1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x):
        chunks = x.chunk(self.chunks, dim=self.dim)
        return torch.cat([self.fn(c) for c in chunks], dim=self.dim)


################################################################################


class ReformerEnc(nn.Module):
    def __init__(
            self,
            dim: int,
            depth: int,
            ff_chunks: int,
            attn_kwargs: Dict,
            ff_kwargs: Dict,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        blocks = []
        norm_type = nn.LayerNorm

        for _ in range(depth):
            self_attn = LSHSelfAttention(dim, causal=False, **attn_kwargs)
            ff = FeedForward(dim, **ff_kwargs)

            normed_self_attn = WithNorm(norm_type, dim, self_attn)
            normed_ff = WithNorm(norm_type, dim, ff)

            if ff_chunks > 1:
                normed_ff = Chunk(ff_chunks, normed_ff, along_dim=-2)

            blocks.append(ReversibleBlock(f=normed_self_attn, g=normed_ff))

        self.layers = ReversibleSequence(nn.ModuleList(blocks))

    def forward(self, x, input_mask=None, kwargs_list=None):
        x = torch.cat([x, x], dim=-1)

        if kwargs_list is not None:
            assert len(kwargs_list) == self.depth, "list_kwargs should be the length of ReversibleSequence"
        else:
            kwargs_list = [dict() for _ in range(self.depth)]

        for kwargs in kwargs_list:  # get all self_attention layers kwargs
            kwargs["f_args"] = {"input_mask": input_mask}

        x = self.layers(x, kwargs_list=kwargs_list)
        return torch.stack(x.chunk(2, dim=-1)).sum(dim=0)


# Inspired by EncoderDecoderBlock from Trax
# https://github.com/google/trax/blob/19777b29668cc92850c593f2c6a8dbf90f951565/trax/models/reformer/reformer.py#L1002
class ReformerDec(nn.Module):
    def __init__(
            self,
            dim: int,
            depth: int,
            ff_chunks: int,
            attn_kwargs: Dict,
            self_attn_kwargs: Dict,
            ff_kwargs: Dict,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.attention_matrices_ = []

        blocks = []
        norm_type = nn.LayerNorm

        for i in range(depth):
            self_attn = LSHSelfAttention(dim, causal=True, **self_attn_kwargs)
            attn = MultiheadAttentionWrapper(dim, self.attention_matrices_, **attn_kwargs)
            # causal has to be false because context is appended to input sequence
            ff = FeedForward(dim, **ff_kwargs)

            normed_self_attn = WithNorm(norm_type, dim, self_attn)
            normed_attn = WithNorm(norm_type, dim, attn)
            normed_ff = WithNorm(norm_type, dim, ff)

            if ff_chunks > 1:
                normed_ff = Chunk(ff_chunks, normed_ff, along_dim=-2)

            blocks += [ReversibleHalfResidual(normed_self_attn),
                       ReversibleSwap(),
                       ReversibleHalfResidual(normed_attn),
                       ReversibleSwap(),
                       ReversibleHalfResidual(normed_ff),
                       ReversibleSwap()]

        self.block_len = 6
        self.layers = ReversibleSequence(nn.ModuleList(blocks))

    def forward(self, x, keys, key_padding_mask=None, input_mask=None, kwargs_list=None):
        x = torch.cat([x, x], dim=-1)

        if kwargs_list is not None:
            assert len(kwargs_list) == self.block_len * self.depth, "list_kwargs should be the length of ReversibleSequence"
        else:
            kwargs_list = [dict() for _ in range(self.block_len * self.depth)]

        for kwargs in kwargs_list[2::6]:  # get all mid_attention kwargs
            kwargs["key"] = keys
            kwargs["value"] = keys
            kwargs["key_padding_mask"] = key_padding_mask

        for kwargs in kwargs_list[::6]:  # get all self_attention layers kwargs
            kwargs["input_mask"] = input_mask

        self.attention_matrices_.clear()
        x = self.layers(x, kwargs_list=kwargs_list)
        output = torch.stack(x.chunk(2, dim=-1)).sum(dim=0)
        return output, self.attention_matrices_


class MultiheadAttentionWrapper(nn.Module):
    """This class is introduced to change the order of forward parameters to be
     compatible with custom reversed framework
     """
    def __init__(self, dim: int, attention_matrices: Optional[List[torch.Tensor]] = None, **kwargs):
        super().__init__()
        self.layer = nn.MultiheadAttention(dim, **kwargs)
        self.attention_matrices_ = attention_matrices

    def forward(self, query, **kwargs):
        """
        query: tensor of shape (batch_size, spec_seq_length, embed_dim)
        kwargs['key']: tensor of shape (batch_size, text_seq_length, embed_dim)
        returns: tensor of shape (batch_size, spec_seq_length, embed_dim
        """
        assert 'key' in kwargs, "forward expects keyword argument 'key'"

        query = query.transpose(1, 0)
        key = kwargs['key'].transpose(1, 0)
        value = kwargs['key'].transpose(1, 0)
        kwargs = {k: v for k, v in kwargs.items() if k != "key" and k != "value"}
        attn_out, attention_matrix = self.layer.forward(query, key, value, **kwargs)
        if not self.training and self.attention_matrices_ is not None:
            self.attention_matrices_.append(attention_matrix)
        attn_out = attn_out.transpose(1, 0)
        return attn_out
