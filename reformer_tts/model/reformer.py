import torch
from torch import nn

from reformer_pytorch import LSHSelfAttention

from .modules import FeedForward
from .reversible import ReversibleSequence, ReversibleBlock, ReversibleHalfResidual, ReversibleSwap
from .config import FeedForwardConfig, LSHSelfAttentionConfig


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
            attn_kwargs: LSHSelfAttention,
            ff_kwargs: FeedForwardConfig,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        blocks = []
        norm_type = nn.LayerNorm

        print(depth)
        for _ in range(depth):
            self_attn = LSHSelfAttention(dim, causal=False, **attn_kwargs)
            ff = FeedForward(dim, **ff_kwargs)

            normed_self_attn = WithNorm(norm_type, dim, self_attn)
            normed_ff = WithNorm(norm_type, dim, ff)

            if ff_chunks > 1:
                normed_ff = Chunk(ff_chunks, normed_ff, along_dim=-2)

            blocks.append(ReversibleBlock(f=normed_self_attn, g=normed_ff))

        self.layers = ReversibleSequence(nn.ModuleList(blocks))

    def forward(self, x, list_kwargs=None):
        x = torch.cat([x, x], dim=-1)
        x = self.layers(x, list_kwargs=list_kwargs)
        return torch.stack(x.chunk(2, dim=-1)).sum(dim=0)


# Inspired by EncoderDecoderBlock from Trax
# https://github.com/google/trax/blob/19777b29668cc92850c593f2c6a8dbf90f951565/trax/models/reformer/reformer.py#L1002
class ReformerDec(nn.Module):
    def __init__(
            self,
            dim: int,
            depth: int,
            ff_chunks: int,
            attn_kwargs: LSHSelfAttentionConfig,
            self_attn_kwargs: LSHSelfAttentionConfig,
            ff_kwargs: FeedForwardConfig,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        blocks = []
        norm_type = nn.LayerNorm

        for _ in range(depth):
            self_attn = LSHSelfAttention(dim, causal=True, **self_attn_kwargs)
            attn = LSHSelfAttention(dim, causal=False, **attn_kwargs)
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

    def forward(self, x, keys, list_kwargs=None):
        x = torch.cat([x, x], dim=-1)

        if list_kwargs is not None:
            assert len(list_kwargs) == self.block_len * self.depth, "list_kwargs should be the length of ReversibleSequence"
        else:
            list_kwargs = [{}] * (6 * self.depth)

        for kwargs in list_kwargs[2::6]:  # get all mid_attention kwargs
            kwargs["keys"] = keys

        x = self.layers(x, list_kwargs=list_kwargs)
        return torch.stack(x.chunk(2, dim=-1)).sum(dim=0)
