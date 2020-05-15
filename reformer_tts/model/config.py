from dataclasses import dataclass


@dataclass
class FeedForwardConfig:
    hidden: int = 2048
    dropout: float = 0.


@dataclass
class LSHSelfAttentionConfig:
    heads: int = 8
    bucket_size: int = 64
    n_hashes: int = 8
    add_local_attn_hash: bool = False
    attn_chunks: int = 1
    random_rotations_per_head: bool = False
    attend_across_buckets: bool = True
    allow_duplicate_attention: bool = True
    num_mem_kv: int = 0
    one_value_head: bool = False
    use_full_attn: bool = False
    full_attn_thres: int = None
    return_attn: bool = False
    post_attn_dropout: float = 0.
    dropout: float = 0.

@dataclass
class MultiheadAttentionConfig:
    num_heads: int = 8
    dropout: float = 0.
    bias: bool = True
    add_bias_kv: bool = False
    add_zero_attn: bool = False
    kdim: int = None
    vdim: int = None

@dataclass
class ReformerEncConfig:
    depth: int = 6
    ff_chunks: int = 100
    attn_kwargs: LSHSelfAttentionConfig = LSHSelfAttentionConfig()
    ff_kwargs: FeedForwardConfig = FeedForwardConfig()


@dataclass
class ReformerDecConfig:
    depth: int = 6
    ff_chunks: int = 100
    attn_kwargs: MultiheadAttentionConfig = MultiheadAttentionConfig()
    self_attn_kwargs: LSHSelfAttentionConfig = LSHSelfAttentionConfig()
    ff_kwargs: FeedForwardConfig = FeedForwardConfig()


@dataclass
class EncoderPreNetConfig:
    dropout: float = 0.5


@dataclass
class DecoderPreNetConfig:
    hidden_size: int = 256
    dropout: float = 0.5


@dataclass
class PostConvNetConfig:
    dropout: float = 0.  # 0.1 originally in Transformer-TTS


@dataclass
class ReformerTTSConfig:
    num_mel_coeffs: int
    dict_size: int = 75
    embedding_dim: int = 512
    pad_base: int = 128  # should be divisible by 2x any bucket size from sub-configs
    enc_prenet_kwargs: EncoderPreNetConfig = EncoderPreNetConfig()
    enc_reformer_kwargs: ReformerEncConfig = ReformerEncConfig()
    dec_prenet_kwargs: DecoderPreNetConfig = DecoderPreNetConfig()
    dec_reformer_kwargs: ReformerDecConfig = ReformerDecConfig()
    postnet_kwargs: PostConvNetConfig = PostConvNetConfig()
