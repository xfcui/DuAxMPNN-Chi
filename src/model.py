"""GNN model definitions for PCQM4Mv2: embeddings, conv/virt/depth/head kernels, and TriAxMPNN."""

import jax
import numpy as np
import jax.numpy as jnp
import equinox as eqx
from jax.ops import segment_sum

# dataset.py feature dimensions
# 10 atom features, 6/2/3/4 hop bond features; node_embd has 17 floats in HDF5, model uses 14 (RWPE12 + EN/GC)
from dataset import (
    NODE_FEAT_VOCAB_SIZES,
    NODE_FEAT_TOTAL_VOCAB,
    EDGE_FEAT_VOCAB_SIZES,
    EDGE_FEAT_TOTAL_VOCAB,
)

DROPOUT         = 0.1
EPSILON         = 1e-6
MINMAX_RATIO    = 20**.5
WIDTH_ACT_SCALE = 4
VERBOSE         = False  # set True to print parameter counts and per-forward kernel stats


def _log(*args, **kwargs) -> None:
    """Print when :data:`VERBOSE` is enabled."""
    if VERBOSE:
        msg = " ".join(map(str, args))
        try:
            from tqdm import tqdm
            tqdm.write(msg)
        except ImportError:
            print(*args, **kwargs)

EMBED_POS  = 12    # RWPE12 only (ignore coord/en/geom auxiliaries)
COORD_DIM  = 3     # 3D coords occupy node_embd[:, 12:15] (train-only, unread as raw input)
EMBED_ELEC = 2
# Explicit continuous-feature column layout in node_embd:
#   [RWPE(0:12), coord(12:15), en(15), gc(16), chiral_sign(17), chiral_vol(18)]
ELEC_LO = EMBED_POS + COORD_DIM          # 15
ELEC_HI = ELEC_LO + EMBED_ELEC           # 17
CHIRAL_SIGN_COL = ELEC_HI                # 17: numbering-invariant CIP R/S in {-1,0,1}
CHIRAL_VOL_COL = ELEC_HI + 1             # 18: train-only signed tetra volume (pseudoscalar)
EMBED_CHIRAL = 1
EDGE_SUFFIXES = list(EDGE_FEAT_VOCAB_SIZES.keys())
EDGE_DIMS_PER_HOP = [
    (EDGE_FEAT_TOTAL_VOCAB[suffix], len(EDGE_FEAT_VOCAB_SIZES[suffix]))
    for suffix in EDGE_SUFFIXES
]

# 1-hop edge storage: graph.py appends neighbor rank after bond_features (..., ring, rank) → rank at column 5.
_ONE_HOP_FEAT_SIZES = list(EDGE_FEAT_VOCAB_SIZES[""])
_ONE_HOP_OFFSETS = np.asarray(
    [1] + list(np.cumsum(_ONE_HOP_FEAT_SIZES[:-1], dtype=np.int32)),
    dtype=np.int32,
)
NEIGHBOR_RANK_EDGE_COL = 5
NEIGHBOR_RANK_NUM = int(_ONE_HOP_FEAT_SIZES[NEIGHBOR_RANK_EDGE_COL])
NEUTRAL_ONEHOP_RANK_COL_TOKEN = int(_ONE_HOP_OFFSETS[NEIGHBOR_RANK_EDGE_COL])
MAX_HOPS = 4
CHIRAL_AUX_WEIGHT = 0.1
CHIRAL_SIGN_DROPOUT = 0.2
CHIRAL_3D_PROB = 0.5
RBF_MAX_VALUE = 2.5


def _split_or_none(key, num):
    """Return a list of subkeys, or ``None`` if no key is provided."""
    if num == 0:
        return []
    if key is None:
        return [None] * num
    return list(jax.random.split(key, num))

def _inverse_softplus(y):
    """NumPy inverse softplus: ``x`` such that ``log(1 + exp(x)) == y`` (``y > 0``)."""
    if y > 20.0:
        return y
    return np.log(np.expm1(y))

def _count_params(model: eqx.Module) -> int:
    """Count the number of parameters in an Equinox module."""
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))

def _clip_with_grad(x, min, max):
    """Clamp ``x`` to ``[min, max]`` while passing gradients only where unclipped."""
    return jax.lax.stop_gradient(x.clip(min, max) - x) + x


class EmbedLayer(eqx.Module):
    """Memory-efficient multi-feature embedding unit."""
    embeddings: jnp.ndarray

    def __init__(self, total_vocab, num_features, width, key, *, init_std=1.0):
        total_dim = int(total_vocab)
        std = init_std / np.sqrt(num_features)
        self.embeddings = jax.random.normal(key, (total_dim, width)) * std

    def __call__(self, x):
        """Sum embedding lookups over the feature dimension."""
        return jnp.sum(self.embeddings[x], axis=-2)

# ReZero: https://arxiv.org/abs/2003.04887
# LayerScale: https://arxiv.org/abs/2103.17239
class ScaleLayer(eqx.Module):
    """Learnable per-channel log-scale; forward multiplies by ``exp(scale)``."""
    scale: jnp.ndarray

    def __init__(self, width, scale_init=1.0):
        self.scale = jnp.full((width,), np.log(scale_init), dtype=jnp.float32)

    def __call__(self, x):
        s = jnp.exp(self.scale)
        s = _clip_with_grad(s, 1e-2, 1)
        return s * x

class LinearLayer(eqx.Module):
    """Bias-free linear projection with configurable init scale."""
    kernel: jnp.ndarray

    def __init__(self, width_in, width_out, key, *, init_std=1.0):
        std = init_std / np.sqrt(width_in)
        self.kernel = jax.random.normal(key, (width_in, width_out)) * std

    def __call__(self, x):
        return x @ self.kernel

class ActLayer(eqx.Module):
    """Softplus activation with learnable bias shift and optional inverted dropout."""
    bias: jnp.ndarray

    def __init__(self, width):
        self.bias = jnp.full((width,), _inverse_softplus(0.5), dtype=jnp.float32)

    def __call__(self, x, key=None):
        """Apply activation, clip output range, and drop units when ``key`` is given."""
        xx = jax.nn.softplus(x + self.bias)
        xx = _clip_with_grad(xx, 1/MINMAX_RATIO, MINMAX_RATIO)
        if key is not None and DROPOUT > 0.0:
            keep = 1.0 - DROPOUT
            mask = jax.random.bernoulli(key, p=keep, shape=xx.shape)
            xx = xx * mask.astype(xx.dtype) / keep
        return xx


class GroupLinearBlock(eqx.Module):
    """Groups-parallel linear: weight ``(num_head, d_in, d_out)``, no bias."""
    num_head: int
    dim_head: int
    linear: LinearLayer
    kernel: jnp.ndarray

    def __init__(self, width_in, width_out, num_head, dim_head, key):
        width_norm = num_head * dim_head
        keys = _split_or_none(key, 2)

        self.num_head = num_head
        self.dim_head = dim_head
        self.linear = LinearLayer(width_in, width_norm, keys[0])
        assert width_out % num_head == 0
        self.kernel = jax.random.normal(keys[1], (num_head, dim_head, width_out // num_head)) / np.sqrt(dim_head)

    def __call__(self, x, norm_bias=None):
        """Project per-head with optional grouped normalization bias."""
        xx = self.linear(x)
        xx = xx.reshape(*x.shape[:-1], self.num_head, -1)
        xx = xx / jnp.sqrt(jnp.mean(jnp.square(xx), axis=-1, keepdims=True) + EPSILON)
        if norm_bias is not None:
            xx = xx + norm_bias.reshape(xx.shape)
        xx = jnp.einsum("...hd,hdf->...hf", xx, self.kernel)
        xx = xx.reshape(*x.shape[:-1], -1)
        return xx

# GLU: https://arxiv.org/abs/1612.08083
# GLU-variants: https://arxiv.org/abs/2002.05202
class GatedLinearBlock(eqx.Module):
    """Gated linear unit with group-linear and optional post-projection."""
    num_head: int
    dim_head: int
    lin_gat:  GroupLinearBlock
    lin_val:  GroupLinearBlock
    act_gat:  ActLayer
    kernel:   jnp.ndarray
    lin_out:  LinearLayer | None

    def __init__(self, width_in, width_out, num_head, dim_head, keep_groups=False, key=None, *, name=None):
        width_norm = num_head * dim_head
        width_act  = num_head * dim_head * WIDTH_ACT_SCALE
        keys = _split_or_none(key, 3 if keep_groups else 4)

        self.num_head = num_head
        self.dim_head = dim_head
        self.lin_gat  = GroupLinearBlock(width_in, width_act, num_head, dim_head, keys[0])
        self.lin_val = GroupLinearBlock(width_in, width_act, num_head, dim_head, keys[1])
        self.act_gat = ActLayer(width_act)
        if keep_groups:
            assert width_out % num_head == 0
            self.kernel = jax.random.normal(keys[2], (num_head, dim_head * WIDTH_ACT_SCALE, width_out // num_head)) / np.sqrt(dim_head * WIDTH_ACT_SCALE)
            self.lin_out = None
        else:
            self.kernel = jax.random.normal(keys[2], (num_head, dim_head * WIDTH_ACT_SCALE, dim_head)) / np.sqrt(dim_head * WIDTH_ACT_SCALE)
            self.lin_out = LinearLayer(width_norm, width_out, keys[3])

        if name is not None:
            _log(f"##params[{name}]:", _count_params(self))

    def __call__(self, x, y=None, gate_bias=None, value_bias=None, key=None):
        """Run gate/value paths and combine them by elementwise product."""
        y = x if y is None else y

        gg = self.lin_gat(x, gate_bias)
        vv = self.lin_val(y, value_bias)
        xx = self.act_gat(gg, key) * vv
        xx = xx.reshape(*x.shape[:-1], self.num_head, -1)
        xx = jnp.einsum("...hd,hdf->...hf", xx, self.kernel)
        xx = xx.reshape(*x.shape[:-1], -1)
        if self.lin_out is not None:
            xx = self.lin_out(xx)
        return xx


class FloatEmbedLayer(eqx.Module):
    """RBF expansion + linear head → ``dim_head`` (production ``rbf_linear``)."""

    num_channels: int = eqx.field(static=True)
    rbf_k: int = eqx.field(static=True)
    rbf_mu: jnp.ndarray
    rbf_log_gamma: jnp.ndarray
    head_linear: LinearLayer

    def __init__(self, num_channels, dim_head, key):
        self.num_channels = int(num_channels)
        dim_head = int(dim_head)
        self.rbf_k = dim_head
        keys = _split_or_none(key, 2)
        mu = np.linspace(-RBF_MAX_VALUE, RBF_MAX_VALUE, dim_head, dtype=np.float32)
        self.rbf_mu = jnp.array(np.tile(mu, (self.num_channels, 1)), dtype=jnp.float32)
        spacing = 2 * RBF_MAX_VALUE / max(dim_head - 1, 1)
        val = 1.0 / (spacing**2)
        self.rbf_log_gamma = jnp.full((self.num_channels,), _inverse_softplus(val), dtype=jnp.float32)
        encode_dim = self.num_channels * dim_head
        self.head_linear = LinearLayer(encode_dim, dim_head, keys[1], init_std=0.1)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        mu = self.rbf_mu
        gamma = jax.nn.softplus(self.rbf_log_gamma)
        phi = jnp.exp(-gamma[:, None] * (x[..., None] - mu) ** 2)
        feat = phi.reshape(*x.shape[:-1], self.num_channels * self.rbf_k)
        return self.head_linear(feat)


# VoVNet: https://arxiv.org/abs/1904.09730
# GNN-AK: https://openreview.net/forum?id=Mspk_WYKoEH
class ConvKernel(eqx.Module):
    """Bond-aware graph convolution with degree normalisation."""
    use_chiral: bool = eqx.field(static=True)
    lora_down:  jnp.ndarray
    lora_up:    jnp.ndarray
    embed_edge: EmbedLayer
    embed_elec: FloatEmbedLayer
    embed_deg:  EmbedLayer
    lin_pre:    LinearLayer
    act_out:    GatedLinearBlock
    rank_phase: EmbedLayer | None
    lin_chiral_down: LinearLayer | None
    lin_chiral_up:   LinearLayer | None

    def __init__(self, num_head, dim_head, edge_total_vocab, edge_num_features, key=None, *, use_chiral: bool = False):
        width_norm = num_head * dim_head
        n_keys = 6 + (3 if use_chiral else 0)
        keys = _split_or_none(key, n_keys)
        self.use_chiral = bool(use_chiral)

        self.lora_down  = jax.random.normal(keys[4], (width_norm, dim_head)) / np.sqrt(width_norm)
        self.lora_up    = jnp.zeros((dim_head, width_norm), dtype=jnp.float32)
        self.embed_edge = EmbedLayer(edge_total_vocab, edge_num_features, dim_head, keys[0], init_std=0.1)
        self.embed_elec = FloatEmbedLayer(2, dim_head, keys[5])
        self.embed_deg  = EmbedLayer(6, 1, width_norm, keys[1], init_std=0.1)
        self.lin_pre    = LinearLayer(width_norm, width_norm, keys[2])
        self.act_out    = GatedLinearBlock(width_norm, width_norm, num_head, dim_head, keep_groups=True, key=keys[3])

        if use_chiral:
            kr = keys[6:9]
            self.rank_phase = EmbedLayer(NEIGHBOR_RANK_NUM, 1, dim_head, kr[0], init_std=0.1)
            self.lin_chiral_down = LinearLayer(width_norm, dim_head, kr[1])
            up = LinearLayer(dim_head, width_norm, kr[2])
            self.lin_chiral_up = eqx.tree_at(lambda l: l.kernel, up, jnp.zeros_like(up.kernel))
        else:
            self.rank_phase = None
            self.lin_chiral_down = None
            self.lin_chiral_up = None

        _log(f"##params[conv]:", _count_params(self), edge_total_vocab, edge_num_features)

    def __call__(self, x, deg, edge_idx, edge_attr, node_elec, chiral_sign=None, gated_chiral_3d=None, key=None):
        """Aggregate neighbor messages weighted by edge and degree embeddings."""
        bond_emb = self.embed_edge(edge_attr)
        src, dst = node_elec[edge_idx[0]], node_elec[edge_idx[1]]
        edge_emb = bond_emb + self.embed_elec(src - dst)

        msg = x[edge_idx[0]] + x[edge_idx[1]]
        msg = self.lin_pre(msg) + (msg @ self.lora_down * edge_emb) @ self.lora_up
        msg = segment_sum(msg, edge_idx[1], len(x))
        if self.use_chiral and (chiral_sign is not None or gated_chiral_3d is not None):
            msg = msg + self._chiral_message(x, edge_idx, edge_attr, chiral_sign, gated_chiral_3d)
        msg = self.act_out(msg, gate_bias=self.embed_deg(deg[:, None]), key=key)
        return msg

    def _chiral_message(self, x, edge_idx, edge_attr, chiral_sign, gated_chiral_3d):
        """Reflection-antisymmetric message: sign[dst] * rank_phase(rank) * value(src).

        The per-centre CIP sign flips between enantiomers, the rank phase makes
        the contribution order-sensitive, and the product (rather than an
        additive offset through a symmetric sum) is what lets the aggregated
        message change between mirror images.
        """
        rank_token = edge_attr[:, NEIGHBOR_RANK_EDGE_COL] - NEUTRAL_ONEHOP_RANK_COL_TOKEN
        rank_token = jnp.clip(rank_token, 0, NEIGHBOR_RANK_NUM - 1)
        phase = self.rank_phase(rank_token[:, None])           # (E, dim_head)
        value = self.lin_chiral_down(x[edge_idx[0]])           # (E, dim_head)
        
        sign_dst = 0.0
        if chiral_sign is not None:
            sign_dst = sign_dst + chiral_sign[edge_idx[1]].reshape(-1, 1)
        if gated_chiral_3d is not None:
            sign_dst = sign_dst + gated_chiral_3d[edge_idx[1]].reshape(-1, 1)

        chiral_msg = sign_dst * phase * value                  # (E, dim_head)
        agg = segment_sum(chiral_msg, edge_idx[1], len(x))     # (N, dim_head)
        return self.lin_chiral_up(agg)                         # (N, width_norm)

# GIN-virtual: https://arxiv.org/abs/2103.09430
class VirtKernel(eqx.Module):
    """Virtual node aggregation for global information exchange."""
    num_head: int
    act_out:  GatedLinearBlock

    def __init__(self, num_head, dim_head, key=None):
        width_norm = num_head * dim_head
        keys = _split_or_none(key, 2)

        self.num_head = num_head
        self.act_out  = GatedLinearBlock(width_norm, width_norm, num_head, dim_head, keep_groups=True, key=keys[1])

        _log("##params[virt]:", _count_params(self))

    def __call__(self, x, batch, batch_size, key=None):
        """Pool node features to graph-level, transform, and broadcast update to each node."""
        msg = segment_sum(x, batch, batch_size)
        msg = self.act_out(msg, key=key)[batch]
        return msg


class SelfMixerKernel(eqx.Module):
    """Position-wise residual self-mixing block."""
    num_head: int = eqx.field(static=True)
    dim_head: int = eqx.field(static=True)
    act: GatedLinearBlock
    sca: ScaleLayer

    def __init__(self, width, num_head, dim_head, key=None):
        self.num_head = num_head
        self.dim_head = dim_head
        self.act = GatedLinearBlock(width, width, num_head, dim_head, key=key)
        self.sca = ScaleLayer(width, scale_init=1.0)

        _log("##params[self_mixer]:", _count_params(self))

    def __call__(self, x, key=None):
        return self.sca(x) + self.act(x, key=key)

class LayerMixerKernel(eqx.Module):
    """Mixes k-hop convolutions and virtual node information."""
    num_head: int = eqx.field(static=True)
    dim_head: int = eqx.field(static=True)
    lin_pre:  GroupLinearBlock
    act_virt: VirtKernel
    lin_out:  LinearLayer
    sca_out:  ScaleLayer
    conv: tuple

    def __init__(self, width, num_head, dim_head, edge_dims_per_hop, key=None):
        num_hops = len(edge_dims_per_hop)
        width_norm = num_head * dim_head
        keys = _split_or_none(key, num_hops + 3)

        self.num_head = num_head
        self.dim_head = dim_head
        self.lin_pre  = GroupLinearBlock(width, width_norm, num_head, dim_head, keys[0])
        self.conv = tuple(
            ConvKernel(
                num_head,
                dim_head,
                edge_dims_per_hop[i][0],
                edge_dims_per_hop[i][1],
                key=keys[i + 3],
                use_chiral=(i == 0),
            )
            for i in range(num_hops)
        )
        self.act_virt = VirtKernel(num_head, dim_head, key=keys[1])
        self.lin_out = LinearLayer(width_norm, width, keys[2], init_std=1/(1 + (num_hops + 1) * .75**2)**.5)
        self.sca_out = ScaleLayer(width, scale_init=1.0)

        _log("##params[layer_mixer]:", _count_params(self))

    def __call__(self, x, edges, batch, batch_size, node_elec, chiral_sign=None, gated_chiral_3d=None, key=None):
        """Run multi-hop convolutions, mix with virtual-node broadcast, then residual-add."""
        keys = _split_or_none(key, len(self.conv) + 1)

        xx = self.lin_pre(x)
        for i, (conv, (idx, attr, deg)) in enumerate(zip(self.conv, edges)):
            if i == 0:
                xx = xx + conv(xx, deg, idx, attr, node_elec, chiral_sign=chiral_sign, gated_chiral_3d=gated_chiral_3d, key=keys[i])
            else:
                xx = xx + conv(xx, deg, idx, attr, node_elec, key=keys[i])
        xx = xx + self.act_virt(xx, batch, batch_size, key=keys[-1])[batch]
        xx = xx + self.sca_out(x) + self.lin_out(xx)
        return xx

class DepthMixerKernel(eqx.Module):
    """Cross-layer dense aggregation: projects all prior layer outputs and gates them into the current one."""
    num_head: int
    dim_head: int
    kernel:   jnp.ndarray | None
    act_out:  GatedLinearBlock

    def __init__(self, depth, width, num_head, dim_head, key=None):
        width_neck = dim_head * num_head // 4
        width_cat  = width + width_neck * depth
        keys = _split_or_none(key, 2)

        self.num_head = num_head
        self.dim_head = dim_head
        if depth > 0:
            self.kernel = jax.random.normal(keys[0], (depth, width, width_neck)) / np.sqrt(width)
        else:
            self.kernel = None
        self.act_out = GatedLinearBlock(width_cat, width, num_head, dim_head, key=keys[1])

        _log(f"##params[depth_mixer]:", _count_params(self), width_cat)

    def __call__(self, x, x_lst, key=None):
        """Gate current features against the sum of all previous layer outputs."""
        if self.kernel is None:
            xx = x
        else:
            xx = jnp.concatenate(x_lst, axis=-1)
            xx = xx.reshape(*x.shape[:-1], -1, x.shape[-1])
            xx = jnp.einsum("...hd,hdf->...hf", xx, self.kernel)
            xx = xx.reshape(*x.shape[:-1], -1)
            xx = jnp.concatenate([x, xx], axis=-1)
        xx = self.act_out(xx, key=key)
        return xx


class HeadKernel(eqx.Module):
    """Readout head: virtual + node pooling → scalar prediction."""
    num_head: int
    dim_head: int
    kernel:   jnp.ndarray
    act_out:  GatedLinearBlock
    readout_scale: jnp.ndarray
    readout_bias:  jnp.ndarray

    def __init__(self, depth, width, num_head, dim_head, key=None):
        width_neck = dim_head * num_head // 4
        width_cat  = width + width_neck * depth
        keys = _split_or_none(key, 3)

        self.num_head = num_head
        self.dim_head = dim_head
        if depth > 0:
            self.kernel = jax.random.normal(keys[0], (depth, width, width_neck)) / np.sqrt(width)
        else:
            self.kernel = jnp.zeros((0, width, width_neck), dtype=jnp.float32)
        self.act_out  = GatedLinearBlock(width_cat, 1, num_head*2, dim_head*2, key=keys[2])
        self.readout_scale = jnp.asarray(1.162127, dtype=jnp.float32)
        self.readout_bias  = jnp.asarray(5.689452, dtype=jnp.float32)

        _log("##params[head]:", _count_params(self), width_cat)

    def __call__(self, x, x_lst, batch, batch_size, key=None):
        """Sum-pool nodes, fuse virtual node, project to scalar, and apply output affine."""
        if self.kernel.shape[0] == 0:
            xx = x
        else:
            xx = jnp.concatenate(x_lst, axis=-1)
            xx = xx.reshape(*x.shape[:-1], -1, x.shape[-1])
            xx = jnp.einsum("...hd,hdf->...hf", xx, self.kernel)
            xx = xx.reshape(*x.shape[:-1], -1)
            xx = jnp.concatenate([xx, x], axis=-1)
        yy = segment_sum(xx, batch, batch_size)
        yy = self.act_out(yy, key=key)
        yy = yy * self.readout_scale + self.readout_bias
        return yy


# GIN: https://openreview.net/forum?id=ryGs6iA5Km
# DenseNet: https://arxiv.org/abs/1608.06993
# AttnRes: https://arxiv.org/abs/2603.15031
class Chiral3DInputEmbed(eqx.Module):
    """Continuous 3D chirality RBF expansion and projection."""
    rbf_mu: jnp.ndarray
    rbf_log_gamma: jnp.ndarray
    linear: LinearLayer

    def __init__(self, key):
        mu = np.linspace(0.0, 1.0, 16, dtype=np.float32)
        self.rbf_mu = jnp.array(mu)
        spacing = 1.0 / 15.0
        val = 1.0 / (spacing**2)
        self.rbf_log_gamma = jnp.array(_inverse_softplus(val))
        self.linear = LinearLayer(16, 1, key, init_std=0.1)

    def __call__(self, vol):
        abs_vol = jnp.abs(vol)
        gamma = jax.nn.softplus(self.rbf_log_gamma)
        phi = jnp.exp(-gamma * (abs_vol - self.rbf_mu) ** 2)
        proj = self.linear(phi)
        sign = jnp.sign(vol)
        return sign * proj


class TriAxMPNN(eqx.Module):
    """TriAxMPNN for PCQM4Mv2: tri-axis message passing with fixed C1 chirality stack."""
    depth: int
    width: int
    num_head: int
    dim_head: int
    chiral_aux_weight: float = eqx.field(static=True)

    atom_embed: EmbedLayer
    atom_pos:   GatedLinearBlock
    chiral_3d_embed: Chiral3DInputEmbed
    layer_mix:  tuple
    depth_mix:  tuple
    head: HeadKernel
    chiral_head: LinearLayer

    def __init__(self, depth, width, num_head, dim_head, key):
        assert key is not None
        assert depth >= 1
        self.chiral_aux_weight = CHIRAL_AUX_WEIGHT

        n_sub = depth * 2 + 3 + 2  # atom embed + pos + chiral_3d + chiral_head + layers + head
        keys = _split_or_none(key, n_sub)

        self.depth = depth
        self.width = width
        self.num_head = num_head
        self.dim_head = dim_head
        edge_dims = EDGE_DIMS_PER_HOP[:MAX_HOPS]
        head_cross_depth = depth - 1

        _log(
            f"#model={self.__class__.__name__}, "
            f"depth={self.depth}, width={self.width}, "
            f"num_head={self.num_head}, dim_head={self.dim_head}, "
            f"max_hops={MAX_HOPS}, depth_mode=dense, cont_embed=rbf_linear, "
            f"elec_mode=edge_diff, edge_fuse=lora, chiral=C1"
        )

        curr = 0
        self.atom_embed = EmbedLayer(NODE_FEAT_TOTAL_VOCAB, len(NODE_FEAT_VOCAB_SIZES), width, keys[curr])
        curr += 1
        self.atom_pos = GatedLinearBlock(EMBED_POS, width, num_head, dim_head, 1, key=keys[curr])
        curr += 1
        self.chiral_3d_embed = Chiral3DInputEmbed(keys[curr])
        curr += 1
        layer_mix: list[LayerMixerKernel] = []
        depth_mix: list[DepthMixerKernel] = []
        for i in range(depth):
            layer_mix.append(
                LayerMixerKernel(width, num_head, dim_head, edge_dims, key=keys[curr])
            )
            curr += 1
            depth_mix.append(DepthMixerKernel(i, width, num_head, dim_head, key=keys[curr]))
            curr += 1
        self.layer_mix = tuple(layer_mix)
        self.depth_mix = tuple(depth_mix)
        self.head = HeadKernel(head_cross_depth, width, num_head, dim_head, key=keys[curr])
        curr += 1
        self.chiral_head = LinearLayer(width, 1, keys[curr], init_std=0.1)

        _log("#params:", _count_params(self))
        _log()

    def __call__(self, batch, training=False, key=None, return_aux=False):
        """Forward pass over a padded batch dict produced by the dataloader."""
        node_feat = batch["node_feat"]
        node_embd_full = batch["node_embd"]
        node_embd = node_embd_full[..., :EMBED_POS]
        node_elec = node_embd_full[..., ELEC_LO:ELEC_HI]
        chiral_sign = None
        if node_embd_full.shape[-1] > CHIRAL_SIGN_COL:
            chiral_sign = node_embd_full[..., CHIRAL_SIGN_COL:CHIRAL_SIGN_COL + 1]
        graph_id = batch["node_batch"]
        batch_size = batch["batch_n_graphs"] + 1
        chiral_vol = None
        if node_embd_full.shape[-1] > CHIRAL_VOL_COL:
            chiral_vol = node_embd_full[..., CHIRAL_VOL_COL:CHIRAL_VOL_COL + 1]

        gated_chiral_3d = self.chiral_3d_embed(chiral_vol) if chiral_vol is not None else None

        if training and key is not None and gated_chiral_3d is not None:
            key, mode_key = jax.random.split(key)
            a = CHIRAL_3D_PROB
            d = CHIRAL_SIGN_DROPOUT
            p_2d = 1.0 - a
            u = jax.random.uniform(mode_key, (batch_size,))
            gate_graph = (u >= p_2d).astype(gated_chiral_3d.dtype)
            gated_chiral_3d = gated_chiral_3d * gate_graph[graph_id][:, None]
            if chiral_sign is not None:
                sign_keep = ~((u >= p_2d) & (u < p_2d + a * d))
                chiral_sign = chiral_sign * sign_keep.astype(chiral_sign.dtype)[graph_id][:, None]
        elif gated_chiral_3d is not None:
            gated_chiral_3d = jnp.zeros_like(gated_chiral_3d)

        edges = self._get_edge(batch)
        keys = _split_or_none(key if training else None, self.depth * 2 + 1)

        _log(
            "#kernel: nodes={}, {}".format(
                node_feat.shape[0],
                ", ".join(
                    "{}_edges={}".format(
                        "1hop" if suffix == "" else suffix[1:],
                        batch[f"edge{suffix}_index"].shape[1],
                    )
                    for suffix in EDGE_SUFFIXES[:MAX_HOPS]
                ),
            )
        )

        x_depth = self.atom_embed(node_feat) + self.atom_pos(node_embd) / 10
        lst_layer: list[jax.Array] = []
        lst_depth: list[jax.Array] = []
        for i in range(self.depth):
            k_layer = keys[i * 2]
            k_depth = keys[i * 2 + 1]
            x_layer = self.layer_mix[i](
                x_depth, edges, graph_id, batch_size, node_elec,
                chiral_sign=chiral_sign, gated_chiral_3d=gated_chiral_3d, key=k_layer
            )
            lst_layer.append(x_layer)
            x_depth = self.depth_mix[i](x_layer, lst_layer[:-1], key=k_depth)
            lst_depth.append(x_depth)
        y = self.head(x_depth, lst_depth[:-1], graph_id, batch_size, key=keys[-1])[1:]
        if return_aux:
            return y, self.chiral_head(x_depth)
        return y

    def _get_edge(self, batch):
        """Build edge tuples ``(edge_index, edge_attr, degree)`` per hop."""
        num_nodes = batch["node_feat"].shape[0]
        edges = []
        for suffix in EDGE_SUFFIXES[:MAX_HOPS]:
            edge_index = batch[f"edge{suffix}_index"]
            edge_attr = batch[f"edge{suffix}_feat"]
            n_edges = batch[f"edge{suffix}_batch"].shape[0]
            deg = segment_sum(
                jnp.ones((n_edges, 1), dtype=edge_index.dtype),
                edge_index[1],
                num_nodes,
            ).squeeze(-1).clip(1, None)
            edges.append((edge_index, edge_attr, deg))
        return edges


def get_model(key, *, depth: int = 5):
    """Create TriAxMPNN (depth=5, width=256, heads=16)."""
    assert key is not None
    return TriAxMPNN(depth=depth, width=256, num_head=16, dim_head=16, key=key)
