"""Optimizer construction, per-parameter LR/WD multipliers, and LR schedules."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from optax._src import base as optax_base


def _attr_segment_names(path: tuple[Any, ...]) -> tuple[str, ...]:
    """Attribute names along a JAX pytree path (ignore sequence/dict keys)."""
    out: list[str] = []
    for k in path:
        if isinstance(k, jtu.GetAttrKey):
            out.append(k.name)
    return tuple(out)


def lr_multiplier_for_param_path(path: tuple[Any, ...]) -> float:
    """Per-parameter LR scale relative to the global optimizer LR.

    - 0.5×: ``ScaleLayer.scale``; ``HeadKernel`` readout scalars; atom embedding table and
      atom position encoder (``atom_embed``, ``atom_pos``); RBF kernel parameters (``rbf_mu``, ``rbf_log_gamma``).
    - 4×: ConvKernel bond/degree embedding tensors, LoRA factors; HeadKernel ``act_out`` only.
    """
    names = _attr_segment_names(path)
    if not names:
        return 1.0
    leaf = names[-1]
    if leaf in ("scale", "readout_scale", "readout_bias", "rbf_mu", "rbf_log_gamma"):
        return 0.5
    if names[0] in ("atom_embed", "atom_pos"):
        return 0.5
    if names[0] == "head":
        if "act_out" in names:
            return 4.0
    if "conv" in names:
        if any(name in ("lora_down", "lora_up") for name in names):
            return 4.0
        if leaf == "embeddings" and ("embed_edge" in names or "embed_deg" in names):
            return 4.0
    return 1.0


def wd_multiplier_for_param_path(path: tuple[Any, ...]) -> float:
    """Per-parameter weight-decay scale relative to the global WD.

    - 1.0×: ``kernel`` (linear / grouped linear weights).
    - 0.5×: embedding tables.
    - 0.0×: ``scale``, ``bias``, readout scalars, ``lora_down`` / ``lora_up`` (product decay
      is applied separately via :func:`_add_lora_product_decay`), RBF kernel parameters (``rbf_mu``, ``rbf_log_gamma``), and any unknown leaf names.
    """
    names = _attr_segment_names(path)
    if not names:
        return 0.0
    leaf = names[-1]
    if leaf in ("scale", "bias", "lora_down", "lora_up", "readout_scale", "readout_bias"):
        return 0.0
    if leaf == "embeddings":
        return 0.5
    if leaf == "kernel":
        return 1.0
    return 0.0


def per_param_lr_multiplier_tree(params: Any) -> Any:
    """PyTree matching ``params`` with a positive float LR multiplier per array leaf."""
    return jtu.tree_map_with_path(
        lambda path, leaf: lr_multiplier_for_param_path(path),
        params,
    )


def per_param_wd_multiplier_tree(params: Any) -> Any:
    """PyTree matching ``params`` with a non-negative float WD multiplier per array leaf."""
    return jtu.tree_map_with_path(
        lambda path, leaf: wd_multiplier_for_param_path(path),
        params,
    )


def _scale_updates_by_lr_multipliers(multipliers: Any) -> optax.GradientTransformation:
    """Multiply each update leaf by the corresponding scalar (fixed multipliers)."""

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params
        scaled = jtu.tree_map(lambda u, m: u * m, updates, multipliers)
        return scaled, state

    return optax.GradientTransformation(init_fn, update_fn)


def _make_lr_schedule(steps_per_epoch: int, k: int, peak_lr: float) -> Callable[[jax.Array], jax.Array]:
    """JAX schedule ``count -> lr`` matching :func:`get_scheduled_hparams` (LR branch)."""
    gr = (1.0 + jnp.sqrt(5.0)) / 2.0
    k_f = float(k)
    spe = float(steps_per_epoch)

    def schedule(count: jax.Array) -> jax.Array:
        epoch_frac = count.astype(jnp.float32) / spe
        period = jnp.floor(epoch_frac / k_f).astype(jnp.int32)
        t = jnp.fmod(epoch_frac, k_f) / k_f
        warmup = peak_lr * t
        constant = peak_lr
        exp = jnp.maximum(period.astype(jnp.float32) - 2.0, 0.0)
        start = peak_lr / jnp.power(gr, exp)
        end = start / (gr * gr)
        cosine = end + 0.5 * (start - end) * (1.0 + jnp.cos(jnp.pi * t))
        return jnp.where(period == 0, warmup, jnp.where(period == 1, constant, cosine))

    return schedule


def _add_scaled_decayed_weights(
    weight_decay: float,
    wd_multiplier_tree: Any,
) -> optax.GradientTransformation:
    """Add ``weight_decay * wd_mult * param`` to each gradient leaf.

    ``wd_multiplier_tree`` matches trainable params; multiplier ``0`` disables decay for that leaf.
    """

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(optax_base.NO_PARAMS_MSG)
        s = jnp.asarray(weight_decay, dtype=jnp.float32)
        new_state = state

        def _scaled_decay(g, m, p):
            if g is None:
                return None
            m_arr = jnp.asarray(m, dtype=p.dtype)
            s_arr = jnp.astype(s, p.dtype)
            return g + s_arr * m_arr * p

        new_updates = jax.tree.map(
            _scaled_decay,
            updates,
            wd_multiplier_tree,
            params,
            is_leaf=lambda x: x is None,
        )
        return new_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


def _add_lora_product_decay(
    weight_decay: float,
    lora_product_wd_multiplier: float,
) -> optax.GradientTransformation:
    """Add decoupled weight decay on ``lora_down @ lora_up`` for sibling LoRA pairs.

    For each pair (A, B) = (``lora_down``, ``lora_up``), adds to gradients the terms from
    ``(wd/2) * mult * ||A @ B||_F^2``, i.e. ``wd * mult * A @ B @ B.T`` on A and
    ``wd * mult * A.T @ A @ B`` on B. Pairs are found by flattening the param tree with paths
    and matching leaves whose final segment is ``lora_down`` / ``lora_up`` under the same parent.
    """

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(optax_base.NO_PARAMS_MSG)
        s = jnp.asarray(weight_decay, dtype=jnp.float32)
        new_state = state

        mult = jnp.asarray(lora_product_wd_multiplier, dtype=jnp.float32)
        coeff = jnp.astype(s, jnp.float32) * mult

        pl_p, _treedef_p = jtu.tree_flatten_with_path(
            params, is_leaf=lambda x: x is None
        )
        pl_u, treedef_u = jtu.tree_flatten_with_path(
            updates, is_leaf=lambda x: x is None
        )
        paths_p = [p for p, _ in pl_p]
        flat_p = [l for _, l in pl_p]
        paths_u = [p for p, _ in pl_u]
        flat_u = [l for _, l in pl_u]
        if len(flat_p) != len(flat_u) or paths_p != paths_u or _treedef_p != treedef_u:
            raise ValueError("params and updates must have identical pytree structure for LoRA WD")

        parent_to_idx: dict[tuple[Any, ...], dict[str, int]] = {}
        for i, path in enumerate(paths_p):
            if not path:
                continue
            last = path[-1]
            if isinstance(last, jtu.GetAttrKey) and last.name in ("lora_down", "lora_up"):
                parent = path[:-1]
                parent_to_idx.setdefault(parent, {})[last.name] = i

        new_flat = list(flat_u)
        for _parent, idx_map in parent_to_idx.items():
            if "lora_down" not in idx_map or "lora_up" not in idx_map:
                continue
            i_d = idx_map["lora_down"]
            i_u = idx_map["lora_up"]
            A = flat_p[i_d]
            B = flat_p[i_u]
            g_d = new_flat[i_d]
            g_u = new_flat[i_u]
            if A is None or B is None or g_d is None or g_u is None:
                continue
            c = jnp.astype(coeff, A.dtype)
            new_flat[i_d] = g_d + c * (A @ B @ B.T)
            new_flat[i_u] = g_u + c * (A.T @ A @ B)

        new_updates = jtu.tree_unflatten(treedef_u, new_flat)
        return new_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


def make_optimizer(
    learning_rate: float | Callable[[jax.Array], jax.Array],
    weight_decay: float,
    wd_multiplier_tree: Any,
    lr_multiplier_tree: Any,
    *,
    lora_product_wd_multiplier: float = 0.5,
) -> optax.GradientTransformation:
    """Custom Adan-based optimizer with weight decay.

    Adan (Adaptive Nesterov Momentum) is a fast-converging optimizer that uses
    first, second, and third-order moments to adapt the step size.

    Args:
        learning_rate: Global learning rate (scalar or schedule ``count -> lr``).
        weight_decay: Constant L2 regularisation coefficient; per-parameter masking via
            ``wd_multiplier_tree`` (and LoRA product decay separately).
        wd_multiplier_tree: PyTree matching trainable params; per-leaf multiplier for decay.
        lr_multiplier_tree: PyTree matching trainable params; each leaf is a positive
            float factor applied before ``scale_by_learning_rate``.
        lora_product_wd_multiplier: Scale for ``||lora_down @ lora_up||_F^2`` decay relative
            to global ``weight_decay`` (LoRA factors themselves use 0× in ``wd_multiplier_tree``).

    Returns:
        An optax.GradientTransformation implementing the optimizer chain.
    """
    wd_transform = _add_scaled_decayed_weights(weight_decay, wd_multiplier_tree)
    lora_wd_transform = _add_lora_product_decay(weight_decay, lora_product_wd_multiplier)
    return optax.chain(
        optax.scale_by_adan(),
        wd_transform,
        lora_wd_transform,
        optax.scale_by_learning_rate(learning_rate),
        _scale_updates_by_lr_multipliers(lr_multiplier_tree),
    )


def get_scheduled_hparams(
    epoch_fractional: float,
    k: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[float, float]:
    """Return scheduled learning rate and constant weight decay for a fractional epoch.

    Args:
        epoch_fractional: Current fractional epoch (e.g., epoch + batch_idx / steps_per_epoch).
        k: Period length in epochs.
        learning_rate: Peak learning rate (held constant in period 1; cosine decay from period 2).
        weight_decay: Constant weight decay (optimizer applies per-parameter multipliers separately).

    Returns:
        Tuple of (learning_rate, weight_decay) for this step.
    """
    gr = (1 + math.sqrt(5)) / 2
    period = int(epoch_fractional // k)  # 0-indexed period number
    t = (epoch_fractional % k) / k       # fractional position within period [0, 1)

    if period == 0:
        current_lr = learning_rate * t
    elif period == 1:
        current_lr = learning_rate
    else:
        # Cosine decay: period N>=2 matches former period N-1 (exponent shift by -1)
        lr_start = learning_rate / gr ** (period - 2)
        lr_end = lr_start / gr ** 2
        current_lr = lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * t))

    return current_lr, weight_decay
