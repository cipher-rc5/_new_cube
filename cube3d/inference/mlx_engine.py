"""Optional Apple Silicon (MLX) inference backend for cube3d.

Scope and intent
-----------------
This module provides ``MlxEngine``, an *opt-in* inference engine that runs the
autoregressive GPT decode loop of the DualStreamRoformer on Apple's MLX / Metal
framework. It is intended only for Apple Silicon machines and is selected via
``--backend mlx`` in ``generate.py``.

Design (from the migration plan):

* Only the GPT forward/decode loop runs in MLX. Everything else -- CLIP text
  conditioning, token/codebook embedding setup, and the shape decode -- stays in
  PyTorch. The torch/MPS path (``cube3d.inference.engine.Engine``) remains the
  default and the source of truth for correctness.
* ``MlxEngine`` *composes* a torch ``Engine`` internally. The torch engine owns:
  - tokenization + CLIP conditioning (``prepare_inputs``),
  - the VQ codebook -> ``wte`` setup,
  - ``run_shape_decode`` (mesh extraction).
  The handoff boundary between MLX and torch is ``output_ids`` -- integer token
  IDs only. No live tensor objects are shared across the two frameworks, which
  keeps the boundary clean and avoids dtype/device aliasing bugs.
* The MLX side reimplements the dual-stream RoFormer decode: embedding lookup,
  the dual-stream attention blocks, the single-stream blocks, RoPE, a KV cache,
  the final norm + lm_head, then CFG mixing + sampling identical to the torch
  ``run_gpt``.

Parity
------
The torch path is authoritative. Where the MLX port of a submodule is ambiguous
from the source, it is implemented faithfully to the torch reference and any
assumption is flagged with a ``TODO(parity)`` comment so the later unit test
(MLX-vs-torch) can pin it down. Sampling mirrors
``cube3d.inference.logits_postprocesses.process_logits`` so deterministic
(argmax) runs can match torch exactly; top_p sampling will differ in RNG.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# --- MLX import guard: module must be safe to import when mlx is absent. ------
try:
    import mlx.core as mx
    import mlx.nn as mlx_nn

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None  # ty: ignore[invalid-assignment]  # optional dep
    mlx_nn = None  # ty: ignore[invalid-assignment]  # optional dep

from cube3d.inference.engine import Engine


# ----------------------------------------------------------------------------
# MLX KV cache
# ----------------------------------------------------------------------------
#
# Mirrors ``cube3d.model.transformers.cache.Cache`` but for MLX arrays.
# Critically: the cache stores PRE-RoPE keys. RoPE is reapplied to the cached
# k-slice at decode time, matching the torch reference
# (``cube3d/model/transformers/rope.py`` lines 80-81 apply RoPE to k AFTER it
# is read from cache).
#
# Update-strategy microbench (see ``_dev/mlx_cache_bench.py``) on a [2,12,1025,128]
# cache with mx.eval forced each iteration:
#   Strategy A (mx.where one-hot mask):     ~0.062s / 200 updates
#   Strategy B (mx.concatenate two slices): ~0.058s / 200 updates
# Strategy B is chosen — marginally faster and simpler to reason about; the
# concatenate avoids materializing a [T,] index range each step.
@dataclass
class MlxCache:
    """KV cache slot, mirroring torch ``Cache`` shape semantics.

    Attributes:
        key_states:   [B, nH, T_max, head_dim], PRE-RoPE keys.
        value_states: [B, nH, T_max, head_dim].

    For a dual block T_max = S + max_shape_tokens; for a single block
    T_max = max_shape_tokens. ``mx.array`` is opaquely typed so the annotation
    uses ``Any`` to keep this module importable without mlx installed.
    """
    key_states: Any
    value_states: Any

    def update(self, pos: int, k_new: Any, v_new: Any) -> None:
        """Write k_new, v_new at slot ``pos`` along axis -2 (sequence axis).

        ``k_new`` / ``v_new`` are shape [B, nH, 1, head_dim]. Uses
        ``mx.concatenate`` (Strategy B above) since MLX has no in-place
        ``index_copy_``.
        """
        self.key_states = mx.concatenate(
            [self.key_states[..., :pos, :], k_new, self.key_states[..., pos + 1 :, :]],
            axis=-2,
        )
        self.value_states = mx.concatenate(
            [self.value_states[..., :pos, :], v_new, self.value_states[..., pos + 1 :, :]],
            axis=-2,
        )

    def write_slice(self, lo: int, hi: int, k_slice: Any, v_slice: Any) -> None:
        """Bulk write at positions [lo, hi) along axis -2. Used at prefill."""
        self.key_states = mx.concatenate(
            [self.key_states[..., :lo, :], k_slice, self.key_states[..., hi:, :]],
            axis=-2,
        )
        self.value_states = mx.concatenate(
            [self.value_states[..., :lo, :], v_slice, self.value_states[..., hi:, :]],
            axis=-2,
        )


# ----------------------------------------------------------------------------
# torch state-dict -> MLX parameter-key mapping
# ----------------------------------------------------------------------------
#
# The MLX weight store in this engine is a flat dict keyed by the *torch*
# DualStreamRoformer state_dict names, with the values converted to mx.array.
# Because we keep the torch names verbatim, the "mapping" is mostly the identity;
# ``torch_to_mlx_key`` exists as the single documented place where any rename
# would live, and so the parity test can assert the contract.
#
# Observed torch parameter names (from dual_stream_roformer.py and the
# transformer submodules), for n_embd=2048, n_head=16, head_dim=128:
#
#   text_proj.weight                                   [n_embd, text_embed_dim]
#   shape_proj.weight / shape_proj.bias                (used to seed wte; not
#                                                       needed at decode time)
#   lm_head.weight                                     [vocab, n_embd]
#   bbox_proj.weight / bbox_proj.bias                  (only if cfg.use_bbox)
#
#   transformer.wte.weight                             [vocab, n_embd]
#   transformer.ln_f.*                                 (elementwise_affine=False
#                                                       -> NO weight/bias saved)
#
#   Dual-stream block i (transformer.dual_blocks.{i}.):
#     ln_1.* , ln_2.*                                  (affine=False -> absent)
#     attn.pre_x.c_qk.weight                           [2*n_embd, n_embd]
#     attn.pre_x.q_norm.weight                         [head_dim]   (RMSNorm)
#     attn.pre_x.k_norm.weight                         [head_dim]
#     attn.pre_x.c_v.{weight,bias}                     bias only if cfg.bias
#     attn.pre_c.c_qk.weight    (if not cond_pre_only)
#     attn.pre_c.c_k.{weight,bias} (if cond_pre_only)  -- last layer only
#     attn.pre_c.q_norm.weight  (if not cond_pre_only)
#     attn.pre_c.k_norm.weight
#     attn.pre_c.c_v.{weight,bias}
#     post_1.c_proj.{weight,bias}
#     post_1.ln_3.*                                    (affine=False -> absent)
#     post_1.mlp.{gate_proj,up_proj,down_proj}.{weight,bias}
#     post_2.* (same as post_1, only if not cond_pre_only)
#
#   Single-stream block i (transformer.single_blocks.{i}.):  (n_single_layer=0
#   for open_model_v0.5, so usually none)
#     ln_1.* , ln_2.*                                  (affine=False -> absent)
#     attn.c_qk.weight , attn.c_v.{weight,bias} , attn.c_proj.{weight,bias}
#     attn.q_norm.weight , attn.k_norm.weight
#     mlp.{gate_proj,up_proj,down_proj}.{weight,bias}
#
# NOTE: nn.Linear stores weight as [out, in] and applies y = x @ W.T + b. MLX
# linear math here uses the same convention via ``_linear`` below, so weights are
# used as-is with no transpose.
def torch_to_mlx_key(name: str) -> str:
    """Map a torch DualStreamRoformer state_dict key to its MLX store key.

    The MLX store intentionally preserves the torch names, so this is the
    identity today. It is kept as an explicit, documented seam: if the MLX
    module layout ever diverges from the torch names, the rename belongs here and
    the parity test can assert against it.

    Args:
        name: A parameter name from ``DualStreamRoformer.state_dict()``.

    Returns:
        The corresponding key in the MLX weight store.
    """
    return name


class MlxEngine:
    """Apple-Silicon GPT-decode engine. Mirrors the public surface of Engine.

    Public methods mirror ``cube3d.inference.engine.Engine``:
        __init__(config_path, gpt_ckpt_path, shape_ckpt_path)
        t2s(...)
        run_gpt(...)            -> output_ids (torch.LongTensor)
        run_shape_decode(...)   -> mesh vertices/faces (delegated to torch)
    """

    def __init__(
        self,
        config_path: str,
        gpt_ckpt_path: str,
        shape_ckpt_path: str,
    ):
        """Initialize the MLX engine.

        Args:
            config_path: Path to the configuration YAML file.
            gpt_ckpt_path: Path to the GPT safetensors checkpoint.
            shape_ckpt_path: Path to the shape encoder/decoder checkpoint.

        Raises:
            RuntimeError: If MLX is not installed/importable.
        """
        if not MLX_AVAILABLE:
            raise RuntimeError(
                "MlxEngine requires the 'mlx' package (Apple Silicon only). "
                "Install with `uv sync --extra mlx` or use --backend torch."
            )

        # Compose a torch Engine for conditioning, codebook/wte setup, and the
        # shape decode. The torch engine runs on MPS (Apple GPU via PyTorch).
        # Importing locally keeps this engine torch-backed but MLX-driven.
        self._torch_engine = Engine(
            config_path,
            gpt_ckpt_path,
            shape_ckpt_path,
            device=torch.device("mps"),
        )

        self.cfg = self._torch_engine.cfg
        self.gpt_cfg = self._torch_engine.gpt_model.cfg
        self.max_new_tokens = self._torch_engine.max_new_tokens
        self.min_id = self._torch_engine.min_id
        self.max_id = self._torch_engine.max_id

        self.n_head = self.gpt_cfg.n_head
        self.n_embd = self.gpt_cfg.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.rope_theta = self.gpt_cfg.rope_theta
        self.eps = self.gpt_cfg.eps
        self.n_layer = self.gpt_cfg.n_layer
        self.n_single_layer = self.gpt_cfg.n_single_layer

        # Load GPT weights into MLX arrays keyed by torch names. We pull from the
        # already-loaded torch module (which had the VQ codebook copied into wte)
        # rather than re-reading the safetensors, so the MLX path sees the exact
        # same weights as torch -- important for parity.
        self._params = self._load_mlx_params_from_torch(self._torch_engine.gpt_model)

        # Free the big GPT decoder weights from MPS now that MLX owns them.
        # ``prepare_inputs`` still needs ``text_proj``, ``shape_proj``, ``bbox_proj``,
        # ``shape_bos_id`` and a one-shot ``encode_token`` (BOS) on the torch side, so
        # those small modules stay on MPS. The dual/single transformer blocks --
        # which are the multi-GB majority of the checkpoint -- are moved to CPU
        # and dropped from the live tensor graph. The MPS allocator's cache is
        # then released so the freed pages stop counting as "other allocations".
        self._free_torch_gpt_decoder_from_mps()

    # ---- weight loading ----------------------------------------------------
    def _load_mlx_params_from_torch(self, gpt_model: torch.nn.Module) -> dict:
        """Convert the torch GPT state_dict into a flat MLX-array store.

        Values are cast to float32 in MLX. MLX on Metal is fp32-friendly and
        this keeps decode numerically close to the torch reference; the parity
        test compares against the torch path.

        Args:
            gpt_model: The loaded DualStreamRoformer (codebook already in wte).

        Returns:
            dict[str, mx.array]: store keyed by ``torch_to_mlx_key(name)``.
        """
        store = {}
        sd = gpt_model.state_dict()
        for name, tensor in sd.items():
            arr = tensor.detach().to(torch.float32).cpu().numpy()
            store[torch_to_mlx_key(name)] = mx.array(arr)
        return store

    def _free_torch_gpt_decoder_from_mps(self) -> None:
        """Move the torch GPT decoder blocks off MPS once MLX has ingested them.

        The torch ``DualStreamRoformer`` is ~3-4 GB on disk and lives on MPS
        after ``Engine.__init__``. Once ``_load_mlx_params_from_torch`` has
        copied every parameter into ``self._params`` as ``mx.array``, the
        decoder blocks on the torch side are pure dead weight for this engine
        -- the MLX path runs the entire decode, and only a few small projections
        (``text_proj``, ``shape_proj``, ``bbox_proj``) plus ``shape_bos_id`` are
        still needed for ``prepare_inputs``. We move the big modules to CPU and
        release the MPS allocator cache so the freed pages stop showing up as
        "other allocations" in MPS memory reports.
        """
        gpt = self._torch_engine.gpt_model
        # transformer.{dual_blocks, single_blocks, ln_f} + lm_head are the bulk
        # of the weights and dead to this engine (MLX owns the decode). Move
        # them to CPU. transformer.wte stays on the torch device because
        # Engine.encode_input still calls gpt_model.encode_token with an
        # MPS-resident token tensor for the BOS embedding (engine.py:240).
        transformer = getattr(gpt, "transformer", None)
        if transformer is not None:
            for sub_attr in ("dual_blocks", "single_blocks", "ln_f"):
                sub = getattr(transformer, sub_attr, None)
                if sub is not None:
                    sub.to("cpu")
        lm_head = getattr(gpt, "lm_head", None)
        if lm_head is not None:
            lm_head.to("cpu")
        self._empty_mps_cache()

    @staticmethod
    def _empty_mps_cache() -> None:
        """Release the torch MPS allocator cache if MPS is the active backend.

        ``torch.mps.empty_cache`` is a no-op on non-MPS builds, but guarding
        against ``mps`` availability avoids hard-failing on systems where
        ``torch.backends.mps`` is unavailable.
        """
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _encode_token_mlx(self, token_ids: np.ndarray, double_for_cfg: bool) -> Any:
        """MLX embedding lookup matching torch ``gpt_model.encode_token``.

        Avoids the per-decode-step MPS round-trip (allocate token tensor on
        MPS -> wte gather on MPS -> numpy copy -> mx.array).

        Args:
            token_ids: int array of shape ``[B/2]`` (one sampled id per
                conditional batch row).
            double_for_cfg: when True, the result is concatenated with itself
                along the batch axis so cond/uncond halves share the embedding
                (matches torch CFG path lines 335-338).

        Returns:
            mx.array of shape ``[B, 1, n_embd]`` where ``B`` is the full batch
            size after CFG doubling, or ``[B/2, 1, n_embd]`` otherwise.
        """
        wte = self._params[torch_to_mlx_key("transformer.wte.weight")]
        ids = mx.array(token_ids.astype(np.int32))
        rows = wte[ids, :]  # [B/2, n_embd]
        out = rows[:, None, :]  # [B/2, 1, n_embd]
        if double_for_cfg:
            out = mx.concatenate([out, out], axis=0)
        return out

    def _get(self, name: str, optional: bool = False):
        key = torch_to_mlx_key(name)
        if key not in self._params:
            if optional:
                return None
            raise KeyError(f"MLX param store missing expected key: {key}")
        return self._params[key]

    # ---- MLX math primitives ----------------------------------------------
    @staticmethod
    def _linear(x, weight, bias=None):
        """y = x @ W.T + b, matching torch nn.Linear (weight is [out, in])."""
        y = x @ weight.T
        if bias is not None:
            y = y + bias
        return y

    def _rms_norm(self, x, weight):
        """RMSNorm matching cube3d norm.fused_rms_norm (computed in fp32)."""
        var = mx.mean(x * x, axis=-1, keepdims=True)
        x = x * mx.rsqrt(var + self.eps)
        return x * weight

    def _layer_norm_no_affine(self, x):
        """LayerNorm with elementwise_affine=False (no weight/bias), fp32."""
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return (x - mean) * mx.rsqrt(var + self.eps)

    def _precompute_freqs_cis(self, position_ids):
        """MLX analogue of rope.precompute_freqs_cis.

        Returns cos/sin tensors of shape [B, T, head_dim/2] used to rotate the
        interleaved (real, imag) pairs of q/k. position_ids is an mx.array of
        shape [B, T].
        """
        dim = self.head_dim
        half = dim // 2
        freqs = 1.0 / (self.rope_theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
        # [B, T, half]
        angles = position_ids.astype(mx.float32)[..., None] * freqs[None, None, :]
        return mx.cos(angles), mx.sin(angles)

    @staticmethod
    def _apply_rope_at_positions(x, cos, sin, positions):
        """Apply rotary embedding to x at the given absolute positions.

        Args:
            x: [B, nH, T, head_dim] tokens to rotate.
            cos, sin: [B, T_full, half] precomputed full-table cos/sin.
            positions: int slice indices into [0, T_full). One of:
                - None: take the last T rows of cos/sin (legacy prefill behavior;
                  positions = T_full - T .. T_full - 1).
                - int: single position; T must equal 1.
                - tuple (lo, hi): contiguous range, T == hi - lo.
        Returns:
            x with RoPE applied, same shape.
        """
        b, nh, t, d = x.shape
        half = d // 2
        x_pairs = x.reshape(b, nh, t, half, 2)
        x_re = x_pairs[..., 0]
        x_im = x_pairs[..., 1]

        if positions is None:
            c = cos[:, -t:, :]
            s = sin[:, -t:, :]
        elif isinstance(positions, int):
            c = cos[:, positions : positions + 1, :]
            s = sin[:, positions : positions + 1, :]
        else:
            lo, hi = positions
            c = cos[:, lo:hi, :]
            s = sin[:, lo:hi, :]
        # broadcast over heads: [B, 1, t, half]
        c = c[:, None, :, :]
        s = s[:, None, :, :]

        out_re = x_re * c - x_im * s
        out_im = x_re * s + x_im * c
        out = mx.stack([out_re, out_im], axis=-1).reshape(b, nh, t, d)
        return out

    @classmethod
    def _apply_rope(cls, x, cos, sin, pos_index=None):
        """Backwards-compat wrapper: prefill (pos_index=None) or single-position decode.

        Implemented in terms of ``_apply_rope_at_positions`` so prefill behavior
        is unchanged.
        """
        if pos_index is None:
            return cls._apply_rope_at_positions(x, cos, sin, positions=None)
        return cls._apply_rope_at_positions(x, cos, sin, positions=int(pos_index))

    def _init_mlx_kv_cache(
        self,
        batch_size: int,
        cond_len: int,
        max_shape_tokens: int,
        n_dual: int,
        n_single: int,
    ) -> list:
        """Allocate the MLX KV cache list. Mirrors ``DualStreamRoformer.init_kv_cache``.

        Dual blocks get T_max = cond_len + max_shape_tokens (cond k/v live at
        slots 0..cond_len-1, x-stream k/v at cond_len..cond_len+L-1).
        Single blocks get T_max = max_shape_tokens (x-stream only).
        """
        nh = self.n_head
        hd = self.head_dim
        dual_T = cond_len + max_shape_tokens
        single_T = max_shape_tokens
        out = []
        for _ in range(n_dual):
            out.append(
                MlxCache(
                    key_states=mx.zeros((batch_size, nh, dual_T, hd), dtype=mx.float32),
                    value_states=mx.zeros((batch_size, nh, dual_T, hd), dtype=mx.float32),
                )
            )
        for _ in range(n_single):
            out.append(
                MlxCache(
                    key_states=mx.zeros((batch_size, nh, single_T, hd), dtype=mx.float32),
                    value_states=mx.zeros((batch_size, nh, single_T, hd), dtype=mx.float32),
                )
            )
        return out

    def _sdpa(self, q, k, v, attn_mask=None):
        """Scaled dot-product attention. q/k/v: [B, nH, Tq/Tk, head_dim].

        attn_mask: boolean [Tq, Tk] (True = keep) or None.
        """
        scale = 1.0 / np.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale  # [B, nH, Tq, Tk]
        if attn_mask is not None:
            neg = mx.array(-1e9, dtype=scores.dtype)
            scores = mx.where(attn_mask, scores, neg)
        weights = mx.softmax(scores, axis=-1)
        return weights @ v  # [B, nH, Tq, head_dim]

    def _to_mha(self, x):
        """[B, T, C] -> [B, nH, T, head_dim]."""
        b, t, _ = x.shape
        return x.reshape(b, t, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

    def _from_mha(self, y):
        """[B, nH, T, head_dim] -> [B, T, C]."""
        b, nh, t, hd = y.shape
        return y.transpose(0, 2, 1, 3).reshape(b, t, nh * hd)

    # ---- dual-stream block --------------------------------------------------
    def _dual_block(
        self,
        idx,
        h,
        c,
        cos,
        sin,
        attn_mask,
        cond_pre_only,
        kv_cache=None,
        curr_pos_id=None,
        decode=False,
        cond_len=0,
    ):
        """Dual-stream RoFormer block. Three modes:

        * decode=False, kv_cache=None  -> full prefill, no cache.
        * decode=False, kv_cache!=None -> prefill that also seeds the cache
          (writes pre-RoPE cond k/v at slots [0, S) and x-stream k/v at
          [S, S+L)).
        * decode=True                  -> single-token decode. ``h`` is
          [B, 1, C] for the new x-stream token (cond is None). Writes pre-RoPE
          x-stream k/v at slot ``curr_pos_id + S`` then attends q against the
          entire cache k/v with RoPE re-applied via the full d_freqs_cis table.
        """
        p = f"transformer.dual_blocks.{idx}."

        if decode:
            # Single-token x-stream decode. cond_n is not used.
            h_n = self._layer_norm_no_affine(h)
            qk = self._linear(h_n, self._get(p + "attn.pre_x.c_qk.weight"))
            qx, kx = mx.split(qk, 2, axis=-1)
            qx = self._rms_norm(self._to_mha(qx), self._get(p + "attn.pre_x.q_norm.weight"))
            kx = self._rms_norm(self._to_mha(kx), self._get(p + "attn.pre_x.k_norm.weight"))
            vx = self._to_mha(
                self._linear(
                    h_n,
                    self._get(p + "attn.pre_x.c_v.weight"),
                    self._get(p + "attn.pre_x.c_v.bias", optional=True),
                )
            )

            # Write the pre-RoPE x-stream k/v at slot curr_pos_id + S.
            assert curr_pos_id is not None and kv_cache is not None
            pos = int(curr_pos_id)
            slot = pos + int(cond_len)
            kv_cache.update(slot, kx, vx)

            # Read back the full cache: cond at [0, S), x-stream at [S, S+L_so_far],
            # zeros after. Attention mask cuts off the trailing zeros.
            k_full = kv_cache.key_states
            v_full = kv_cache.value_states
            T_full = k_full.shape[2]
            L_so_far = pos + 1  # number of x-stream tokens emitted so far
            valid_len = int(cond_len) + L_so_far

            # RoPE on q at single position (curr_pos_id + S), and on the entire
            # cached k slice over positions [0, T_full). cos/sin already encode
            # cond positions == 0 and x-stream positions 0..L-1, padded to T_full.
            q = self._apply_rope_at_positions(qx, cos, sin, positions=slot)
            k_rot = self._apply_rope_at_positions(k_full, cos, sin, positions=(0, T_full))

            # Build [1, T_full] mask: True over [0, valid_len), False after.
            idx_arr = mx.arange(T_full)
            row_mask = (idx_arr < valid_len)[None, :]  # broadcasts to [1, T_full]
            y = self._sdpa(q, k_rot, v_full, attn_mask=row_mask)
            y = self._from_mha(y)

            # post_1 on x stream (cond stays None for cond_pre_only or just
            # untouched otherwise; in decode the caller passes c=None).
            h = self._post(p + "post_1.", h, y)
            return h, None

        # ---- non-decode: full prefill (with or without cache) ---------------
        h_n = self._layer_norm_no_affine(h)
        c_n = self._layer_norm_no_affine(c) if c is not None else None

        # pre_x: query stream (always has query)
        qk = self._linear(h_n, self._get(p + "attn.pre_x.c_qk.weight"))
        qx, kx = mx.split(qk, 2, axis=-1)
        qx = self._rms_norm(self._to_mha(qx), self._get(p + "attn.pre_x.q_norm.weight"))
        kx = self._rms_norm(self._to_mha(kx), self._get(p + "attn.pre_x.k_norm.weight"))
        vx = self._to_mha(
            self._linear(
                h_n,
                self._get(p + "attn.pre_x.c_v.weight"),
                self._get(p + "attn.pre_x.c_v.bias", optional=True),
            )
        )

        # pre_c: condition stream
        if cond_pre_only:
            # last layer: condition contributes only key/value (c_k, no query)
            kc = self._linear(
                c_n,
                self._get(p + "attn.pre_c.c_k.weight"),
                self._get(p + "attn.pre_c.c_k.bias", optional=True),
            )
            kc = self._rms_norm(self._to_mha(kc), self._get(p + "attn.pre_c.k_norm.weight"))
            vc = self._to_mha(
                self._linear(
                    c_n,
                    self._get(p + "attn.pre_c.c_v.weight"),
                    self._get(p + "attn.pre_c.c_v.bias", optional=True),
                )
            )
            q = qx
        else:
            qkc = self._linear(c_n, self._get(p + "attn.pre_c.c_qk.weight"))
            qc, kc = mx.split(qkc, 2, axis=-1)
            qc = self._rms_norm(self._to_mha(qc), self._get(p + "attn.pre_c.q_norm.weight"))
            kc = self._rms_norm(self._to_mha(kc), self._get(p + "attn.pre_c.k_norm.weight"))
            vc = self._to_mha(
                self._linear(
                    c_n,
                    self._get(p + "attn.pre_c.c_v.weight"),
                    self._get(p + "attn.pre_c.c_v.bias", optional=True),
                )
            )
            # prepend condition stream: [cond ; x]
            q = mx.concatenate([qc, qx], axis=2)

        k = mx.concatenate([kc, kx], axis=2)
        v = mx.concatenate([vc, vx], axis=2)

        # If a cache is provided at prefill, seed it with the PRE-RoPE k/v.
        # This matches torch dual_stream_attention.py:197-198 which writes the
        # pre-RoPE k/v into the cache before applying RoPE downstream.
        if kv_cache is not None:
            s_len = c.shape[1] if c is not None else 0
            l_len = h.shape[1]
            kv_cache.write_slice(0, s_len + l_len, k, v)

        # RoPE on q and k (full sequence; cond positions are 0).
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        s = c.shape[1] if c is not None else 0
        # In the last (cond_pre_only) layer the query is only the x-stream
        # (length L) while keys span [cond ; x] (length S+L). The causal mask is
        # [S+L, S+L]; select its x-stream rows so it broadcasts against scores of
        # shape [B, nH, L, S+L]. Other layers query the full [cond ; x] sequence.
        block_mask = attn_mask[s:, :] if cond_pre_only else attn_mask
        y = self._sdpa(q, k, v, attn_mask=block_mask)
        y = self._from_mha(y)

        if y.shape[1] == h.shape[1]:
            y_c, y_x = None, y
        else:
            y_c = y[:, :s, :]
            y_x = y[:, s:, :]

        # post_1 on x stream
        h = self._post(p + "post_1.", h, y_x)
        if y_c is not None and not cond_pre_only:
            c = self._post(p + "post_2.", c, y_c)
        else:
            c = None if cond_pre_only else c
        return h, c

    def _post(self, p, x, a):
        """DismantledPostAttention: x = x + c_proj(a); x = x + mlp(ln_3(x))."""
        x = x + self._linear(
            a,
            self._get(p + "c_proj.weight"),
            self._get(p + "c_proj.bias", optional=True),
        )
        xn = self._layer_norm_no_affine(x)  # ln_3 (affine=False)
        gate = self._linear(
            xn,
            self._get(p + "mlp.gate_proj.weight"),
            self._get(p + "mlp.gate_proj.bias", optional=True),
        )
        up = self._linear(
            xn,
            self._get(p + "mlp.up_proj.weight"),
            self._get(p + "mlp.up_proj.bias", optional=True),
        )
        act = (gate * mx.sigmoid(gate)) * up  # SiLU(gate) * up
        down = self._linear(
            act,
            self._get(p + "mlp.down_proj.weight"),
            self._get(p + "mlp.down_proj.bias", optional=True),
        )
        return x + down

    def _single_block(
        self,
        idx,
        h,
        cos,
        sin,
        attn_mask,
        kv_cache=None,
        curr_pos_id=None,
        decode=False,
    ):
        """Single-stream RoFormer block. Same three modes as _dual_block but
        no cond stream.
        """
        p = f"transformer.single_blocks.{idx}."

        if decode:
            h_n = self._layer_norm_no_affine(h)
            qk = self._linear(h_n, self._get(p + "attn.c_qk.weight"))
            q1, k1 = mx.split(qk, 2, axis=-1)
            q1 = self._rms_norm(self._to_mha(q1), self._get(p + "attn.q_norm.weight"))
            k1 = self._rms_norm(self._to_mha(k1), self._get(p + "attn.k_norm.weight"))
            v1 = self._to_mha(
                self._linear(
                    h_n,
                    self._get(p + "attn.c_v.weight"),
                    self._get(p + "attn.c_v.bias", optional=True),
                )
            )
            assert curr_pos_id is not None and kv_cache is not None
            slot = int(curr_pos_id)
            kv_cache.update(slot, k1, v1)

            k_full = kv_cache.key_states
            v_full = kv_cache.value_states
            T_full = k_full.shape[2]
            valid_len = slot + 1

            q = self._apply_rope_at_positions(q1, cos, sin, positions=slot)
            k_rot = self._apply_rope_at_positions(k_full, cos, sin, positions=(0, T_full))

            idx_arr = mx.arange(T_full)
            row_mask = (idx_arr < valid_len)[None, :]
            y = self._sdpa(q, k_rot, v_full, attn_mask=row_mask)
            y = self._from_mha(y)
            h = h + self._linear(
                y,
                self._get(p + "attn.c_proj.weight"),
                self._get(p + "attn.c_proj.bias", optional=True),
            )
            hn = self._layer_norm_no_affine(h)
            gate = self._linear(hn, self._get(p + "mlp.gate_proj.weight"),
                                self._get(p + "mlp.gate_proj.bias", optional=True))
            up = self._linear(hn, self._get(p + "mlp.up_proj.weight"),
                              self._get(p + "mlp.up_proj.bias", optional=True))
            act = (gate * mx.sigmoid(gate)) * up
            down = self._linear(act, self._get(p + "mlp.down_proj.weight"),
                                self._get(p + "mlp.down_proj.bias", optional=True))
            return h + down

        # Non-decode (prefill, with or without cache).
        h_n = self._layer_norm_no_affine(h)
        qk = self._linear(h_n, self._get(p + "attn.c_qk.weight"))
        q, k = mx.split(qk, 2, axis=-1)
        q = self._rms_norm(self._to_mha(q), self._get(p + "attn.q_norm.weight"))
        k = self._rms_norm(self._to_mha(k), self._get(p + "attn.k_norm.weight"))
        v = self._to_mha(
            self._linear(
                h_n,
                self._get(p + "attn.c_v.weight"),
                self._get(p + "attn.c_v.bias", optional=True),
            )
        )
        if kv_cache is not None:
            l_len = h.shape[1]
            kv_cache.write_slice(0, l_len, k, v)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)
        y = self._sdpa(q, k, v, attn_mask=attn_mask)
        y = self._from_mha(y)
        h = h + self._linear(
            y,
            self._get(p + "attn.c_proj.weight"),
            self._get(p + "attn.c_proj.bias", optional=True),
        )
        # SwiGLU MLP (single-stream uses ln_2)
        hn = self._layer_norm_no_affine(h)
        gate = self._linear(hn, self._get(p + "mlp.gate_proj.weight"),
                            self._get(p + "mlp.gate_proj.bias", optional=True))
        up = self._linear(hn, self._get(p + "mlp.up_proj.weight"),
                          self._get(p + "mlp.up_proj.bias", optional=True))
        act = (gate * mx.sigmoid(gate)) * up
        down = self._linear(act, self._get(p + "mlp.down_proj.weight"),
                            self._get(p + "mlp.down_proj.bias", optional=True))
        return h + down

    def _freqs_cis_for_seq(self, batch_size: int, cond_len: int, max_new_tokens: int):
        """Precompute d_cos/d_sin (dual) and s_cos/s_sin (single) tables covering
        the full S + max_new_tokens range. Returns (d_cos, d_sin, s_cos, s_sin).

        Dual table positions: [0]*S concat [0..max_new_tokens-1].
        Single table positions: [0..max_new_tokens-1].
        """
        b = batch_size
        s = cond_len
        L = max_new_tokens
        d_pos = mx.concatenate(
            [
                mx.zeros((b, s), dtype=mx.int32),
                mx.broadcast_to(mx.arange(L)[None, :], (b, L)).astype(mx.int32),
            ],
            axis=1,
        )
        s_pos = mx.broadcast_to(mx.arange(L)[None, :], (b, L)).astype(mx.int32)
        d_cos, d_sin = self._precompute_freqs_cis(d_pos)
        s_cos, s_sin = self._precompute_freqs_cis(s_pos)
        return d_cos, d_sin, s_cos, s_sin

    def _gpt_forward_prefill(self, embed_mx, cond_mx, kv_cache, d_cos, d_sin, s_cos, s_sin):
        """Full prefill that also seeds the KV cache with pre-RoPE k/v.

        embed_mx: [B, L_prefill, C] where L_prefill is just the BOS embed (length 1)
        cond_mx:  [B, S, C]
        kv_cache: list[MlxCache] of length n_dual + n_single
        d_cos/d_sin: dual freqs of length S + max_new_tokens
        s_cos/s_sin: single freqs of length max_new_tokens
        """
        b, l, _ = embed_mx.shape
        s = cond_mx.shape[1]
        # Slice freqs to current prefill length.
        # Dual: positions are cond zeros then 0..l-1, so the relevant freq slice
        # is d_cos[:, :s+l, :].
        d_cos_pref = d_cos[:, : s + l, :]
        d_sin_pref = d_sin[:, : s + l, :]
        s_cos_pref = s_cos[:, :l, :]
        s_sin_pref = s_sin[:, :l, :]

        full = s + l
        idx = mx.arange(full)
        attn_mask = idx[:, None] >= idx[None, :]

        h = embed_mx
        c = cond_mx
        for i in range(self.n_layer):
            cond_pre_only = i == self.n_layer - 1
            h, c = self._dual_block(
                i, h, c, d_cos_pref, d_sin_pref, attn_mask, cond_pre_only,
                kv_cache=kv_cache[i],
                curr_pos_id=None,
                decode=False,
                cond_len=s,
            )

        x_idx = mx.arange(l)
        x_mask = x_idx[:, None] >= x_idx[None, :]
        for j in range(self.n_single_layer):
            h = self._single_block(
                j, h, s_cos_pref, s_sin_pref, x_mask,
                kv_cache=kv_cache[self.n_layer + j],
                curr_pos_id=None,
                decode=False,
            )

        h = self._layer_norm_no_affine(h)
        logits = self._linear(h, self._get("lm_head.weight"))
        return logits

    def _gpt_forward_decode(self, next_embed_mx, kv_cache, curr_pos_id, cond_len, d_cos, d_sin, s_cos, s_sin):
        """Single-token decode forward using the cache.

        next_embed_mx: [B, 1, C] embedding of the most recently sampled token
        curr_pos_id:   int position index in [0, max_new_tokens); x-stream slot
                       is curr_pos_id + cond_len for dual blocks, curr_pos_id
                       for single blocks.
        """
        h = next_embed_mx
        for i in range(self.n_layer):
            cond_pre_only = i == self.n_layer - 1
            h, _ = self._dual_block(
                i, h, None, d_cos, d_sin, attn_mask=None,
                cond_pre_only=cond_pre_only,
                kv_cache=kv_cache[i],
                curr_pos_id=curr_pos_id,
                decode=True,
                cond_len=cond_len,
            )
        for j in range(self.n_single_layer):
            h = self._single_block(
                j, h, s_cos, s_sin, attn_mask=None,
                kv_cache=kv_cache[self.n_layer + j],
                curr_pos_id=curr_pos_id,
                decode=True,
            )
        h = self._layer_norm_no_affine(h)
        logits = self._linear(h, self._get("lm_head.weight"))
        return logits

    def _gpt_forward_full(self, embed_mx, cond_mx):
        """Full (non-cached) forward of the dual-stream RoFormer in MLX.

        Args:
            embed_mx: [B, L, C] shape-token embeddings (mx.array).
            cond_mx: [B, S, C] conditioning embeddings (mx.array).

        Returns:
            logits mx.array of shape [B, L, vocab].
        """
        b, l, _ = embed_mx.shape
        s = cond_mx.shape[1]

        # Causal mask over [S+L, S+L]; condition occupies the first S positions.
        full = s + l
        # lower-triangular True
        idx = mx.arange(full)
        attn_mask = idx[:, None] >= idx[None, :]  # [full, full] bool

        # position ids: x-stream uses 0..l-1; for the dual blocks the condition
        # positions are 0 (matching torch which concatenates zeros for cond).
        d_pos = mx.concatenate(
            [mx.zeros((b, s), dtype=mx.int32), mx.broadcast_to(mx.arange(l)[None, :], (b, l)).astype(mx.int32)],
            axis=1,
        )
        s_pos = mx.broadcast_to(mx.arange(l)[None, :], (b, l)).astype(mx.int32)
        d_cos, d_sin = self._precompute_freqs_cis(d_pos)
        s_cos, s_sin = self._precompute_freqs_cis(s_pos)

        h = embed_mx
        c = cond_mx
        for i in range(self.n_layer):
            cond_pre_only = i == self.n_layer - 1
            # dual blocks attend over [cond ; x]; mask is the full lower-tri.
            h, c = self._dual_block(i, h, c, d_cos, d_sin, attn_mask, cond_pre_only)

        # single blocks operate on x-stream only (no condition), causal over L.
        x_idx = mx.arange(l)
        x_mask = x_idx[:, None] >= x_idx[None, :]
        for i in range(self.n_single_layer):
            h = self._single_block(i, h, s_cos, s_sin, x_mask)

        h = self._layer_norm_no_affine(h)  # ln_f (affine=False)
        logits = self._linear(h, self._get("lm_head.weight"))
        return logits

    # ---- public surface ----------------------------------------------------
    def run_gpt(
        self,
        prompts: list,
        use_kv_cache: bool,
        guidance_scale: float = 3.0,
        top_p: float | None = None,
        bounding_box_xyz: Optional[Tuple[float]] = None,
    ) -> torch.Tensor:
        """Autoregressive GPT decode in MLX. Returns output_ids (torch.LongTensor).

        Mirrors Engine.run_gpt: classifier-free-guidance mixing with a per-step
        gamma schedule and process_logits-equivalent sampling.

        When ``use_kv_cache=True`` runs the incremental KV-cache decode path
        (O(N) total cost over the decode). When ``use_kv_cache=False`` falls
        back to the legacy full-recompute path (O(N^2)) which is kept for
        debugging and as a non-regression reference.
        """
        # Conditioning is computed by the torch engine (CLIP + projections).
        embed_t, cond_t = self._torch_engine.prepare_inputs(
            prompts, guidance_scale, bounding_box_xyz
        )
        batch_size, input_seq_len, dim = embed_t.shape

        cond_mx = mx.array(cond_t.detach().to(torch.float32).cpu().numpy())
        cond_len = cond_mx.shape[1]
        del cond_t

        if not use_kv_cache:
            # Legacy O(N^2) full-recompute path -- kept for parity verification.
            max_seq_len = input_seq_len + self.max_new_tokens
            embed_buffer = np.zeros((batch_size, max_seq_len, dim), dtype=np.float32)
            embed_buffer[:, :input_seq_len, :] = (
                embed_t.detach().to(torch.float32).cpu().numpy()
            )
            embed_buffer = mx.array(embed_buffer)
            # Free CLIP/conditioning intermediates from MPS before the decode
            # loop. MLX now owns embed_buffer; the torch copy is dead.
            del embed_t
            self._empty_mps_cache()

            output_ids = []
            for i in tqdm(
                range(self.max_new_tokens),
                desc="generating",
                mininterval=0.5,
            ):
                cur_len = input_seq_len + i
                logits = self._gpt_forward_full(embed_buffer[:, :cur_len, :], cond_mx)
                logits = logits[:, cur_len - 1, :]
                logits = logits[:, self.min_id : self.max_id]
                if guidance_scale > 0.0:
                    half = logits.shape[0] // 2
                    cond_logits = logits[:half, :]
                    uncond_logits = logits[half:, :]
                    gamma = guidance_scale * (self.max_new_tokens - i) / self.max_new_tokens
                    logits = (1 + gamma) * cond_logits - gamma * uncond_logits
                next_id = self._sample(logits, top_p)
                output_ids.append(next_id)

                next_embed_mx = self._encode_token_mlx(
                    next_id, double_for_cfg=guidance_scale > 0.0
                )  # [B, 1, C]
                embed_buffer[:, cur_len, :] = next_embed_mx[:, 0, :]

            ids = np.stack(output_ids, axis=1)
            return torch.from_numpy(ids).long()

        # ---- KV-cache decode path -------------------------------------------
        # Step 0: prefill with the initial BOS embedding (length input_seq_len,
        # which is 1 from encode_input). Seeds the cache.
        embed_mx = mx.array(embed_t.detach().to(torch.float32).cpu().numpy())
        del embed_t
        self._empty_mps_cache()

        max_shape_tokens = self.max_new_tokens + 1  # +1 for BOS token slot
        kv_cache = self._init_mlx_kv_cache(
            batch_size, cond_len, max_shape_tokens, self.n_layer, self.n_single_layer
        )
        d_cos, d_sin, s_cos, s_sin = self._freqs_cis_for_seq(
            batch_size, cond_len, max_shape_tokens
        )

        prefill_logits = self._gpt_forward_prefill(
            embed_mx, cond_mx, kv_cache, d_cos, d_sin, s_cos, s_sin
        )
        # Take the logits at the last prefill position.
        logits = prefill_logits[:, input_seq_len - 1, :]
        logits = logits[:, self.min_id : self.max_id]
        if guidance_scale > 0.0:
            half = logits.shape[0] // 2
            cond_logits = logits[:half, :]
            uncond_logits = logits[half:, :]
            gamma = guidance_scale * (self.max_new_tokens - 0) / self.max_new_tokens
            logits = (1 + gamma) * cond_logits - gamma * uncond_logits

        next_id = self._sample(logits, top_p)
        output_ids = [next_id]

        # Steps 1..N-1: decode loop, one token per iteration.
        for i in tqdm(
            range(1, self.max_new_tokens),
            desc="generating",
            mininterval=0.5,
        ):
            # Embed the previous token directly in MLX. The torch ``wte`` has
            # been moved to CPU, but we have the exact same weight matrix in
            # ``self._params`` as an mx.array, so no MPS allocation is needed.
            prev_id = output_ids[-1]
            next_embed_mx = self._encode_token_mlx(
                prev_id, double_for_cfg=guidance_scale > 0.0
            )  # [B, 1, C]

            # curr_pos_id is i (the x-stream slot of the new token).
            logits = self._gpt_forward_decode(
                next_embed_mx, kv_cache, curr_pos_id=i, cond_len=cond_len,
                d_cos=d_cos, d_sin=d_sin, s_cos=s_cos, s_sin=s_sin,
            )
            logits = logits[:, 0, :]  # [B, vocab]
            logits = logits[:, self.min_id : self.max_id]
            if guidance_scale > 0.0:
                half = logits.shape[0] // 2
                cond_logits = logits[:half, :]
                uncond_logits = logits[half:, :]
                gamma = guidance_scale * (self.max_new_tokens - i) / self.max_new_tokens
                logits = (1 + gamma) * cond_logits - gamma * uncond_logits
            next_id = self._sample(logits, top_p)
            output_ids.append(next_id)

        ids = np.stack(output_ids, axis=1)  # [b/2, max_new_tokens]
        return torch.from_numpy(ids).long()

    def _sample(self, logits_mx, top_p):
        """Sampling mirroring logits_postprocesses.process_logits.

        Deterministic argmax when top_p is None (parity with torch). For top_p,
        nucleus filtering then multinomial sampling (RNG differs from torch).
        """
        logits = np.array(logits_mx)  # [b, vocab]
        if top_p is None:
            return np.argmax(logits, axis=-1).astype(np.int64)
        # TODO(parity): torch top_p path hardcodes top_p=0.9 inside
        # top_p_filtering regardless of the passed value; replicate that quirk so
        # the parity test compares like-for-like.
        effective_p = 0.9
        out = np.empty(logits.shape[0], dtype=np.int64)
        for b in range(logits.shape[0]):
            row = logits[b]
            order = np.argsort(-row)
            sorted_logits = row[order]
            probs = _softmax(sorted_logits)
            cum = np.cumsum(probs)
            remove = cum > effective_p
            remove[0] = False
            sorted_logits = np.where(remove, -np.inf, sorted_logits)
            p = _softmax(sorted_logits)
            choice = np.random.choice(len(p), p=p)
            out[b] = order[choice]
        return out

    def run_shape_decode(
        self,
        output_ids: torch.Tensor,
        resolution_base: float = 8.0,
        chunk_size: int = 250_000,
    ):
        """Delegate shape decode to the torch engine (stays in PyTorch)."""
        output_ids = output_ids.to(self._torch_engine.device)
        try:
            return self._torch_engine.run_shape_decode(
                output_ids, resolution_base, chunk_size
            )
        finally:
            # Shape decode allocates large grid-sampling tensors on MPS that
            # would otherwise linger in the allocator cache between calls.
            self._empty_mps_cache()

    def t2s(
        self,
        prompts: list,
        use_kv_cache: bool,
        guidance_scale: float = 3.0,
        resolution_base: float = 8.0,
        chunk_size: int = 250_000,
        top_p: float | None = None,
        bounding_box_xyz: Optional[Tuple[float]] = None,
    ):
        """Text-to-shape: MLX GPT decode, then torch shape decode."""
        output_ids = self.run_gpt(
            prompts, use_kv_cache, guidance_scale, top_p, bounding_box_xyz
        )
        return self.run_shape_decode(output_ids, resolution_base, chunk_size)


def _softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)
