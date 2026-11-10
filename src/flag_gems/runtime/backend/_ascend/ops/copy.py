# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ascend override for ``aten::copy_``.

Background
----------
The generic implementation in ``flag_gems/ops/copy.py`` handles every case it
cannot run with Triton by calling::

    torch.ops.aten.copy_.default.redispatch(_FALLBACK_KEYSET, dst, src, ...)

with ``_FALLBACK_KEYSET = {CompositeExplicitAutograd}``.  On Ascend that call
does not reach the vendor kernel: FlagGems' own ``copy_`` is registered at the
``PrivateUse1`` key and the dispatcher re-enters it, so any fallback scenario
(CPU <-> NPU copies, overlapping views sharing a storage, quantized tensors,
``float8_e8m0fnu``, >2**31 elements, ...) ends in ``RecursionError``.  Because
``Tensor.cpu()`` / ``Tensor.to("cpu")`` are themselves implemented on top of
``aten::copy_``, this recursion also breaks plain D2H transfers, not just
explicit ``copy_`` calls.

Fix
---
Keep the Triton path for the layouts it supports, and for everything else call
the *captured vendor kernel* instead of re-dispatching.  ``torch.library.
get_kernel`` is documented for exactly this use case ("retrieve the original
kernel ... then register another kernel ... that calls into the original
kernel").  This module is imported before ``flag_gems.enable()`` registers the
FlagGems kernel, so the handle captured below is torch_npu's CANN kernel.

Note: a captured ``SafeKernelFunction`` is only obtainable through the
``torch.library.get_kernel`` API (torch >= 2.5, verified on 2.10.0+cpu); if the
handle is unavailable the wrapper fails loudly instead of silently recursing.
"""

import torch
import triton
import triton.language as tl

from flag_gems.ops.copy import _FLOAT8_E8M0FNU
from flag_gems.ops.copy import _can_use_triton
from flag_gems.ops.copy import copy_ as _generic_copy_

_PRIVATEUSE1_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.PrivateUse1)

_MAX_TRITON_NUMEL = 2**31 - 1

#: Element types the shared Ascend pointwise kernel is known to mishandle.
_UNSUPPORTED_TRITON_DTYPES = tuple(
    dt for dt in (getattr(torch, "int64", None), getattr(torch, "uint64", None)) if dt is not None
)

#: 1-D copy kernel tuning (measured on Ascend 910C, bf16 4096x2048).
_COPY_1D_BLOCK = 32768
_COPY_1D_WARPS = 1


@triton.jit
def _copy_1d_kernel(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=mask), mask=mask)


def _triton_copy_1d(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Same-dtype contiguous copy through a dedicated vectorised kernel.

    The shared ``pointwise_dynamic`` kernel is ~5.5x slower than the vendor
    copy for a 4096x2048 bf16 copy; this flat kernel lands at ~1.5x.
    """
    n = src.numel()
    grid = (triton.cdiv(n, _COPY_1D_BLOCK),)
    _copy_1d_kernel[grid](
        src, dst, n, BLOCK=_COPY_1D_BLOCK, num_warps=_COPY_1D_WARPS
    )
    return dst


def _capture_native_copy():
    """Return torch_npu's own ``aten::copy_`` kernel handle (pre-override)."""
    try:
        kernel = torch.library.get_kernel("aten::copy_", "PrivateUse1")
    except Exception:  # pragma: no cover - very old torch / no vendor kernel
        return None
    # ``debug`` is a string such as "registered at ../torch_npu/...".
    # Never accept a FlagGems kernel here: that would re-introduce recursion.
    if "flag_gems" in (getattr(kernel, "debug", "") or ""):
        return None
    return kernel


_NATIVE_COPY_ = _capture_native_copy()


def _can_use_triton_path(dst: torch.Tensor, src) -> bool:
    """Whether the generic Triton pointwise copy is known to be correct.

    Empirically (torch 2.10 / torch_npu 2.10.0.post2 / triton 3.5.1), the
    shared pointwise kernel returns wrong values for strided (non-contiguous)
    sources of some shapes on Ascend, e.g. a ``(8, 4)`` view with stride
    ``(8, 2)``.  Restrict the Triton path to the layouts that are covered by
    the correctness matrix (both sides contiguous; a single-element source may
    broadcast) and let the vendor kernel handle every other layout.
    """
    if not isinstance(src, torch.Tensor):
        # Scalar src is converted by the generic implementation; let it do so.
        return isinstance(src, (int, float, bool)) and dst.device.type == "npu"
    if dst.device.type != "npu" or src.device.type != "npu":
        # Kernel arguments must all live on the NPU.  Cross-device copies
        # (including Tensor.cpu()) go to the vendor kernel.
        return False
    if dst.layout != torch.strided or src.layout != torch.strided:
        return False
    if not (src.is_contiguous() and dst.is_contiguous()):
        return False
    if src.shape != dst.shape and src.numel() != 1:
        # Non-trivial broadcasting is left to the vendor kernel.
        return False
    if dst.is_quantized or src.is_quantized:
        return False
    if src.is_complex() or dst.is_complex():
        return False
    # 64-bit integer elementwise kernels are not usable on Ascend: the shared
    # pointwise kernel silently produces wrong values for int64 (checked
    # against CPU: sums differ), so those go through the vendor copy.
    if src.dtype in _UNSUPPORTED_TRITON_DTYPES or dst.dtype in _UNSUPPORTED_TRITON_DTYPES:
        return False
    if _FLOAT8_E8M0FNU is not None and (
        src.dtype == _FLOAT8_E8M0FNU or dst.dtype == _FLOAT8_E8M0FNU
    ):
        return False
    if src.numel() > _MAX_TRITON_NUMEL or dst.numel() > _MAX_TRITON_NUMEL:
        return False
    if torch._C._is_alias_of(dst, src):
        # Overlapping views have PyTorch-defined (and vendor-checked) semantics;
        # mirror native behaviour through the vendor kernel.
        return False
    return _can_use_triton(dst, src)


def _native_copy(dst: torch.Tensor, src, non_blocking: bool):
    if _NATIVE_COPY_ is None:
        raise RuntimeError(
            "FlagGems Ascend copy_: cannot fall back to the native CANN "
            "aten::copy_ kernel (torch.library.get_kernel unavailable or the "
            f"captured kernel is a FlagGems one); dst={dst!r} src={src!r}"
        )
    return _NATIVE_COPY_.call_boxed(
        _PRIVATEUSE1_KEYSET, dst, src, bool(non_blocking)
    )


def copy_(dst: torch.Tensor, src: torch.Tensor, non_blocking: bool = False):
    if dst.numel() == 0 and dst.layout == torch.strided:
        if dst._is_zerotensor():
            raise RuntimeError("ZeroTensors are immutable. Call clone() before copy_.")
        if isinstance(src, torch.Tensor) and src._is_zerotensor():
            return dst.zero_()
        src_shape = src.shape if isinstance(src, torch.Tensor) else ()
        broadcast_shape = torch.broadcast_shapes(dst.shape, src_shape)
        if torch.Size(broadcast_shape) != dst.shape:
            raise RuntimeError(
                f"The broadcast shape {broadcast_shape} does not match "
                f"destination shape {tuple(dst.shape)}"
            )
        return dst
    if _can_use_triton_path(dst, src):
        if (
            isinstance(src, torch.Tensor)
            and src.shape == dst.shape
            and src.dtype == dst.dtype
        ):
            # Fast path: same-dtype contiguous 1-D copy kernel.
            return _triton_copy_1d(dst, src)
        # Cast / single-element broadcast: the shared pointwise kernel.
        return _generic_copy_(dst, src, non_blocking=non_blocking)
    return _native_copy(dst, src, non_blocking)


def copy(template: torch.Tensor, src: torch.Tensor, *, non_blocking=False):
    out = torch.empty_strided(
        template.size(), template.stride(), dtype=template.dtype, device=template.device
    )
    return copy_(out, src, non_blocking=bool(non_blocking))
