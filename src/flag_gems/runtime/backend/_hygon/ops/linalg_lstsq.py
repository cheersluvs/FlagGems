import importlib
import logging

import torch
import triton

logger = logging.getLogger("flag_gems." + __name__)

# NOT `import flag_gems.ops.linalg_lstsq as _generic`: flag_gems/ops/__init__.py
# does `from .linalg_lstsq import linalg_lstsq`, which rebinds that attribute on
# the package to the FUNCTION, and `import a.b as c` binds by attribute lookup,
# so _generic would be the function. import_module resolves via sys.modules.
_generic = importlib.import_module("flag_gems.ops.linalg_lstsq")

# ---------------------------------------------------------------------------
# One Hygon device fact: 64KB of shared memory per block. Upstream's compact-WY
# update config asks for 2*BLOCK_R*BLOCK_C*esize + BLOCK_R*P*esize
# = 2*128*64*4 + 128*16*4 = 73728 bytes, 8KB over, and 14 of the suite's cases
# died with
#
#     triton.runtime.errors.OutOfResources: out of resource: shared memory,
#     Required: 73728, Hardware limit: 65536
#
# Every one of them is a compact-WY shape (square, underdetermined, tall
# blocked, rank-deficient square); the monolithic and blocked-TSQR paths fit and
# pass. So BLOCK_C is derived from the limit Triton actually enforces, which on
# this device is 65536 and yields 16 (24576 bytes).
#
# Nothing else is rebound. The other two shared-memory-sensitive tiles
# (_TARGET_TILE_BYTES for the panel kernel, _TARGET_STACK_ROWS for the reduce)
# fit here today and their shapes all pass, and a backend override is not the
# place to change configuration that has not been measured on the device.
# ---------------------------------------------------------------------------


def _smem_per_block() -> int:
    """Shared memory Triton will let one block use, in bytes.

    From Triton's driver, not torch's device properties: Triton opts into the
    larger dynamic limit, so on NVIDIA torch reports 49152 (the static default)
    while Triton enforces 164-228KB. Both agree at 65536 here, but taking the
    wrong one would matter if this file were ever reused elsewhere.
    """
    try:
        drv = triton.runtime.driver.active
        props = drv.utils.get_device_properties(drv.get_current_device())
        return int(props["max_shared_mem"])
    except Exception:
        pass
    try:
        v = torch.cuda.get_device_properties(0).shared_memory_per_block
        if v:
            return int(v)
    except Exception:
        pass
    logger.warning("GEMS_HYGON LINALG_LSTSQ: smem limit unknown, assuming 64KB")
    return 64 * 1024


def _derive_bc(ubr: int, es: int) -> int:
    """Largest BLOCK_C leaving two blocks resident per SM.

    Fitting is not the same as fitting well: measured on a MetaX part with the
    same 64KB limit, bc=16 (24576 B, two blocks per SM) ran the update in
    0.0797 ms against bc=32 (40960 B, one block) at 0.0977 -- so the target is
    HALF the limit, not the whole of it.
    """
    bc = _generic._WY_BLOCK_C
    target = _smem_per_block() // 2
    while bc > 8 and 2 * ubr * bc * es + ubr * _generic._WY_PANEL * es > target:
        bc //= 2
    return bc


_BC = None


def _bc() -> int:
    """BLOCK_C for both dtypes, derived on FIRST USE -- never at import.

    Deriving it at import would query the Triton driver while flag_gems is
    still loading its vendor ops, i.e. before torch has touched the device.
    Nothing here needs to run before the operator is first called.
    """
    global _BC
    if _BC is None:
        bc32 = _derive_bc(min(_generic._WY_BLOCK_R, 128), 4)
        bc64 = _derive_bc(64, 8)
        # The generic driver computes the WY update grid from the MODULE
        # constant _WY_BLOCK_C while taking BLOCK_C from _wy_cfg -- and _wy_cfg
        # returns a LITERAL 64 for float64. Rebinding only the constant would
        # therefore leave float64 launching a 64-wide tile against a grid sized
        # for 16, silently under-updating the trailing block. Both are rebound,
        # which is only sound while the two dtypes derive the same value. They
        # do here (both 16). Fail loudly if a device ever makes them differ.
        if bc32 != bc64:
            raise RuntimeError(
                "hygon linalg_lstsq: fp32 and fp64 derive different BLOCK_C "
                f"({bc32} vs {bc64}); the generic grid uses a single module "
                "constant, so they must match. Fix the grid in "
                "flag_gems/ops/linalg_lstsq.py to use the per-dtype value "
                "instead."
            )
        _BC = bc32
        _generic._WY_BLOCK_C = _BC  # keep the generic grid in step with _wy_cfg
        logger.debug(
            "GEMS_HYGON LINALG_LSTSQ: BLOCK_C=%d (smem %d B)",
            _BC,
            _smem_per_block(),
        )
    return _BC


def _wy_cfg_hygon(dt):
    """(panel BLOCK_R, update BLOCK_R, BLOCK_C, num_stages), BLOCK_C derived."""
    if dt == torch.float64:
        return 256, 64, _bc(), 2
    return _generic._WY_BLOCK_R, min(_generic._WY_BLOCK_R, 128), _bc(), 3


# ---- apply, once, at import (a pure Python rebind; no device access) ----
_generic._wy_cfg = _wy_cfg_hygon


def linalg_lstsq(A, b, rcond=None, driver=None):
    """Hygon specialization: generic kernels, device-derived WY block width."""
    logger.debug("GEMS_HYGON LINALG_LSTSQ")
    _bc()  # derive the device-dependent config now, on a live context
    return _generic.linalg_lstsq(A, b, rcond=rcond, driver=driver)
