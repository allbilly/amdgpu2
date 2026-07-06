# Adding Renoir (gfx90c) support to tinygrad

This document records the patches applied to a fresh `tinygrad` clone to make
it run on the AMD Renoir APU (`gfx90c`, GFX IP version `(9, 3, 0)`) on this
machine, plus the limitations and follow-up work needed.

## Result

`Tensor([1.0, 2.0, 3.0, 4.0]) + Tensor([5.0, 6.0, 7.0, 8.0])` →
`[6. 8. 10. 12.]` — a real tinygrad kernel executed on the Renoir GPU via
`libdrm_amdgpu` and KFD, and produced the correct result.

A subsequent test hung the GPU (KFD is no longer accepting opens, and
`/dev/dri/renderD128` is also stuck). Cannot recover without `sudo rmmod
amdgpu`. The teardown/cleanup path is the likely culprit, not the kernel
itself.

## What was discovered about the box

| Item | Value | Source |
|---|---|---|
| `gfx_target_version` (KFD) | `90012` | `kfd/topology/nodes/1/properties` |
| GC IP version | `(9, 3, 0)` | `ip_discovery/die/0/GC/0/{major,minor,revision}` |
| MP1 IP version | `(12, 0, 0)` | `ip_discovery/die/0/MP1/0/...` |
| NBIF IP version | `(2, 5, 0)` | `ip_discovery/die/0/NBIF/0/...` |
| SDMA0 IP version | `(4, 1, 2)` | `ip_discovery/die/0/SDMA0/0/...` |
| MMHUB IP version | `(1, 5, 0)` | `ip_discovery/die/0/MMHUB/0/...` |
| Firmware in `/lib/firmware/amdgpu/` | `renoir_*.bin.xz` (symlinks to `green_sardine_*.bin.xz`) | filesystem |
| KFD | `/dev/kfd` present, 1 GPU node (id=2465), 1 CPU node | `/sys/devices/virtual/kfd/kfd/topology/nodes/` |
| Driver | `amdgpu` 6.x kernel module loaded | `lsmod` |
| `cwsr_size` (kernel-reported) | 2441216 (≈2.4 MB) | KFD properties |

The kernel hardcodes `gfx_target_version = 90012` for Renoir in
`drivers/gpu/drm/amd/amdkfd/kfd_device.c` (in the `case IP_VERSION(9, 3, 0)`
arm of the switch). The format is `MMmmRR` and is decoupled from the actual
IP discovery version.

## Why each patch is needed

### Patch A: arch string override
- File: `tinygrad/runtime/ops_amd.py:960-962`
- The kernel reports `gfx_target_version = 90012`, which tinygrad decodes as
  `(9, 0, 12)` and builds `arch = "gfx90c"` via the formula `"gfx%d%x%x" %
  (9, 0, 12)`. We override the integer mapping (`90012 → 90000`) and the arch
  string so that the target becomes `(9, 0, 0)` and the arch string becomes
  the LLVM-recognised `"gfx90c"`. The `(9, 0, 0)` tuple is then added to the
  supported-targets assertion.

### Patch B: GC register file pin
- File: `tinygrad/runtime/ops_amd.py:992-995`
- `import_asic_regs("gc", (9, 3, 0))` fails because `reg_files["gc"]` only
  has `(9, 4, 3)` for the GFX9 family. We pin GFX9 to `(9, 4, 3)` (closest
  available) when the actual IP version is not in the supported set.

### Patch C: NBIO register file pin
- File: `tinygrad/runtime/ops_amd.py:1000-1005`
- Same problem for `nbio`. The kernel reports `NBIF_HWIP = (2, 5, 0)`, but
  tinygrad's `reg_files["nbio"]` jumps straight to `(4, 3, 0)` as the
  smallest. We pin GFX9 to `(7, 2, 0)` (Vega20's NBIO).

### Patch D: skip HDP flush in `memory_barrier()` for APU
- File: `tinygrad/runtime/ops_amd.py:133-141`
- The Vega20 NBIO register set is **wrong** for Renoir's NBIO 2.3 even
  though the import succeeds. The `BIF_BX_PF0_GPU_HDP_FLUSH_*` register
  addresses returned (`0xe26` for instance 0) point at unrelated hardware
  on Renoir. The `wait_reg_mem` PM4 packet that polls the HDP flush
  register spins forever on the wrong address.
- Renoir is an APU: the GPU and CPU share the system memory coherency
  domain, so the HDP flush is not needed. We skip the HDP flush when the
  device has `local_mem_size == 0` (the APU marker).
- Verification: without this patch, even a single `q.signal(...).submit(...)`
  blocks the queue (signal never advances past 1). With the patch, kernels
  run end-to-end.

## What still needs to be done (limitations of this patch set)

1. **NBIO register layout is wrong for Renoir even with the version pin**.
   The HDP flush happens to be skipped (patch D), but any other NBIO
   register access will silently hit the wrong address. The proper fix is
   to generate `nbio_2_3_0.py` (or at minimum a Renoir-specific override of
   the HDP register offsets).

2. **GC register layout uses Vega20 offsets** (patch B). This works
   because Vega20 and Renoir have the same SH reg space for `COMPUTE_*`
   registers — which is all the user-space PM4 layer needs. The hidden
   cost is that the GC_INFO-derived `cu_per_simd_array` etc. use Renoir's
   real values, but the register *addresses* come from Vega20. If the
   driver ever needs to read a GC register (e.g. for profiling via
   `PMCSample`), it will use the wrong address.

3. **No `gc_9_3_0` firmware entries in `tinygrad/runtime/autogen/am/fw.py`**.
   The KFD path doesn't load firmware itself (the kernel does), so this
   isn't a blocker for `KFDIface`. The `AMDev` (PCIIface / AM path) path
   *would* need it, and is therefore still broken on Renoir.

4. **No `mp_13_0_10.py`** for the SMU. Same situation as above: not needed
   for the KFD path, but would block the AM path.

5. **The kernel's CWSR setup differs from tinygrad's CWSR size calculation**.
   The kernel reports `cwsr_size=2441216` (~2.4 MB), but tinygrad computes
   `wg_data_size + ctl_stack_size = ~4.3 MB` (using `vgpr_size_per_cu=0x80000
   for GFX9` and the cu_count from KFD). The first test worked, so the
   difference is not fatal, but the values are inconsistent. A later kernel
   run hung — this is the most likely cause of the hang.

6. **`.xz` firmware files are not auto-extracted by `fetch_fw()`**. The
   `fetch_fw` helper only handles `.zst`:
   ```python
   if sys.version_info >= (3,14) and (p:=pathlib.Path(f"/lib/firmware/{path}/{name}.zst")).is_file():
       from compression.zstd import decompress
       if hashlib.sha256(b:=decompress(p.read_bytes())).hexdigest() == sha256: return b
   ```
   Renoir's firmware on this Fedora box is `.xz` compressed. The KFD path
   doesn't use `fetch_fw` (the kernel handles firmware load), so this isn't
   blocking. The `AMDev` path would need either a path to extract `.xz`
   firmware or pre-extracted firmware placed in `/lib/firmware/amdgpu/`.

7. **No proper GPU recovery**. Once the GPU hangs, only `sudo rmmod amdgpu`
   can reset it. Tinygrad's `can_recover` flag is only set for the AM path,
   not for KFD.

8. **The final tear-down/cleanup of the KFD queue is what hung on the
   second kernel run**. The first kernel finished cleanly. Likely culprits:
   the CWSR teardown or the second `release_mem` for a different
   `signal.value_addr` page.

## Verified end-to-end

```
DEV=AMD:LLVM LLVM_PATH=/home/linuxbrew/.linuxbrew/lib python3 -c "
import os
os.environ['DEV'] = 'AMD:LLVM'
os.environ['LLVM_PATH'] = '/home/linuxbrew/.linuxbrew/lib'
import sys; sys.path.insert(0, '/usr/lib64/python3.14/site-packages')
import numpy as np
from tinygrad.tensor import Tensor
a = Tensor([1.0, 2.0, 3.0, 4.0])
b = Tensor([5.0, 6.0, 7.0, 8.0])
print('a+b =', (a + b).numpy())
"
# → a+b = [ 6.  8. 10. 12.]
```

## Files changed
- `tinygrad/tinygrad/runtime/ops_amd.py` — 4 patches (A, B, C, D above)

## Recommended next steps
1. Generate a real `nbio_2_3_0.py` from the Renoir kernel headers
   (`drivers/gpu/drm/amd/include/asic_reg/nbio/nbio_2_3_0_offset.h` and
   `..._sh_mask.h`).
2. Generate `mp_13_0_10.py` from `smu_v13_0_10_pmfw.h` and
   `smu_v13_0_10_ppsmc.h`.
3. Add Renoir firmware hashes to `fw.py` (use `sha256sum` after `xzcat`).
4. Investigate the KFD teardown hang — possibly related to `cwsr_size`
   mismatch (#5 above).
5. Once stable, these patches could be sent upstream.
