# Terascale eGPU (`examples_egpu_terrascale`)

Bare-metal bring-up scaffold for **pre-GCN** AMD cards over TinyGPU (same transport
as `examples_egpu/`), targeting:

| Product | ASIC | Family (`radeon_family.h`) | TeraScale | PCI (examples) |
|---------|------|----------------------------|-----------|----------------|
| HD 5570 | Redwood | `CHIP_REDWOOD` | 2 (Evergreen) | `1002:68D9` (+ 68D8/68DA…) |
| HD 4850 | RV770 | `CHIP_RV770` | 1 (R700) | `1002:9442` |

Linux driver: **`drm/radeon`** (not `amdgpu`). Sources under
`ref/linux/drivers/gpu/drm/radeon/` — especially `evergreen.c`, `r600.c`,
`rv770.c`, `evergreend.h`, `r600d.h`.

User-space compute reference: Mesa **r600g** `evergreen_compute.c`
(OpenCL on Evergreen).

## Status

`add.py` is offline-first, and the HD 4850 CP smoke path has been exercised
with AGP-mapped host sysmem while local VRAM remains unusable:

```bash
python3 examples_egpu_terrascale/add.py --selftest --chip=hd5570
python3 examples_egpu_terrascale/add.py --selftest --chip=hd4850
python3 examples_egpu_terrascale/add.py --dry-run --chip=hd5570
python3 examples_egpu_terrascale/add.py --list-chips
# when card + TinyGPU are up:
python3 examples_egpu_terrascale/add.py --probe --chip=hd5570
```

| Piece | HD 5570 (Evergreen) | HD 4850 (RV770) |
|-------|---------------------|-----------------|
| TinyGPU PCI/MMIO | ready | ready |
| Chip / PCI ID table | ready | ready |
| `r600_cp_resume` MMIO sequence (dry-run) | ready | ready |
| Evergreen LS compute IB (Mesa-shaped) | ready | n/a |
| Real r600 CF/ALU shader (`llvm -march=r600`) | **TODO** (stub blob) | **TODO** |
| ATOM / MC / CP boot on TinyGPU | **TODO** | **TODO** |
| RAT / global buffer bindings | **TODO** | n/a |

For RV770, `--cp-mem-write-test` does not use local VRAM: it places the CP ring,
writeback page, and `MEM_WRITE` result buffer in contiguous host sysmem behind
the AGP aperture. BAR0 is deliberately lazy and only mapped for `--vram-probe`
or an explicitly requested BAR0 probe. It is strictly a CP payload-write test,
not GPU arithmetic; the default command refuses to CPU-offload an add. This is
the supported diagnostic path while GDDR3 writes return a floating-bus value.

## Evergreen compute path (HD 5570)

Mirrors Mesa `evergreen_emit_cs_shader` + `evergreen_emit_dispatch`:

1. `PACKET3_SET_CONTEXT_REG` → `SQ_PGM_START_LS` / `SQ_PGM_RESOURCES_LS` (`va >> 8`)
2. `PACKET3_SET_CONFIG_REG` → `VGT_COMPUTE_START_*`, `VGT_COMPUTE_THREAD_GROUP_SIZE`
3. `PACKET3_SET_CONTEXT_REG` → `SPI_COMPUTE_NUM_THREAD_{X,Y,Z}`, `SQ_LDS_ALLOC`
4. `PACKET3_DISPATCH_DIRECT` with **compute bit** (Mesa `PKT3C`) and initiator `1`

Registers: `ref/linux/.../evergreend.h` (`VGT_COMPUTE_*`, `SQ_PGM_START_LS`, …).
CP ring bring-up: `r600_cp_resume` / `r600_cp_start` in `r600.c`.

ISA is **r600 CF+ALU** (VLIW), not GCN VOP2. Do not reuse `examples_egpu` GCN
shaders. Next step: assemble with `llvm-mc`/`llc` `-march=r600 -mcpu=redwood`.

## RV770 path (HD 4850)

Shares R600 CP (`r600_cp_resume`) but **no Evergreen LS compute**. A genuine
RV770 pixel shader source now lives in `rv770_add.ll`; it compiles to four R600
hardware `ADD` ALU instructions plus a color export. Its matching vertex shader
is `rv770_vs.ll`, which exports clip position and the two vec4 operands to the
pixel stage:

```bash
python3 examples_egpu_terrascale/add.py --compile-rv770-add
AMD_BOOT_ATOM=0 python3 examples_egpu_terrascale/add.py --gpu-add-preflight
```

The compiler inspection does not touch hardware. The preflight allocates the
48-byte VS, 64-byte PS, three-vertex input buffer, and FP32 output target in
contiguous AGP sysmem, but deliberately issues no graphics packets. The
remaining work is to bind that state through the RV770 graphics draw pipeline
and read the GPU-produced result. Until that exists, default `add.py` refuses
to replace GPU arithmetic with a CPU calculation.

## vs Polaris (`examples_egpu`)

| | Polaris RX570 | Terascale HD 5xxx/4xxx |
|--|---------------|-------------------------|
| Driver ref | `amdgpu` / gfx8 | `radeon` / r600+evergreen |
| Compute | MEC + `COMPUTE_*` SH regs | GFX CP + LS (`SQ_PGM_START_LS`) |
| ISA | GCN3 | r600 VLIW |
| Ring | KCQ/MQD (VI) | classic `CP_RB_*` |

VRAM/AGP lessons from Polaris still apply on eGPU: prefer host-visible buffers
until BAR0/HDP writeback is proven on each card.

## Reference repos / docs (websearch)

Ranked for this bring-up (HD 5570 Evergreen compute + HD 4850 R700 CP):

| Rank | Repo / doc | Why |
|-----:|------------|-----|
| 1 | [`mesa/mesa`](https://gitlab.freedesktop.org/mesa/mesa) `src/gallium/drivers/r600/evergreen_compute.c` | LS compute IB: `SQ_PGM_START_LS`, `DISPATCH_DIRECT`, RAT/LDS |
| 2 | `ref/linux` → [`torvalds/linux`](https://github.com/torvalds/linux) `drivers/gpu/drm/radeon/` | `evergreen.c` / `r600.c` / `rv770.c`, `evergreend.h`, `r600_cs.c` |
| 3 | [`llvm/llvm-project`](https://github.com/llvm/llvm-project) AMDGPU `r600` | `-march=r600 -mcpu=redwood` / `rv770` shader codegen |
| 4 | [libclc](https://github.com/libclc/libclc) + Mesa **Clover** / **Rusticl** (`RUSTICL_ENABLE=r600`) | OpenCL → r600 binary path (historical GalliumCompute) |
| 5 | [CLRX/CLRX-mirror](https://github.com/CLRX/CLRX-mirror) | GalliumCompute binary / asm notes (more GCN-focused; still useful) |
| 6 | [X.Org AMD docs](https://www.x.org/docs/AMD/old/) | [r600 ISA](https://www.x.org/docs/AMD/old/r600isa.pdf), [Evergreen accel](https://www.x.org/docs/AMD/old/evergreen_cayman_programming_guide.pdf) |
| 7 | TechPowerUp ISA PDFs | [R700 ISA](https://www.techpowerup.com/gpu-specs/docs/ati-r700-isa.pdf), [Evergreen ISA](https://www.techpowerup.com/gpu-specs/docs/ati-evergreen-isa.pdf) |
| 8 | [RadeonFeature](https://www.x.org/wiki/RadeonFeature/) / [GalliumCompute](https://wiki.freedesktop.org/dri/GalliumCompute/) | Family decoder ring + OpenCL stack overview |

**Not useful here:** `amdgpu`/ROCm (GCN+), TrustOS `neural.rs` (RDNA), Polaris `examples_egpu` GCN shaders.

DeepWiki: ask `mesa/mesa`, `torvalds/linux`, `llvm/llvm-project` (listed in root `AGENTS.md`).
