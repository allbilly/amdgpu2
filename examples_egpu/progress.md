# RX570 eGPU Bring-Up Progress (TinyGPU / M1 Mac)

**Goal:** Run vector-add on **AMD RX570 (Polaris10 / gfx803, `1002:67df`)** via **TinyGPU.app** bare-metal MMIO/PM4 — not macOS `AMDRadeon*` kexts.

**Last updated:** 2026-07-08

## Status

| Item | State |
|------|--------|
| **Solved** | ATOM `asic_init` completes — VRAM trains (`MEMSIZE=4096`, `MISC0\|0x80`, `trained=True`) |
| **Blocker** | No CPU-visible VRAM data path after training — BAR0 and MM_INDEX dead on TinyGPU/USB4 |
| **Danger** | `LoadUcodes` / unproven GART DMA **kernel-panics macOS** (full reboot; not a soft USB4 disconnect) |
| **Next** | (1) single-page GART DMA probe, (2) TrustOS-style direct MMIO firmware upload, (3) compute |
| **Safe** | `--probe`, `--selftest`, `--boot-stage=atom`, `--boot-stage=pre-fw` |

---

## Solved: ATOM VRAM training

### What looked like a hardware problem

`atom_replay.py` ran ~5k MMIO writes then hit **stuck backward-JMP loops** — mislabeled as "memory-training polls." Linux `atom.c` **aborts** `asic_init` after ~20s on the same pattern (`ctx->abort`, `-EINVAL`). Our escape hatch `AMD_ATOM_JUMP_BAIL=1` produced **fake** state (`MEMSIZE=0x10`, `MISC0=0x1800`), not real GDDR5.

**Actual root cause:** two bugs in the `atom_replay.py` bytecode VM. After fixing both, `ATOM_CMD_INIT` completes in ~0.4 s with **1008 MMIO writes**:

```
MEMSIZE = 4096 (0x1000)   MISC0 = 0x50609190 (bit 0x80 set)   trained = True
```

Confirmed via `add.py --boot-stage=atom` and `--boot-stage=pre-fw` (`trained=True`).

No Linux golden trace was needed — the VBIOS bytecode was fine; the interpreter mis-decoded operands.

### What "trained" means

Linux `amdgpu` and our `vram_training_ok()` agree on the same proof registers:

| Register | Offset | Trained (RX570) |
|----------|--------|-----------------|
| `mmCONFIG_MEMSIZE` | `0x150a` | **4096** (MB) |
| `mmMC_SEQ_MISC0` | `0xa80` | **bit `0x80` set** |
| `mmMC_VM_FB_LOCATION` | `0x809` | Valid FB base/top |

`vram_trained()` in `polaris_boot.py` requires **both** valid `CONFIG_MEMSIZE` (not 0/0xffff, ≥128 MB) **and** `MISC0|0x80`. Do not trust `MC_IO_DEBUG_UP_13` bit 23 alone — that only means MC ucode ran once.

### Two-layer training (Linux Polaris VI)

```
Layer 1 — ATOM asic_init (VBIOS bytecode, software interpreter)     ← we run this in atom_replay.py
  amdgpu_atom_execute_table(ATOM_CMD_INIT)
    → CallTable 5: MemoryControllerInit
    → MMIO polls until CONFIG_MEMSIZE + MC regs valid

Layer 2 — MC microcode (polaris10_mc.bin)                         ← optional after Layer 1
  gmc_v8_0_polaris_mc_load_microcode()
    → upload via MC_SEQ_SUP_*
    → poll mmMC_SEQ_MISC0 until bit 0x80
```

On M1 + TinyGPU there is no x86 VBIOS POST and no Linux `amdgpu` — we **must cold-boot Layer 1 ourselves** via `atom_replay.py` reading the ROM image.

### Bug 1 — `ATOM_ARG_ID` did not dereference the ROM

`atom.c` `ATOM_ARG_ID`: `val = U32(idx + gctx->data_block)` — it **reads the dword at ROM offset** `idx + data_block`. Our code used `val = idx + g.data_block` (the *address*, not the *contents*). This fed garbage into data-table-driven loops. A `data_block += ID[...]` loop counter never converged (`data_block` marched `0xa894 → … → 0xfffe` doubling each step instead of indexing a table), so the `CMP data_block == remainder` exit at `0xd2e8` never hit → infinite loop.

Fix (`_get_src_int`, `ATOM_ARG_ID`):

```python
off = (idx + g.data_block) & 0xffff
val = _u32(bios, off) if off + 4 <= len(bios) else 0
```

### Bug 2 — missing WS special registers `ATOM_WS_OR_MASK` / `ATOM_WS_AND_MASK`

`atom.h`: `ATOM_WS_SHIFT=0x43`, `ATOM_WS_OR_MASK=0x44`, `ATOM_WS_AND_MASK=0x45`, `ATOM_WS_FB_WINDOW=0x46`, `ATOM_WS_ATTRIBUTES=0x47`. Our map had `FB_WINDOW=0x46` but **omitted `0x44`/`0x45`** and let `ws[]` shadow the special regs. Per `atom.c` the `0x40–0x48` switch **takes priority** over `ws[idx]`; `OR_MASK = 1<<shift`, `AND_MASK = ~(1<<shift)` are **read-only** derived values. Mask-building loops (bit set/clear on MC regs) produced wrong masks. Fixed read + write paths so specials win and OR/AND masks compute from `shift`.

### VBIOS parser additions (supporting ATOM, from NootedRed + linux headers)

| Function | Purpose |
|----------|---------|
| `check_atom_bios()` | `0xAA55` + `ATOM`/`MOTA` magic |
| `mdt_offset()` / `MDT_IDX_*` | Master data table lookup (`VRAM_INFO=0x1C`, etc.) |
| `parse_firmware_info()` | `main_call_parser`, `bios_scratch_reg_start` |
| `parse_vram_info()` | GDDR5 size, channels, `mc_phyinit_off` from ROM |
| `atom_info()` | Extended dump when `DEBUG=1` |

Inspect ROM offline:

```bash
python3 -c "from atom_replay import atom_info; print(atom_info(open('/tmp/rx570.rom','rb').read()))"
```

---

## Blocker: trained VRAM, dead CPU data path

GDDR5 trains (MC regs look sane), but the host cannot read/write framebuffer memory:

| Path | Result |
|------|--------|
| BAR0 (`dev.vram[off]`) | Fixed constant at every offset; writes do not stick |
| MM_INDEX (`pos \| 0x80000000`) | Same constant — floating/aliased BAR, not VRAM |
| MM_INDEX at MC base (`FB_LOC 0xf4fff400` → `0xf400000000 + off`) | Still dead |
| Reprogram FB to 0-based + SYS aperture | Still dead |

Post-train MC routing: `FB_LOCATION=0xf4fff400`, `FB_OFFSET=0`, `BIF_FB_EN=0x3`, `MC_ARB_RAMCFG=0x692`.

**Conclusion:** TinyGPU's BAR0 mapping does not reach trained VRAM. Transport/aperture limitation — not a training gap. May need TinyGPU-side BAR handling or an aperture-window register we have not found.

`LoadUcodes` cannot proceed: Linux puts TOC + scratch in **VRAM**; SMC DMA-reads them. We have no CPU path to populate those buffers.

**`test_gtt_load.py` post-mortem:** `AMD_BOOT_FW_LAYOUT=gtt` + RLC-only `LoadUcodes` **kernel-panicked macOS** mid-run. Do not retry full GTT LoadUcodes until a tiny GART DMA probe passes.

---

## Safety

Risky runs cause **whole macOS kernel panic** (machine reboots). After reboot the eGPU may show `pci=0xffff` until physically replugged — that is fallout, not the failure mode itself.

| Command | Safe? | Notes |
|---------|-------|-------|
| `python3 add.py --probe` | Yes | Few reads; stop if `pci=0xffff` |
| `python3 add.py --selftest` | Yes | Transport only |
| `python3 add.py --boot-stage=atom` | Low–medium | ~5k MMIO |
| `python3 add.py --boot-stage=pre-fw` | Medium | Full boot except LoadUcodes |
| `python3 add.py` (full) | Not recommended | Skips LoadUcodes when BAR0 dead, but still heavy MMIO |
| `AMD_BOOT_LOADUCODES_UNTRAINED=1` | **Unsafe** | Forces LoadUcodes → kernel panic |
| `test_gtt_load.py` | **Unsafe** | Proven macOS panic |

Gates in `load_ip_firmware_prereqs()` refuse LoadUcodes unless BAR0 or MM_INDEX probe passes (or forced via env).

Default `add.py` now: ATOM train → SMC boot → **skip LoadUcodes** → compute attempt → expected `AssertionError` (no MEC fw). No 30–120 s hang, no panic.

---

## Next steps (ordered)

1. **GART-sysmem DMA probe** — map one `alloc_sysmem` page into GART; single SDMA/engine read; PCI health check; abort on `0xffff`. Do **not** run full LoadUcodes first.
2. **Direct MMIO firmware load** — port TrustOS `polaris_sdma_full_init` style upload (RLC/MEC via registers, bypass `PPSMC_MSG_LoadUcodes`) once step 1 works.
3. **Compute** — only after firmware is resident.

Open question: can SMC DMA to trained VRAM MC addresses even though CPU BAR0 is dead? Probe step 1 answers the GART/sysmem side.

---

## Linux boot order (Polaris VI — our target)

```
amdgpu_device_init()
  ATOM_CMD_INIT                    # Layer 1 — done via atom_replay.py
  gmc_v8_0_hw_init
    mc_program
    gmc_v8_0_polaris_mc_load_microcode   # poll MISC0 bit 0x80
    gmc_v8_0_gart_enable                 # before LoadUcodes
  polaris10_start_smu
  smu7_request_smu_load_fw               # TOC/scratch VRAM, fw_buf GTT
  gfx_v8_0_hw_init / compute
```

**LoadUcodes message sequence:** `SMU_DRAM` 0x252/0x253 → build TOC → `DRV_DRAM` 0x250/0x251 → `LoadUcodes` 0x254 → poll `UcodeLoadStatus` @ soft_regs+0x6c.

Our `polaris_boot.boot()` order:

```
vi_common_init → enable_vbios_rom → ATOM asic_init
→ gmc_sw_init → start_smc
→ mc_program → load_mc_firmware → gart_enable
→ load_ip_firmware (only if prereqs pass)
→ enable_compute → init_compute_queue
```

---

## Hardware & key files

| Item | Value |
|------|-------|
| GPU | RX570 Polaris10, `1002:67df` |
| Host | M1 Mac, USB4 eGPU, TinyGPU.app → `APLRemotePCIDevice` |
| Transport template | **`allbilly/nvgpu`** ← **`tinygrad/tinygrad`** `APLRemotePCIDevice` in `runtime/support/system.py` |
| BARs | BAR0 VRAM, BAR2 doorbells, BAR5 MMIO |
| Linux ref | `ref/linux/drivers/gpu/drm/amd/` |

| File | Role |
|------|------|
| `add.py` | Transport, `PolarisDevice`, CLI, PM4 |
| `polaris_boot.py` | VI boot: SMC, MC, GART, LoadUcodes gates |
| `atom_replay.py` | ATOM `asic_init` interpreter |
| `diag_bar0.py` | BAR0 aperture diagnosis |
| `test_gtt_load.py` | **Unsafe** — GTT LoadUcodes experiment |
| `shaders/egpu-add4.s` | gfx803 add kernel |

---

## What works

- [x] GPU enumeration (`--probe`, `--reset`)
- [x] ATOM `asic_init` → VRAM trained (`trained=True`)
- [x] SMC upload + mailbox (`resp=0x1`, segmented upload)
- [x] VBIOS ROM read (`0xe974aa55`)
- [x] Golden regs + doorbells (`vi_common_init`)
- [x] LoadUcodes safety gates (skip when BAR0/MM dead)
- [x] MMIO drain (`AMD_MMIO_DRAIN_EVERY`)

## Todo

- [x] ATOM training (`atom_replay.py` bugs fixed)
- [x] SMC boot
- [ ] CPU-visible VRAM path (BAR0 or MM_INDEX) **or** proven GART DMA
- [ ] `load_ip_firmware` / firmware resident
- [ ] Vector-add via `add.py`

---

## References — DeepWiki re-rank (2026-07-08, latest)

Scored **0–10** for the **current** blocker: trained VRAM, dead BAR0/MM_INDEX, GART-sysmem DMA probe, direct MMIO fw upload, avoid macOS kernel panic. **Not** ATOM training (solved). DeepWiki MCP + code review (`examples_egpu/add.py` lineage).

### `tinygrad/tinygrad` review (DeepWiki 2026-07-08)

**Useful for transport only — not Polaris boot.**

| Layer | tinygrad path | RX570 / gfx803? |
|-------|----------------|-----------------|
| **TinyGPU transport** | `runtime/support/system.py` → `APLRemotePCIDevice`: unix socket to TinyGPU.app, `MAP_BAR`, `MAP_SYSMEM_FD`, `read_config`/`write_config` | **Yes** — vendored into `examples_egpu/add.py` (via nvgpu) |
| **PrepareDMA** | TinyGPU.app driver (`TinyGPUDriverUserClient.cpp`); phys segs written into sysmem shm | **Yes** — GART PTE targets need these paddrs |
| **Setup** | `extra/setup_tinygpu_osx.sh` (referenced by `add.py` on connect failure) | **Yes** |
| **AMD compute boot** | `runtime/ops_amd.py` → `AMDDevice` asserts `gfx90402` / `gfx90500` / `gfx11+` only | **No gfx803** |
| **Bare-metal AMD init** | `runtime/support/am/amdev.py` → `AMDev` PSP→MP1→MMHUB (RDNA) | **Wrong path** — use linux VI + `polaris_boot.py` |
| **Linux driver path** | `KFDIface` + `/dev/kfd` | **N/A** on M1 TinyGPU |

DeepWiki rated **2/10** for Polaris bring-up (correct for `AMDev`/`ops_amd`; underrates transport). **Adjusted: 7/10** — authoritative upstream for the same TinyGPU stack nvgpu and `examples_egpu/add.py` use; ignore `AMDev` for RX570.

### `GatoAmarilloBicolor/AMDstracted-GPU` review (DeepWiki 2026-07-08)

**Useful as modular VI/Polaris reference — not M1/TinyGPU transport.**

| Layer | AMDstracted-GPU path | RX570 / M1 TinyGPU? |
|-------|----------------------|---------------------|
| **Architecture** | HAL + IP-block lifecycle (`early_init` → `hw_init`); `OBJGPU` with `mmio_base`, `ip_blocks[]` | **Reference** — mirrors linux VI ordering |
| **Polaris VI** | `gmc_v8_0.c`, `gfx_v8_0.c` (`CHIP_POLARIS10/11/12`), `vi.c` + SDMA v2.4/v3.0 | **Yes** — gfx803/GCN3 blocks present |
| **GART / MC** | `gmc_v8_0_mc_program`, `gmc_v8_0_polaris_mc_load_microcode`, TLB flush | **Yes** — same regs as linux `gmc_v8_0.c` |
| **Firmware** | `polaris10_mc/rlc/mec/mec2/smc.bin` via `gfx_v8_0_init_microcode`, `amdgpu_cgs.c` | **Yes** — same bins; no TrustOS-style MMIO bypass |
| **ATOM** | `amdgpu_atombios.h` / `atom.h` in VI path | **Partial** — uses kernel ATOM, not bare `atom_replay.py` |
| **Command rings** | `amdgpu_command_submit_hal`, doorbells, GFX/compute/SDMA rings | **Post-fw** — needs working memory path first |
| **Platform** | Linux DRM ioctl (`DRM_IOCTL_AMDGPU_CS`); Haiku/FreeBSD direct MMIO; sim fallback | **No macOS, Apple Silicon, TinyGPU, or USB4** |
| **Tests** | 11/11 pass (mostly hardware simulation per DW) | Not validated on RX570 eGPU |

DeepWiki rated **7–10** (generic cross-platform amdgpu bring-up; claims “production-ready” HAL). **Adjusted: 6/10** — cleaner IP-block navigation than spelunking full `torvalds/linux`, and confirms our `polaris_boot.py` ordering (GMC → GART → SMC → gfx fw). Does **not** solve BAR0-dead, GART-sysmem DMA on TinyGPU, or macOS panic gates; linux amdgpu remains canonical and this repo is derivative.

### Full ranking

| Rank | Score | Repo | Verdict |
|------|-------|------|---------|
| 1 | **10** | **torvalds/linux** | Canonical AMD: `gmc_v8_0_gart_enable`, `smu7_request_smu_load_fw`, `amdgpu_device_mm_access`, `polaris10_smumgr.c`. DW: 6. |
| 2 | **10** | **allbilly/amdgpu** | **This repo** — `examples_egpu/`, `polaris_boot.py`, `atom_replay.py`, GART, `sysmem_dma_flush`, LoadUcodes gates. DW: 2. |
| 3 | **9** | **ROCm/amdgpu** | Same amdgpu tree as linux (DW index missed Polaris; interchangeable). |
| 4 | **8** | **nathan237/TrustOS** | `firmware.rs`: direct MMIO RLC/MEC/SDMA, bypass `LoadUcodes`; `polaris_gmc_init` golden L2. DW: 7. |
| 5 | **8** | **allbilly/nvgpu** | **Applied TinyGPU template** — working NV bare-metal on M1: `examples/add.py`, `middle_nv.py`, probe/selftest, sysmem DMA. `examples_egpu/add.py` mirrors this. DW: 2 (NV-only). |
| 6 | **7** | **tinygrad/tinygrad** | **Upstream transport** — `APLRemotePCIDevice`, `MAP_BAR`, `MAP_SYSMEM_FD`, PrepareDMA, `setup_tinygpu_osx.sh`. **Do not use** `AMDev`/`ops_amd` for gfx803. DW: 2 → **7** for transport. |
| 7 | **7** | **geerlingguy/raspberry-pi-pcie-devices** | [#756](https://github.com/geerlingguy/raspberry-pi-pcie-devices/discussions/756) ARM DMA coherency → `sysmem_dma_flush`. DW: 0. |
| 8 | **6** | **GatoAmarilloBicolor/AMDstracted-GPU** | HAL + `gmc_v8_0`/`gfx_v8_0` Polaris VI ref; GART + polaris10 fw bins. No macOS/TinyGPU. DW: 7–10 → **6**. |
| 9 | **5** | **komen205/polaris30-smu-bist** | `1002:67DF` UEFI SMU7 BIST after DMA works. DW: 10 — overrated; x86 only. |
| 10 | **4** | **allbilly/AArch64-Explore-GPU** | AArch64 / Apple-Silicon GPU bring-up notes. |
| 11 | **4** | **xCuri0/ReBarUEFI** | BAR sizing theory; PC UEFI only. DW: 9 — overrated for M1. |
| 12 | **4** | **Aitbytes/proxmox-amd-gpu-passthrough** | `67DF` reset / Code 43 symptom parallel. DW: 10 — overrated. |
| 13 | **3** | **allbilly/mesa-mesa** | Mesa/radeonsi reference post-fw. |
| 14 | **3** | **tinygrad/7900xtx** | `polaris10_mec.bin` PM4 notes — after fw loads. |
| 15 | **3** | **boopdotpng/tenstorrent-docs** | Host-memory DMA model contrast. |
| 16 | **2** | **kc9zda/atombios-inspect** | Offline ROM audit (training solved). |
| 17 | **2** | **ChefKissInc/NootedRed** | `ATOMBIOS.hpp` ported; Vega iGPU kext. |
| 18 | **2** | **allbilly/miaow** | GCN Southern Islands RTL sim (gfx803-adjacent). |
| 19 | **2** | **vosen/amdgpu_debug** | Post-boot rocgdb only. |
| 20 | **1** | **Zile995/…** / **heavyarms2112/atitool** / **Andybf/AtomBiosEditor** | VFIO / Linux-only / offline editor. |
| 21 | **1** | **gem5** / **mgpusim** / **gpgpu-sim** / **miaow (VRG)** / **rdna-sim** | Simulators. |
| 22 | **0** | **NootRX** / **WhateverGreen** / **VirtualSMC** / **Hackintosh** / **ZLUDA** / **coreboot** | Wrong layer. |
| 23 | **0** | **allbilly/applegpu** / **amd_scheduler** / **ml_workload** / **allbilly/tinygrad** fork | Other stacks (forks duplicate upstream tinygrad). |

**Takeaway:** Tier-1 AMD = **linux + this repo + TrustOS `firmware.rs`**. Tier-1b transport = **`nvgpu` (working example) + `tinygrad/tinygrad` (upstream `APLRemotePCIDevice` / TinyGPU.app)**. Tier-2 VI reference = **AMDstracted-GPU** (`gmc_v8_0.c`, `gfx_v8_0.c` — optional cleaner read vs full linux tree). Tier-2 DMA = **rpi-pcie #756**. Chain: `tinygrad/system.py` → `nvgpu/add.py` → `examples_egpu/add.py` → `polaris_boot.py` (linux VI boot, not `AMDev`).

### Primary files to read

| Source | Path | Use for |
|--------|------|---------|
| Linux VI boot | `torvalds/linux` → `gmc_v8_0.c`, `atom.c`, `polaris10_smumgr.c` | GART, LoadUcodes, MM_INDEX fallback |
| VI HAL ref (optional) | `GatoAmarilloBicolor/AMDstracted-GPU` → `gmc_v8_0.c`, `gfx_v8_0.c`, `vi.c` | IP-block lifecycle, polaris10 fw load order |
| TrustOS fw | `nathan237/TrustOS` → `kernel/.../firmware.rs` | Direct MMIO upload, GMC golden regs |
| nvgpu (applied) | `allbilly/nvgpu` → `examples/add.py`, `TODO.md` | Bare-metal eGPU pattern on TinyGPU |
| **tinygrad (upstream)** | `tinygrad/tinygrad` → `runtime/support/system.py`, `extra/setup_tinygpu_osx.sh` | `APLRemotePCIDevice`, BAR/sysmem RPC, TinyGPU install |
| Local | `examples_egpu/add.py`, `polaris_boot.py`, `diag_bar0.py` | AMD port + VI boot |

Skip: `AMDev` / `ops_amd` bare-metal boot (RDNA-only), macOS kexts, VFIO, VBIOS editors, simulators.

---

## Test commands

```bash
cd examples_egpu

# After macOS panic reboot (replug eGPU if pci=0xffff):
python3 add.py --reset
python3 add.py --probe

AMD_BOOT_VBIOS_FILE=/tmp/rx570.rom \
  python3 add.py --boot-stage=atom      # trained=True expected

AMD_BOOT_VBIOS_FILE=/tmp/rx570.rom \
  python3 add.py --boot-stage=pre-fw    # check bar0/mm/load_ok

python3 diag_bar0.py                    # BAR0 diagnosis

# DO NOT until GART probe passes:
# python3 add.py
# python3 test_gtt_load.py
# AMD_BOOT_LOADUCODES_UNTRAINED=1 python3 add.py
```

---

## Key environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `AMD_BOOT_LOADUCODES_UNTRAINED` | `0` | `1` = force LoadUcodes (panic risk) |
| `AMD_BOOT_FW_LAYOUT` | `auto` | `vram` / `hybrid` / `gtt` |
| `AMD_BOOT_SYSMEM_FLUSH` | `1` | `msync` before SMC DMA read |
| `AMD_BOOT_SMC_UPLOAD` | `segmented` | SMC fw upload mode |
| `AMD_BOOT_SMC_FLUSH_READ` | `0` | Skip risky post-upload SMC RAM read |
| `AMD_MMIO_DRAIN_EVERY` | `128` | Drain TinyGPU MMIO queue |
| `AMD_BOOT_VBIOS_FILE` | — | Path to `rx570.rom` |
| `AMD_ATOM_JUMP_BAIL` | `0` | `1` = fake-complete ATOM (obsolete now) |
| `DEBUG` | `0` | Verbose logging |
