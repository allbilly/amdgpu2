# RX570 eGPU Bring-Up Progress (TinyGPU / M1 Mac)

**Goal:** Run vector-add on **AMD RX570 (Polaris10 / gfx803, `1002:67df rev 0xef`)** via **TinyGPU.app** bare-metal MMIO/PM4 — not macOS `AMDRadeon*` kexts.

**Last updated:** 2026-07-07 (evening session — Linux order + AGP layout)

---

## Latest Session (2026-07-07 PM) — Hybrid VRAM+AGP layout, MM_INDEX path

### Research (linux amdgpu + tinygrad + DeepWiki)

**Linux `smu7_init` / `smu7_request_smu_load_fw` buffer domains (confirmed):**
| Buffer | Domain | SMC address |
|--------|--------|-------------|
| `header_buffer` (TOC) | **VRAM** (`AMDGPU_GEM_DOMAIN_VRAM`) | `header_buffer.mc_addr` |
| `smu_buffer` (scratch) | **VRAM** | `smu_buffer.mc_addr` |
| `fw_buf` (IP images) | **GTT** | `fw_buf_mc + offset` via `amdgpu_gmc_agp_addr` = `agp_start + dma_address` |

**VRAM CPU write when BAR0 dead:** Linux `amdgpu_device_vram_access` falls back to `amdgpu_device_mm_access` using `mmMM_INDEX` (0x0) / `mmMM_DATA` (0x1) with `pos | 0x80000000`. VI HDP flush/invalidate via `mmHDP_MEM_COHERENCY_FLUSH_CNTL` / `mmHDP_DEBUG0`.

**4GB VRAM addressing:** `vram_start=0`, `vram_end=0xffffffff`, `vram_visible_mc=0xf0000000` (last 256MB BAR window) — correct for full 4GB.

### Code changes this session (continued)

| Change | Detail |
|--------|--------|
| **Hybrid uses GART not AGP** | Linux VI has AGP disabled (`agp_size=0`); fw_buf uses GART VA + PTE bind |
| **GART before LoadUcodes** | `gart_enable()` moved before `load_ip_firmware()` (eGPU needs HW VM live) |
| **Contiguous sysmem** | `alloc_sysmem(contiguous=True)` for firmware buffers |
| **CONFIG_MEMSIZE** | Program `mmCONFIG_MEMSIZE` from `AMD_VRAM_MB` when hardware reads 0 |
| **Enhanced `--probe`** | Tests BAR0, MM_INDEX, sysmem paddr/AGP after `mc_program` |

### Env vars (new/updated)

| Variable | Default | Purpose |
|----------|---------|---------|
| `AMD_BOOT_FW_LAYOUT` | `auto` | `vram`/`hybrid`/`gtt`/`agp` |
| `AMD_BOOT_FORCE_HYBRID` | `0` | Try hybrid even if MM_INDEX probe fails |
| `AMD_BOOT_HDP_NONSURFACE` | `1` | Program HDP_NONSURFACE in mc_program |
| `AMD_BOOT_AGP_RAW_PHYS` | `0` | Legacy all-sysmem layout only |

### Test commands

```bash
cd examples_egpu
python3 add.py --reset && python3 add.py --probe
PYTHONUNBUFFERED=1 DEBUG=1 python3 add.py
PYTHONUNBUFFERED=1 DEBUG=1 AMD_BOOT_FORCE_HYBRID=1 python3 add.py
AMD_BOOT_FW_MINIMAL=1 AMD_BOOT_FW_MASK=0x400 DEBUG=1 python3 add.py  # RLC only
```

### Still blocked

- MM_INDEX VRAM probe failed on last GPU-attached run (readback ≠ written)
- GPU fell off PCIe (`vid=0xffff`) during reset — needs cable replug to retest
- If MM_INDEX works: hybrid should fix LoadUcodes (SMC reads TOC from VRAM, fw from AGP)
- If AGP DMA unreachable over TB: need TinyGPU IOMMU mapping verification

---

## Previous Session (2026-07-07 PM) — Linux-aligned boot, blocked at LoadUcodes

### Research (linux amdgpu + tinygrad + DeepWiki)

**Linux `amdgpu_device_init` order (Polaris VI):**
1. `gmc_v8_0_mc_init` / `amdgpu_ucode_create_bo` (sw_init)
2. phase1: `vi_common_hw_init` (common, IH)
3. **`amdgpu_device_fw_loading`** → `polaris10_start_smu` → **`smu7_request_smu_load_fw`** (LoadUcodes)
4. phase2: `gmc_v8_0_hw_init` → `mc_program` → MC ucode → **`gart_enable`**

**Key:** `LoadUcodes` runs **before** `gmc_v8_0_gart_enable`. GART is not required for the SMC message itself.

**Linux buffer layout (`smu7_init` / `smu7_request_smu_load_fw`):**
| Buffer | Domain | CPU write | SMC address |
|--------|--------|-----------|-------------|
| `header_buffer` (TOC) | **VRAM** | `memcpy_toio(kaddr)` via BAR0 | `mc_addr` in VRAM |
| `smu_buffer` (scratch) | **VRAM** | BAR0 | `mc_addr` in VRAM |
| `fw_buf` (IP images) | **GTT** | sysmem | `fw_buf_mc + offset` (GART VA or AGP) |

**tinygrad:** `APLRemotePCIDevice.alloc_sysmem` → TinyGPU `PrepareDMA` writes phys addrs to shm. **gfx803/Polaris not supported** by `AMDev` (gfx9+ only). Our bare-metal path is correct approach.

### Bugs fixed this session

| Fix | Detail |
|-----|--------|
| `mmBIF_FB_EN` | `0x1024` → **`0x1524`** (`bif_5_0_d.h`) |
| VRAM visible MC base | `vram_visible_mc = 0xF0000000` (256MB BAR window at end of 4GB) |
| `MC_VM_AGP_BOT/TOP` | Was zero; now programmed from `agp_start`/`agp_end` |
| numpy int32 overflow | `rreg()`/`reg()` cast to Python `int` before `<< 24` |
| Boot order | `load_ip_firmware` **before** `gart_enable` (matches Linux) |
| Auto layout | BAR0 fail → `agp` sysmem layout (`AMD_BOOT_FW_LAYOUT=auto`) |

### BAR0 framebuffer is dead on this eGPU

After `mc_program` + `BIF_FB_EN=0x3`: writes to BAR0 read back **`0xffffffff`**. Host cannot populate VRAM. Linux relies on BAR0 for `header_buffer`/`smu_buffer`.

`MC_IO_DEBUG_UP_13` bit 23 **set** (VBIOS MC ucode loaded). `CONFIG_MEMSIZE=0`, `MISC0` bit `0x80` clear.

### Layouts tried for LoadUcodes (all fail)

| Layout | SMC addresses | Result |
|--------|---------------|--------|
| VRAM | `0xF00xxxxx` via BAR0 upload | BAR0 writes don't stick |
| GART | `0xff001xxxxx` | PTE table in dead VRAM |
| GART self-map sysmem | PTE at `0xff00000000` | Still timeout |
| AGP (`agp_start + paddr`) | `0x1000xxxxx` | Timeout |
| Raw phys | `0x4000`, `0x19c000` | Timeout |
| RLC-only `mask=0x400` | Same | Timeout |

SMC accepts `SMU_DRAM`/`DRV_DRAM` (`resp=0x1`) but **`LoadUcodes` hangs** (`RESP=0`, `UcodeLoadStatus=0`, PC ≈ `0x3a6c0`).

### Root cause (refined)

SMC cannot **read** the firmware buffers we point it at — not a message-ID bug anymore. Linux needs **working VRAM BAR0** for TOC header + scratch; we don't have that on M1+eGPU without VBIOS `asic_init` memory training.

### Next steps

1. **Verify TinyGPU `PrepareDMA` phys addrs are GPU-reachable** (not just host-local)
2. **VBIOS replay** — run ATOM `asic_init` so BAR0 + `CONFIG_MEMSIZE` work
3. **HDP_NONSURFACE** path to write VRAM without BAR0
4. **Hybrid:** VRAM MC addrs for header/smu (per Linux) + working write path

---

### Critical bug fixed: wrong PPSMC message IDs for DRV_DRAM

`polaris_boot.py` had `PPSMC_MSG_DRV_DRAM_ADDR_HI/LO = 0x255/0x256` (wrong). Correct per `smu7_ppsmc.h`:

| Message | ID |
|---------|-----|
| `PPSMC_MSG_DRV_DRAM_ADDR_HI` | `0x250` |
| `PPSMC_MSG_DRV_DRAM_ADDR_LO` | `0x251` |
| `PPSMC_MSG_SMU_DRAM_ADDR_HI` | `0x252` |
| `PPSMC_MSG_SMU_DRAM_ADDR_LO` | `0x253` |
| `PPSMC_MSG_LoadUcodes` | `0x254` |

**Proof:** Wrong IDs gave `DRV_DRAM resp=0xfe` (UnknownCmd). After fix, all setup messages return `resp=0x1`.

### Current failure: `PPSMC_MSG_LoadUcodes` (0x254)

```text
SMC SMU_DRAM_HI/LO resp=0x1
SMC DRV_DRAM_HI/LO resp=0x1
SMC LoadUcodes resp=0x0 (async)  → UcodeLoadStatus=garbage after 120s
SMC PC ≈ 0x3a6c0 (appears hung)
```

### Linux amdgpu init order (VI / Polaris10) — from `ref/linux`

1. `gmc_v8_0_mc_init` + `vram_gtt_location` (sw_init)
2. `amdgpu_ucode_create_bo` — **fw_buf in GTT**, header/smu_buffer in VRAM
3. phase1 hw_init (common, IH)
4. `amdgpu_device_fw_loading` → `polaris10_start_smu` + `smu7_request_smu_load_fw`
5. phase2 → `gmc_v8_0_hw_init` → `mc_program` → MC ucode → `gart_enable`

### Changes applied this session

- Fixed `smc_send_msg` IndentationError
- Firmware images + TOC + smu_dram staging → **GTT** (`gart_start+` MC addrs)
- GART PTE flags `0x73` (was `0x17`) per `amdgpu_ttm_tt_pte_flags` + `gmc_v8_0`
- **GART page table in VRAM** (`amdgpu_gart_table_vram_alloc`) not sysmem
- Boot order: `gart_enable` → `load_ip_firmware` → `mc_program` (fw before GMC hw_init)
- `LoadUcodes` treated async; poll `UcodeLoadStatus` at soft_regs+0x6c

### Bugs fixed this continuation

- **`mmCONFIG_MEMSIZE` wrong register**: was `0x5428` (DCE), correct is **`0x150a`** (BIF). Both read 0 on this eGPU (no VBIOS training).
- **`smc_soft_reg` treated 0 as invalid**: `smc_read()` filters `0` → `UcodeLoadStatus=0` showed as `None`. Now uses raw `smc_rreg`.
- **`AMD_BOOT_FW_MINIMAL` env check**: `int(x)== "1"` was always false.
- **GART `VM_CONTEXT0_CNTL`**: `0x11` (was `0x9`).

### Root cause hypothesis

`PPSMC_MSG_LoadUcodes` hangs (SMC PC ~`0x3a6c0`, `RESP=0`) because **GPU-side VRAM is not trained** (`CONFIG_MEMSIZE=0`, `MC_SEQ_MISC0` bit `0x80` never sets). SMC cannot fetch firmware from VRAM MC addresses. Linux runs VBIOS `asic_init` / MC training before driver load.

### Next investigation

- [ ] Fix MC ucode training on eGPU (or VBIOS replay via `enable_vbios_rom`)
- [ ] Verify TinyGPU sysmem DMA for GTT path once MC works
- [ ] `AMD_BOOT_FW_MINIMAL=1 AMD_BOOT_FW_MASK=0x400` for single-ucode debug

---

## Earlier Session (2026-07-07 AM) — SMC BOOT WORKING

### Root cause fix: wrong `mmSMC_MSG_ARG_0`

Polaris (smu_7_1_1) uses `mmSMC_MSG_ARG_0 = 0xa4`, **not** `0x96` (`0x96` is `mmSMC_MESSAGE_1`). Writing `PPSMC_MSG_Test` arg `0x20000` to `0x96` corrupted the message interface → perpetual `RESP=0` timeout.

### Working SMC boot

```text
stage=smc smc_running=True PC=0x20558 FLAGS=0x1 STATUS=0x3 RESP=0x1
```

Combined with **segmented upload** (`AMD_BOOT_SMC_UPLOAD=segmented`, sync=4096 dwords): firmware readback verified, GPU stays on PCI.

### Next blocker: IP firmware load + compute

Full `add.py` reaches `load_ip_firmware` (`PPSMC_MSG_LoadUcodes`) then times out. Also fixed: `alloc_sysmem_buffer` unpack, GART PTE fill via byte slices.

---

With `AMD_BOOT_SMC_UPLOAD=chunked` + `AMD_MMIO_DRAIN_EVERY=128`, the full ~130 KiB SMC upload completes and **PCI stays `1002:67df`** through protection-mode handshake. Previous crashes were misattributed to timeouts; many were **PCIe device loss** (`0xffff`).

### Current failure mode (GPU online)

| Step | Result |
|------|--------|
| Upload (chunked + drain) | Completes, PCI OK |
| `RCU_INTERRUPTS_ENABLED` | Already set at boot (`EVENTS=0xf0080`) |
| `PPSMC_MSG_Test` @ `0x20000` | **30 s timeout**, `RESP=0x0`, `pci_online=True` |
| Non-protection fallback | Also times out; `FLAGS=0xaaaa5555` (garbage read) |

**Conclusion:** Firmware image is likely **not landing in SMC RAM** without `pc_sync` barriers during upload. Message interface never responds because protected firmware never starts.

### Upload verify matrix

| Mode | `AMD_BOOT_SMC_SYNC` | GPU after upload | Readback @ `0x20000` |
|------|---------------------|------------------|----------------------|
| `pc_sync` | 32768 | Online | **Mismatch** (no mid-upload barrier for 32490 dwords) |
| `pc_sync` | 4096 | **Offline** at upload finish | — |
| `chunked` + drain | 64 | Online | Not verified; msg timeout |

The `sync=4096` crash at "upload finish" was likely **`smc_flush_upload()` SMC RAM read** (`mmio_sync_smc_data`), not the `pc_sync` barriers. **Fix:** `AMD_BOOT_SMC_FLUSH_READ=0` (new default).

### New defaults (this session)

| Variable | New default | Notes |
|----------|-------------|-------|
| `AMD_BOOT_SMC_UPLOAD` | `segmented` | Burst per segment + 1 PC barrier each |
| `AMD_BOOT_SMC_SYNC` | `4096` | Dwords per segment (~32 segments for 130 KiB FW) |
| `AMD_BOOT_SMC_PC_PAUSE_MS` | `15` | Pause after each SMC PC read |
| `AMD_BOOT_SMC_FLUSH_READ` | `0` | Skip post-upload SMC RAM read |
| `AMD_BOOT_SMC_SETTLE_MS` | `250` | Pause between upload segments |

`pc_sync` @ 8192 still knocks GPU off USB4 in ~1.5 s — too aggressive even with flush read disabled.

---

## Hardware & Transport

| Item | Value |
|------|-------|
| GPU | RX570 Polaris10, PCI `1002:67df` |
| Host | M1 Mac, USB4 eGPU enclosure |
| Transport | TinyGPU.app → `APLRemotePCIDevice` unix socket |
| BAR layout | BAR0 VRAM, BAR2 doorbells, BAR5 MMIO (`fmt='I'`) |
| Linux reference | `ref/linux/` (local torvalds/linux tree) |

---

## Key Files

| File | Role |
|------|------|
| `examples_egpu/add.py` | TinyGPU PCI transport, `PolarisDevice`, CLI, PM4 builder |
| `examples_egpu/polaris_boot.py` | VI boot: SMC, MC ucode, golden regs, GART, compute queue |
| `shaders/egpu-add4.s` | gfx803 add kernel |
| `ref/linux/drivers/gpu/drm/amd/` | amdgpu init order, `polaris10_smumgr.c`, `vi.c`, `gmc_v8_0.c` |

---

## What Works

- [x] GPU enumeration after USB4 replug / `--reset`
- [x] BAR0/MMIO probe (`--probe`, `--selftest`)
- [x] `alloc_sysmem` segfault fixed (`FileIOInterface.mmap` + `MAP_FAILED` check)
- [x] Chunked SMC upload completes without immediate GPU drop (when avoiding SMC reads)
- [x] VBIOS ROM read via SMC ind-port (`ROM[0]=0xe974aa55`, valid `55AA` signature)
- [x] Golden register init + doorbell aperture (`vi_common_init`)
- [x] PCI health checks during boot (`pci_online()` / `_check_pci()`)
- [x] MMIO write draining (`AMD_MMIO_DRAIN_EVERY`) — GRBM read RPC after N writes

---

## Current Blocker: SMC Firmware Won't Start

Upload likely completes, but **SMC never runs driver firmware**:

| Symptom | Typical value |
|---------|---------------|
| SMC PC | Stuck `0x80`–`0x88` (ROM idle), not `≥ 0x20100` |
| `FIRMWARE_FLAGS` | `0x0` |
| `RCU_INTERRUPTS_ENABLED` (`0x10000`) | Rarely sets without risky SMC reads |
| `PPSMC_MSG_Test` | Timeout — `SMC_RESP` stays `0` |
| `boot_seq_done` (`EVENTS` bit `0x80`) | Sometimes set after replug; not sufficient alone |
| SMC RAM readback | Often `0xaaaa5555` — unreliable on TinyGPU |
| PCI | Stays `0x1002` until SMC read storm or `PPSMC_MSG_Test` knocks GPU off (`0xffff`) |

**End-to-end add kernel test** (`[11,22,33,44] + [10,20,30,40]`) is blocked on SMC.

---

## Critical Discoveries

1. **TinyGPU MMIO writes are fire-and-forget** — client `_bulk_write` does not wait for server ack. Mitigation: periodic `pci.drain_mmio()` (GRBM read RPC) via `AMD_MMIO_DRAIN_EVERY`.

2. **SMC indirect reads are dangerous on M1 eGPU** — `smc_rreg()` during upload or aggressive post-upload polling can knock the GPU off USB4. Safe: `mmio_sync_ind_port()` (read `mmSMC_IND_ACCESS_CNTL` / `mmSMC_IND_INDEX_11` only).

3. **`pc_sync` upload** (SMC PC read barriers) can get past `RCU_INTERRUPTS_ENABLED` but tends to crash the GPU during or after `PPSMC_MSG_Test`. Use only sparingly (`AMD_BOOT_SMC_FINAL_PC_SYNC=1`).

4. **Card is in SMC protection mode** — `SMU_FIRMWARE` has `SMU_MODE` (`0x10000`). Use `polaris10_start_smu_in_protection_mode`, not non-protection (unless forced).

5. **Firmware selection** — `SMU_SEL` bit 17: `1` → `polaris10_smc.bin`, `0` → `polaris10_smc_sk.bin`. Override: `AMD_SMC_FW=...`.

6. **No VBIOS/ACPI handoff on M1** — `boot_seq_done` may be `0` at cold boot; linux gets pre-SMU init from VBIOS/ATOM on PC.

7. **Hackintosh / WhateverGreen / macOS kexts** — not applicable to TinyGPU bare-metal path.

---

## Linux Init Order (reference)

From `amdgpu_device_init` in `ref/linux/`:

1. **sw_init:** read VBIOS ROM (`vi_read_bios_from_rom`), `amdgpu_atombios_init`
2. **hw_init phase1:** `vi_common_hw_init` (golden regs, ASPM, doorbell)
3. **`amdgpu_device_fw_loading`:** `amdgpu_pm_load_smu_firmware` → `polaris10_start_smu`
4. **hw_init phase2:** GMC (`gmc_v8_0_hw_init`), GFX, etc.

Our port order in `polaris_boot.boot()`:

```
vi_common_init → start_smc → mc_program → load_mc_firmware → gart_enable → load_ip_firmware → enable_compute → init_compute_queue
```

---

## Test Commands

```bash
cd examples_egpu

# After USB4 replug (required if pci=0xffff):
python3 add.py --reset
python3 add.py --probe

# Incremental boot stages
python3 add.py --boot-stage=smc
python3 add.py --boot-stage=mc
python3 add.py                    # full boot + add kernel

# Recommended SMC test (segmented upload — burst + 1 PC barrier per segment)
AMD_BOOT_SMC_UPLOAD=segmented \
AMD_BOOT_SMC_SYNC=4096 \
AMD_BOOT_SMC_PC_PAUSE_MS=15 \
AMD_BOOT_SMC_FLUSH_READ=0 \
AMD_MMIO_DRAIN_EVERY=64 \
AMD_BOOT_SMC_POLL_MS=50 \
AMD_BOOT_SMC_SETTLE_MS=250 \
DEBUG=1 python3 add.py --boot-stage=smc

# Legacy: chunked upload (GPU stays online but firmware may not stick)
AMD_BOOT_SMC_UPLOAD=chunked AMD_MMIO_DRAIN_EVERY=128 DEBUG=1 python3 add.py --boot-stage=smc

# If RCU wait times out but GPU stays online
AMD_BOOT_PROT_SKIP_RCU=1 DEBUG=1 python3 add.py --boot-stage=smc

# Risky: one SMC PC read after upload only
AMD_BOOT_SMC_UPLOAD=hybrid AMD_BOOT_SMC_FINAL_PC_SYNC=1 DEBUG=1 python3 add.py --boot-stage=smc

# Non-protection path (when boot_seq_done is set)
AMD_BOOT_SMC_PROT=0 AMD_BOOT_WAIT_BOOT_SEQ_DONE=1 DEBUG=1 python3 add.py --boot-stage=smc
```

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `AMD_MMIO_DRAIN_EVERY` | `128` | GRBM read RPC every N MMIO writes (drain TinyGPU queue) |
| `AMD_BOOT_SMC_UPLOAD` | `pc_sync` | `pc_sync`, `chunked`, `linux`, `per_addr`, `hybrid` |
| `AMD_BOOT_SMC_SYNC` | `8192` | PC barrier interval (dwords); must be ≤ ~32000 for 130 KiB FW |
| `AMD_BOOT_SMC_PC_PAUSE_MS` | `15` | Sleep after each SMC PC read during upload |
| `AMD_BOOT_SMC_FLUSH_READ` | `0` | Post-upload SMC RAM read in `smc_flush_upload` (risky) |
| `AMD_BOOT_SMC_POLL_MS` | `25` | Post-upload poll interval (ms), backoff to 250 ms |
| `AMD_BOOT_SMC_SETTLE_MS` | `250` | Pause after reset deassert before handshake |
| `AMD_BOOT_SMC_TIMEOUT_S` | `60` | Generic firmware wait timeout (seconds) |
| `AMD_BOOT_RCU_TIMEOUT_S` | `30` | `RCU_INTERRUPTS_ENABLED` wait |
| `AMD_BOOT_SMC_MSG_TIMEOUT_S` | `30` | `PPSMC_MSG_Test` / SMC_RESP wait |
| `AMD_BOOT_SMC_PROT` | `auto` | `auto`, `1` (protection), `0` (non-protection) |
| `AMD_BOOT_PROT_SKIP_RCU` | `0` | Skip RCU wait, try message anyway |
| `AMD_BOOT_SMC_VERIFY` | `0` | SMC RAM readback verify (unreliable on TinyGPU) |
| `AMD_BOOT_SMC_FINAL_PC_SYNC` | `0` | Single SMC PC barrier after hybrid upload |
| `AMD_SMC_FW` | (auto) | Override firmware blob name |
| `AMD_BOOT_GOLDEN` | `1` | Apply polaris10 golden regs before SMC |
| `AMD_BOOT_ROM_ENABLE` | `1` | Enable VBIOS ROM (`vi_read_disabled_bios` path) |
| `DEBUG` | `0` | Verbose boot logging |

---

## Session History (summary)

| Session | Result |
|---------|--------|
| alloc_sysmem fix | Segfault resolved |
| MMIO sync discovery | Fire-and-forget writes; read barriers required |
| Protection mode port | Aligned with `vegam_start_smu_in_protection_mode` |
| `pc_sync` upload | Passes RCU sometimes; crashes GPU on msg or mid-upload SMC read |
| Chunked + drain | Upload completes; firmware still doesn't execute |
| Timeout / PCI health | Early abort on `pci=0xffff`; slower polling reduces read storms |

---

## Next Steps

1. **Replug USB4** whenever `pci=0xffff` — TinyGPU PCI reset cannot recover a missing device.

2. **Validate upload with drain** — confirm GPU stays online through full chunked upload + protection handshake.

3. **Minimize SMC indirect reads** after upload — poll `SMC_RESP` (direct MMIO) only; use `AMD_BOOT_SMC_POLL_MS=50` or higher.

4. **If RCU never sets without `pc_sync`** — investigate whether a single post-upload SMC read or longer settle (`AMD_BOOT_SMC_SETTLE_MS=1000`) is enough to trigger protection-mode `PPSMC_MSG_Test`.

5. **Port additional pre-SMU init** from `vi_common_hw_init` / VBIOS path (ASPM, `vi_program_aspm`) if handshake still fails.

6. **After SMC runs** — fix GPU VM mapping for DMA pages (GART), then run full `add.py` kernel test.

7. **Do not pursue** WhateverGreen, Hackintosh EFI, or macOS `AMDRadeonX4000` for this path.

---

## Todo

- [x] Fix `alloc_sysmem` segfault
- [x] **SMC boot working** (segmented upload + `mmSMC_MSG_ARG_0=0xa4`)
- [ ] `load_ip_firmware` / `PPSMC_MSG_LoadUcodes`
- [ ] Fix GPU VM mapping for DMA pages
- [ ] Run full `add.py` kernel test
