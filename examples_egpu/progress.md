# RX570 eGPU Bring-Up Progress (TinyGPU / M1 Mac)

**Goal:** Run vector-add on **AMD RX570 (Polaris10 / gfx803, `1002:67df`)** via **TinyGPU.app** bare-metal MMIO/PM4 — not macOS `AMDRadeon*` kexts.

**Last updated:** 2026-07-08 ~22:03 — APCIE panic #5; session stopped at `fw-mec`

## Status

| Item | State |
|------|--------|
| **Solved** | ATOM `asic_init` — VRAM trains (`MEMSIZE=4096`, `MISC0\|0x80`, `trained=True`) |
| **Solved** | Direct MMIO firmware upload path (bypasses SMC `LoadUcodes`) |
| **Solved** | Staged fw upload: `atom` → `fw-mec` completes without panic (~32s total) |
| **Solved** | SRBM / KCQ direct / GART PTE — verified in **earlier** sessions (not re-validated this reboot) |
| **Blocker** | **CP rings never drain** — `PQ_RPTR=0x0`; `SCRATCH` stuck at `0xCAFEDEAD` (earlier `kcq-ring-test`) |
| **Blocker** | **GPU-side GART DMA unproven** — CPU `pte_ok=True` ≠ device can read `0xff00…` sysmem |
| **Fixed** | **macOS USB4 APCIE MSI panic** — device IRQs now masked (PCI MSI/INTx + GPU IH/CP interrupt block) before firmware runs; see *Fix: interrupt masking* below |
| **Session #5** | **Crashed during `fw-start`→`gart-probe`→`kcq-direct` chain** — only `atom`+`fw-mec` completed (pre interrupt-mask fix) |
| **Current stage** | Interrupt-mask fix landed — device IRQs masked before firmware runs; re-run staged boot to verify no panic |
| **Next** | Replug → `fw-start` alone → `gart-probe` alone → `kcq-direct` alone → `AMD_BOOT_RING_TEST=1 kcq-ring-test` (panic trigger now removed) |
| **Safe (this session)** | `--probe`, `--selftest`, `atom`, `fw-mec` (upload only, MEC stays halted) |
| **Risky** | `fw-start` (unhalt MEC), `gart-probe`, `kcq-direct`, anything with ring dispatch |
| **Gated** | `kiq-map`, `kcq-ring-test` (`AMD_BOOT_RING_TEST=1`), `add` (`AMD_BOOT_ADD=1`), `AMD_BOOT_FULL=1` |

---

## Current stage & blocker (2026-07-08 ~22:03)

### Session #5 — panic during chained boot

**This reboot, before crash:**

| Step | Command | Result | Time |
|------|---------|--------|------|
| 1 | `--probe` | `pci=1002:67df` cold (`MEMSIZE=0`, MEC halted) | ~0.5s |
| 2 | `--boot-stage=atom` | `trained=True` `MEMSIZE=0x1000` | ~2.8s |
| 3 | `--boot-stage=fw-mec` | `CP_MEC_CNTL=0x50000000` (halted) | ~29s |
| 4 | `fw-start && gart-probe && kcq-direct` | **macOS kernel panic** — no stage output captured | killed @141s |

**Furthest safe point this session:** end of **`fw-mec`** — firmware uploaded, MEC still halted.

```
atom → fw-mec → fw-start → gart-probe → kcq-direct → kcq-ring-test → add
 ✓       ✓        ?           ?              ?              ✗           ✗
         ↑ YOU ARE HERE (MEC halted, fw resident)
```

### Fix: interrupt masking (2026-07-08 — panic root cause)

The `apciec unhandled interrupts (0x200000)` panic was the eGPU **asserting an IRQ
to the macOS USB4 bridge**, which TinyGPU.app leaves unhandled. Once CP/MEC/RLC
firmware goes live it raises MSIs the `AppleT8103PCIe` bridge cannot route →
kernel panic. Disabling the BAR2 doorbell (`AMD_BOOT_NO_DOORBELL=1`) only removed
*one* MSI trigger; the firmware itself still asserted interrupts on unhalt.

**Fix — mask every device interrupt source, keep the GPU polling-only:**

| Layer | Where | What |
|-------|-------|------|
| PCI config | `RemotePCIDevice.mask_msi()` (`add.py`, called in `PolarisDevice.__init__`) | Set PCI command **Interrupt Disable** (bit 10); walk cap list, clear **MSI Enable** / **MSI-X Enable**. Bus-master (DMA) left on. |
| GPU IH block | `PolarisBoot.disable_gpu_interrupts()` (`polaris_boot.py`) | `tonga_ih_disable_interrupts`: `IH_RB_CNTL` RB_ENABLE+ENABLE_INTR=0, zero RPTR/WPTR, IH doorbell off |
| CP / compute pipe | same method | `CP_INT_CNTL_RING0=0`, `CPC_INT_CNTL=0` (no EOP/priv/error IRQ requests) |

Wired into `vi_common_init` (baseline), and **before every unhalt**:
`unhalt_loaded_firmware`, `load_ip_firmware_direct` (unhalt block), `enable_compute`.
Enabled by default on darwin; toggle with `AMD_BOOT_MASK_INTERRUPTS=0/1`.

### Remaining blocker

#### A) Platform — APCIE MSI panic (FIXED, verify on HW)

Prior signature:

```
apciec[pcic0-bridge] unhandled interrupts (0x200000 out of 0x220000)
@APCIECPort.cpp:2056  (AppleT8103PCIeC / USB4)
```

| When it fired | Now handled by |
|---------------|----------------|
| `fw-start` (MEC unhalt) | `disable_gpu_interrupts("pre-unhalt")` before `cp_*_enable(True)` |
| `kcq-direct` (`enable_compute` + HQD) | `disable_gpu_interrupts("pre-enable-compute")` + PCI MSI mask |
| `kcq-ring-test` / doorbell | `AMD_BOOT_NO_DOORBELL=1` (darwin) + MSI enable cleared in config |
| Chaining stages | Still avoid — settle rules below unchanged |

**Rule (still advised):** one `python3 add.py --boot-stage=…` per shell invocation;
wait 5–10s between steps after `fw-start`. The IRQ mask removes the panic trigger,
but USB4 link settle timing is still worth respecting.

#### B) Functional — rings never execute PM4 (earlier session, pre-#5)

From last successful boot through `kcq-direct` (earlier tonight):

| Stage | Result |
|-------|--------|
| `fw-start` | `CP_MEC_CNTL=0x10000000` |
| `gart-probe` | `pte_ok=True` `pte=0x48077` |
| `kcq-direct` | `KIQ=1` `KCQ=1` |
| `kcq-ring-test` | `ring_ok=False` `SCRATCH=0xcafedead` `PQ_RPTR=0x0` → panic |

HQD `ACTIVE=1` only means registers committed — **MEC never fetched the ring**. Until `SCRATCH=0xDEADBEEF`, vector-add is pointless.

Likely causes (unchanged):

1. GPU cannot DMA-read GART sysmem (`0xff00…` ring/MQD/wptr) despite CPU PTE self-test passing.
2. Direct KCQ bypasses Linux `MAP_QUEUES` / `SET_RESOURCES` — scheduler may not know about the queue.
3. TinyGPU `PrepareDMA` / M1 USB4 lacks device-coherent mapping for PTE table + buffers.

### Mitigations already in code

| Change | Status |
|--------|--------|
| `AMD_BOOT_NO_DOORBELL=1` (darwin default) | Skip BAR2 MSI path |
| `boot_use_mmio_wptr()` | MMIO `CP_HQD_PQ_WPTR` when no doorbell |
| `AMD_BOOT_RING_TEST=1` / `AMD_BOOT_ADD=1` gates | Block ring dispatch / vector-add |
| `boot_minimal_for_compute()` + `enable_compute()` | Fixed on `skip_fw` path |

**Not proven:** MMIO-only `kcq-ring-test` — panic #4/#5 prevented completion.

### Open questions (priority)

1. Does **`fw-start` alone** panic, or only when followed immediately by GART/HQD? (isolate unhalt vs compute setup)
2. **SDMA GPU readback** from GART VA — only test that proves device DMA (not `gart-probe`)
3. Can we **`fw-start` with MEC still halted** for compute queues? (probably not — need ME1 running)
4. TinyGPU-side: mask GPU MSI at bridge before unhalt?

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

### VRAM data path dead after training (BAR0 / MM_INDEX)

Even with `trained=True`, CPU cannot read/write VRAM:

| Path | Result |
|------|--------|
| BAR0 | constant garbage per session (`0x36e94e32`, `0xdbaeea31`, …) — writes ignored |
| MM_INDEX | same constant at all offsets |
| SMC `LoadUcodes` | times out — SMC DMA-reads TOC from VRAM MC addresses |

**Workaround:** GART sysmem for compute buffers + **direct MMIO** firmware upload (no SMC DMA).

### 2026-07-08 evening — direct MMIO + KIQ/KCQ port 

**Done in code** (`ref/linux`):

- `load_ip_firmware_direct()` — TrustOS-style MMIO ucode upload (RLC/PFP/CE/ME/MEC/SDMA)
- `ViMqd` + `mqd_init_vi` / `mqd_commit_vi` from `gfx_v8_0.c` + `vi_structs.h`
- KIQ at `me=1, pipe=1, queue=0` (KCQ uses `pipe=0`) per `amdgpu_gfx_kiq_acquire`
- **SRBM bug fixed:** `srbm_select` had wrong `SRBM_GFX_CNTL` field layout (`vi.c` uses pipe@0, me@2, vmid@4, queue@8 — we had them scrambled)

**Last HW session (2026-07-08 late PM) — staged boot without panic:**

| Stage | Result | Notes |
|-------|--------|-------|
| `fw-mec` | ✓ ~32s | `CP_MEC_CNTL=0x50000000` (halted) |
| `fw-start` | ✓ ~4s | `CP_MEC_CNTL=0x10000000` (ME1 only) |
| `kiq` | ✓ ~33s | `KIQ_HQD_ACTIVE=0x1`, `KCQ=0x0` (expected) |
| `kiq-map` | ✓ ~6s **no crash** | `skip_fw=True`; `KCQ_HQD_ACTIVE` still `0x0` |
| `AMD_BOOT_FULL=1` | ✗ no panic | `result=[0,0,0,0]` — KCQ never activated |

**Observed after `kiq-map` doorbell (`DEBUG=1`):**

```
KIQ_HQD_ACTIVE=0x1
KIQ PQ_WPTR=0x100   (256 dwords — ring commit + doorbell accepted)
KIQ PQ_RPTR=0x0     (MEC never consumed KIQ ring)
CP_HQD_ERROR=0x0
KCQ_HQD_ACTIVE=0x0
GART: 0xff00000000–0xff0fffffff, kcq_mqd=0xff00110000
```

**Interpretation:** MAP_QUEUES PM4 is in the ring and the doorbell updates WPTR, but **KIQ firmware does not advance RPTR**. Likely causes: (1) GPU cannot read GART-backed sysmem ring/MQD, (2) MEC/KIQ scheduler not fetching, (3) need TrustOS direct-HQD path instead of KIQ.

**Crash post-mortem (2026-07-08 PM #1):** wrong `srbm_select` + KIQ doorbell → kernel panic.

**Crash post-mortem (2026-07-08 PM #3):** `kiq-map` re-uploaded full MEC (~26s) then rang KIQ doorbell → kernel panic. Linux `gfx_v8_0_kcq_resume` order: KCQ MQD in memory only → `set_mec_doorbell_range` → `kiq_kcq_enable` → `amdgpu_ring_commit` (flush + doorbell). Fixes:

| Issue | Fix |
|-------|-----|
| `kiq-map` re-uploads firmware | **`skip_fw=True`** if `compute_fw_loaded()` (ME1 already running) |
| Doorbell without flush/settle | `_ring_commit`: HDP flush, `sysmem_dma_flush`, `mmio_settle`, drain before/after doorbell |
| Wrong MEC doorbell range upper | `DOORBELL_MEC_RING7=0x17` not `MEC_RING0+8` |
| Duplicate `enable_compute` | Removed from `_boot_stage_kiq` when using full fw path |

**Fixes (2026-07-08 late PM — DeepWiki + `ref/linux` audit):**

| Issue | Fix |
|-------|-----|
| KIQ ring commit alignment | `VI_RING_ALIGN_MASK=0xff` (256-dword pad per `gfx_v8_0_ring_funcs_kiq`) |
| Missing wptr CPU shadow | `_publish_wptr()` before doorbell (`gfx_v8_0_ring_set_wptr_compute`) |
| MQD `rptr_report` used `wptr_gpu` | Separate `rptr_gpu` / `wptr_gpu` in `ComputeQueue.init()` |
| `skip_fw` GART at wrong VA | `boot_minimal_for_compute()` calls **`gmc_sw_init()`** before `gart_enable()` |
| GART PTE in VRAM when BAR0 writes probe ok | **`boot_minimal` forces `AMD_BOOT_GART_SYSMEM=1`** (host PTE table) |
| `vi_common_init` on hot GPU | Removed from `boot_minimal` (avoid golden-reg reset mid-session) |
| RLC safe mode | `rlc_exit_safe_mode()` before `kiq_setting` (TrustOS / `gfx_v8_0_unset_safe_mode`) |
| PM4 ring padding | `VI_PKT3_NOP` (`PACKET3_NOP, 0x3FFF`) not `PACKET2(0)` |
| Doorbell BAR2 index | `ring_doorbell(index)` uses **`index >> 2`** (VI byte offset → dword slot) |
| Compute buffers on GART | `ComputeQueue._gtt` when `gart_pte_sysmem` is set |

**Fixes (2026-07-08 evening — DeepWiki tier-1 repos + code audit):**

| Issue | Fix |
|-------|-----|
| GART PTE table not flushed to device | `_gart_pte_flush()` after PTE build + every `map_sysmem_gpu()` |
| Wrong doorbell enable | Removed `CP_PQ_STATUS | (1<<28)`; use `set_mec_doorbell_range()` only (bit 1) |
| No GART validation before KIQ | `probe_gart_dma()` + `--boot-stage=gart-probe` |
| KCQ stuck after MAP_QUEUES | `AMD_BOOT_KCQ_DIRECT=1`, `--boot-stage=kcq-direct`, auto-fallback in `setup_with_kiq` |
| HDP before TLB invalidate | `hdp_flush()` in `map_sysmem_gpu()` after PTE writes |

Linux VI path: `gfx_v7_0_cp_gfx_load_microcode` (PFP/CE/ME), `gfx_v7_0_cp_compute_load_microcode` (MEC), `cik_sdma_load_microcode` (SDMA). KCQ resume: `gfx_v8_0_kcq_init_queue` (MQD only) → `set_mec_doorbell_range` → `gfx_v8_0_kiq_kcq_enable` → `amdgpu_ring_commit`.

### Staged verification plan (do NOT skip steps)

```bash
# 1. Safe — confirm eGPU back after replug
python3 add.py --probe

# 2. ATOM only (~5k MMIO)
python3 add.py --boot-stage=atom

# 3. RLC only (~4k MMIO) — safest firmware probe
python3 add.py --boot-stage=fw-rlc

# 4. + PFP/CE/ME (~12k MMIO)
python3 add.py --boot-stage=fw-cp

# 5. + MEC upload only — stay halted (~15s with settle pauses)
AMD_MMIO_DRAIN_EVERY=32 python3 add.py --boot-stage=fw-mec

# 6. Settle + unhalt ME1 — RUN ALONE, wait 10s before next step
python3 add.py --boot-stage=fw-start
sleep 10

# 7. GART PTE self-map — RUN ALONE
python3 add.py --boot-stage=gart-probe

# 8. KCQ direct HQD — RUN ALONE
python3 add.py --boot-stage=kcq-direct

# 9. KCQ ring test — gated; MMIO wptr only on darwin
AMD_BOOT_RING_TEST=1 python3 add.py --boot-stage=kcq-ring-test

# 10. Vector-add — only after ring_ok=True
AMD_BOOT_ADD=1 python3 add.py --boot-stage=add

# DO NOT chain: fw-start && gart-probe && kcq-direct  (caused panic #5)
```

---

## Historical: VRAM Not Trained blocker (resolved 2026-07-08)

Layer 1 ATOM `asic_init` was stuck due to two `atom_replay.py` interpreter bugs (not hardware). Fixed — see **Solved: ATOM VRAM training** above. Old "Path A / Path B" Linux golden trace was unnecessary.

---

## ⚠️ STOP — Safety

**Interrupt-mask fix (above) removes the APCIE MSI panic trigger** — the device no
longer asserts IRQs to the USB4 bridge. Firmware unhalt / compute setup should no
longer kernel-panic. Residual risk is USB4 link drop (replug), not full reboot.
Still run one stage per shell and respect settle timing until verified on HW.

| Command | Safe? |
|---------|-------|
| `--probe`, `--selftest` | ✅ |
| `--boot-stage=atom`, `--boot-stage=pre-fw` | ⚠️ low–medium |
| `--boot-stage=fw-direct` | ⚠️ high MMIO volume |
| `--boot-stage=fw-mec` | ⚠️ medium | ~30s MMIO; MEC stays halted — **safe endpoint this session** |
| `--boot-stage=fw-start` | ⚠️ **high** | MEC unhalt — may trigger APCIE MSI (panic #5 suspect) |
| `--boot-stage=gart-probe` | ⚠️ medium | GART PTE setup |
| `--boot-stage=kcq-direct` | ⚠️ **high** | KIQ+KCQ HQD commit |
| `--boot-stage=kcq-ring-test` | ❌ **gated** — `AMD_BOOT_RING_TEST=1`; caused APCIE panic with doorbell |
| `--boot-stage=add` | ❌ **gated** — `AMD_BOOT_ADD=1` |
| **`add.py` default** | ❌ **blocked** — requires `AMD_BOOT_FULL=1` |
| `AMD_BOOT_LOADUCODES_UNTRAINED=1` | ❌ never |

---

## Latest Session (2026-07-08) — Docs + VBIOS parser

### Research conclusions

| Topic | Verdict |
|-------|---------|
| **ChefKiss NootedRed / NootRX** | No VRAM training — useful `ATOMBIOS.hpp` parsers only |
| **TrustOS** | SDMA milestone assumes VBIOS already POST'd on x86 |
| **Aitbytes VFIO** | Same `1002:67DF`; PCI rescan workaround |

### Code — `atom_replay.py` (from NootedRed + linux headers)

| Addition | Purpose |
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

1. **Replug eGPU** — `--probe` must show `pci=1002:67df`.
2. **`atom`** then **`fw-mec`** — confirmed safe this session (~32s total).
3. **`fw-start` alone** — wait 10s; note if panic happens here (isolates MEC unhalt).
4. **`gart-probe` alone** — CPU PTE check only.
5. **`kcq-direct` alone** — HQD commit; expect `KCQ=1`.
6. **`AMD_BOOT_RING_TEST=1 kcq-ring-test`** — only if step 5 survives; watch `SCRATCH`.
7. If ring stuck → **SDMA GPU readback** from GART VA before any more dispatch attempts.
8. **Never chain stages** in one shell command on USB4.

Open question: panic #5 hit during chained `fw-start→gart→kcq` — is **`fw-start` unhalt** the trigger, or **`kcq-direct` HQD setup**?

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
- [x] Staged direct MMIO fw: `fw-mec`, `fw-start` (ME1 unhalt, no panic)
- [x] KIQ MQD commit → `KIQ_HQD_ACTIVE=0x1` (SRBM fix verified)
- [x] `kiq-map` with `skip_fw` — fast, no kernel panic (KCQ still inactive)
- [x] GART PTE flush + `gart-probe` + KCQ direct fallback (code; HW pending)
- [x] MMIO drain (`AMD_MMIO_DRAIN_EVERY`)

## Todo

- [x] ATOM training (`atom_replay.py` bugs fixed)
- [x] SMC boot
- [ ] CPU-visible VRAM path (BAR0 or MM_INDEX) **or** proven GART DMA
- [ ] `load_ip_firmware` / firmware resident
- [ ] Vector-add via `add.py`

---

## References — DeepWiki re-rank (2026-07-08, latest)

Scored **0–10** for the **current** blocker: **KIQ ring not draining / KCQ not activating** on GART-sysmem; need GART DMA proof or TrustOS direct-HQD fallback. DeepWiki + `ref/linux` `gfx_v8_0.c` audit (2026-07-08 late PM).

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
| `AMD_BOOT_FW_WRITE_PAUSE_MS` | `8` (MEC) | ms sleep every drain during large upload |
| `AMD_MMIO_SETTLE_ROUNDS` | `30` | heavy settle loops before unhalt |
| `AMD_MMIO_SETTLE_MS` | `100` | ms per settle round |
| `AMD_BOOT_MEC2_HALT` | `1` | keep MEC2 halted (ME1 only) |
| `AMD_BOOT_KIQ_MAP` | `0` | `1` = allow `--boot-stage=kiq-map` MAP_QUEUES doorbell |
| `AMD_BOOT_KCQ_DIRECT` | `auto` | `1` = force direct KCQ HQD; `auto` = fallback after MAP_QUEUES fails |
| `AMD_BOOT_NO_DOORBELL` | `1` on darwin | `1` = skip BAR2 doorbell (prevents APCIE MSI panic) |
| `AMD_BOOT_MASK_INTERRUPTS` | `1` on darwin | `1` = mask PCI MSI/INTx + GPU IH/CP interrupts (prevents APCIE panic on unhalt) |
| `AMD_BOOT_MMIO_WPTR` | auto | `1` when `NO_DOORBELL`; MMIO `CP_HQD_PQ_WPTR` (TrustOS path) |
| `AMD_BOOT_RING_TEST` | `0` | `1` = allow `--boot-stage=kcq-ring-test` |
| `AMD_BOOT_ADD` | `0` | `1` = allow `--boot-stage=add` |
| `AMD_BOOT_GART_SYSMEM` | `auto` | `1` = host PTE table (forced in `boot_minimal_for_compute`) |
| `AMD_BOOT_KCQ_ACTIVE_TIMEOUT_S` | `5` | poll for KCQ active after MAP_QUEUES |
| `AMD_BOOT_DOORBELL_SETTLE_MS` | `10`/`50` | sleep after wptr signal (10 when no doorbell) |
| `AMD_BOOT_VBIOS_FILE` | — | Path to `rx570.rom` |
| `AMD_ATOM_JUMP_BAIL` | `0` | `1` = fake-complete ATOM (obsolete now) |
| `DEBUG` | `0` | Verbose logging |
