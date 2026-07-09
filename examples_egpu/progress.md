# RX570 eGPU Bring-Up Progress (TinyGPU / M1 Mac)

**Goal:** Run vector-add on **AMD RX570 (Polaris10 / gfx803, `1002:67df`)** via **TinyGPU.app** bare-metal MMIO/PM4 — not macOS `AMDRadeon*` kexts.

**Last updated:** 2026-07-10 ~02:10 — **session #21: `add.py` vector-add PASS**

## Current status

**PASS:** KCQ `ring_ok=True` (`SCRATCH=0xDEADBEEF`) + **vector-add**
`result=[11.0, 22.0, 33.0, 44.0]` on AGP host buffers (VRAM still dead).

```bash
# Full success path:
AMD_BOOT_ADD=1 AMD_BOOT_MASK_INTERRUPTS=1 AMD_BOOT_KCQ_DIRECT=1 \
  AMD_BOOT_COMPUTE_AGP=1 AMD_BOOT_KIQ_MAP=0 AMD_BOOT_SKIP_KIQ=1 \
  AMD_BOOT_DOORBELL_HIT=1 AMD_BOOT_MEC_SMC_UCODE=1 \
  AMD_BOOT_FW_LAYOUT=agp AMD_BOOT_LOADUCODES_UNTRAINED=1 AMD_BOOT_FW_MINIMAL=1 \
  python3 add.py --boot-stage=add
```

### Session #21 — KCQ execute + vector-add

DeepWiki rerank (AGENTS.md, helpfulness for this bring-up):

| Rank | Repo | Score | Why |
|------|------|------:|-----|
| 1 | torvalds/linux / ROCm/amdgpu | 10 | gfx_v8 MQD/PRELOAD, SET_UCONFIG offset, SH_MEM VMID0, IB emit |
| 2 | allbilly/amdgpu | 10 | this tree |
| 3 | nathan237/TrustOS | 9 | AGP DMA / Polaris bare-metal journal |
| 4 | tinygrad/tinygrad (+7900xtx) | 6–7 | PM4/IB patterns; not gfx803 boot |
| 5+ | passthrough / Hackintosh / sims / ZLUDA / … | 0–3 | N/A for MEC KCQ |

| Fix | Result |
|-----|--------|
| **SET_UCONFIG offset: drop `>>2`** | Was `0x10` not `0x40` → ring drained, SCRATCH untouched. Now `ring_ok=True` |
| **`submit_ib`: real `INDIRECT_BUFFER` (0x3F)** | Was NOP `0x10` + inline words → RPTR stuck |
| **AGP buffers for shader/a/b/out** | Match KCQ aperture (VRAM dead) |
| **`COMPUTE_PGM_LO/HI >>= 8`** | Linux 256-byte units |
| **gfx803 `flat_*` shader** (not gfx900 `global_*`) | Assembled llvm `-mcpu=gfx803` |
| **RSRC1 VGPRS=3 / USER_SGPR=6** | Flat shader needs v0–v13 |
| **`init_sh_mem_vmid0` (UC + BASES=0)** | **Unblocked flat stores** → add PASS |
| PACKET3 count = `len(vals)-1` | Header encoding |

Still true: VRAM BAR0/MM writes fail after HDP; use AGP for rings/data.

### Session #19 — F32 execute unblocked (`write_ok=True`)

DeepWiki: almost all AGENTS.md repos **N/A** for Polaris F32 execute; only
TrustOS / linux / this tree / tinygrad remain useful.

| Finding | Detail |
|---------|--------|
| Linux Polaris SDMA fw | Loaded by **SMC `LoadUcodes`** (`smu7_request_smu_load_fw`), not CIK MMIO |
| AGP TOC LoadUcodes | **Works** — `UcodeLoadStatus=0x6` (SDMA0\|1) with `AMD_BOOT_FW_LAYOUT=agp` |
| After SMC load | Same execute stall — rules out “direct MMIO ucode corrupt” as sole cause |
| NOP pad | `FETCH` walks past packet into NOPs; still no retire |
| `STATUS2` | `F32_INSTR_PTR=0`, `CMD_OP=0` while stuck |
| `POWER_CNTL` | Still reads 0 (golden `0x3c800` ignored) |

**Code:** `AMD_BOOT_SDMA_SMC_UCODE=1` → SMC LoadUcodes (AGP TOC, mask SDMA only);
`AMD_BOOT_SDMA_NOP_PAD` trailing NOPs in probe.

```bash
AMD_BOOT_SDMA_PROBE=1 AMD_BOOT_SDMA_AGP=1 AMD_BOOT_SDMA_SMC_UCODE=1 \
  AMD_BOOT_SDMA_PHASE=0xff0f \
  python3 add.py --boot-stage=sdma-probe
```

### Session #17 — DeepWiki rerank + execute-path experiments

Tried (all keep `fetch_ok`, none get `write_ok`):

| Knob | Result |
|------|--------|
| `RB_PRIV=1` (TrustOS bare-metal) | applied (`RB=0x830015`); no retire |
| Linux golden `CHICKEN=0x00810007` | sticks; no retire |
| `ATC_L1` on/off, `AUTO_CTXSW` on/off | no change |
| `IB=0` / golden `0x100` | no change |
| `COUNT=0` vs `1`, dst=ring+0x100 | no change |
| `SRBM_WRITE` → `DUMMY_REG` | fetch OK, DUMMY stays 0 — **execute broken even without host DMA** |
| `CSA_ADDR` programmed | sticks; CSA page unchanged |
| `POWER_CNTL` golden `0x3c800` | **does not stick** (reads 0) — same as GDDR dead |
| `PHASE=0` (default) | WPTR bump → `CTX_STATUS.EXPIRED=1` |
| `PHASE=0xff0f` | **clears EXPIRED**; still no retire |

**Implication:** blocker is F32 **packet execute**, not host-write aperture / TinyGPU DMA dir.
TrustOS journal never recorded a single “Level B PASS” fix after the same mid-debug
state; milestone doc claims success but the journal stops at fetch-OK / retire-fail.

**Code landed:** golden CHICKEN/CLK; `RB_PRIV` default on; richer probe
(`CTX`/`EXPIRED`/`PHASE`/`CHICKEN`/`SYS_APR`); `AMD_BOOT_SDMA_PHASE` /
`AMD_BOOT_SDMA_ATC` / `AMD_BOOT_SDMA_AUTO_CTXSW` / `AMD_BOOT_SDMA_DST` /
`AMD_BOOT_SDMA_COUNT` / `AMD_BOOT_SDMA_IB=golden|0|1`.

### Session #16 breakthrough — preserve `SDMA0_CNTL` preamble

DeepWiki + TrustOS journal: clobbering `SDMA0_CNTL→0x1` wiped `MC_*REQ_CREDIT`
and left `RPTR_FETCH=0` forever. **RMW clear AUTO_CTXSW only** → `CNTL=0x8010402`
→ `RPTR_FETCH` moves (`0x4` on NOP, `0x18` on NOP+WRITE).

| Check | Result |
|-------|--------|
| AGP ring fetch | **`fetch_ok`** — `RPTR_FETCH` reaches packet end |
| GART ring fetch | **same** |
| `WRITE_LINEAR` → host | **stuck** — dst stays `0xCAFEDEAD` |
| Same-page write (ring+0x100) | **stuck** — not a bad dst address |
| `SRBM_WRITE` → scratch | **stuck** — fetch works, execute does not |
| TinyGPU DMA dir | `kIOMemoryDirectionInOut` (bidirectional) — not the cause |

**Code landed:** `_sdma_disable_auto_ctxsw` preserve mode; `IB=0`; `PHASE*=0`;
NOP-first packet; `fetch_ok` / `RPTR_FETCH` in probe output; refuse dead-VRAM ring.

### Session #15 — panic root cause + VRAM autopsy

**23:32 panic** (`apciec 0x200000`): `AMD_BOOT_SDMA_VRAM=1` with GDDR dead —
`RB_BASE=0xf4…` emitted as ≥32-bit PCIe TLP. **Do not retry VRAM ring.**

| Check | Result |
|-------|--------|
| ATOM `asic_init` | ✓ `MEMSIZE=4096`, `MISC0` trained bit, `FB_LOC=0xf4fff400` |
| `MC_SEQ_STATUS_M` | `0x3` = PWRUP only; **`CMD_RDY=0`** |
| BAR0 / MM_INDEX | after HDP flush → constant `0xbde1aebe` |

### What works

| step | status |
|------|--------|
| TinyGPU PCI enumerate + MMIO | OK (`1002:67df`, BAR0/2/5 up) |
| CPU-side GART PTE build (`gart-probe`) | OK (64-bit PTEs) |
| SDMA ucode MMIO upload (no SMC) | OK (~3k words, F32 halted) |
| Host ring **fetch** (AGP/GART) | **`fetch_ok`** after CNTL preserve |
| `sdma_soft_reset()` after stall | recovers engine |
| `--reset` after replug | OK |

### What fails

| step | symptom |
|------|---------|
| SDMA packet execute / retire | `PKT_RDY=1`, `RB_RPTR=0`, dst/`DUMMY` unchanged |
| VRAM-backed anything | `CMD_RDY=0`, BAR0 writes vanish |
| Full boot with bad addresses | kernel panic `apciec 0x200000` |

### Recovery protocol (before next HW attempt)

```bash
python3 add.py --probe
AMD_BOOT_SDMA_PROBE=1 AMD_BOOT_SDMA_AGP=1 AMD_BOOT_SDMA_VRAM=0 \
  AMD_BOOT_SDMA_PHASE=0xff0f \
  python3 add.py --boot-stage=sdma-probe
```
Success: `write_ok=True dst=0xdeadbeef` (not just `fetch_ok` / `EXP=False`).

### Still to try (ordered)

1. Why F32 fetches but never retires even `SRBM_WRITE` (no host DMA) — ucode /
   scheduler / missing RLC handshake / `POWER_CNTL` stuck at 0
2. TrustOS late-F32 + doorbell values (`0x100001E0`) without reintroducing MC0
3. Compare live `STATUS2.CMD_OP` / F32 instr ptr while stuck
4. Only then vector-add (`AMD_BOOT_ADD=1`)


## Reference repos — helpfulness (reranked, session #20)

Blocker now: **MEC/KCQ ring fetch** (`ACTIVE=1`, `RPTR=0`). SDMA execute is PASS.

| Rank | Repo | Score | Why |
|------|------|-------|-----|
| **1** | `torvalds/linux` / `ROCm/amdgpu` | **10/10** | `PRELOAD_SIZE` mask; `mqd_commit` poll; KFD `DOORBELL_EN`; gfx_v7 MEC load |
| **2** | `allbilly/amdgpu` | **10/10** | This tree — AGP SDMA proof + PRELOAD bugfix |
| **3** | `nathan237/TrustOS` | **8/10** | Same class of fetch-OK/retire issues; no recorded SCRATCH=DEADBEEF |
| **4** | `tinygrad/tinygrad` | **5/10** | MEC setup_ring / doorbell_en patterns |
| **5** | pascal-egpu / SMU-BIST / NootedRed / rpi / tenstorrent | **2–3/10** | Weak analogy |
| **6** | Passthrough / Hackintosh / sims / ZLUDA / … | **0/10** | DeepWiki N/A |

**Next:** why MEC stays busy without advancing RPTR after correct `PS=0x5300` —
SMC LoadUcodes for MEC+JT (vs direct MMIO), RLC CP jump table, or real doorbell.

**Linux init order (confirmed from `gmc_v8_0_hw_init` + `sdma_v3_0_gfx_resume`):**
```
atom_asic_init (VBIOS POST — trains VRAM, sets MEMSIZE)
→ gmc_sw_init
→ gmc_v8_0_hw_init: golden_regs + mc_program + polaris10_mc.bin (if not trained by VBIOS) + gart_enable
→ SDMA ucode (MMIO ok without SMC per gfx_v7_0 direct path)
→ sdma_v3_0_gfx_resume: ring in GART, AUTO_CTXSW only if RLC loaded
→ sdma_v3_0_enable(unhalt) + ring_test_ring
```

**Session #13 code fix:** `_sdma_disable_auto_ctxsw()` — clears `SDMA0_CNTL.AUTO_CTXSW_ENABLE`
(TrustOS: without RLC, F32 silently refuses to fetch ring). Called before SDMA fw upload
and ring setup.

**09:08 panic:** overnight spontaneous `apciec 0x200000`.
**~09:12:** replug + `--reset` OK (`CONFIG_MEMSIZE=0`).
**~09:15 crash:** again before `sdma-probe` reported `write_ok` — see **Current blocker** above.

## Session #11 — bit 21 decoded: **"Request address is greater than 32 bits"**

Public T8103 panic reports (StackOverflow "MacOS kext panic Request address is greater
than 32 bits") show the *handled* version of our exact interrupt:

```
apciec[pcic1-bridge]::handleInterrupt: Request address is greater than 32 bits
linksts=0x99000001 pcielint=0x00220000 ... @AppleT8103PCIeCPort.cpp:1301
```

Same port interrupt mask `0x220000`; bit 21 (`0x200000`) = **inbound (device→host)
request whose bus address exceeds 32 bits**. On newer macOS the eGPU port has no
handler registered for it → `"unhandled interrupts (0x200000 out of 0x220000)"` panic.
Apple Silicon gives each PCIe device a DART; TinyGPU's `PrepareDMA` hands back small
32-bit IOVAs (`0x4000`, `0x88000`, …). So the panic fires whenever the GPU emits a TLP
with a **garbage / ≥4 GB address** — not on every DMA:

- Malformed GART walk (4-byte PTEs, VA root) → walker TLPs at garbage 64-bit addresses
  → panic (#5–#8). The 64-bit-PTE fix (session #9) attacks exactly this.
- Stale/uninitialized engine state (EOP/MQD/ring bases are 64-bit regs full of junk)
  → spontaneous panics with zero host activity.
- Correct 32-bit IOVA DMA (NV path) does NOT panic — consistent.

**Rules going forward:** every address the device can ever emit must be a <4 GB IOVA
from `PrepareDMA`; never unhalt an engine whose base/rptr/wptr addresses aren't
programmed; quiesce rings before halt/unhalt (already done for SDMA).

Also: `--boot-stage=sdma-probe` cold path can now skip the ATOM replay
(`AMD_BOOT_SDMA_ATOM=0`) — SDMA + AGP-to-host needs no VRAM training, and the ATOM
replay is minutes of MMIO with its own crash history. Order of first HW attempt:
`AMD_BOOT_SDMA_PROBE=1 AMD_BOOT_SDMA_ATOM=0 add.py --boot-stage=sdma-probe`
(no ATOM, no SMC, no MEC, AGP addressing, one WRITE_LINEAR).

### Session #11 results — first sdma-probe runs that DON'T panic

Ran the minimal probe (no ATOM, no SMC, no MEC) on hardware. **No kernel panic** —
first time an SDMA unhalt + ring fetch attempt did not kill the machine. But no data
either; the engine wedges on its first MC read:

| run | result |
|-----|--------|
| AGP mode, VBIOS-default `MC_VM_MX_L1_TLB_CNTL=0x503` | `rptr=0`, `STATUS=0x4496e446` — `IDLE=0`, `MC_RD_IDLE=0` (fetch read stuck in-flight) |
| AGP mode + `mc_setup_tlb_apertures()` (L1 TLB on, SYSTEM_ACCESS_MODE=3, ADV_MODEL on — gmc_v8_0_gart_enable values) | same stall |
| GART mode (64-bit PTEs verified by CPU: `pte=0x88077 == expected`) | same stall |

`sdma_soft_reset()` (SRBM bits, sdma_v3_0_soft_reset) recovers the engine to
`IDLE=1` every time — added to the probe as automatic cleanup, since a wedged
in-flight read is a prime suspect for the "spontaneous" delayed panics.

**Interpretation:** with the minimal (ATOM-skipped) boot, the device→host read never
leaves the chip / never completes — no panic, no data, engine stuck. In earlier FULL
boots (ATOM+SMC+MC) the reads clearly DID leave the chip (they panicked the root port
with garbage addresses). So the MC-hub/BIF path for outbound system reads needs some
init that only the ATOM asic_init (or SMC/MC fw) performs. Next: keep the GART-mode
probe (every emitted address is a CPU-verified <4 GB IOVA → cannot trip bit 21) but
run the ATOM replay first (`AMD_BOOT_SDMA_ATOM=1`), still no SMC / no MEC.

**01:21:38 panic (delayed again):** same `apciec 0x200000`, ~11 min after the last
probe — even though both probe runs ended with `sdma_soft_reset()` and a verified
`IDLE=1`. So the SRBM soft reset does not cancel whatever transaction is latched in
the MC hub / BIF, or the delayed panic has a different source entirely (macOS
periodically touching the stale device is one candidate — the un-posted GPU is
enumerated but half-dead). The wedged-read cleanup stays (it's correct per
sdma_v3_0_soft_reset), but it is demonstrably NOT sufficient to stop the delayed
panics. Conclusion stands: **only a boot state in which the GPU can actually
complete host reads is safe to leave on the link.**

Also fixed while reading gmc_v8_0_gart_enable line-by-line: our
`_gart_program_vm` wrote `MC_VM_MX_L1_TLB_CNTL=0x98000b` (SYSTEM_ACCESS_MODE=1, no
ENABLE_ADVANCED_DRIVER_MODEL). Linux uses MODE=3 (not-in-sys) + ADV_MODEL=1 —
without ADV_MODEL the VM/aperture path is not fully active on gfx8. Now programmed
field-by-field exactly like Linux (`mc_setup_tlb_apertures()`), shared by the GART
and AGP paths.

### Panic timeline update (01:28 / 01:35) — half-initialized GPU is itself a panic source

- **01:28:15**: panic ~30 s after the reboot from the 01:21 panic, with **zero** host
  commands — macOS re-enumerated the stale, half-dead eGPU (unified log shows apciec
  bridge enumeration + DART SID mapping right around it) and the root port tripped.
- **01:32**: AGP-mode sdma-probe (no ATOM) ran with **no panic** — engine wedged on
  the fetch again (`IDLE=0, MC_RD_IDLE=0`), auto `sdma_soft_reset()` → `IDLE=1`.
- **~01:33**: sdma-probe with `AMD_BOOT_SDMA_ATOM=1` — ATOM replay aborted mid-table
  (`ATOM jump loop stuck op=73 iters=512`): `AMD_ATOM_JUMP_MAX=512` was too small; a
  training/poll loop needed more. Training DID progress: `MISC0 0x0 → 0x50609162`.
- **01:35:07**: panic again — GPU left **mid-asic_init** (worse than stale).

Lesson: an interrupted ATOM replay leaves the most dangerous state of all. Never
abort it with a tight iteration cap; use the default budget and let it finish.

- **01:41:49**: panic ~20 s after a read-only `--probe`. The mid-asic_init GPU state
  **persists across host reboots** (enclosure keeps the card powered), so every new
  macOS boot re-enumerates the same broken device and panics within seconds-to-minutes
  of any contact — a reboot loop. Recovery plan: `--reset` (vi_asic_pci_config_reset
  via cfg 0x7c) to return the ASIC to power-on defaults, then a **complete** ATOM
  asic_init with the default jump budget (no `AMD_ATOM_JUMP_MAX` cap), then the GART
  sdma-probe. If `--reset` cannot stop the spontaneous panics, the eGPU enclosure
  needs a physical power-cycle before the next attempt.

### Session #12 — replug + Linux `gmc_v8_0_hw_init` gap closed

After replug (`CONFIG_MEMSIZE=0`, clean PCI). Code changes this session:

- **`gmc_hw_init_for_dma()`** — mirrors Linux `gmc_v8_0_hw_init` minus SMC: `mc_program`
  → `load_mc_firmware()` (polaris10_mc.bin) → re-`mc_program` if ATOM trained VRAM
  → `mc_program_apertures` + `mc_setup_tlb_apertures` + `gmc_program_vm_l2`.
  Root cause of ring-fetch stall without ATOM: we never loaded MC ucode.
- **`gmc_program_vm_l2()`** — VM_L2_CNTL/2/3 from `gmc_v8_0_gart_enable`; AGP path was
  missing this entirely.
- **`boot_sdma_minimal()`** default `AMD_BOOT_SDMA_ATOM=1` (full jump budget, no cap).
- **`sdma-probe`** default GART mode (`AMD_BOOT_SDMA_AGP=0`) — verified <4GB IOVAs.
- Next HW command after replug:
  `AMD_BOOT_SDMA_PROBE=1 python3 add.py --boot-stage=sdma-probe`

Read `/Library/Logs/DiagnosticReports/panic-full-*.panic`. **Every** panic (23:57, 00:04,
00:09, 00:16) is byte-identical:

```
panic(...): "apciec[pcic0-bridge] unhandled interrupts (0x200000 out of 0x220000)" @APCIECPort.cpp:2056
T8103 / AppleT8103PCIeC (USB4 root port); triggering task around TinyGPU
```

Bit `0x200000` (bit 21) is the Apple T8103 PCIe **root-port error interrupt** on a
device-initiated transaction it can't service; the port enables `0x220000` and has no
handler for `0x200000` → panic. Masking the **GPU's** MSI/INTx + IH/CP interrupts does
not help — it's the **bridge** reacting to an outbound (GPU→host) TLP, not a GPU IRQ.

**Key facts that narrow it down:**

- DeepWiki (tinygrad) confirms `MAP_SYSMEM_FD` returns the **device DMA addresses** from
  `IODMACommand::PrepareForDMA` — correct to program directly. So addressing *values* are right.
- The **NV** eGPU path DMAs host memory over this same TinyGPU/USB4 transport **without**
  this panic → device→host DMA is NOT fundamentally broken here.
- `gart-probe` (CPU-only PTE build, **no** SMC, **no** engine, **no** device DMA) never panics.
- Panics only appear once we (a) `start_smc` in a full boot, (b) unhalt an engine, or
  (c) let an engine do its first host read.

**New root-cause hypothesis — FB-aperture collision on the page-table walk.** GCN routes
an MC address to local VRAM when it falls inside the framebuffer aperture. We set
`FB_LOCATION` to `vram_start=0 .. vram_end=0xFFFFFFFF` (4 GB). The DMA addresses TinyGPU
hands back are **small** (`0x4000`, `0x88000`, `0x114000`) — i.e. **inside** `[0, 4 GB)`.
With `PTE_REQUEST_PHYSICAL=1`, the page-table **walker** reads the PTE at that small
physical address, which the MC routes to **VRAM**, not PCIe → it gets garbage → the data
access goes to a bogus address → the root port times out → `0x200000`. (Data pages carry
the PTE `SYSTEM` bit so they'd route to PCIe, but the *walk itself* does not.)

**Fix direction (this session): avoid the page-table walk entirely — use the AGP aperture.**
`amdgpu_gmc_agp_addr` maps a linear MC window `agp_start + dma_addr` straight to system
memory with **no page table**. `agp_start = 4 GB` (above the FB aperture), so AGP MC
addresses route to PCIe/host by construction. Program the SDMA ring base + dst as AGP MC
addresses → the only device transactions are (1) ring fetch read and (2) WRITE_LINEAR —
both routed to host via AGP, no walk, no FB collision.

Also: the SDMA proof no longer starts the **SMC** (unneeded — ucode is uploaded by MMIO;
SMC boot is an extra DMA/crash surface) and never touches RLC/CP/MEC.

**Code added:**

- `boot_sdma_minimal()` — ATOM → `mc_program` (FB/AGP/system apertures) → GART/AGP →
  SDMA-only fw (halted). No SMC, no RLC/CP/MEC.
- `probe_sdma_dma(use_agp)` — `AMD_BOOT_SDMA_AGP=1` (default): ring + dst via
  `agp_mc_addr(paddr)`, no GART page-table dependency.
- `sdma-probe` stage: cold GPU → `boot_sdma_minimal`; hot GPU → SDMA-only upload, no MEC.

Prior 64-bit-PTE + physical-base + ring-teardown fixes stay (correct for the GART path;
still needed if AGP is disabled).

## Session #9 — GART page table was malformed (still-valid fix)

Found the real reason **every** GPU→host DMA read panicked (KCQ #6, SDMA #7/#8, chained #5):
our GART page table was **wrong on two counts**, so the MC's very first page-table walk
read a bogus host address → `apciec 0x200000` completion timeout → macOS reboot.

| Bug | Was | Correct (gfx8 / `gmc_v8_0_gart_enable`) |
|-----|-----|------------------------------------------|
| **PTE width** | 4-byte entries, 4-byte stride | **8-byte (64-bit) PTEs** — `amdgpu_gmc_set_pte_pde` writes 64-bit; a 4-byte table means the walker reads every entry at the wrong offset |
| **Page-table base** | `PAGE_TABLE_BASE_ADDR = gart_start` (a GART **VA**, self-mapped) | **physical DMA address** of the PTE table — the root pointer is dereferenced by the MC directly, not through the table |
| **L2 request mode** | `VM_L2_CNTL4 = 0` | **PDE/PTE_REQUEST_PHYSICAL = 1** (ctx0+ctx1) so the walker reads the table + PTEs from **host RAM** (`/* enable PTE/PDE in system memory */`) |

Confirmed PTE size = 8 bytes and the physical-request requirement via DeepWiki
(`torvalds/linux` gmc_v8) and `ref/linux/.../gmc_v8_0.c:865-883`.

**Fix applied (`polaris_boot.py`):**

- `GART_PTE_SIZE=8`, `GART_PTE_ADDR_MASK` (`[47:12]`), `_encode_pte()` builds 64-bit PTEs.
- `gart_enable` (sysmem): 8-byte table sized `gart_size/4K * 8` (512 KB / 256 MB aperture),
  **contiguous** host alloc (walker needs `base_phys + off`), base = `paddrs[0]`,
  `_gart_program_vm(..., pte_physical=True)`.
- `_gart_program_vm`: sets `VM_L2_CNTL4` ctx0+ctx1 PDE/PTE_REQUEST_PHYSICAL when physical.
- `_gart_write_pte` / `map_sysmem_gpu` / `probe_gart_dma`: 8-byte stride + `<Q>` packing.
- `_paddrs_contiguous()` guard on the table alloc.

**Verified on HW (CPU-side, no device DMA, no panic):**

```
polaris: GART PTE table in host RAM base_phys=0x4000 entries=65536 bytes=0x80000
gart_probe pte_ok=True src_va=0xff00100000 paddr=0x88000 pte=0x88077 expected=0x88077
```

**Also this session:** `sdma-probe` cold path is now **SDMA-only (no RLC/CP/MEC upload,
no ME1 unhalt)** — SDMA is independent of the graphics pipe, so a pure DMA proof no
longer needs the whole compute bring-up (smaller crash surface).

**Crash note:** the staged run got cleanly through `fw-mec` → `fw-start` (ME1 live,
`CP_MEC_CNTL=0x10000000`, printed and exited 0) then the machine rebooted shortly after
— consistent with the known residual risk of leaving **ME1 unhalted** on USB4. The new
SDMA-only `sdma-probe` avoids ME1 entirely; that is the next HW test.

**Next HW test (cold, single shell, self-contained):**

```
AMD_BOOT_VBIOS_FILE=/tmp/rx570.rom AMD_BOOT_SDMA_PROBE=1 python3 add.py --boot-stage=sdma-probe
```

boots ATOM→SMC→MC→GART(64-bit phys table)→SDMA-only(halted)→ring in GART→unhalt SDMA→
WRITE_LINEAR→poll dst. If the PTE fix is right, `write_ok=True` with no panic — the first
proven device→host DMA on this stack.

### Linux amdgpu SDMA read (ref/linux `sdma_v3_0.c`, `gmc_v8_0.c`) — audited 2026-07-09 ~00:10

Confirmed our path now matches Linux for a host-memory ring:

| Detail | Linux | Ours |
|--------|-------|------|
| GART PTE size | 8 bytes (`amdgpu_gmc_set_pte_pde`) | ✅ 8-byte |
| PTE encode | `paddr[47:12] \| flags(0x77)` | ✅ `_encode_pte` |
| Page-table base | `PAGE_TABLE_BASE_ADDR = table_paddr>>12` (phys) | ✅ `gart_pte_phys` |
| Host-mem walk | `VM_L2_CNTL4` PDE/PTE_REQUEST_PHYSICAL=1 | ✅ ctx0+ctx1 set |
| ctx0 depth | flat, `PAGE_TABLE_DEPTH=0` | ✅ `VM_CONTEXT0_CNTL=0x11` |
| Ring base | `RB_BASE = gpu_addr>>8`, `RB_BASE_HI = gpu_addr>>40` | ✅ |
| wptr (no doorbell) | `WREG32(SDMA0_GFX_RB_WPTR, wptr<<2)` | ✅ `_sdma_gfx_ring_commit` |
| Order | `enable(false)`(gfx_stop+HALT) → program RB while halted → `RB_ENABLE=1` → `enable(true)` | ✅ `sdma_enable(False)` → `_sdma_gfx_ring_setup` → unhalt at end |
| Ring test pkt | 5-dw WRITE_LINEAR, `COUNT(1)`, `0xDEADBEEF` | ✅ |

tinygrad `AMDev` (DeepWiki) confirms the same model: ring lives in a GPUVM VA; the page
table is in **host memory**; `VM_CONTEXT0_PAGE_TABLE_BASE_ADDR` = **physical** address of
the root table (no framebuffer/AGP offset). So the design is correct; the only open
question is whether M1/USB4 can service the device→host **read** of the table+ring.

Extra hardening in `_sdma_gfx_ring_setup`: clear `RPTR_WRITEBACK_ENABLE`, zero
`RB_RPTR_ADDR`, disable `RB_WPTR_POLL_CNTL` and `SDMA0_GFX_DOORBELL` so the *only*
engine-initiated device DMA is the ring fetch itself (rptr writeback / wptr poll would
each add another host access that could independently time out).

### Bug fixed (2026-07-08 ~23:57): cold `sdma-probe` reached `gart_probe pte_ok=True`
(64-bit table accepted) then raised `SDMA ucode not resident` — `load_ip_firmware_direct`
uploaded SDMA but never set `_sdma_fw_resident`, so `probe_sdma_dma()` bailed **before**
the WRITE_LINEAR. Now set on the direct path too. GPU came back cold after (USB4 re-enum);
next run reaches the actual device DMA.

**Hardened SDMA ring setup (2026-07-09 ~00:05):** `_sdma_gfx_ring_setup` now disables
every *engine-initiated* device DMA so the probe's only device→host access is the ring
fetch itself: cleared `RPTR_WRITEBACK_ENABLE` (else the engine DMA-writes the rptr report
to host), zeroed `RB_RPTR_ADDR_HI/LO`, disabled `RB_WPTR_POLL_CNTL.ENABLE` (else it
DMA-reads a host wptr shadow) and `GFX_DOORBELL.ENABLE`. This isolates the one read we
actually want to prove.

**Prior line:** 2026-07-08 ~23:28 — `fw-sdma` FIXED (no panic); panic #8 at `sdma-probe` ring fetch

## Session #8 — `fw-sdma` fixed; panic #8 is the SDMA ring fetch itself (2026-07-08 ~23:28)

**`fw-sdma` no longer panics.** The code fix from Session #7 works on a **hot GPU**
(`CP_MEC_CNTL=0x10000000`, ME1 live, SMC up):

```
atom → fw-mec (0x50000000) → fw-start (0x10000000) → fw-sdma
stage=fw-sdma sdma_only=True unhalt=False CP_MEC_CNTL=0x10000000 F32_CNTL=0x1
  polaris: SDMA-only firmware loaded (polaris10_sdma.bin, polaris10_sdma1.bin)
```

- **SDMA-only upload** (2×3109 words, ~1s) — MEC never re-halted/re-uploaded.
- SDMA left **halted** (`F32_CNTL=0x1`), GFX rings torn down (`RB_ENABLE=0`, `RB_BASE=0`).
- No kernel panic. This is exactly the incremental, safe path the Session #7 fix planned.

**Then `AMD_BOOT_SDMA_PROBE=1 sdma-probe` kernel-panicked (panic #8).** This is the
`_sdma_gfx_ring_setup` → `sdma_enable(True)` → WRITE_LINEAR path: the moment the SDMA
F32 is unhalted **with `RB_ENABLE=1` pointing at a GART-sysmem ring**, the engine
DMA-reads that ring from host memory over USB4 → `apciec 0x200000` completion timeout.

**Conclusion — the wall is device→host DMA *read*, not firmware upload.**
`fw-sdma` (all MMIO, no DMA) is safe. `sdma-probe` fails at the **first ring fetch**,
i.e. the GPU reading the RB from a host physical address. This matches every prior
panic (KCQ MQD preload #6, chained boot #5): **any GPU→host read over M1/USB4 times
out at the bridge.** CPU-side checks (`gart-probe` PTE self-map) pass because they
never involve a device read.

**New hypothesis (addressing, not transport):** on Apple Silicon a PCIe device behind
the USB4 **DART** sees *IOVA*, not host physical addresses. If `MAP_SYSMEM` hands us
host paddrs but the GART PTEs / ring base must carry **DART-translated IOVAs** for the
GPU to reach host RAM, every ring fetch lands on an unmapped DART address → completion
timeout → panic. Next work is in code/transport (verify the DMA address domain), **not**
another blind `sdma-probe` (it reboots the machine).

**Fix applied this session (code):**

| Change | File | What |
|--------|------|------|
| `_sdma_gfx_ring_disable(off)` | `polaris_boot.py` | Mirror `sdma_v3_0_gfx_stop`: clear `RB_ENABLE`+`IB_ENABLE`, zero `RB_BASE`/`RB_BASE_HI`/`RPTR`/`WPTR`, per instance |
| `sdma_enable(False)` | `polaris_boot.py` | gfx-stop **both** instances before F32 HALT (so next unhalt can't fetch stale `RB_BASE`) |
| `load_sdma_firmware_only(unhalt=False)` | `polaris_boot.py` | SDMA-only MMIO upload; never touches live CP/MEC (fixes panic #7 hot re-upload) |
| `sdma_fw_resident()` | `polaris_boot.py` | Track ucode residency separate from unhalt state |
| `probe_sdma_dma()` | `polaris_boot.py` | No longer requires pre-unhalt; ring setup unhalts only after `RB_BASE` is a valid GART VA |
| `fw-sdma` stage | `add.py` | Hot GPU → `load_sdma_firmware_only`; default `unhalt=False` |
| `sdma-probe` stage | `add.py` | Hot GPU → GART + SDMA-only upload (halted), never re-boot MEC |

**Status:** `fw-sdma` = ✅ safe now. `sdma-probe` = ❌ **panic #8 (ring fetch)** — the
device→host DMA read wall. Investigate DART/IOVA addressing before re-running.

**Prior line:** 2026-07-08 ~23:10 — APCIE panic #7 during `fw-sdma` (never reached `sdma-probe`)

## Session #7 — `fw-sdma` panic before `sdma-probe` (2026-07-08 ~23:08)

Attempted staged DMA proof: `fw-sdma` → `AMD_BOOT_SDMA_PROBE=1 sdma-probe`.

**Pre-crash state** (`--probe` immediately before):

```
pci=1002:67df  CP_MEC_CNTL=0x10000000 (ME1 already running)
SMC running=True  CONFIG_MEMSIZE=0x1000 (trained)
```

GPU was **hot** from an earlier session (`fw-start` / `kiq`), not cold.

**What ran:** `python3 add.py --boot-stage=fw-sdma`

- Calls `boot_through_fw_direct(FW_COMPUTE_MIN | SDMA0 | SDMA1)` with **`unhalt=True`**
  (default `AMD_BOOT_FW_UNHALT=1`).
- Full re-bootstrap on a live GPU: ATOM → SMC → MC → GART → **halt running MEC** →
  re-upload RLC + PFP + CE + ME + **MEC** + SDMA0 + SDMA1 (~200 s MMIO) → unhalt CP + MEC +
  **SDMA F32**.
- Process killed @~200 s — **macOS kernel panic / reboot** (no stage output captured).
- **`sdma-probe` never ran.**

**Root cause (same APCIE class as panic #6, new trigger):**

| Factor | Problem |
|--------|---------|
| **Hot re-upload** | `fw-sdma` is not incremental — it halts a **live ME1** and re-streams the entire compute firmware blob on USB4. High MMIO volume + mid-session MEC halt is fragile. |
| **SDMA unhalt without ring teardown** | Linux `sdma_v2_4_gfx_stop` clears `RB_ENABLE` + `IB_ENABLE` **before** F32 halt. Our `load_ip_firmware_direct` only sets `F32_CNTL.HALT`; stale `RB_BASE` / `RB_ENABLE` may survive. Unhalting SDMA (`sdma_enable(True)`) can make F32 **immediately DMA-read a garbage ring address** → device→host read → `apciec 0x200000` panic. |
| **Wrong prerequisite path** | `sdma-probe` needs SDMA ucode resident, but `fw-sdma` as written is a full fw re-bootstrap, not “upload SDMA bins only, stay safe.” |

**Not the failure mode we intended to test:** we never reached the gated `sdma-probe`
WRITE_LINEAR experiment. The crash is at SDMA **firmware upload + unhalt**, not at the
deliberate ring-commit path in `probe_sdma_dma()`.

**Fix needed (code, not yet applied):**

1. `fw-sdma` on hot GPU: **SDMA-only upload** if MEC already running (`skip_fw` pattern);
   do not re-halt/re-upload MEC.
2. Before any SDMA halt/unhalt: call `_sdma_gfx_ring_disable()` (mirror
   `sdma_v2_4_gfx_stop`) — clear `RB_ENABLE`, `IB_ENABLE`, zero `RB_BASE`.
3. Default `fw-sdma` to **`unhalt=False`**; let `sdma-probe` unhalt only after ring is
   mapped in GART.
4. Or fold SDMA upload into `sdma-probe` itself with `unhalt=False` upload + controlled
   ring setup + single unhalt at the end.

**Current blocker:** cannot run `sdma-probe` until SDMA fw can be loaded **without panic**.
Until then, KCQ activation (`AMD_BOOT_KCQ_ACTIVATE=1`) and vector-add remain blocked.

## Session #6 — panic root cause + fix (2026-07-08 ~22:40)

`--boot-stage=kiq` kernel-panicked macOS (reboot; BARs re-enumerated). Root cause is a
regression introduced after commit `dcb3ef5` ("result 0 but no more crash"):

- The WIP commit added a **direct KCQ HQD commit** path — `mqd_init_vi(..., activate=True)`
  sets `CP_HQD_PERSISTENT_STATE.PRELOAD_REQ` + `CP_HQD_ACTIVE=1`. Committing it makes the
  MEC **preload the queue context by DMA-reading the MQD/ring from GART host sysmem**.
- On M1/USB4 (`AppleT8103PCIe`) that device→host read is not serviceable → PCIe completion
  timeout → `apciec unhandled interrupts (0x200000)` → kernel panic. Masking the GPU's own
  MSI/INTx (session #5 fix) does not help: it is the *bridge's* error interrupt on a failed
  downstream transaction, not the GPU's own IRQ.
- The `dcb3ef5` "no crash" version left the KCQ MQD **in memory only** (no activation), which
  is exactly why `kiq` used to be safe. Activating the KIQ alone (no `PRELOAD_REQ`, empty ring)
  was and remains safe.

**Fix (`polaris_boot.boot_allow_hqd_activation`):** any HQD *activation* (the direct-KCQ
`activate`/`PRELOAD_REQ` commit and its MAP_QUEUES auto-fallback) is now gated. Default =
staged-in-memory only, no activation. `kiq`/`kcq-direct` are inspection-only again and no
longer crash. Activation requires an explicit opt-in (`AMD_BOOT_KCQ_ACTIVATE=1`, or the
already-gated `AMD_BOOT_RING_TEST=1` / `AMD_BOOT_ADD=1` / `AMD_BOOT_FULL=1`). Verified on HW:
`kiq` runs to completion, `KIQ_HQD_ACTIVE=0x1`, `KCQ_HQD_ACTIVE=0x0`, no panic.

**Still blocked:** vector-add needs a live KCQ → needs proven device→host DMA → needs
`sdma-probe` → needs SDMA fw loaded safely. **Panic #7 showed `fw-sdma` itself is unsafe**
on a hot GPU; fix SDMA upload path before retrying.

**Prior line:** 2026-07-08 ~22:40 — panic #6 KCQ activation gated; `kiq` safe

## Status

| Item | State |
|------|--------|
| **Solved** | ATOM `asic_init` — VRAM trains (`MEMSIZE=4096`, `MISC0\|0x80`, `trained=True`) |
| **Solved** | Direct MMIO firmware upload path (bypasses SMC `LoadUcodes`) |
| **Solved** | Staged fw upload: `atom` → `fw-mec` completes without panic (~32s total) |
| **Solved** | SRBM / KCQ direct / GART PTE — verified in **earlier** sessions (not re-validated this reboot) |
| **Blocker** | **CP rings never drain** — `PQ_RPTR=0x0`; `SCRATCH` stuck at `0xCAFEDEAD` (earlier `kcq-ring-test`) |
| **Blocker** | **GPU-side GART DMA unproven** — `sdma-probe` never ran; blocked by unsafe `fw-sdma` |
| **Blocker** | **`fw-sdma` panics on hot GPU** — full MEC re-upload + SDMA unhalt without `RB_ENABLE` teardown (panic #7) |
| **Fixed** | **KCQ HQD activation gated** — `kiq` safe (`KIQ=1`, `KCQ=0`); panic #6 root-caused |
| **Fixed** | **macOS USB4 APCIE MSI panic (IRQ path)** — interrupt mask before unhalt; does not fix DMA completion timeout |
| **Session #7** | **`fw-sdma` panic @~200s** on hot GPU (`CP_MEC_CNTL=0x10000000`); `sdma-probe` not reached |
| **Furthest safe point** | After replug: `atom` → `fw-mec` → `fw-start` → `kiq` (no KCQ activation) |
| **Next (code)** | Fix `fw-sdma`: SDMA-only incremental upload, `sdma_v2_4_gfx_stop` before halt, `unhalt=False` default |
| **Next (HW)** | Only after fix: `AMD_BOOT_SDMA_PROBE=1 sdma-probe` → then consider `AMD_BOOT_KCQ_ACTIVATE=1` |
| **Safe** | `--probe`, `--selftest`, `atom`, `fw-mec` (MEC halted), `kiq` (KCQ staged only) |
| **Unsafe** | **`fw-sdma` on hot GPU** (proven panic #7), `AMD_BOOT_KCQ_ACTIVATE=1`, chained stages |
| **Gated** | `sdma-probe`, `kcq-ring-test`, `add`, `AMD_BOOT_FULL=1` |

---

## Current stage & blocker (2026-07-08 ~23:10)

### Session #7 — `fw-sdma` panic (sdma-probe prerequisite failed)

| Step | Command | Result |
|------|---------|--------|
| 1 | `--probe` | `pci=1002:67df` `CP_MEC_CNTL=0x10000000` ME1 hot, trained |
| 2 | `--boot-stage=fw-sdma` | **kernel panic @~200s** — no output |
| 3 | `AMD_BOOT_SDMA_PROBE=1 sdma-probe` | **not reached** |

```
atom → fw-mec → fw-start → kiq  →  fw-sdma  →  sdma-probe  →  KCQ activate  →  add
 ✓       ✓        ✓       ✓          ✗            ✗               ✗            ✗
                              ↑ panic #7 (hot re-upload + SDMA unhalt)
```

**Unified APCIE panic mechanism (sessions #5–#7):**

Any operation that makes the GPU **DMA-read host sysmem** over M1/USB4 can trigger
`apciec unhandled interrupts (0x200000)` — completion timeout on the bridge, not GPU MSI.
Known triggers:

| Trigger | Session |
|---------|---------|
| KCQ HQD `PRELOAD_REQ` + `CP_HQD_ACTIVE` | #6 (fixed — gated) |
| SDMA F32 unhalt with stale `RB_ENABLE` / garbage `RB_BASE` | **#7** |
| SDMA ring commit / MEC ring fetch (untested) | pending `sdma-probe` |
| Chained `fw-start → gart → kcq` on hot GPU | #5 |

Interrupt masking fixes the **IRQ** path only; it does **not** fix DMA completion timeout.

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

# 8. SDMA fw — **UNSAFE on hot GPU until code fix (panic #7)**
#    Need: SDMA-only upload, RB_ENABLE cleared, unhalt=False until sdma-probe sets ring
# python3 add.py --boot-stage=fw-sdma
# sleep 10
# AMD_BOOT_SDMA_PROBE=1 python3 add.py --boot-stage=sdma-probe

# 9. KCQ direct HQD — RUN ALONE (activation gated unless AMD_BOOT_KCQ_ACTIVATE=1)
python3 add.py --boot-stage=kcq-direct

# 10. KCQ ring test — gated; MMIO wptr only on darwin
AMD_BOOT_RING_TEST=1 python3 add.py --boot-stage=kcq-ring-test

# 11. Vector-add — only after ring_ok=True and sdma-probe write_ok=True
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
| `--boot-stage=gart-probe` | ⚠️ medium | GART PTE setup (CPU-only) |
| `--boot-stage=fw-sdma` | ❌ **unsafe on hot GPU** | Panic #7 — full MEC re-upload + SDMA unhalt w/o ring teardown |
| `--boot-stage=sdma-probe` | ❌ **blocked** | Needs safe SDMA upload first; then `AMD_BOOT_SDMA_PROBE=1` |
| `--boot-stage=kcq-direct` | ⚠️ medium | KIQ+staged KCQ (activation gated) |
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

1. **Replug eGPU** after panic — `--probe` must show `pci=1002:67df`.
2. **Code fix `fw-sdma`** before any more SDMA experiments (see Session #7).
3. Cold path only until fix: `atom` → `fw-mec` → `fw-start` → `kiq` (each alone, sleep 10s).
4. **Do not run `fw-sdma` on hot GPU** — proven panic #7.
5. After fix: `AMD_BOOT_SDMA_PROBE=1 sdma-probe` — the real DMA proof.
6. Only if `sdma-probe write_ok=True`: consider `AMD_BOOT_KCQ_ACTIVATE=1`.
7. **Never chain stages** in one shell command on USB4.

Open question: is M1/USB4 device→host DMA broken entirely, or only when ring addresses
are wrong? Cannot answer until `fw-sdma` is fixed enough to reach `sdma-probe`.

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
- [x] KCQ activation gate (`boot_allow_hqd_activation`) — panic #6 fixed
- [x] `sdma-probe` code path (WRITE_LINEAR + CPU readback) — **HW blocked by `fw-sdma` panic**

## Todo

- [x] ATOM training (`atom_replay.py` bugs fixed)
- [x] SMC boot
- [ ] **Fix `fw-sdma`**: SDMA-only incremental upload, `sdma_v2_4_gfx_stop`, `unhalt=False`
- [ ] Run `sdma-probe` on HW — prove or disprove GART device DMA
- [ ] CPU-visible VRAM path (BAR0 or MM_INDEX) **or** proven GART DMA
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
| `AMD_BOOT_SDMA_PROBE` | `0` | `1` = allow `--boot-stage=sdma-probe` (device DMA; panic risk) |
| `AMD_BOOT_SDMA_PROBE_TIMEOUT_S` | `5` | poll dst buffer after SDMA WRITE_LINEAR |
| `AMD_BOOT_KCQ_ACTIVATE` | `0` | `1` = allow KCQ HQD activation (after sdma-probe passes) |
| `AMD_BOOT_DOORBELL_SETTLE_MS` | `10`/`50` | sleep after wptr signal (10 when no doorbell) |
| `AMD_BOOT_VBIOS_FILE` | — | Path to `rx570.rom` |
| `AMD_ATOM_JUMP_BAIL` | `0` | `1` = fake-complete ATOM (obsolete now) |
| `DEBUG` | `0` | Verbose logging |
