# HD 4850 eGPU Bring-Up Progress (TinyGPU / M1 Mac)

**Last updated:** 2026-07-11

## Current blocker

**Local GDDR3 VRAM does not retain writes** (stable float bus on FB@0+BIF).  
**AGP + CP `--test` PASS.**

Linux **radeon does not fix this in driver code** — RV770 boot/resume only runs `atom_asic_init` (VBIOS), then `rv770_mc_program` (apertures / `BIF_FB_EN` / HDP). No GDDR3 trainer, no `mc.bin`.

---

## Linux radeon (local `ref/linux/.../radeon/`)

| Step | What it does for VRAM |
|------|------------------------|
| `atom_asic_init` | FWI def SCLK/MCLK → `ASIC_Init` only |
| `rv770_mc_init` | Read `CONFIG_MEMSIZE` / channels — assumes DRAM live |
| `rv770_mc_program` | HDP clear, `rv515_mc_stop` (BIF=0→blackout), apertures, `mc_resume` (clear blackout→BIF=3) |
| DPM later | May `SetVoltage` / program MPLL regs — **not** first-time GDDR3 train |

`SetMemoryClock` flags in `atombios.h`: `FIRST_TIME_CHANGE_CLOCK=0x08000000`, `SKIP_SW_PROGRAM_PLL=0x10000000`. **Driver never sets these**; only VBIOS nested calls do.

Honest take: copying `mc_program` cannot fix float-bus if BIF/blackout already sane.

---

## Best software strategy so far

### `AMD_ATOM_REPAIR_AFTER_MPLLINIT=1` (now **default ON**)

1. Let `MemoryPLLInit` write **CLKF=0** (VBIOS power-up window)
2. **Immediately repair MPLL → CLKF=73** before nested DLL/Training/DeviceInit
3. Rest of `SetMemoryClock` continues with live clock

Results (synth + this hook):

- `MISC0=0x3000422a`, AGP **PASS** (unlike `PATCH_MPLL`)
- IO_DEBUG: **57** pairs before repair (CLKF=0), **159** after (CLKF=73)
- VRAM still **float** (`0x5555555d` / `0x5d555555`) — not sticky

### Also tried

| Experiment | Result |
|------------|--------|
| `PATCH_MPLL` (replace CLKF=0 writes) | Breaks MISC0/AGP; BIF hang |
| Post-hoc-only repair (old default) | AGP OK; all train at CLKF=0 |
| Nested-PS `finish_memory` | Safe; no stick |
| `SetMemoryClock(SKIP\|FIRST\|mclk)=0x180183e4` after good post | No CLKF=0 writes; MISC0 OK; no stick |
| `mc_program`-style FB@0+BIF | Decode works (float visible); writes don’t retain |

---

## Status

| Path | Status |
|------|--------|
| `--atom` (repair-after-MemoryPLLInit) + `--cp-mem-write-test` | **PASS** |
| VRAM stick (MM/BAR0) | **FAIL** (float) |
| `--cp-mem-write-test` | **PASS** — CP writes a supplied payload to AGP-mapped host sysmem; this is **not GPU add** |
| Default `add.py` | **REFUSES** — RV770 GPU ALU/shader path is not implemented; no CPU fallback |
| `AMD_ATOM_PATCH_MPLL` | **Do not use** |

`add.py` now maps BAR0 lazily: only `--vram-probe` (or an explicit
`AMD_BOOT_PROBE_BAR0=1` probe) opens it. Normal boot leaves `BIF_FB_EN=0`, parks
the FB range above the AGP aperture, and puts the CP ring, writeback page, and
diagnostic payload output in contiguous host memory. Thus local VRAM is not a
prerequisite for the RV770 CP write smoke test.

## Diagnosis

This is not an `add.py` allocation or Linux aperture-programming bug. The
Linux RV770 sequence is `atom_asic_init` followed by `rv770_mc_program`: it
clears HDP, programs `MC_VM_*`, releases MC blackout, and enables
`BIF_FB_EN`. The probe reproduces that state yet reads the stable floating
pattern after a write. Those registers choose a route; they cannot make an
unpowered/untrained GDDR3 device retain data. The remaining VRAM investigation
is therefore ATOM power/training or board hardware (especially MVDD/GDDR3), not
CP or AGP setup.

## Real RV770 add status

`rv770_add.ll` now compiles with LLVM's `-march=r600 -mcpu=rv770` backend to a
64-byte pixel shader containing exactly four hardware `ADD` instructions and a
real `SQ_EXPORT_PIXEL` color export. `rv770_vs.ll` compiles to a 48-byte vertex
shader with `SQ_EXPORT_POS`, `SQ_EXPORT_PARAM[0]`, and
`SQ_EXPORT_PARAM[1]`; `add.py` relocates its CF-export GPRs to Mesa's R700
fetch ABI (`GPR1..3`, because `GPR0` contains the vertex index).  A hand-built
80-byte R700 VFETCH shader reads the three 48-byte vertex attributes from
resource 160, and the PM4 draw contains only graphics packets—no
`PKT3_MEM_WRITE` result payload.

The draw has now been submitted on the attached `1002:9442` card.  CP consumes
the complete 108-dword graphics stream (`CP_RPTR == CP_WPTR`) and the card
remains reachable afterwards, but the AGP `CB_COLOR0` target remains all zero.
Therefore **GPU add is still failing**, not silently falling back to CPU.  The
next debug target is the remaining Linux RV770 graphics initialization/context
state or the 3D color-write route to the AGP aperture.

`AMD_BOOT_ATOM=0 python3 examples_egpu_terrascale/add.py --gpu-add-preflight`
has passed on the attached card: CP/ring initialization succeeds and returns
AGP addresses for the VS, PS, vertex buffer, and FP32 color target. It makes no
graphics submission; this verifies the exact input/output allocation topology
for the next draw-stage implementation.

## Recipe

```bash
rm -f $TMPDIR/amd_usb4.lock
python3 examples_egpu_terrascale/add.py --clock-probe   # prefer CHG=True
python3 examples_egpu_terrascale/add.py --atom           # REPAIR_AFTER_MPLLINIT default 1
AMD_BOOT_ATOM=0 python3 examples_egpu_terrascale/add.py --test
AMD_BOOT_ATOM=0 python3 examples_egpu_terrascale/add.py --vram-probe
```

## GPU-add debug infrastructure (new)

Following `plan.md` Phase 1-4, the graphics-add path now isolates the pipeline:

- **`CB_COLOR_CONTROL = 0x00CC0000`** (was `0` → CB disabled). Encoded by
  `rv770_cb_color_control(rop3=0xCC, special_op=0)`; `CB_COLOR0_INFO` by
  `rv770_color_info_rgba32_float()` (RGBA32_FLOAT, no CMASK/FMASK).
- **Completion fence**: separate AGP page + `SURFACE_SYNC` + `EVENT_WRITE_EOP`
  (CACHE_FLUSH_AND_INV_TS, 32-bit seq write, no IRQ). `run_add` polls the fence,
  not the color target. Fallback `--gpu-add-fence-mode=wait-memwrite` uses
  `WAIT_UNTIL` + a CP `MEM_WRITE` **to the fence only**.
- **Canary**: color target filled with `0xA5` before each draw, so outcomes are
  `canary-intact` (no write) vs `wrote-zero` vs `expected`.
- **Stage ladder**: `--gpu-add-stage={cp,constant,param0,add}`.
  - `cp` — fence only; proves completion + CPU visibility.
  - `constant` — `rv770_constant_ps.ll` exports `{0.25,-0.5,3.0,1.0}`.
  - `param0` — `rv770_param0_ps.ll` exports interpolated PARAM0.
  - `add` — `rv770_add.ll` exports PARAM0+PARAM1.
- **Validation**: `validate_gpu_add_pm4` rejects any `MEM_WRITE` to the color
  target, requires exactly one completion fence, and (for graphics stages)
  requires the fence after the draw.
- **Diagnostics**: `--gpu-add-dump-pm4` (offline decoder) and
  `--gpu-add-dump-registers` (GRBM/CB/SQ/DB snapshot). Full R700 graphics
  context defaults are opt-in via `--gpu-add-full-gfx-init`.

Hardware evidence: `--gpu-add-stage=cp --gpu-add-fence-mode=wait-memwrite`
passes, proving the separate fence page and CPU/AGP visibility.  A graphics
stage with the diagnostic raw fence reaches the post-draw fence, but the
`0xA5` color canary remains unchanged and `GRBM_STATUS` reports active
SH/VGT/SPI/PA units.  The preferred EOP fence times out because graphics does
not reach EOP; this is now distinct from a missing color write.  No CP packet
writes the color allocation.  Default `add.py` remains intentionally failing
until a GPU-produced color result is verified.

An additional `--gpu-add-stage=stream` experiment relocates the four FP32 ADDs
into a VS and configures `VGT_STRMOUT_BUFFER_0` to the AGP result page.  It also
leaves the canary unchanged.  Mesa's allocator shows a key constraint:
`PIPE_USAGE_IMMUTABLE` shader BOs are placed in `RADEON_DOMAIN_VRAM`, while
this direct path puts VS/PS/fetch code in AGP.  Persistent SH activity is
consistent with RV770 shader instruction fetch not accepting that AGP
placement; this is the leading no-VRAM blocker under investigation.

The streamout route was also exercised on hardware.  Its VS contains four
real FP32 ADDs and programs `VGT_STRMOUT_BUFFER_0` at the AGP result page, but
the result canary remains intact.  Mesa's R600 source confirms immutable
shader BOs are allocated in VRAM, whereas GTT/AGP is used for ordinary buffer
resources.  This supports the current diagnosis that CP/AGP DMA works while
the RV770 shader instruction path is not usable from this AGP-only setup; no
CPU or CP arithmetic fallback has been enabled.

Offline `python3 examples_egpu_terrascale/add.py --selftest` passes (R600 llc
present: constant/param0/add PS all compile, stage ladder + validator OK).

Env: `AMD_ATOM_REPAIR_AFTER_MPLLINIT=1` (default), `AMD_ATOM_PATCH_MPLL=0`.

### 2026-07-11 empty-shader isolation

Added opt-in `AMD_GPU_ADD_EMPTY_VS=1` alongside `AMD_GPU_ADD_EMPTY_PS=1`.
Both programs are then minimal `CF_END` blobs and the vertex resource/fetch is
skipped; this is diagnostic only.  On the HD 4850 the combined empty-shader
test with `--gpu-add-fence-mode=wait-memwrite` still timed out, with the color
canary intact (`CP_RPTR==CP_WPTR`, `GRBM_STATUS=0xb2303028`).  The hang is
therefore not specific to the compiled arithmetic or normal vertex-fetch
shader.  No CPU result or CP color write was introduced.

Corrected CP resume to Linux's `CP_ME_CNTL=0` instead of the prior `0xFF`
(reserved low bits).  A constant graphics run still timed out with an
untouched canary, so this parser-state mismatch was not sufficient either.

Found and corrected one real register-constant bug: `PA_CL_CLIP_CNTL` was
mistakenly set to `0x28038` (a non-clip register); Mesa/Linux define it as
`0x28810`.  The full-context constant-stage retest still timed out, so this
fix is necessary but not sufficient.  It remains in the normal code path.

Moved the essential clip/depth/scissor defaults from the optional full-context
replay into every graphics draw packet.  The normal constant-stage test still
timed out with an intact canary (`GRBM_STATUS=0xb2703028`), so stale raster
state is not the remaining blocker.

Ported additional non-context RV770 defaults from `rv770_gpu_init`:
`SQ_MS_FIFO_SIZES`, `SX_EXPORT_BUFFER_SIZES`, `PA_SC_FIFO_SIZE`, and
`SPI_CONFIG_CNTL_1`.  The constant-stage draw still timed out with the canary
intact, so missing FIFO/export defaults were not sufficient.

The isolation state was tightened so an empty VS advertises zero SPI outputs
instead of the normal position/parameter linkage; the same hardware timeout
(`GRBM_STATUS=0xb2303028`) persisted.  This rules out a stale VS-output
declaration as the immediate cause.

The empty-VS probe now also uses a minimal CF_END fetch program (instead of
the normal VTX fetch microprogram).  It still times out with the same status
and untouched canary, so neither the normal fetch shader nor its linkage is
the sole trigger.  This strengthens the evidence that graphics instruction
execution/state on this card is unavailable from the AGP-only setup.

Added diagnostic `AMD_GPU_ADD_POST_DELAY_S` because the raw fence is written by
CP immediately after the draw and is not a graphics completion signal.  A
5-second delay after the raw-fence constant draw still left the 0xA5 canary
untouched (`GRBM_STATUS=0xa2703028`).  The missing CB write is therefore not
just a polling race.

Matched the kernel soft-reset hold time (50 ms instead of the previous 100 us)
for the RV770 graphics-unit reset.  A normal constant-stage run still timed
out (`GRBM_STATUS=0xb2703028`), so reset pulse duration was not the fix.

## Next (VRAM likely beyond Linux driver)

1. Diff IO_DEBUG **content** (57 @ CLKF=0 vs 159 @ CLKF=73) vs desktop-posted dump
2. MMIO-replay nested `GPIOPinControl` write list (MVDD GPIO)
3. Hardware: eGPU **MVDD / GDDR3** power (float `0x55` pattern strongly suggests undriven DQ)

The latest direct test also confirms `--vram-probe` still fails after the full
ATOM memory sequence: MM_INDEX and BAR0 reads return floating values rather
than the written pattern.  This matters because Mesa places immutable R600
shader BOs in VRAM; the AGP-only path can prove CP DMA but cannot yet prove
shader instruction fetch.  The remaining route to a genuine add is therefore
either restoring the card's GDDR3/VRAM path or proving an R700 GTT shader-fetch
configuration; no CPU arithmetic fallback is acceptable.

After dock recovery the HD 4850 re-enumerated with the AGP aperture at
`0x80000000` instead of `0x0`.  The safe CP stage passed at that address, but
a fresh constant graphics test still timed out with an intact canary
(`GRBM_STATUS=0xb27030a8`), so the failure is not limited to low AGP addresses.

The smallest valid graphics probe (`AMD_GPU_ADD_CONSTANT_VS=1` with an empty
PS and no normal vertex fetch) reaches the raw CP fence but leaves the canary
untouched even after a 3-second delay (`GRBM_STATUS=0xa2703028`).  This rules
out normal VFETCH and PS arithmetic as the first failure; the VS/raster/CB
path still produces no observable AGP write.

Ran `AMD_BOOT_VRAM_IO_DEBUG=1` after widening the indexed snapshot to 0x200
entries.  The memory-tail replay changes 167 IO_DEBUG entries (mostly
indices `0x140–0x1ff`), matching the expected large post-MPLL programming
window rather than the earlier truncated zero-diff.  VRAM still fails
(`0x57545d55` readback), so the replay is actively programming the MC but the
resulting GDDR3 bus data remains corrupt.

The explicit VRAM probe now characterizes the failure: BAR0 reads a stable
`555d5655aaa2a9aa`, but writing `0xa5a55a5a` reads back `0x55565d55` through
both BAR0 and MM_INDEX.  This is a responding but incorrectly trained memory
bus, not a floating MMIO register.  An opt-in `AMD_BOOT_VRAM_SET_VOLTAGE=1`
replay of ATOM SetVoltage for MVDDC/MVDDQ was added and tested; both tables
returned success but performed zero writes, so this VBIOS does not expose the
rail change through that revision.  The probe still fails.

The BIOS command header is `SetVoltage` crev=2, so the first replay used the
wrong rev1 index/mode encoding.  Corrected the opt-in path to require an
explicit `AMD_BOOT_VRAM_MVDD_MV` value and encode rev2's type/mode/u16-mV
layout; it refuses to guess a potentially damaging voltage.  No voltage was
applied by default.

An opt-in rev2 `SetVoltage` max-level query for MVDDC/MVDDQ returned unchanged
`ps0=0x6`/level 0 with zero observable table writes on warm boot.  A cold CHG
boot is therefore needed to capture/replay the board's actual MVDD GPIO or
voltage object.

With the dock still warm, `AMD_BOOT_VRAM_REPLAY_POWER=1` correctly refused to
touch the rails because `MISC0=0x30004222` (required cold/unpatched value is
`0x3000422a`).  The guard is working; a cold CHG boot remains necessary for
the captured replay.

After replug, fixed `--vram-probe` to run ASIC_Init whenever CHG is asserted;
previously it incorrectly skipped ATOM because reset `CLKF=50` looked nonzero.
The cold run now executes the captured power replay successfully
(`GPIOPinControl writes=6`, `SetVoltage writes=2`) and completes all memory
tables, but BAR0/MM_INDEX still fail.  Readback changed to
`0x17575d14` (BAR0 bytes `145d5717...`) rather than the prior `0x55565d55`,
proving the replay changes the memory-controller state but does not restore
correct GDDR3 data integrity.

The cold capture exposed exact pre-training payloads: `GPIOPinControl` ps0
`0x0101002f`, followed by rev2 `SetVoltage` ps0 `0x04630001`.  Added an
explicit `AMD_BOOT_VRAM_REPLAY_POWER=1` path to replay those payloads before
the existing memory-training sequence; it is intentionally not enabled by
default because it drives board-specific power GPIOs.

The unmodified default command (`python3 add.py`, stage `add`, EOP fence) was
also rerun after recovery.  It reaches the expected RV770 setup and consumes
the full CP stream, but times out before EOP with the canary intact
(`GRBM_STATUS=0xb2703028`).  Thus the shipped default remains an honest
GPU-only failure rather than silently falling back to CPU computation.

Added `AMD_BOOT_VRAM_MCLK_10KHZ` for controlled clock experiments.  Replayed
the memory tail at 500 MHz and 250 MHz; both produced the same bad readback
`0x17575d14` as 993 MHz.  The fault is not marginal timing at the requested
MCLK; it persists through a 4x clock reduction.

Added `AMD_BOOT_VRAM_SWEEP=1` and tested offsets 0, 4, 0x100, and 0x1000.
All writes fail, with offset 0x100 reading constant `0x55555555` and offsets
0/0x1000 aliasing `0x17575d14`.  This shows address/data-line corruption (not
just one bad location), consistent with a GDDR3 bus/power or training failure.

Implemented `AMD_BOOT_VRAM_FAULT_MAP=1` with 816 tests: zero/one/inverse and
walking-bit data at offsets 0, 4, 8, ..., 0x1000, recording MM_INDEX, BAR0,
and XOR.  It reports 815 bad rows; written data never changes the readback.
The zero-pattern address map is:
`0x0->57545d55`, `0x4->a8aba2aa`, `0x8->57555d55`, `0x10->57555d1d`,
`0x20->17555555`, `0x40->1757155c`, `0x80->57545d55`,
`0x100->55555555`, `0x200->55555554`, `0x400->51555555`,
`0x800->57545d55`, `0x1000->57545d55`.
This localizes the fault as a fixed/corrupt memory response with address
dependent aliases, rather than a software write-protocol issue.
