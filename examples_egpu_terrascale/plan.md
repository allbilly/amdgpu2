
Today 7:24 AM

Pasted text(8).txt
Document
# Terascale eGPU (examples_egpu_terrascale)

Bare-metal bring-up scaffold for **pre-GCN** AMD cards over TinyGPU (same transport
as examples_egpu/), targeting:

| Product | ASIC | Family (radeon_family.h) | TeraScale | PCI (examples) |
|---------|------|----------------------------|-----------|----------------|
| HD 5570 | Redwood | CHIP_REDWOOD | 2 (Evergreen) | 1002:68D9 (+ 68D8/68DA…) |
| HD 4850 | RV770 | CHIP_RV770 | 1 (R700) | 1002:9442 |

Linux driver: **drm/radeon** (not amdgpu). Sources under
ref/linux/drivers/gpu/drm/radeon/ — especially evergreen.c, r600.c,
rv770.c, evergreend.h, r600d.h.

User-space compute reference: Mesa **r600g** evergreen_compute.c
(OpenCL on Evergreen).

## Status

add.py is offline-first, and the HD 4850 CP smoke path has been exercised
with AGP-mapped host sysmem while local VRAM remains unusable:

bash
python3 examples_egpu_terrascale/add.py --selftest --chip=hd5570
python3 examples_egpu_terrascale/add.py --selftest --chip=hd4850
python3 examples_egpu_terrascale/add.py --dry-run --chip=hd5570
python3 examples_egpu_terrascale/add.py --list-chips
# when card + TinyGPU are up:
python3 examples_egpu_terrascale/add.py --probe --chip=hd5570


| Piece | HD 5570 (Evergreen) | HD 4850 (RV770) |
|-------|---------------------|-----------------|
| TinyGPU PCI/MMIO | ready | ready |
| Chip / PCI ID table | ready | ready |
| r600_cp_resume MMIO sequence (dry-run) | ready | ready |
| Evergreen LS compute IB (Mesa-shaped) | ready | n/a |
| Real r600 CF/ALU shader (llvm -march=r600) | **TODO** (stub blob) | **TODO** |
| ATOM / MC / CP boot on TinyGPU | **TODO** | **TODO** |
| RAT / global buffer bindings | **TODO** | n/a |

For RV770, --cp-mem-write-test does not use local VRAM: it places the CP ring,
writeback page, and MEM_WRITE result buffer in contiguous host sysmem behind
the AGP aperture. BAR0 is deliberately lazy and only mapped for --vram-probe
or an explicitly requested BAR0 probe. It is strictly a CP payload-write test,
not GPU arithmetic; the default command refuses to CPU-offload an add. This is
the supported diagnostic path while GDDR3 writes return a floating-bus value.

## Evergreen compute path (HD 5570)

Mirrors Mesa evergreen_emit_cs_shader + evergreen_emit_dispatch:

1. PACKET3_SET_CONTEXT_REG → SQ_PGM_START_LS / SQ_PGM_RESOURCES_LS (va >> 8)
2. PACKET3_SET_CONFIG_REG → VGT_COMPUTE_START_*, VGT_COMPUTE_THREAD_GROUP_SIZE
3. PACKET3_SET_CONTEXT_REG → SPI_COMPUTE_NUM_THREAD_{X,Y,Z}, SQ_LDS_ALLOC
4. PACKET3_DISPATCH_DIRECT with **compute bit** (Mesa PKT3C) and initiator 1

Registers: ref/linux/.../evergreend.h (VGT_COMPUTE_*, SQ_PGM_START_LS, …).
CP ring bring-up: r600_cp_resume / r600_cp_start in r600.c.

ISA is **r600 CF+ALU** (VLIW), not GCN VOP2. Do not reuse examples_egpu GCN
shaders. Next step: assemble with llvm-mc/llc -march=r600 -mcpu=redwood.

## RV770 path (HD 4850)

Shares R600 CP (r600_cp_resume) but **no Evergreen LS compute**. A genuine
RV770 pixel shader source now lives in rv770_add.ll; it compiles to four R600
hardware ADD ALU instructions plus a color export. Its matching vertex shader
is rv770_vs.ll, which exports clip position and the two vec4 operands to the
pixel stage:

bash
python3 examples_egpu_terrascale/add.py --compile-rv770-add
AMD_BOOT_ATOM=0 python3 examples_egpu_terrascale/add.py --gpu-add-preflight


The compiler inspection does not touch hardware. The preflight allocates the
48-byte VS, 64-byte PS, three-vertex input buffer, and FP32 output target in
contiguous AGP sysmem, but deliberately issues no graphics packets. The
remaining work is to bind that state through the RV770 graphics draw pipeline
and read the GPU-produced result. Until that exists, default add.py refuses
to replace GPU arithmetic with a CPU calculation.

## vs Polaris (examples_egpu)

| | Polaris RX570 | Terascale HD 5xxx/4xxx |
|--|---------------|-------------------------|
| Driver ref | amdgpu / gfx8 | radeon / r600+evergreen |
| Compute | MEC + COMPUTE_* SH regs | GFX CP + LS (SQ_PGM_START_LS) |
| ISA | GCN3 | r600 VLIW |
| Ring | KCQ/MQD (VI) | classic CP_RB_* |

VRAM/AGP lessons from Polaris still apply on eGPU: prefer host-visible buffers
until BAR0/HDP writeback is proven on each card.

## Reference repos / docs (websearch)

Ranked for this bring-up (HD 5570 Evergreen compute + HD 4850 R700 CP):

| Rank | Repo / doc | Why |
|-----:|------------|-----|
| 1 | [mesa/mesa](https://gitlab.freedesktop.org/mesa/mesa) src/gallium/drivers/r600/evergreen_compute.c | LS compute IB: SQ_PGM_START_LS, DISPATCH_DIRECT, RAT/LDS |
| 2 | ref/linux → [torvalds/linux](https://github.com/torvalds/linux) drivers/gpu/drm/radeon/ | evergreen.c / r600.c / rv770.c, evergreend.h, r600_cs.c |
| 3 | [llvm/llvm-project](https://github.com/llvm/llvm-project) AMDGPU r600 | -march=r600 -mcpu=redwood / rv770 shader codegen |
| 4 | [libclc](https://github.com/libclc/libclc) + Mesa **Clover** / **Rusticl** (RUSTICL_ENABLE=r600) | OpenCL → r600 binary path (historical GalliumCompute) |
| 5 | [CLRX/CLRX-mirror](https://github.com/CLRX/CLRX-mirror) | GalliumCompute binary / asm notes (more GCN-focused; still useful) |
| 6 | [X.Org AMD docs](https://www.x.org/docs/AMD/old/) | [r600 ISA](https://www.x.org/docs/AMD/old/r600isa.pdf), [Evergreen accel](https://www.x.org/docs/AMD/old/evergreen_cayman_programming_guide.pdf) |
| 7 | TechPowerUp ISA PDFs | [R700 ISA](https://www.techpowerup.com/gpu-specs/docs/ati-r700-isa.pdf), [Evergreen ISA](https://www.techpowerup.com/gpu-specs/docs/ati-evergreen-isa.pdf) |
| 8 | [RadeonFeature](https://www.x.org/wiki/RadeonFeature/) / [GalliumCompute](https://wiki.freedesktop.org/dri/GalliumCompute/) | Family decoder ring + OpenCL stack overview |

**Not useful here:** amdgpu/ROCm (GCN+), TrustOS neural.rs (RDNA), Polaris examples_egpu GCN shaders.

DeepWiki: ask mesa/mesa, torvalds/linux, llvm/llvm-project (listed in root AGENTS.md).


# HD 4850 eGPU Bring-Up Progress (TinyGPU / M1 Mac)

**Last updated:** 2026-07-11

## Current blocker

**Local GDDR3 VRAM does not retain writes** (stable float bus on FB@0+BIF).  
**AGP + CP --test PASS.**

Linux **radeon does not fix this in driver code** — RV770 boot/resume only runs atom_asic_init (VBIOS), then rv770_mc_program (apertures / BIF_FB_EN / HDP). No GDDR3 trainer, no mc.bin.

---

## Linux radeon (local ref/linux/.../radeon/)

| Step | What it does for VRAM |
|------|------------------------|
| atom_asic_init | FWI def SCLK/MCLK → ASIC_Init only |
| rv770_mc_init | Read CONFIG_MEMSIZE / channels — assumes DRAM live |
| rv770_mc_program | HDP clear, rv515_mc_stop (BIF=0→blackout), apertures, mc_resume (clear blackout→BIF=3) |
| DPM later | May SetVoltage / program MPLL regs — **not** first-time GDDR3 train |

SetMemoryClock flags in atombios.h: FIRST_TIME_CHANGE_CLOCK=0x08000000, SKIP_SW_PROGRAM_PLL=0x10000000. **Driver never sets these**; only VBIOS nested calls do.

Honest take: copying mc_program cannot fix float-bus if BIF/blackout already sane.

---

## Best software strategy so far

### AMD_ATOM_REPAIR_AFTER_MPLLINIT=1 (now **default ON**)

1. Let MemoryPLLInit write **CLKF=0** (VBIOS power-up window)
2. **Immediately repair MPLL → CLKF=73** before nested DLL/Training/DeviceInit
3. Rest of SetMemoryClock continues with live clock

Results (synth + this hook):

- MISC0=0x3000422a, AGP **PASS** (unlike PATCH_MPLL)
- IO_DEBUG: **57** pairs before repair (CLKF=0), **159** after (CLKF=73)
- VRAM still **float** (0x5555555d / 0x5d555555) — not sticky

### Also tried

| Experiment | Result |
|------------|--------|
| PATCH_MPLL (replace CLKF=0 writes) | Breaks MISC0/AGP; BIF hang |
| Post-hoc-only repair (old default) | AGP OK; all train at CLKF=0 |
| Nested-PS finish_memory | Safe; no stick |
| SetMemoryClock(SKIP\|FIRST\|mclk)=0x180183e4 after good post | No CLKF=0 writes; MISC0 OK; no stick |
| mc_program-style FB@0+BIF | Decode works (float visible); writes don’t retain |

---

## Status

| Path | Status |
|------|--------|
| --atom (repair-after-MemoryPLLInit) + --cp-mem-write-test | **PASS** |
| VRAM stick (MM/BAR0) | **FAIL** (float) |
| --cp-mem-write-test | **PASS** — CP writes a supplied payload to AGP-mapped host sysmem; this is **not GPU add** |
| Default add.py | **REFUSES** — RV770 GPU ALU/shader path is not implemented; no CPU fallback |
| AMD_ATOM_PATCH_MPLL | **Do not use** |

add.py now maps BAR0 lazily: only --vram-probe (or an explicit
AMD_BOOT_PROBE_BAR0=1 probe) opens it. Normal boot leaves BIF_FB_EN=0, parks
the FB range above the AGP aperture, and puts the CP ring, writeback page, and
diagnostic payload output in contiguous host memory. Thus local VRAM is not a
prerequisite for the RV770 CP write smoke test.

## Diagnosis

This is not an add.py allocation or Linux aperture-programming bug. The
Linux RV770 sequence is atom_asic_init followed by rv770_mc_program: it
clears HDP, programs MC_VM_*, releases MC blackout, and enables
BIF_FB_EN. The probe reproduces that state yet reads the stable floating
pattern after a write. Those registers choose a route; they cannot make an
unpowered/untrained GDDR3 device retain data. The remaining VRAM investigation
is therefore ATOM power/training or board hardware (especially MVDD/GDDR3), not
CP or AGP setup.

## Real RV770 add status

rv770_add.ll now compiles with LLVM's -march=r600 -mcpu=rv770 backend to a
64-byte pixel shader containing exactly four hardware ADD instructions and a
real SQ_EXPORT_PIXEL color export. rv770_vs.ll compiles to a 48-byte vertex
shader with SQ_EXPORT_POS, SQ_EXPORT_PARAM[0], and
SQ_EXPORT_PARAM[1]; add.py relocates its CF-export GPRs to Mesa's R700
fetch ABI (GPR1..3, because GPR0 contains the vertex index).  A hand-built
80-byte R700 VFETCH shader reads the three 48-byte vertex attributes from
resource 160, and the PM4 draw contains only graphics packets—no
PKT3_MEM_WRITE result payload.

The draw has now been submitted on the attached 1002:9442 card.  CP consumes
the complete 108-dword graphics stream (CP_RPTR == CP_WPTR) and the card
remains reachable afterwards, but the AGP CB_COLOR0 target remains all zero.
Therefore **GPU add is still failing**, not silently falling back to CPU.  The
next debug target is the remaining Linux RV770 graphics initialization/context
state or the 3D color-write route to the AGP aperture.

AMD_BOOT_ATOM=0 python3 examples_egpu_terrascale/add.py --gpu-add-preflight
has passed on the attached card: CP/ring initialization succeeds and returns
AGP addresses for the VS, PS, vertex buffer, and FP32 color target. It makes no
graphics submission; this verifies the exact input/output allocation topology
for the next draw-stage implementation.

## Recipe

bash
rm -f $TMPDIR/amd_usb4.lock
python3 examples_egpu_terrascale/add.py --clock-probe   # prefer CHG=True
python3 examples_egpu_terrascale/add.py --atom           # REPAIR_AFTER_MPLLINIT default 1
AMD_BOOT_ATOM=0 python3 examples_egpu_terrascale/add.py --test
AMD_BOOT_ATOM=0 python3 examples_egpu_terrascale/add.py --vram-probe


Env: AMD_ATOM_REPAIR_AFTER_MPLLINIT=1 (default), AMD_ATOM_PATCH_MPLL=0.

## Next (VRAM likely beyond Linux driver)

1. Diff IO_DEBUG **content** (57 @ CLKF=0 vs 159 @ CLKF=73) vs desktop-posted dump
2. MMIO-replay nested GPIOPinControl write list (MVDD GPIO)
3. Hardware: eGPU **MVDD / GDDR3** power (float 0x55 pattern strongly suggests undriven DQ)

explain what next

Today 11:29 AM

Pasted text(9).txt
Document
# HD 4850 eGPU Bring-Up Progress (TinyGPU / M1 Mac)

**Last updated:** 2026-07-11

## Current blocker

**Local GDDR3 VRAM does not retain writes** (stable float bus on FB@0+BIF).  
**AGP + CP --test PASS.**

Linux **radeon does not fix this in driver code** — RV770 boot/resume only runs atom_asic_init (VBIOS), then rv770_mc_program (apertures / BIF_FB_EN / HDP). No GDDR3 trainer, no mc.bin.

---

## Linux radeon (local ref/linux/.../radeon/)

| Step | What it does for VRAM |
|------|------------------------|
| atom_asic_init | FWI def SCLK/MCLK → ASIC_Init only |
| rv770_mc_init | Read CONFIG_MEMSIZE / channels — assumes DRAM live |
| rv770_mc_program | HDP clear, rv515_mc_stop (BIF=0→blackout), apertures, mc_resume (clear blackout→BIF=3) |
| DPM later | May SetVoltage / program MPLL regs — **not** first-time GDDR3 train |

SetMemoryClock flags in atombios.h: FIRST_TIME_CHANGE_CLOCK=0x08000000, SKIP_SW_PROGRAM_PLL=0x10000000. **Driver never sets these**; only VBIOS nested calls do.

Honest take: copying mc_program cannot fix float-bus if BIF/blackout already sane.

---

## Best software strategy so far

### AMD_ATOM_REPAIR_AFTER_MPLLINIT=1 (now **default ON**)

1. Let MemoryPLLInit write **CLKF=0** (VBIOS power-up window)
2. **Immediately repair MPLL → CLKF=73** before nested DLL/Training/DeviceInit
3. Rest of SetMemoryClock continues with live clock

Results (synth + this hook):

- MISC0=0x3000422a, AGP **PASS** (unlike PATCH_MPLL)
- IO_DEBUG: **57** pairs before repair (CLKF=0), **159** after (CLKF=73)
- VRAM still **float** (0x5555555d / 0x5d555555) — not sticky

### Also tried

| Experiment | Result |
|------------|--------|
| PATCH_MPLL (replace CLKF=0 writes) | Breaks MISC0/AGP; BIF hang |
| Post-hoc-only repair (old default) | AGP OK; all train at CLKF=0 |
| Nested-PS finish_memory | Safe; no stick |
| SetMemoryClock(SKIP\|FIRST\|mclk)=0x180183e4 after good post | No CLKF=0 writes; MISC0 OK; no stick |
| mc_program-style FB@0+BIF | Decode works (float visible); writes don’t retain |

---

## Status

| Path | Status |
|------|--------|
| --atom (repair-after-MemoryPLLInit) + --cp-mem-write-test | **PASS** |
| VRAM stick (MM/BAR0) | **FAIL** (float) |
| --cp-mem-write-test | **PASS** — CP writes a supplied payload to AGP-mapped host sysmem; this is **not GPU add** |
| Default add.py | **REFUSES** — RV770 GPU ALU/shader path is not implemented; no CPU fallback |
| AMD_ATOM_PATCH_MPLL | **Do not use** |

add.py now maps BAR0 lazily: only --vram-probe (or an explicit
AMD_BOOT_PROBE_BAR0=1 probe) opens it. Normal boot leaves BIF_FB_EN=0, parks
the FB range above the AGP aperture, and puts the CP ring, writeback page, and
diagnostic payload output in contiguous host memory. Thus local VRAM is not a
prerequisite for the RV770 CP write smoke test.

## Diagnosis

This is not an add.py allocation or Linux aperture-programming bug. The
Linux RV770 sequence is atom_asic_init followed by rv770_mc_program: it
clears HDP, programs MC_VM_*, releases MC blackout, and enables
BIF_FB_EN. The probe reproduces that state yet reads the stable floating
pattern after a write. Those registers choose a route; they cannot make an
unpowered/untrained GDDR3 device retain data. The remaining VRAM investigation
is therefore ATOM power/training or board hardware (especially MVDD/GDDR3), not
CP or AGP setup.

## Real RV770 add status

rv770_add.ll now compiles with LLVM's -march=r600 -mcpu=rv770 backend to a
64-byte pixel shader containing exactly four hardware ADD instructions and a
real SQ_EXPORT_PIXEL color export. rv770_vs.ll compiles to a 48-byte vertex
shader with SQ_EXPORT_POS, SQ_EXPORT_PARAM[0], and
SQ_EXPORT_PARAM[1]; add.py relocates its CF-export GPRs to Mesa's R700
fetch ABI (GPR1..3, because GPR0 contains the vertex index).  A hand-built
80-byte R700 VFETCH shader reads the three 48-byte vertex attributes from
resource 160, and the PM4 draw contains only graphics packets—no
PKT3_MEM_WRITE result payload.

The draw has now been submitted on the attached 1002:9442 card.  CP consumes
the complete 108-dword graphics stream (CP_RPTR == CP_WPTR) and the card
remains reachable afterwards, but the AGP CB_COLOR0 target remains all zero.
Therefore **GPU add is still failing**, not silently falling back to CPU.  The
next debug target is the remaining Linux RV770 graphics initialization/context
state or the 3D color-write route to the AGP aperture.

AMD_BOOT_ATOM=0 python3 examples_egpu_terrascale/add.py --gpu-add-preflight
has passed on the attached card: CP/ring initialization succeeds and returns
AGP addresses for the VS, PS, vertex buffer, and FP32 color target. It makes no
graphics submission; this verifies the exact input/output allocation topology
for the next draw-stage implementation.

## Recipe

bash
rm -f $TMPDIR/amd_usb4.lock
python3 examples_egpu_terrascale/add.py --clock-probe   # prefer CHG=True
python3 examples_egpu_terrascale/add.py --atom           # REPAIR_AFTER_MPLLINIT default 1
AMD_BOOT_ATOM=0 python3 examples_egpu_terrascale/add.py --test
AMD_BOOT_ATOM=0 python3 examples_egpu_terrascale/add.py --vram-probe


## GPU-add debug infrastructure (new)

Following plan.md Phase 1-4, the graphics-add path now isolates the pipeline:

- **CB_COLOR_CONTROL = 0x00CC0000** (was 0 → CB disabled). Encoded by
  rv770_cb_color_control(rop3=0xCC, special_op=0); CB_COLOR0_INFO by
  rv770_color_info_rgba32_float() (RGBA32_FLOAT, no CMASK/FMASK).
- **Completion fence**: separate AGP page + SURFACE_SYNC + EVENT_WRITE_EOP
  (CACHE_FLUSH_AND_INV_TS, 32-bit seq write, no IRQ). run_add polls the fence,
  not the color target. Fallback --gpu-add-fence-mode=wait-memwrite uses
  WAIT_UNTIL + a CP MEM_WRITE **to the fence only**.
- **Canary**: color target filled with 0xA5 before each draw, so outcomes are
  canary-intact (no write) vs wrote-zero vs expected.
- **Stage ladder**: --gpu-add-stage={cp,constant,param0,add}.
  - cp — fence only; proves completion + CPU visibility.
  - constant — rv770_constant_ps.ll exports {0.25,-0.5,3.0,1.0}.
  - param0 — rv770_param0_ps.ll exports interpolated PARAM0.
  - add — rv770_add.ll exports PARAM0+PARAM1.
- **Validation**: validate_gpu_add_pm4 rejects any MEM_WRITE to the color
  target, requires exactly one completion fence, and (for graphics stages)
  requires the fence after the draw.
- **Diagnostics**: --gpu-add-dump-pm4 (offline decoder) and
  --gpu-add-dump-registers (GRBM/CB/SQ/DB snapshot). Full R700 graphics
  context defaults are opt-in via --gpu-add-full-gfx-init.

Hardware evidence: --gpu-add-stage=cp --gpu-add-fence-mode=wait-memwrite
passes, proving the separate fence page and CPU/AGP visibility.  A graphics
stage with the diagnostic raw fence reaches the post-draw fence, but the
0xA5 color canary remains unchanged and GRBM_STATUS reports active
SH/VGT/SPI/PA units.  The preferred EOP fence times out because graphics does
not reach EOP; this is now distinct from a missing color write.  No CP packet
writes the color allocation.  Default add.py remains intentionally failing
until a GPU-produced color result is verified.

An additional --gpu-add-stage=stream experiment relocates the four FP32 ADDs
into a VS and configures VGT_STRMOUT_BUFFER_0 to the AGP result page.  It also
leaves the canary unchanged.  Mesa's allocator shows a key constraint:
PIPE_USAGE_IMMUTABLE shader BOs are placed in RADEON_DOMAIN_VRAM, while
this direct path puts VS/PS/fetch code in AGP.  Persistent SH activity is
consistent with RV770 shader instruction fetch not accepting that AGP
placement; this is the leading no-VRAM blocker under investigation.

The streamout route was also exercised on hardware.  Its VS contains four
real FP32 ADDs and programs VGT_STRMOUT_BUFFER_0 at the AGP result page, but
the result canary remains intact.  Mesa's R600 source confirms immutable
shader BOs are allocated in VRAM, whereas GTT/AGP is used for ordinary buffer
resources.  This supports the current diagnosis that CP/AGP DMA works while
the RV770 shader instruction path is not usable from this AGP-only setup; no
CPU or CP arithmetic fallback has been enabled.

Offline python3 examples_egpu_terrascale/add.py --selftest passes (R600 llc
present: constant/param0/add PS all compile, stage ladder + validator OK).

Env: AMD_ATOM_REPAIR_AFTER_MPLLINIT=1 (default), AMD_ATOM_PATCH_MPLL=0.

## Next (VRAM likely beyond Linux driver)

1. Diff IO_DEBUG **content** (57 @ CLKF=0 vs 159 @ CLKF=73) vs desktop-posted dump
2. MMIO-replay nested GPIOPinControl write list (MVDD GPIO)
3. Hardware: eGPU **MVDD / GDDR3** power (float 0x55 pattern strongly suggests undriven DQ)# HD 4850 eGPU Bring-Up Progress (TinyGPU / M1 Mac)

**Last updated:** 2026-07-11

## Current blocker

**Local GDDR3 VRAM does not retain writes** (stable float bus on FB@0+BIF).  
**AGP + CP --test PASS.**

Linux **radeon does not fix this in driver code** — RV770 boot/resume only runs atom_asic_init (VBIOS), then rv770_mc_program (apertures / BIF_FB_EN / HDP). No GDDR3 trainer, no mc.bin.

---

## Linux radeon (local ref/linux/.../radeon/)

| Step | What it does for VRAM |
|------|------------------------|
| atom_asic_init | FWI def SCLK/MCLK → ASIC_Init only |
| rv770_mc_init | Read CONFIG_MEMSIZE / channels — assumes DRAM live |
| rv770_mc_program | HDP clear, rv515_mc_stop (BIF=0→blackout), apertures, mc_resume (clear blackout→BIF=3) |
| DPM later | May SetVoltage / program MPLL regs — **not** first-time GDDR3 train |

SetMemoryClock flags in atombios.h: FIRST_TIME_CHANGE_CLOCK=0x08000000, SKIP_SW_PROGRAM_PLL=0x10000000. **Driver never sets these**; only VBIOS nested calls do.

Honest take: copying mc_program cannot fix float-bus if BIF/blackout already sane.

---

## Best software strategy so far

### AMD_ATOM_REPAIR_AFTER_MPLLINIT=1 (now **default ON**)

1. Let MemoryPLLInit write **CLKF=0** (VBIOS power-up window)
2. **Immediately repair MPLL → CLKF=73** before nested DLL/Training/DeviceInit
3. Rest of SetMemoryClock continues with live clock

Results (synth + this hook):

- MISC0=0x3000422a, AGP **PASS** (unlike PATCH_MPLL)
- IO_DEBUG: **57** pairs before repair (CLKF=0), **159** after (CLKF=73)
- VRAM still **float** (0x5555555d / 0x5d555555) — not sticky

### Also tried

| Experiment | Result |
|------------|--------|
| PATCH_MPLL (replace CLKF=0 writes) | Breaks MISC0/AGP; BIF hang |
| Post-hoc-only repair (old default) | AGP OK; all train at CLKF=0 |
| Nested-PS finish_memory | Safe; no stick |
| SetMemoryClock(SKIP\|FIRST\|mclk)=0x180183e4 after good post | No CLKF=0 writes; MISC0 OK; no stick |
| mc_program-style FB@0+BIF | Decode works (float visible); writes don’t retain |

---

## Status

| Path | Status |
|------|--------|
| --atom (repair-after-MemoryPLLInit) + --cp-mem-write-test | **PASS** |
| VRAM stick (MM/BAR0) | **FAIL** (float) |
| --cp-mem-write-test | **PASS** — CP writes a supplied payload to AGP-mapped host sysmem; this is **not GPU add** |
| Default add.py | **REFUSES** — RV770 GPU ALU/shader path is not implemented; no CPU fallback |
| AMD_ATOM_PATCH_MPLL | **Do not use** |

add.py now maps BAR0 lazily: only --vram-probe (or an explicit
AMD_BOOT_PROBE_BAR0=1 probe) opens it. Normal boot leaves BIF_FB_EN=0, parks
the FB range above the AGP aperture, and puts the CP ring, writeback page, and
diagnostic payload output in contiguous host memory. Thus local VRAM is not a
prerequisite for the RV770 CP write smoke test.

## Diagnosis

This is not an add.py allocation or Linux aperture-programming bug. The
Linux RV770 sequence is atom_asic_init followed by rv770_mc_program: it
clears HDP, programs MC_VM_*, releases MC blackout, and enables
BIF_FB_EN. The probe reproduces that state yet reads the stable floating
pattern after a write. Those registers choose a route; they cannot make an
unpowered/untrained GDDR3 device retain data. The remaining VRAM investigation
is therefore ATOM power/training or board hardware (especially MVDD/GDDR3), not
CP or AGP setup.

## Real RV770 add status

rv770_add.ll now compiles with LLVM's -march=r600 -mcpu=rv770 backend to a
64-byte pixel shader containing exactly four hardware ADD instructions and a
real SQ_EXPORT_PIXEL color export. rv770_vs.ll compiles to a 48-byte vertex
shader with SQ_EXPORT_POS, SQ_EXPORT_PARAM[0], and
SQ_EXPORT_PARAM[1]; add.py relocates its CF-export GPRs to Mesa's R700
fetch ABI (GPR1..3, because GPR0 contains the vertex index).  A hand-built
80-byte R700 VFETCH shader reads the three 48-byte vertex attributes from
resource 160, and the PM4 draw contains only graphics packets—no
PKT3_MEM_WRITE result payload.

The draw has now been submitted on the attached 1002:9442 card.  CP consumes
the complete 108-dword graphics stream (CP_RPTR == CP_WPTR) and the card
remains reachable afterwards, but the AGP CB_COLOR0 target remains all zero.
Therefore **GPU add is still failing**, not silently falling back to CPU.  The
next debug target is the remaining Linux RV770 graphics initialization/context
state or the 3D color-write route to the AGP aperture.

AMD_BOOT_ATOM=0 python3 examples_egpu_terrascale/add.py --gpu-add-preflight
has passed on the attached card: CP/ring initialization succeeds and returns
AGP addresses for the VS, PS, vertex buffer, and FP32 color target. It makes no
graphics submission; this verifies the exact input/output allocation topology
for the next draw-stage implementation.

## Recipe

bash
rm -f $TMPDIR/amd_usb4.lock
python3 examples_egpu_terrascale/add.py --clock-probe   # prefer CHG=True
python3 examples_egpu_terrascale/add.py --atom           # REPAIR_AFTER_MPLLINIT default 1
AMD_BOOT_ATOM=0 python3 examples_egpu_terrascale/add.py --test
AMD_BOOT_ATOM=0 python3 examples_egpu_terrascale/add.py --vram-probe


## GPU-add debug infrastructure (new)

Following plan.md Phase 1-4, the graphics-add path now isolates the pipeline:

- **CB_COLOR_CONTROL = 0x00CC0000** (was 0 → CB disabled). Encoded by
  rv770_cb_color_control(rop3=0xCC, special_op=0); CB_COLOR0_INFO by
  rv770_color_info_rgba32_float() (RGBA32_FLOAT, no CMASK/FMASK).
- **Completion fence**: separate AGP page + SURFACE_SYNC + EVENT_WRITE_EOP
  (CACHE_FLUSH_AND_INV_TS, 32-bit seq write, no IRQ). run_add polls the fence,
  not the color target. Fallback --gpu-add-fence-mode=wait-memwrite uses
  WAIT_UNTIL + a CP MEM_WRITE **to the fence only**.
- **Canary**: color target filled with 0xA5 before each draw, so outcomes are
  canary-intact (no write) vs wrote-zero vs expected.
- **Stage ladder**: --gpu-add-stage={cp,constant,param0,add}.
  - cp — fence only; proves completion + CPU visibility.
  - constant — rv770_constant_ps.ll exports {0.25,-0.5,3.0,1.0}.
  - param0 — rv770_param0_ps.ll exports interpolated PARAM0.
  - add — rv770_add.ll exports PARAM0+PARAM1.
- **Validation**: validate_gpu_add_pm4 rejects any MEM_WRITE to the color
  target, requires exactly one completion fence, and (for graphics stages)
  requires the fence after the draw.
- **Diagnostics**: --gpu-add-dump-pm4 (offline decoder) and
  --gpu-add-dump-registers (GRBM/CB/SQ/DB snapshot). Full R700 graphics
  context defaults are opt-in via --gpu-add-full-gfx-init.

Hardware evidence: --gpu-add-stage=cp --gpu-add-fence-mode=wait-memwrite
passes, proving the separate fence page and CPU/AGP visibility.  A graphics
stage with the diagnostic raw fence reaches the post-draw fence, but the
0xA5 color canary remains unchanged and GRBM_STATUS reports active
SH/VGT/SPI/PA units.  The preferred EOP fence times out because graphics does
not reach EOP; this is now distinct from a missing color write.  No CP packet
writes the color allocation.  Default add.py remains intentionally failing
until a GPU-produced color result is verified.

An additional --gpu-add-stage=stream experiment relocates the four FP32 ADDs
into a VS and configures VGT_STRMOUT_BUFFER_0 to the AGP result page.  It also
leaves the canary unchanged.  Mesa's allocator shows a key constraint:
PIPE_USAGE_IMMUTABLE shader BOs are placed in RADEON_DOMAIN_VRAM, while
this direct path puts VS/PS/fetch code in AGP.  Persistent SH activity is
consistent with RV770 shader instruction fetch not accepting that AGP
placement; this is the leading no-VRAM blocker under investigation.

The streamout route was also exercised on hardware.  Its VS contains four
real FP32 ADDs and programs VGT_STRMOUT_BUFFER_0 at the AGP result page, but
the result canary remains intact.  Mesa's R600 source confirms immutable
shader BOs are allocated in VRAM, whereas GTT/AGP is used for ordinary buffer
resources.  This supports the current diagnosis that CP/AGP DMA works while
the RV770 shader instruction path is not usable from this AGP-only setup; no
CPU or CP arithmetic fallback has been enabled.

Offline python3 examples_egpu_terrascale/add.py --selftest passes (R600 llc
present: constant/param0/add PS all compile, stage ladder + validator OK).

Env: AMD_ATOM_REPAIR_AFTER_MPLLINIT=1 (default), AMD_ATOM_PATCH_MPLL=0.

## Next (VRAM likely beyond Linux driver)

1. Diff IO_DEBUG **content** (57 @ CLKF=0 vs 159 @ CLKF=73) vs desktop-posted dump
2. MMIO-replay nested GPIOPinControl write list (MVDD GPIO)
3. Hardware: eGPU **MVDD / GDDR3** power (float 0x55 pattern strongly suggests undriven DQ)


review the plan

give me a plan file

rv770_add_repair_plan.md

It incorporates the current add.py implementation and the revised AGP shader-fetch diagnosis. 

 



Pasted markdown(1).md
File
revise the plan to make add.py works

Worked for 1m 6s

Analyzed
rv770_add_repair_plan.md

The revision removes already-exhausted graphics-state experiments and establishes two decisive paths:

prove shader execution from GTT on a working Linux RV770 setup; or

restore a minimal stable VRAM region and place shader code, inputs, and output there while retaining the working AGP CP ring and fence.

Based on the latest empty-shader, streamout, high-address AGP, reset, and VRAM-training results, the minimal-VRAM topology is now identified as the most likely shortest route to a working add.py. 

 



Pasted markdown(2).md
File
The VRAM issue is now conclusively isolated to hardware-level GDDR3 address/data corruption:

  - Cold ATOM/MVDD replay executes successfully.
  - Memory training completes.
  - 250–993 MHz clocks all fail identically.
  - Multiple VRAM offsets show aliases and constant patterns.

  The remaining blocker is physical GDDR3/MVDD signal or board integrity; software initialization and PM4 changes cannot make add.py produce a valid GPU result while VRAM
  is corrupted.

revise the plan again

rv770_add_repair_plan.md

The revision now treats the software path as complete enough to resume later, but blocks graphics submission until execution memory is qualified. It removes further PM4/shader tuning as the main path and focuses on:

board-level GDDR3/MVDD diagnosis or card replacement;

a formal VRAM retention qualification suite;

an all-VRAM first working topology after repair;

preservation of the working AGP CP ring and fence;

explicit failure messages instead of suggesting software can overcome the current corruption. 

 


Library
/
rv770_add_repair_plan.md


# RV770 HD 4850 TinyGPU GPU-Add Plan — Hardware-Gated Revision

**Target:** `examples_egpu_terrascale/add.py`  
**Hardware:** Radeon HD 4850 (`1002:9442`, RV770/R700) over TinyGPU on M1 Mac  
**Revision date:** 2026-07-11

## Executive conclusion

The remaining blocker is no longer a software-initialization or PM4 problem.

The card’s local GDDR3 path has been isolated to physical address/data corruption:

- cold ATOM and board-specific MVDD/GPIO replay execute successfully;
- the complete memory-training sequence runs;
- memory clocks from 250 MHz through 993 MHz fail in the same way;
- BAR0 and MM_INDEX agree on corrupted readback;
- multiple offsets show aliases and constant patterns;
- changing software timing, PM4, shader state, clocks, and initialization order does not restore retention.

Therefore:

> `add.py` cannot produce a valid RV770 shader result on this physical HD 4850 until the GDDR3/MVDD/address-data path is repaired or the card is replaced.

This plan no longer treats further shader, PM4, ATOM, or clock experimentation as the main path to success.

---

# 1. Project state to preserve

The current software work is valuable and should be frozen as the known-good bring-up baseline.

## 1.1 Proven working

- PCI configuration and MMIO access
- TinyGPU transport
- AGP aperture programming
- CP firmware upload
- CP ring bring-up
- CP scratch test
- CP writes to AGP-backed host memory
- separate AGP fence page
- PM4 validation
- offline RV770 shader compilation
- real four-ADD RV770 shader binary
- constant, PARAM0, ADD, streamout, and empty-shader diagnostics
- canary-based output classification
- no CPU result fallback
- no CP result write to the output allocation

## 1.2 Proven failing on this board

- VRAM write retention
- BAR0 readback
- MM_INDEX readback
- multiple VRAM offsets
- 250, 500, and 993 MHz memory clocks
- cold ATOM memory initialization
- board-specific power replay
- full training sequence
- graphics draw completion
- streamout result production
- EOP retirement after graphics work

## 1.3 Preserve the baseline commit

Create a checkpoint containing all current diagnostics and guards.

Suggested commit:

```bash
git add examples_egpu_terrascale
git commit -m "rv770: freeze software bring-up at confirmed hardware VRAM fault"
```

The checkpoint should keep:

```text
--selftest
--test
--cp-mem-write-test
--gpu-add-stage=cp
--gpu-add-stage=constant
--gpu-add-stage=param0
--gpu-add-stage=add
--gpu-add-stage=stream
--vram-probe
--gpu-add-dump-pm4
--gpu-add-dump-registers
```

---

# 2. Change the default user-facing behavior

The default command should no longer imply that more PM4 work may make the current board succeed.

## 2.1 Add a hardware-integrity gate

Before submitting any RV770 graphics stage, require a VRAM integrity qualification unless a future known-good GTT shader path is explicitly selected.

Implement:

```python
def require_rv770_execution_memory(self) -> None:
    if self.vram_qualified:
        return

    if self.gtt_shader_execution_qualified:
        return

    raise RuntimeError(
        "RV770 GPU add is unavailable on this board: local GDDR3 fails "
        "address/data integrity tests, and shader execution from AGP/GTT has "
        "not been qualified. Repair or replace the card, or use a known-good "
        "RV770/Redwood device."
    )
```

## 2.2 Add persistent qualification state

Track:

```python
self.vram_qualified = False
self.gtt_shader_execution_qualified = False
```

These values must not be inferred from:

- `CONFIG_MEMSIZE`;
- successful ATOM return;
- successful CP test;
- a nonzero memory clock;
- BAR0 responding;
- one matching read.

They should be set only after explicit tests pass.

## 2.3 Update status text

Replace language such as:

```text
next debug target is graphics initialization or CB routing
```

with:

```text
graphics software bring-up is blocked by confirmed board-level GDDR3
address/data corruption. Further PM4 changes are not expected to help.
```

Replace:

```text
default add.py refuses because the RV770 shader path is not implemented
```

with:

```text
default add.py contains a genuine RV770 shader path but fails closed because
this board has no qualified execution memory.
```

---

# 3. Stop conditions for software experimentation

The following work should stop on this card until hardware integrity changes.

## 3.1 Do not continue changing

- shader binaries;
- VS/PS linkage;
- empty shader variants;
- VFETCH variants;
- CB state;
- streamout state;
- viewport or scissor values;
- CP parser state;
- EOP timing;
- raw-fence delays;
- reset pulse duration;
- AGP base address;
- memory clock frequency;
- ATOM table ordering;
- speculative MVDD values;
- unverified GPIO payloads;
- `BIF_FB_EN` permutations.

These experiments have either already been exhausted or cannot overcome physical address/data corruption.

## 3.2 Do not weaken safety gates

Keep:

- explicit opt-in for board power replay;
- VBIOS hash guard;
- PCI/subsystem-ID guard;
- cold-state guard;
- no guessed voltage values;
- no automatic BAR0 probing;
- no CPU arithmetic fallback;
- no CP result upload.

## 3.3 Do not report partial CP success as GPU add

The following remain diagnostics only:

```text
CP ring test
CP MEM_WRITE
raw post-draw breadcrumb
AGP fence write
```

None proves shader execution or arithmetic.

---

# 4. Hardware repair track

The next meaningful work is board-level diagnosis and repair.

## 4.1 Primary fault classes

Rank the physical causes as:

1. missing or unstable MVDD/MVDDQ;
2. GDDR3 address-line fault;
3. GDDR3 data-line fault;
4. damaged or cracked BGA joint;
5. damaged memory chip;
6. damaged termination or series resistor;
7. damaged memory-controller package connection;
8. board corrosion, contamination, or trace damage;
9. incorrect strap or board-identification circuitry;
10. power-sequencing failure.

The observed aliases and constant patterns are especially consistent with:

- stuck address bits;
- stuck or shorted data bits;
- missing memory rail;
- incomplete chip select;
- BGA connectivity failure.

## 4.2 Establish a known-good reference

Use the same card on a desktop motherboard if possible.

Record:

- whether it displays video;
- whether a desktop BIOS POSTs it;
- whether Linux reports stable VRAM;
- whether a VRAM stress test passes;
- MVDD/MVDDQ voltage;
- memory clock waveform;
- GDDR3 reset/enable timing.

Interpretation:

```text
fails identically in desktop
    → board/card hardware fault is confirmed independently of TinyGPU

works in desktop
    → eGPU power delivery, reset, reference clock, or board sequencing differs
```

## 4.3 Measure memory power rails

Using suitable equipment, measure:

- MVDD;
- MVDDQ if separate;
- VDDCI or related I/O rail if present;
- rail rise time;
- rail sequencing;
- ripple under load;
- voltage before, during, and after ATOM training.

Capture:

```text
cold power-on
after GPIO replay
after SetVoltage
during MemoryPLLInit
after training
during VRAM write test
```

Compare against:

- board schematic if available;
- memory-chip datasheet;
- known-good desktop operation;
- another identical card.

## 4.4 Inspect the board

Inspect:

- all GDDR3 chips;
- memory power MOSFETs and inductors;
- decoupling capacitors;
- resistor networks;
- series resistors on address/data lines;
- strap resistors;
- corrosion;
- cracked solder joints;
- burned or mechanically damaged areas;
- missing components.

Use magnification around:

- GPU package;
- memory chips;
- memory power section;
- PCIe power input;
- eGPU adapter power path.

## 4.5 Thermal and pressure diagnostics

As non-destructive diagnostics only:

- compare cold versus warmed behavior;
- apply controlled cooling to individual memory chips;
- apply controlled heating within safe limits;
- apply light mechanical pressure only as a diagnostic;
- observe whether corruption patterns change.

A pattern change with temperature or pressure strongly suggests BGA or package connectivity.

Do not treat uncontrolled reflow as a repair method.

## 4.6 Isolate bad channels or chips

Use the corruption map to infer channel behavior.

Add a diagnostic output that records, for each offset:

```text
written value
BAR0 value
MM_INDEX value
XOR difference
bits always zero
bits always one
aliased offsets
```

Sweep at least:

```text
0x000000
0x000004
0x000100
0x001000
0x010000
0x100000
```

Use patterns:

```text
0x00000000
0xFFFFFFFF
0xAAAAAAAA
0x55555555
0xA5A55A5A
0x5A5AA5A5
walking 1
walking 0
address-derived value
```

The goal is now hardware localization, not software qualification.

---

# 5. VRAM qualification after repair

After any board repair or card replacement, do not jump directly to `add`.

## 5.1 Required retention suite

A minimum 4 KiB region must pass:

- all-zero;
- all-one;
- alternating-bit patterns;
- walking-one;
- walking-zero;
- address-dependent pattern;
- inverse address-dependent pattern;
- repeated writes;
- delayed reads.

For each pattern:

1. write through BAR0;
2. read through BAR0;
3. read through MM_INDEX;
4. wait 100 ms;
5. read again through both paths;
6. repeat 100 times.

Acceptance:

```text
zero mismatches
zero aliasing
zero stuck bits
identical BAR0 and MM_INDEX data
```

## 5.2 Expand the qualification region

After 4 KiB passes, test:

```text
64 KiB
1 MiB
16 MiB
full accessible VRAM
```

The first GPU add needs only a small stable region, but larger qualification is needed before general use.

## 5.3 Set `vram_qualified`

Only after the minimum retention suite passes:

```python
self.vram_qualified = True
```

Log:

```text
qualified size
tested clock
pattern count
iteration count
BAR0/MM_INDEX agreement
```

---

# 6. Post-repair `add.py` completion path

Once VRAM is qualified, use a simple all-VRAM topology first.

## 6.1 Keep CP infrastructure in AGP

Continue using AGP for:

- CP ring;
- CP writeback page;
- EOP fence;
- diagnostic breadcrumbs.

Those paths already work.

## 6.2 Place all graphics resources in VRAM

For the first successful draw:

```text
fetch shader  → VRAM
vertex shader → VRAM
pixel shader  → VRAM
vertex data   → VRAM
color target  → VRAM
streamout     → VRAM
```

Do not mix AGP and VRAM graphics resources initially.

## 6.3 Add a domain-aware allocator

Implement:

```python
class GpuDomain(enum.Enum):
    AGP = "agp"
    VRAM = "vram"
```

and:

```python
@dataclass
class GpuAllocation:
    gpu_address: int
    size: int
    alignment: int
    domain: GpuDomain
    name: str
```

Provide:

```python
alloc_agp(...)
alloc_vram(...)
write_allocation(...)
read_allocation(...)
verify_allocation(...)
```

## 6.4 Recommended first layout

Use at least one stable 4 KiB VRAM page:

```text
+0x000 fetch shader
+0x100 vertex shader
+0x200 pixel shader
+0x300 vertex data
+0x400 color target
+0x500 streamout target
+0x600 guard pattern
```

Every shader start must be 256-byte aligned.

## 6.5 Verify every upload

Before submission:

- write shader/data;
- read back through BAR0;
- read back through MM_INDEX;
- compare byte-for-byte;
- abort on mismatch.

## 6.6 Run the stage ladder

Run in this order:

```text
constant-minimal
constant
param0
param1
add
streamout
```

Acceptance:

```text
constant-minimal → expected constant
param0           → A
param1           → B
add              → A+B
streamout        → VS-produced A+B
```

## 6.7 Preserve result integrity rules

The PM4 validator must still reject:

- CP writes to the result;
- literal expected-result values in result-write packets;
- CPU upload of the expected result.

Reading GPU-produced VRAM through BAR0 or MM_INDEX is valid.

---

# 7. Optional GTT shader research track

This is now optional research, not the main route to fixing this card.

Use a known-good RV770 under Linux and force shader BOs into GTT.

Interpretation:

```text
GTT shaders work
    → add.py may eventually support AGP-only shader execution

GTT shaders fail
    → RV770 shader code should remain VRAM-only
```

Do not use results from the corrupted HD 4850 to decide this question.

The main implementation should remain:

```text
VRAM-qualified card required for RV770 graphics
```

unless a separate known-good experiment proves otherwise.

---

# 8. Alternative hardware paths

If repairing this card is not practical, the project can still succeed by changing hardware.

## 8.1 Replace the HD 4850

Use another known-good RV770/R700 board.

Required precheck:

```text
desktop POST works
VRAM stress test passes
same PCI family
VBIOS dump available
```

Then reuse the existing `add.py` graphics path.

## 8.2 Use the HD 5570 path

The Redwood HD 5570 has a genuine Evergreen LS compute path.

Advantages:

- no graphics rasterization required;
- direct compute dispatch;
- easier scalar/vector-add topology;
- avoids RV770’s no-compute limitation.

Remaining work:

- real Redwood r600 compute shader;
- RAT/global buffer binding;
- compute resource setup;
- EOP/readback.

This may be a shorter software project than repairing a physically damaged HD 4850.

## 8.3 Keep CP-only HD 4850 diagnostics

Even with failed VRAM, the HD 4850 remains useful for:

- PCI/MMIO tests;
- CP firmware loading;
- AGP DMA;
- PM4 parser experiments;
- ATOM interpreter validation.

It should not be presented as capable of shader arithmetic until execution memory is qualified.

---

# 9. Revised implementation order

1. Freeze the current software baseline.
2. Add the execution-memory qualification gate.
3. Update all status and error messages.
4. Stop further PM4/shader experimentation on this board.
5. Confirm the fault on a desktop host if possible.
6. Measure MVDD/MVDDQ and memory-clock behavior.
7. Inspect and repair board-level faults.
8. Run the 4 KiB VRAM retention qualification.
9. Add the AGP/VRAM domain allocator.
10. Place all graphics resources in VRAM.
11. Make `constant-minimal` pass.
12. Make `param0` and `param1` pass.
13. Make `add` pass.
14. Run 1,000 repeated additions.
15. Reintroduce AGP input/output only as optional optimization.

---

# 10. Final test sequence after repair

## VRAM qualification

```bash
python3 add.py --vram-probe --vram-pattern-suite
```

Required:

```text
4 KiB minimum
100 iterations
zero errors
BAR0 == MM_INDEX
```

## CP baseline

```bash
AMD_BOOT_ATOM=0 python3 add.py --test
```

## Constant draw

```bash
AMD_BOOT_ATOM=0 python3 add.py \
  --gpu-add-stage=constant \
  --gpu-add-code-domain=vram \
  --gpu-add-data-domain=vram \
  --gpu-add-output-domain=vram
```

## Parameter tests

```bash
AMD_BOOT_ATOM=0 python3 add.py \
  --gpu-add-stage=param0 \
  --gpu-add-code-domain=vram \
  --gpu-add-data-domain=vram \
  --gpu-add-output-domain=vram

AMD_BOOT_ATOM=0 python3 add.py \
  --gpu-add-stage=param1 \
  --gpu-add-code-domain=vram \
  --gpu-add-data-domain=vram \
  --gpu-add-output-domain=vram
```

## Final add

```bash
AMD_BOOT_ATOM=0 python3 add.py \
  --gpu-add-stage=add \
  --gpu-add-code-domain=vram \
  --gpu-add-data-domain=vram \
  --gpu-add-output-domain=vram
```

## Stress

```bash
AMD_BOOT_ATOM=0 python3 add.py \
  --gpu-add-stage=add \
  --gpu-add-repeat=1000
```

Acceptance:

```text
1,000 correct GPU-produced results
no fence timeout
no output corruption
no CP result upload
no card reset required
```

---

# 11. Final decision tree

```text
Current card VRAM corrupt
    ↓
software changes cannot qualify shader execution
    ↓
repair card or replace card
    ↓
VRAM retention suite passes?
    ├── no
    │   → continue hardware diagnosis
    │   → or abandon this board
    │
    └── yes
        → place all graphics resources in VRAM
        → constant-minimal
        → PARAM0
        → PARAM1
        → ADD
        → stress test
```

## Final project position

The software scaffold is sufficiently developed to resume once execution memory is valid.

The remaining blocker is external to `add.py`:

> physical GDDR3/MVDD/address-data integrity on the HD 4850 board.

Until that is repaired, `add.py` should fail closed and explain that the board lacks qualified shader execution memory.

## Next action: card-versus-eGPU discriminator

The software fault map is complete (815/816 failures, write-independent
readback, address-dependent aliases, and no improvement from 250--993 MHz).
Do not continue changing shaders, PM4 context, IO_DEBUG values, or guessed
voltages on this board.

1. Test this exact HD 4850 in a conventional PCIe desktop with normal
   auxiliary power: POST/display, Linux `radeon`, and VRAM retention.
2. If it fails there, the fault is on-card (GDDR3, MVDD/MVDDQ, termination,
   traces, or BGA) and the board must be repaired/replaced.
3. If it passes there, compare eGPU rails, reset/reference-clock timing,
   auxiliary power, and grounding.
4. Resume `add.py` graphics testing only after a 4 KiB, 100-iteration VRAM
   retention test passes.  CP/AGP success alone is only a DMA transport test.

Implemented VRAM diagnostics: `AMD_BOOT_VRAM_IO_DEBUG=1` snapshots
`MC_SEQ_IO_DEBUG_INDEX/DATA` around the memory-tail replay and reports changed
entries.  Use it on a cold `CHG=True` boot for the planned CLKF content diff;
`AMD_BOOT_VRAM_MCLK_10KHZ` and `AMD_BOOT_VRAM_SWEEP=1` provide clock and address
experiments.

Latest run: widened IO_DEBUG capture reports 167 changed indexed entries
(`0x140–0x1ff` region), proving the post-MPLL memory replay is not a no-op.
Despite that, BAR0/MM_INDEX writes still read back `0x57545d55`; the next
comparison should use these captured values against a known-good cold desktop
post, not repeat clock changes.
Library
/
rv770_add_repair_plan.md


# RV770 HD 4850 TinyGPU GPU-Add Plan — Hardware-Gated Revision

**Target:** `examples_egpu_terrascale/add.py`  
**Hardware:** Radeon HD 4850 (`1002:9442`, RV770/R700) over TinyGPU on M1 Mac  
**Revision date:** 2026-07-11

## Executive conclusion

The remaining blocker is no longer a software-initialization or PM4 problem.

The card’s local GDDR3 path has been isolated to physical address/data corruption:

- cold ATOM and board-specific MVDD/GPIO replay execute successfully;
- the complete memory-training sequence runs;
- memory clocks from 250 MHz through 993 MHz fail in the same way;
- BAR0 and MM_INDEX agree on corrupted readback;
- multiple offsets show aliases and constant patterns;
- changing software timing, PM4, shader state, clocks, and initialization order does not restore retention.

Therefore:

> `add.py` cannot produce a valid RV770 shader result on this physical HD 4850 until the GDDR3/MVDD/address-data path is repaired or the card is replaced.

This plan no longer treats further shader, PM4, ATOM, or clock experimentation as the main path to success.

---

# 1. Project state to preserve

The current software work is valuable and should be frozen as the known-good bring-up baseline.

## 1.1 Proven working

- PCI configuration and MMIO access
- TinyGPU transport
- AGP aperture programming
- CP firmware upload
- CP ring bring-up
- CP scratch test
- CP writes to AGP-backed host memory
- separate AGP fence page
- PM4 validation
- offline RV770 shader compilation
- real four-ADD RV770 shader binary
- constant, PARAM0, ADD, streamout, and empty-shader diagnostics
- canary-based output classification
- no CPU result fallback
- no CP result write to the output allocation

## 1.2 Proven failing on this board

- VRAM write retention
- BAR0 readback
- MM_INDEX readback
- multiple VRAM offsets
- 250, 500, and 993 MHz memory clocks
- cold ATOM memory initialization
- board-specific power replay
- full training sequence
- graphics draw completion
- streamout result production
- EOP retirement after graphics work

## 1.3 Preserve the baseline commit

Create a checkpoint containing all current diagnostics and guards.

Suggested commit:

```bash
git add examples_egpu_terrascale
git commit -m "rv770: freeze software bring-up at confirmed hardware VRAM fault"
```

The checkpoint should keep:

```text
--selftest
--test
--cp-mem-write-test
--gpu-add-stage=cp
--gpu-add-stage=constant
--gpu-add-stage=param0
--gpu-add-stage=add
--gpu-add-stage=stream
--vram-probe
--gpu-add-dump-pm4
--gpu-add-dump-registers
```

---

# 2. Change the default user-facing behavior

The default command should no longer imply that more PM4 work may make the current board succeed.

## 2.1 Add a hardware-integrity gate

Before submitting any RV770 graphics stage, require a VRAM integrity qualification unless a future known-good GTT shader path is explicitly selected.

Implement:

```python
def require_rv770_execution_memory(self) -> None:
    if self.vram_qualified:
        return

    if self.gtt_shader_execution_qualified:
        return

    raise RuntimeError(
        "RV770 GPU add is unavailable on this board: local GDDR3 fails "
        "address/data integrity tests, and shader execution from AGP/GTT has "
        "not been qualified. Repair or replace the card, or use a known-good "
        "RV770/Redwood device."
    )
```

## 2.2 Add persistent qualification state

Track:

```python
self.vram_qualified = False
self.gtt_shader_execution_qualified = False
```

These values must not be inferred from:

- `CONFIG_MEMSIZE`;
- successful ATOM return;
- successful CP test;
- a nonzero memory clock;
- BAR0 responding;
- one matching read.

They should be set only after explicit tests pass.

## 2.3 Update status text

Replace language such as:

```text
next debug target is graphics initialization or CB routing
```

with:

```text
graphics software bring-up is blocked by confirmed board-level GDDR3
address/data corruption. Further PM4 changes are not expected to help.
```

Replace:

```text
default add.py refuses because the RV770 shader path is not implemented
```

with:

```text
default add.py contains a genuine RV770 shader path but fails closed because
this board has no qualified execution memory.
```

---

# 3. Stop conditions for software experimentation

The following work should stop on this card until hardware integrity changes.

## 3.1 Do not continue changing

- shader binaries;
- VS/PS linkage;
- empty shader variants;
- VFETCH variants;
- CB state;
- streamout state;
- viewport or scissor values;
- CP parser state;
- EOP timing;
- raw-fence delays;
- reset pulse duration;
- AGP base address;
- memory clock frequency;
- ATOM table ordering;
- speculative MVDD values;
- unverified GPIO payloads;
- `BIF_FB_EN` permutations.

These experiments have either already been exhausted or cannot overcome physical address/data corruption.

## 3.2 Do not weaken safety gates

Keep:

- explicit opt-in for board power replay;
- VBIOS hash guard;
- PCI/subsystem-ID guard;
- cold-state guard;
- no guessed voltage values;
- no automatic BAR0 probing;
- no CPU arithmetic fallback;
- no CP result upload.

## 3.3 Do not report partial CP success as GPU add

The following remain diagnostics only:

```text
CP ring test
CP MEM_WRITE
raw post-draw breadcrumb
AGP fence write
```

None proves shader execution or arithmetic.

---

# 4. Hardware repair track

The next meaningful work is board-level diagnosis and repair.

## 4.1 Primary fault classes

Rank the physical causes as:

1. missing or unstable MVDD/MVDDQ;
2. GDDR3 address-line fault;
3. GDDR3 data-line fault;
4. damaged or cracked BGA joint;
5. damaged memory chip;
6. damaged termination or series resistor;
7. damaged memory-controller package connection;
8. board corrosion, contamination, or trace damage;
9. incorrect strap or board-identification circuitry;
10. power-sequencing failure.

The observed aliases and constant patterns are especially consistent with:

- stuck address bits;
- stuck or shorted data bits;
- missing memory rail;
- incomplete chip select;
- BGA connectivity failure.

## 4.2 Establish a known-good reference

Use the same card on a desktop motherboard if possible.

Record:

- whether it displays video;
- whether a desktop BIOS POSTs it;
- whether Linux reports stable VRAM;
- whether a VRAM stress test passes;
- MVDD/MVDDQ voltage;
- memory clock waveform;
- GDDR3 reset/enable timing.

Interpretation:

```text
fails identically in desktop
    → board/card hardware fault is confirmed independently of TinyGPU

works in desktop
    → eGPU power delivery, reset, reference clock, or board sequencing differs
```

## 4.3 Measure memory power rails

Using suitable equipment, measure:

- MVDD;
- MVDDQ if separate;
- VDDCI or related I/O rail if present;
- rail rise time;
- rail sequencing;
- ripple under load;
- voltage before, during, and after ATOM training.

Capture:

```text
cold power-on
after GPIO replay
after SetVoltage
during MemoryPLLInit
after training
during VRAM write test
```

Compare against:

- board schematic if available;
- memory-chip datasheet;
- known-good desktop operation;
- another identical card.

## 4.4 Inspect the board

Inspect:

- all GDDR3 chips;
- memory power MOSFETs and inductors;
- decoupling capacitors;
- resistor networks;
- series resistors on address/data lines;
- strap resistors;
- corrosion;
- cracked solder joints;
- burned or mechanically damaged areas;
- missing components.

Use magnification around:

- GPU package;
- memory chips;
- memory power section;
- PCIe power input;
- eGPU adapter power path.

## 4.5 Thermal and pressure diagnostics

As non-destructive diagnostics only:

- compare cold versus warmed behavior;
- apply controlled cooling to individual memory chips;
- apply controlled heating within safe limits;
- apply light mechanical pressure only as a diagnostic;
- observe whether corruption patterns change.

A pattern change with temperature or pressure strongly suggests BGA or package connectivity.

Do not treat uncontrolled reflow as a repair method.

## 4.6 Isolate bad channels or chips

Use the corruption map to infer channel behavior.

Add a diagnostic output that records, for each offset:

```text
written value
BAR0 value
MM_INDEX value
XOR difference
bits always zero
bits always one
aliased offsets
```

Sweep at least:

```text
0x000000
0x000004
0x000100
0x001000
0x010000
0x100000
```

Use patterns:

```text
0x00000000
0xFFFFFFFF
0xAAAAAAAA
0x55555555
0xA5A55A5A
0x5A5AA5A5
walking 1
walking 0
address-derived value
```

The goal is now hardware localization, not software qualification.

---

# 5. VRAM qualification after repair

After any board repair or card replacement, do not jump directly to `add`.

## 5.1 Required retention suite

A minimum 4 KiB region must pass:

- all-zero;
- all-one;
- alternating-bit patterns;
- walking-one;
- walking-zero;
- address-dependent pattern;
- inverse address-dependent pattern;
- repeated writes;
- delayed reads.

For each pattern:

1. write through BAR0;
2. read through BAR0;
3. read through MM_INDEX;
4. wait 100 ms;
5. read again through both paths;
6. repeat 100 times.

Acceptance:

```text
zero mismatches
zero aliasing
zero stuck bits
identical BAR0 and MM_INDEX data
```

## 5.2 Expand the qualification region

After 4 KiB passes, test:

```text
64 KiB
1 MiB
16 MiB
full accessible VRAM
```

The first GPU add needs only a small stable region, but larger qualification is needed before general use.

## 5.3 Set `vram_qualified`

Only after the minimum retention suite passes:

```python
self.vram_qualified = True
```

Log:

```text
qualified size
tested clock
pattern count
iteration count
BAR0/MM_INDEX agreement
```

---

# 6. Post-repair `add.py` completion path

Once VRAM is qualified, use a simple all-VRAM topology first.

## 6.1 Keep CP infrastructure in AGP

Continue using AGP for:

- CP ring;
- CP writeback page;
- EOP fence;
- diagnostic breadcrumbs.

Those paths already work.

## 6.2 Place all graphics resources in VRAM

For the first successful draw:

```text
fetch shader  → VRAM
vertex shader → VRAM
pixel shader  → VRAM
vertex data   → VRAM
color target  → VRAM
streamout     → VRAM
```

Do not mix AGP and VRAM graphics resources initially.

## 6.3 Add a domain-aware allocator

Implement:

```python
class GpuDomain(enum.Enum):
    AGP = "agp"
    VRAM = "vram"
```

and:

```python
@dataclass
class GpuAllocation:
    gpu_address: int
    size: int
    alignment: int
    domain: GpuDomain
    name: str
```

Provide:

```python
alloc_agp(...)
alloc_vram(...)
write_allocation(...)
read_allocation(...)
verify_allocation(...)
```

## 6.4 Recommended first layout

Use at least one stable 4 KiB VRAM page:

```text
+0x000 fetch shader
+0x100 vertex shader
+0x200 pixel shader
+0x300 vertex data
+0x400 color target
+0x500 streamout target
+0x600 guard pattern
```

Every shader start must be 256-byte aligned.

## 6.5 Verify every upload

Before submission:

- write shader/data;
- read back through BAR0;
- read back through MM_INDEX;
- compare byte-for-byte;
- abort on mismatch.

## 6.6 Run the stage ladder

Run in this order:

```text
constant-minimal
constant
param0
param1
add
streamout
```

Acceptance:

```text
constant-minimal → expected constant
param0           → A
param1           → B
add              → A+B
streamout        → VS-produced A+B
```

## 6.7 Preserve result integrity rules

The PM4 validator must still reject:

- CP writes to the result;
- literal expected-result values in result-write packets;
- CPU upload of the expected result.

Reading GPU-produced VRAM through BAR0 or MM_INDEX is valid.

---

# 7. Optional GTT shader research track

This is now optional research, not the main route to fixing this card.

Use a known-good RV770 under Linux and force shader BOs into GTT.

Interpretation:

```text
GTT shaders work
    → add.py may eventually support AGP-only shader execution

GTT shaders fail
    → RV770 shader code should remain VRAM-only
```

Do not use results from the corrupted HD 4850 to decide this question.

The main implementation should remain:

```text
VRAM-qualified card required for RV770 graphics
```

unless a separate known-good experiment proves otherwise.

---

# 8. Alternative hardware paths

If repairing this card is not practical, the project can still succeed by changing hardware.

## 8.1 Replace the HD 4850

Use another known-good RV770/R700 board.

Required precheck:

```text
desktop POST works
VRAM stress test passes
same PCI family
VBIOS dump available
```

Then reuse the existing `add.py` graphics path.

## 8.2 Use the HD 5570 path

The Redwood HD 5570 has a genuine Evergreen LS compute path.

Advantages:

- no graphics rasterization required;
- direct compute dispatch;
- easier scalar/vector-add topology;
- avoids RV770’s no-compute limitation.

Remaining work:

- real Redwood r600 compute shader;
- RAT/global buffer binding;
- compute resource setup;
- EOP/readback.

This may be a shorter software project than repairing a physically damaged HD 4850.

## 8.3 Keep CP-only HD 4850 diagnostics

Even with failed VRAM, the HD 4850 remains useful for:

- PCI/MMIO tests;
- CP firmware loading;
- AGP DMA;
- PM4 parser experiments;
- ATOM interpreter validation.

It should not be presented as capable of shader arithmetic until execution memory is qualified.

---

# 9. Revised implementation order

1. Freeze the current software baseline.
2. Add the execution-memory qualification gate.
3. Update all status and error messages.
4. Stop further PM4/shader experimentation on this board.
5. Confirm the fault on a desktop host if possible.
6. Measure MVDD/MVDDQ and memory-clock behavior.
7. Inspect and repair board-level faults.
8. Run the 4 KiB VRAM retention qualification.
9. Add the AGP/VRAM domain allocator.
10. Place all graphics resources in VRAM.
11. Make `constant-minimal` pass.
12. Make `param0` and `param1` pass.
13. Make `add` pass.
14. Run 1,000 repeated additions.
15. Reintroduce AGP input/output only as optional optimization.

---

# 10. Final test sequence after repair

## VRAM qualification

```bash
python3 add.py --vram-probe --vram-pattern-suite
```

Required:

```text
4 KiB minimum
100 iterations
zero errors
BAR0 == MM_INDEX
```

## CP baseline

```bash
AMD_BOOT_ATOM=0 python3 add.py --test
```

## Constant draw

```bash
AMD_BOOT_ATOM=0 python3 add.py \
  --gpu-add-stage=constant \
  --gpu-add-code-domain=vram \
  --gpu-add-data-domain=vram \
  --gpu-add-output-domain=vram
```

## Parameter tests

```bash
AMD_BOOT_ATOM=0 python3 add.py \
  --gpu-add-stage=param0 \
  --gpu-add-code-domain=vram \
  --gpu-add-data-domain=vram \
  --gpu-add-output-domain=vram

AMD_BOOT_ATOM=0 python3 add.py \
  --gpu-add-stage=param1 \
  --gpu-add-code-domain=vram \
  --gpu-add-data-domain=vram \
  --gpu-add-output-domain=vram
```

## Final add

```bash
AMD_BOOT_ATOM=0 python3 add.py \
  --gpu-add-stage=add \
  --gpu-add-code-domain=vram \
  --gpu-add-data-domain=vram \
  --gpu-add-output-domain=vram
```

## Stress

```bash
AMD_BOOT_ATOM=0 python3 add.py \
  --gpu-add-stage=add \
  --gpu-add-repeat=1000
```

Acceptance:

```text
1,000 correct GPU-produced results
no fence timeout
no output corruption
no CP result upload
no card reset required
```

---

# 11. Final decision tree

```text
Current card VRAM corrupt
    ↓
software changes cannot qualify shader execution
    ↓
repair card or replace card
    ↓
VRAM retention suite passes?
    ├── no
    │   → continue hardware diagnosis
    │   → or abandon this board
    │
    └── yes
        → place all graphics resources in VRAM
        → constant-minimal
        → PARAM0
        → PARAM1
        → ADD
        → stress test
```

## Final project position

The software scaffold is sufficiently developed to resume once execution memory is valid.

The remaining blocker is external to `add.py`:

> physical GDDR3/MVDD/address-data integrity on the HD 4850 board.

Until that is repaired, `add.py` should fail closed and explain that the board lacks qualified shader execution memory.
