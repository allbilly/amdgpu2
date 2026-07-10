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

Env: `AMD_ATOM_REPAIR_AFTER_MPLLINIT=1` (default), `AMD_ATOM_PATCH_MPLL=0`.

## Next (VRAM likely beyond Linux driver)

1. Diff IO_DEBUG **content** (57 @ CLKF=0 vs 159 @ CLKF=73) vs desktop-posted dump
2. MMIO-replay nested `GPIOPinControl` write list (MVDD GPIO)
3. Hardware: eGPU **MVDD / GDDR3** power (float `0x55` pattern strongly suggests undriven DQ)
