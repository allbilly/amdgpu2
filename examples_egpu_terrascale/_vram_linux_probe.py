"""Cold ATOM nested PS capture; Linux-aligned post (MPLL_TIME, SetVoltage); VRAM stick."""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import struct
import sys
import time
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "examples_egpu"))

from examples_egpu_terrascale.add import *  # noqa: E402
import neural as nl  # noqa: E402

NAMES = {
  0: "ASIC_Init", 3: "VRAM_BlockVenderDetection", 5: "MemoryControllerInit",
  6: "GPIO_PinInit", 7: "MemoryParamAdjust", 9: "GPIOPinControl",
  10: "SetEngineClock", 11: "SetMemoryClock", 14: "ResetMemoryDLL",
  15: "ResetMemoryDevice", 16: "MemoryPLLInit", 18: "AdjustMemoryController",
  52: "VRAM_BlockDetectionByStrap", 59: "MC_Synchronization",
  61: "ComputeMemoryEnginePLL", 63: "DynamicMemorySettings", 64: "MemoryTraining",
  67: "SetVoltage", 72: "MemoryDeviceInit",
}
WATCH = {3, 5, 6, 7, 9, 11, 14, 15, 16, 18, 52, 59, 61, 63, 64, 67, 72}


def main() -> None:
  d = TerrascaleDevice(wait_s=2)
  spll = d.rreg(REG_CG_SPLL_STATUS)
  print(f"pre CHG={bool(spll & SPLL_CHG_STATUS)} CLKF={d.rreg(0x624) & 0x7f} "
        f"MEM={d.rreg(REG_CONFIG_MEMSIZE):#x}")
  if not (spll & SPLL_CHG_STATUS):
    raise SystemExit("need cold CHG=True — power-cycle dock")

  bios_path = os.environ.get("AMD_BOOT_VBIOS_FILE", str(DEFAULT_VBIOS))
  bios = open(bios_path, "rb").read()
  if not nl.check_atom_bios(bios):
    raise RuntimeError("bad VBIOS")

  ctx0 = nl.parse_atom_context(bios)
  sv_off = nl._u16(bios, ctx0.cmd_table + 4 + 2 * 67)
  print(f"SetVoltage off={sv_off:#x} frev={bios[sv_off + 2]} crev={bios[sv_off + 3]}")

  nested: list[dict] = []
  d.clear_mc_blackout()
  d.prepare_spll_refclk()
  os.environ["AMD_ATOM_QUIET"] = "1"
  os.environ["AMD_ATOM_JUMP_BAIL"] = "1"
  os.environ["AMD_ATOM_JUMP_MAX"] = "200000"
  os.environ["AMD_ATOM_JUMP_TIMEOUT_SEC"] = "8"
  os.environ["AMD_ATOM_PATCH_MPLL"] = "0"
  os.environ["AMD_ATOM_SYNTH_SPLL_CHG"] = "0"

  class BootAdapter:
    def __init__(self, dev: TerrascaleDevice):
      self.dev = dev
      self._wcount = 0

    def rreg(self, reg: int) -> int:
      return self.dev.rreg((reg & 0xFFFF) * 4)

    def wreg(self, reg: int, val: int) -> None:
      self.dev.wreg((reg & 0xFFFF) * 4, val & 0xFFFFFFFF)
      self._wcount += 1
      if (self._wcount & 0x3F) == 0 and self.dev.pci.read_config(0, 2) == 0xFFFF:
        raise RuntimeError("pci hung mid-ATOM")

    def mmio_sync_safe(self) -> None:
      with contextlib.suppress(Exception):
        _ = self.dev.rreg(REG_CONFIG_MEMSIZE)

    def post_atom_sync(self) -> None:
      self.mmio_sync_safe()
      time.sleep(0.05)

  class Card(nl.AtomCard):
    def reg_read(self, reg: int) -> int:
      return super().reg_read(self._mmio_reg(reg))

  boot = BootAdapter(d)
  card = Card(boot, debug=False)
  ctx = nl.parse_atom_context(bios)
  exe = nl.AtomExecutor(ctx, card)
  _orig = type(exe)._execute_locked

  def hooked(self, index, ps, ps_size=16):
    if index in WATCH:
      entry = {
        "index": index,
        "name": NAMES.get(index, "?"),
        "depth": ctx.execute_depth,
        "ps": [int(x) & 0xFFFFFFFF for x in ps[:8]],
      }
      nested.append(entry)
      ps_hex = [hex(x) for x in entry["ps"][:4]]
      print(f"  nest d={ctx.execute_depth} [{index}] {entry['name']} ps={ps_hex}",
            flush=True)
    return _orig(self, index, ps, ps_size)

  exe._execute_locked = types.MethodType(hooked, exe)

  hwi = nl._u16(bios, ctx.data_table + nl.ATOM_DATA_FWI_PTR)
  ps = [0] * 16
  ps[0] = nl._u32(bios, hwi + nl.ATOM_FWI_DEFSCLK_PTR)
  ps[1] = nl._u32(bios, hwi + nl.ATOM_FWI_DEFMCLK_PTR)
  print(f"ASIC_Init ps sclk={ps[0]} mclk={ps[1]}")
  t0 = time.time()
  ret = exe.execute_table(nl.ATOM_CMD_INIT, ps, 16)
  print(f"ATOM ret={ret} writes={boot._wcount} t={time.time() - t0:.1f}s")

  misc0 = d.rreg(REG_MC_SEQ_MISC0)
  clkf = d.rreg(0x624) & 0x7F
  print(f"post-ATOM MISC0={misc0:#x} CLKF={clkf} MEM={d.rreg(REG_CONFIG_MEMSIZE):#x}")
  if clkf == 0:
    d.repair_mpll_boot_clock()
    d._wake_mrdck()
  print(f"after repair CLKF={d.rreg(0x624) & 0x7F} MISC0={d.rreg(0x2a00):#x}")

  out = pathlib.Path("/tmp/atom_nested_ps.json")
  out.write_text(json.dumps(nested, indent=2))
  print(f"saved {len(nested)} nested calls -> {out}")

  # Linux GDDR3-only: MPLL_TIME lock/reset defaults (rv770_program_mpll_timing_parameters)
  MPLL_TIME = 0x654
  d.wreg(MPLL_TIME, (100 & 0xFFFF) | ((150 & 0xFFFF) << 16))
  print(f"MPLL_TIME={d.rreg(MPLL_TIME):#x}")

  sv_calls = [c for c in nested if c["index"] == 67]
  print(f"SetVoltage nested calls: {len(sv_calls)}")
  for i, c in enumerate(sv_calls):
    ps_hex = [hex(x) for x in c["ps"][:4]]
    print(f"  SV[{i}] ps={ps_hex}")
    try:
      d.atom_run_cmd(67, "SetVoltage", c["ps"][0])
      d.ensure_mpll_alive()
      d.wreg(R600_BIF_FB_EN, 0)
      print(f"    after SV MISC0={d.rreg(0x2a00):#x} pci={d.pci.read_config(0, 2):#x}")
    except Exception as e:
      print(f"    SV fail {e}")

  # GPIOPinControl nested replay (memory power GPIOs often here)
  gpio_calls = [c for c in nested if c["index"] == 9]
  print(f"GPIOPinControl nested: {len(gpio_calls)}")
  for i, c in enumerate(gpio_calls):
    ps_hex = [hex(x) for x in c["ps"][:4]]
    print(f"  GPIO[{i}] ps={ps_hex}")
    try:
      d.atom_run_cmd(9, "GPIOPinControl", c["ps"][0])
      d.ensure_mpll_alive()
      d.wreg(R600_BIF_FB_EN, 0)
    except Exception as e:
      print(f"    GPIO fail {e}")

  crev = bios[sv_off + 3]
  print(f"Linux-style SetVoltage crev={crev} (MVDDC/MVDDQ)")
  if crev == 1:
    for vtype, name in ((1, "VDDC"), (2, "MVDDC"), (3, "MVDDQ")):
      # TYPE | (MODE_ALL_SOURCE=1 << 8) | (index=0 << 16)
      ps0 = vtype | (0x1 << 8)
      try:
        d.atom_run_cmd(67, f"SetVoltage-{name}", ps0)
        print(f"  {name} ok MISC0={d.rreg(0x2a00):#x}")
      except Exception as e:
        print(f"  {name} {e}")

  try:
    d.finish_memory_after_mpll()
  except Exception as e:
    print(f"finish_memory {e}")
  d.ensure_mpll_alive()

  d.wreg(R600_BIF_FB_EN, 0)
  d.program_agp()
  d.load_cp_fw()
  d.cp_resume()
  ring = d.ring_test(timeout_s=1.0)
  print(f"ring={ring} MISC0={d.rreg(0x2a00):#x}")

  ok = False
  if d.rreg(0x2a00) == 0x3000422A and d.pci.read_config(0, 2) != 0xFFFF:
    d.program_mc_vram_linux(enable_bif=False)
    d.ensure_mpll_alive()
    d.wreg(R600_BIF_FB_EN, R600_FB_READ_EN | R600_FB_WRITE_EN)
    time.sleep(0.02)
    if d.pci.read_config(0, 2) == 0xFFFF:
      print("HUNG BIF")
    else:
      if d.vram is not None:
        print(f"BAR0={bytes(d.vram[0:8]).hex()}")
      pat = 0xA5A55A5A
      d.wreg(0, 0x80000000)
      d.wreg(4, pat)
      d.wreg(REG_HDP_DEBUG1, 0)
      time.sleep(0.05)
      d.wreg(0, 0x80000000)
      got = d.rreg(4)
      ok = got == pat
      print(f"MM stick={ok} got={got:#x}")
      if d.vram is not None and d.pci.read_config(0, 2) != 0xFFFF:
        d.vram[0:4] = struct.pack("<I", 0xDEADBEEF)
        d.wreg(REG_HDP_DEBUG1, 0)
        time.sleep(0.05)
        bg = struct.unpack("<I", bytes(d.vram[0:4]))[0]
        print(f"BAR0 stick={bg == 0xDEADBEEF} got={bg:#x}")
    d.wreg(R600_BIF_FB_EN, 0)
    d.program_agp()

  d.load_cp_fw()
  d.cp_resume()
  print(f"final ring={d.ring_test(timeout_s=1.0)} vram_stick={ok} "
        f"pci={d.pci.read_config(0, 2):#x}")


if __name__ == "__main__":
  main()
