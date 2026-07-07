#!/usr/bin/env python3
"""Debug ATOM asic_init — find which MMIO op kills TinyGPU."""
import os, sys, traceback
os.environ.setdefault("AMD_BOOT_VBIOS_FILE", "/tmp/rx570.rom")
os.environ.setdefault("AMD_ATOM_QUIET", "1")
os.environ.setdefault("AMD_ATOM_DRAIN_EVERY", "32")
os.environ.setdefault("AMD_MMIO_DRAIN_EVERY", "32")

from add import PolarisDevice
from polaris_boot import PolarisBoot
from atom_replay import (
  read_vbios_rom, parse_atom_context, atom_asic_init,
  clear_asic_init_scratch, AtomCard, AtomExecutor, _u16, _u32,
  ATOM_DATA_FWI_PTR, ATOM_CMD_INIT, ATOM_FWI_DEFSCLK_PTR, ATOM_FWI_DEFMCLK_PTR,
)

class TraceCard(AtomCard):
  def reg_read(self, reg: int) -> int:
    print(f"  atom RREG {reg:#06x}", flush=True)
    try:
      return super().reg_read(reg)
    except Exception as e:
      print(f"FAIL read reg={reg:#06x}: {e}", flush=True)
      raise

def main():
  dev = PolarisDevice()
  boot = PolarisBoot(dev)
  boot.vi_common_init()
  boot.enable_vbios_rom()
  bios = read_vbios_rom(boot)
  clear_asic_init_scratch(boot)
  ctx = parse_atom_context(bios)
  card = TraceCard(boot, debug=False)
  exe = AtomExecutor(ctx, card)
  hwi = _u16(bios, ctx.data_table + ATOM_DATA_FWI_PTR)
  ps = [0] * 16
  ps[0] = _u32(bios, hwi + ATOM_FWI_DEFSCLK_PTR)
  ps[1] = _u32(bios, hwi + ATOM_FWI_DEFMCLK_PTR)
  print(f"starting asic_init writes={ctx.reg_write_count}", flush=True)
  try:
    exe.execute_table(ATOM_CMD_INIT, ps, 16)
    print(f"done writes={ctx.reg_write_count} MEMSIZE={boot.rreg(0x150a):#x}", flush=True)
  except Exception:
    traceback.print_exc()
    print(f"partial writes={ctx.reg_write_count} pci={dev.pci.read_config(0,2)&0xffff:#06x}", flush=True)
    sys.exit(1)

if __name__ == "__main__":
  main()
