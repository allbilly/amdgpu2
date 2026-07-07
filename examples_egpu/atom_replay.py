"""ATOM BIOS interpreter for Polaris eGPU — ports linux amdgpu/atom.c for asic_init.

Linux hot-plug runs amdgpu_atom_asic_init() → amdgpu_atom_execute_table(ATOM_CMD_INIT).
This module reads VBIOS ROM via SMC ind-port and executes the bytecode VM in Python.
"""
from __future__ import annotations
import os, struct, time, json, contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
  from polaris_boot import PolarisBoot

# atom.h / atom.c constants
ATOM_BIOS_MAGIC = 0xAA55
ATOM_ATI_MAGIC_PTR = 0x30
ATOM_ATI_MAGIC = b" 761295520"  # 10 bytes at ROM offset 0x30
ATOM_ROM_TABLE_PTR = 0x48
ATOM_ROM_MAGIC = b"ATOM"
ATOM_ROM_MAGIC_PTR = 4
ATOM_ROM_CMD_PTR = 0x1E
ATOM_ROM_DATA_PTR = 0x20
ATOM_CMD_INIT = 0
ATOM_DATA_FWI_PTR = 0xC
ATOM_DATA_IIO_PTR = 0x32
ATOM_FWI_DEFSCLK_PTR = 8
ATOM_FWI_DEFMCLK_PTR = 0xC
ATOM_CT_SIZE_PTR = 0
ATOM_CT_WS_PTR = 4
ATOM_CT_PS_PTR = 5
ATOM_CT_PS_MASK = 0x7F
ATOM_CT_CODE_PTR = 6
ATOM_OP_CNT = 127
ATOM_OP_EOT = 91
ATOM_ARG_REG, ATOM_ARG_PS, ATOM_ARG_WS = 0, 1, 2
ATOM_ARG_FB, ATOM_ARG_ID, ATOM_ARG_IMM = 3, 4, 5
ATOM_ARG_PLL, ATOM_ARG_MC = 6, 7
ATOM_SRC_DWORD, ATOM_SRC_WORD0, ATOM_SRC_WORD8, ATOM_SRC_WORD16 = 0, 1, 2, 3
ATOM_SRC_BYTE0, ATOM_SRC_BYTE8, ATOM_SRC_BYTE16, ATOM_SRC_BYTE24 = 4, 5, 6, 7
ATOM_IO_MM, ATOM_IO_PCI, ATOM_IO_SYSIO = 0, 1, 2
ATOM_IO_IIO = 0x80
ATOM_UNIT_MICROSEC, ATOM_UNIT_MILLISEC = 0, 1
ATOM_COND_ABOVE, ATOM_COND_ABOVEOREQUAL, ATOM_COND_ALWAYS = 0, 1, 2
ATOM_COND_BELOW, ATOM_COND_BELOWOREQUAL, ATOM_COND_EQUAL = 3, 4, 5
ATOM_COND_NOTEQUAL = 6
# atom-names.h: op 67..73 jump variants (not equal to ATOM_COND_* enum order)
ATOM_JUMP_OP_COND = (2, 5, 3, 0, 4, 1, 6)  # ALWAYS, EQUAL, BELOW, ABOVE, BELOWOREQUAL, ABOVEOREQUAL, NOTEQUAL
ATOM_CASE_MAGIC, ATOM_CASE_END = 0x63, 0x5A5A
ATOM_IIO_START, ATOM_IIO_END = 1, 9
ATOM_WS_QUOTIENT, ATOM_WS_REMAINDER, ATOM_WS_DATAPTR = 0x40, 0x41, 0x42
ATOM_WS_SHIFT, ATOM_WS_FB_WINDOW, ATOM_WS_ATTRIBUTES = 0x43, 0x46, 0x47
ATOM_WS_REGPTR = 0x48
ATOM_EXECUTE_MAX_DEPTH = 32
ATOM_CMD_TIMEOUT_SEC = float(os.environ.get("AMD_ATOM_JUMP_TIMEOUT_SEC", "120"))
ATOM_JUMP_MAX_ITERS = int(os.environ.get("AMD_ATOM_JUMP_MAX", "512"))
ATOM_MAX_WRITES = int(os.environ.get("AMD_ATOM_MAX_WRITES", "65536"))
ATOM_SCRATCH_BYTES = int(os.environ.get("AMD_ATOM_SCRATCH_KB", "20")) << 10
mmBIOS_SCRATCH_7 = 0x5D0
ATOM_S7_ASIC_INIT_COMPLETE_MASK = 0x00000200

ATOM_ARG_MASK = (0xFFFFFFFF, 0xFFFF, 0xFFFF00, 0xFFFF0000, 0xFF, 0xFF00, 0xFF0000, 0xFF000000)
ATOM_ARG_SHIFT = (0, 0, 8, 16, 0, 8, 16, 24)
ATOM_DST_TO_SRC = (
  (0, 0, 0, 0), (1, 2, 3, 0), (1, 2, 3, 0), (1, 2, 3, 0),
  (4, 5, 6, 7), (4, 5, 6, 7), (4, 5, 6, 7), (4, 5, 6, 7),
)
ATOM_DEF_DST = (0, 0, 1, 2, 0, 1, 2, 3)
ATOM_IIO_LEN = (1, 2, 3, 3, 3, 3, 4, 4, 4, 3)


def _u8(b: bytes, off: int) -> int:
  return b[off]

def _u16(b: bytes, off: int) -> int:
  return struct.unpack_from("<H", b, off)[0]

def _u32(b: bytes, off: int) -> int:
  return struct.unpack_from("<I", b, off)[0]

def _cstr(b: bytes, off: int, n: int = 32) -> bytes:
  return bytes(b[off:off + n]).split(b"\x00", 1)[0]


@dataclass
class AtomContext:
  bios: bytes
  cmd_table: int
  data_table: int
  iio: dict[int, int] = field(default_factory=dict)
  data_block: int = 0
  fb_base: int = 0
  divmul: list[int] = field(default_factory=lambda: [0, 0])
  io_attr: int = 0
  reg_block: int = 0
  shift: int = 0
  cs_equal: bool = False
  cs_above: bool = False
  io_mode: int = ATOM_IO_MM
  execute_depth: int = 0
  scratch: list[int] | None = None
  scratch_size_bytes: int = 0
  reg_write_count: int = 0


class AtomCard:
  """MMIO callbacks for ATOM interpreter (cail_reg_read/write)."""

  def __init__(self, boot: "PolarisBoot", debug: bool = False):
    self.boot = boot
    self.debug = debug
    self._n = 0
    self._poll_reads: dict[int, int] = {}
    self._jump_counts: dict[int, int] = {}
    self._drain_every = max(1, int(os.environ.get(
      "AMD_ATOM_DRAIN_EVERY", os.environ.get("AMD_MMIO_DRAIN_EVERY", "32"))))

  def _maybe_drain(self):
    self._n += 1
    if self._n >= self._drain_every:
      self._n = 0
      with contextlib.suppress(Exception):
        self.boot.dev.pci.drain_mmio(bar=5, reg=0x2004)
      ms = int(os.environ.get("AMD_ATOM_SETTLE_MS", "5"))
      if ms > 0:
        time.sleep(ms / 1000.0)

  def _mmio_reg(self, reg: int) -> int:
    """ATOM reg indices are 16-bit; reg_block+offset can wrap (e.g. 0x5c08+0xdc8c→0x3894)."""
    return reg & 0xFFFF

  def reg_read(self, reg: int) -> int:
    reg = self._mmio_reg(reg)
    if reg == 0:
      return 0
    val = self.boot.rreg(reg)
    if val == 0 and os.environ.get("AMD_ATOM_POLL_HACK", "1") != "0":
      n = self._poll_reads.get(reg, 0) + 1
      self._poll_reads[reg] = n
      # VBIOS asic_init spins on low MMIO regs (e.g. 0x0001) until HW sets bits.
      if reg <= 0xff and n > int(os.environ.get("AMD_ATOM_POLL_THRESH", "64")):
        return int(os.environ.get("AMD_ATOM_POLL_VAL", "1"), 0)
    return val

  def reg_write(self, reg: int, val: int):
    reg = self._mmio_reg(reg)
    if self.debug and not os.environ.get("AMD_ATOM_QUIET"):
      print(f"  atom WREG {reg:#06x} = {val:#010x}", flush=True)
    self.boot.wreg(reg, val)
    self._maybe_drain()

  def mc_read(self, reg: int) -> int:
    return self.reg_read(reg)

  def mc_write(self, reg: int, val: int):
    self.reg_write(reg, val)

  def pll_read(self, reg: int) -> int:
    return self.reg_read(reg)

  def pll_write(self, reg: int, val: int):
    self.reg_write(reg, val)


def read_vbios_rom(boot: "PolarisBoot", length: int | None = None) -> bytes:
  """vi_read_bios_from_rom via SMC ind-port."""
  vbios_file = os.environ.get("AMD_BOOT_VBIOS_FILE")
  if vbios_file and os.path.isfile(vbios_file):
    return open(vbios_file, "rb").read()
  from polaris_boot import mmSMC_IND_INDEX_11, mmSMC_IND_DATA_11, ixROM_INDEX, ixROM_DATA
  boot.enable_vbios_rom()
  boot.wreg(mmSMC_IND_INDEX_11, ixROM_INDEX)
  boot.wreg(mmSMC_IND_DATA_11, 0)
  boot.wreg(mmSMC_IND_INDEX_11, ixROM_DATA)
  hdr = bytearray(512)
  for i in range(128):
    struct.pack_into("<I", hdr, i * 4, boot.rreg(mmSMC_IND_DATA_11))
  if _u16(hdr, 0) != ATOM_BIOS_MAGIC:
    raise RuntimeError(f"VBIOS bad magic {_u16(hdr, 0):#x}")
  if hdr[ATOM_ATI_MAGIC_PTR:ATOM_ATI_MAGIC_PTR + len(ATOM_ATI_MAGIC)] != ATOM_ATI_MAGIC:
    raise RuntimeError("VBIOS missing ATI magic")
  if length is None:
    length = _u8(hdr, 2) << 9
  length = (length + 3) & ~3
  rom = bytearray(length)
  rom[:512] = hdr[:512]
  boot.wreg(mmSMC_IND_INDEX_11, ixROM_INDEX)
  boot.wreg(mmSMC_IND_DATA_11, 512)
  boot.wreg(mmSMC_IND_INDEX_11, ixROM_DATA)
  for i in range(512 // 4, length // 4):
    struct.pack_into("<I", rom, i * 4, boot.rreg(mmSMC_IND_DATA_11))
  return bytes(rom)


def _index_iio(bios: bytes, data_table: int) -> dict[int, int]:
  iio: dict[int, int] = {}
  base = _u16(bios, data_table + ATOM_DATA_IIO_PTR) + 4
  while _u8(bios, base) == ATOM_IIO_START:
    idx = _u8(bios, base + 1)
    iio[idx] = base + 2
    base += 2
    while _u8(bios, base) != ATOM_IIO_END:
      base += ATOM_IIO_LEN[_u8(bios, base)]
    base += 3
  return iio


def alloc_atom_scratch(ctx: AtomContext) -> None:
  """amdgpu_atombios_allocate_fb_scratch — 20KB default workspace for ATOM_ARG_FB."""
  if ctx.scratch is not None:
    return
  n = max(ATOM_SCRATCH_BYTES, 4096) // 4
  ctx.scratch = [0] * n
  ctx.scratch_size_bytes = n * 4


def parse_atom_context(bios: bytes) -> AtomContext:
  if _u16(bios, 0) != ATOM_BIOS_MAGIC:
    raise ValueError("invalid BIOS magic")
  if bios[ATOM_ATI_MAGIC_PTR:ATOM_ATI_MAGIC_PTR + len(ATOM_ATI_MAGIC)] != ATOM_ATI_MAGIC:
    raise ValueError("invalid ATI magic")
  base = _u16(bios, ATOM_ROM_TABLE_PTR)
  if bios[base + ATOM_ROM_MAGIC_PTR:base + ATOM_ROM_MAGIC_PTR + 4] != ATOM_ROM_MAGIC:
    raise ValueError("invalid ATOM magic")
  ctx = AtomContext(
    bios=bios,
    cmd_table=_u16(bios, base + ATOM_ROM_CMD_PTR),
    data_table=_u16(bios, base + ATOM_ROM_DATA_PTR),
  )
  ctx.iio = _index_iio(bios, ctx.data_table)
  alloc_atom_scratch(ctx)
  return ctx


def atom_info(bios: bytes) -> dict:
  ctx = parse_atom_context(bios)
  init_off = _u16(bios, ctx.cmd_table + 4 + 2 * ATOM_CMD_INIT)
  hwi = _u16(bios, ctx.data_table + ATOM_DATA_FWI_PTR)
  return {
    "bios_len": len(bios),
    "cmd_table": ctx.cmd_table,
    "data_table": ctx.data_table,
    "asic_init_off": init_off,
    "def_sclk": _u32(bios, hwi + ATOM_FWI_DEFSCLK_PTR) if hwi else 0,
    "def_mclk": _u32(bios, hwi + ATOM_FWI_DEFMCLK_PTR) if hwi else 0,
    "iio_tables": len(ctx.iio),
  }


def need_asic_init(boot: "PolarisBoot") -> bool:
  if os.environ.get("AMD_BOOT_ATOM_FORCE", "0") == "1":
    return True
  mem_mb = boot.rreg(0x150a) & 0xffff
  misc0 = boot.rreg(0xa80)
  if mem_mb not in (0, 0xffff) and (misc0 & 0x80):
    return False
  # Scratch bit can be set without real VRAM training (partial/fake init)
  return True


def clear_asic_init_scratch(boot: "PolarisBoot"):
  """Clear stale ASIC_INIT_COMPLETE so VBIOS reruns MC training."""
  scratch7 = boot.rreg(mmBIOS_SCRATCH_7)
  if scratch7 & ATOM_S7_ASIC_INIT_COMPLETE_MASK:
    boot.wreg(mmBIOS_SCRATCH_7, scratch7 & ~ATOM_S7_ASIC_INIT_COMPLETE_MASK)


class AtomExecutor:
  """Python port of amdgpu atom.c bytecode VM."""

  def __init__(self, ctx: AtomContext, card: AtomCard):
    self.ctx = ctx
    self.card = card
    self.debug = card.debug
    self.ps: list[int] = []
    self.ps_size = 0
    self.ws: list[int] = []
    self._base = 0

  def _iio_execute(self, base: int, index: int, data: int) -> int:
    bios, temp = self.ctx.bios, 0xCDCDCDCD
    while True:
      op = _u8(bios, base)
      if op == 0:  # NOP
        base += 1
      elif op == 2:  # READ
        temp = self.card.reg_read(_u16(bios, base + 1))
        base += 3
      elif op == 3:  # WRITE
        self.card.reg_write(_u16(bios, base + 1), temp)
        base += 3
      elif op == 4:  # CLEAR
        temp &= ~((0xFFFFFFFF >> (32 - _u8(bios, base + 1))) << _u8(bios, base + 2))
        base += 3
      elif op == 5:  # SET
        temp |= (0xFFFFFFFF >> (32 - _u8(bios, base + 1))) << _u8(bios, base + 2)
        base += 3
      elif op == 6:  # MOVE_INDEX
        m = 0xFFFFFFFF >> (32 - _u8(bios, base + 1))
        temp &= ~(m << _u8(bios, base + 3))
        temp |= ((index >> _u8(bios, base + 2)) & m) << _u8(bios, base + 3)
        base += 4
      elif op == 7:  # MOVE_ATTR
        m = 0xFFFFFFFF >> (32 - _u8(bios, base + 1))
        temp &= ~(m << _u8(bios, base + 3))
        temp |= (data & m) << _u8(bios, base + 3)
        base += 4
      elif op == 8:  # MOVE_DATA
        m = 0xFFFFFFFF >> (32 - _u8(bios, base + 1))
        temp = (temp & ~(m << _u8(bios, base + 2))) | ((data & m) << _u8(bios, base + 2))
        base += 4
      elif op == ATOM_IIO_END:
        return temp
      else:
        raise RuntimeError(f"unknown IIO op {op:#x} @ {base:#x}")

  def _get_src_direct(self, align: int, ptr: list[int]) -> int:
    p = ptr[0]
    if align == ATOM_SRC_DWORD:
      v = _u32(self.ctx.bios, p); ptr[0] = p + 4
    elif align in (ATOM_SRC_WORD0, ATOM_SRC_WORD8, ATOM_SRC_WORD16):
      v = _u16(self.ctx.bios, p); ptr[0] = p + 2
    else:
      v = _u8(self.ctx.bios, p); ptr[0] = p + 1
    return v

  def _skip_src(self, attr: int, ptr: list[int]):
    align, arg = (attr >> 3) & 7, attr & 7
    p = ptr[0]
    if arg in (ATOM_ARG_REG, ATOM_ARG_ID):
      ptr[0] = p + 2
    elif arg in (ATOM_ARG_PLL, ATOM_ARG_MC, ATOM_ARG_PS, ATOM_ARG_WS, ATOM_ARG_FB):
      ptr[0] = p + 1
    elif arg == ATOM_ARG_IMM:
      ptr[0] = p + (4 if align == ATOM_SRC_DWORD else 2 if align <= ATOM_SRC_WORD16 else 1)

  def _get_src_int(self, attr: int, ptr: list[int], saved: list[int] | None, do_read: bool) -> int:
    g, bios = self.ctx, self.ctx.bios
    align, arg = (attr >> 3) & 7, attr & 7
    p = ptr[0]
    val = 0
    if arg == ATOM_ARG_REG:
      idx = _u16(bios, p); ptr[0] = p + 2
      idx += g.reg_block
      if g.io_mode == ATOM_IO_MM:
        val = 0 if idx == 0 else self.card.reg_read(idx)
      elif g.io_mode & ATOM_IO_IIO:
        iio = g.iio.get(g.io_mode & 0x7F)
        if not iio:
          raise RuntimeError(f"undefined IIO {g.io_mode & 0x7F}")
        val = self._iio_execute(iio, idx, 0)
      else:
        raise RuntimeError(f"unsupported io_mode {g.io_mode}")
    elif arg == ATOM_ARG_PS:
      idx = _u8(bios, p); ptr[0] = p + 1
      if idx < self.ps_size:
        val = self.ps[idx]
      else:
        val = 0
    elif arg == ATOM_ARG_WS:
      idx = _u8(bios, p); ptr[0] = p + 1
      if idx < len(self.ws):
        val = self.ws[idx]
      else:
        ws_map = {
          ATOM_WS_QUOTIENT: g.divmul[0], ATOM_WS_REMAINDER: g.divmul[1],
          ATOM_WS_DATAPTR: g.data_block, ATOM_WS_SHIFT: g.shift,
          ATOM_WS_FB_WINDOW: g.fb_base, ATOM_WS_ATTRIBUTES: g.io_attr,
          ATOM_WS_REGPTR: g.reg_block,
        }
        val = ws_map.get(idx, 0)
    elif arg == ATOM_ARG_ID:
      idx = _u16(bios, p); ptr[0] = p + 2
      val = idx + g.data_block
    elif arg == ATOM_ARG_FB:
      idx = _u8(bios, p); ptr[0] = p + 1
      off = (g.fb_base // 4) + idx
      if g.scratch and (g.fb_base + idx * 4) <= g.scratch_size_bytes:
        val = g.scratch[off]
    elif arg == ATOM_ARG_IMM:
      val = self._get_src_direct(align, ptr)
    elif arg == ATOM_ARG_PLL:
      idx = _u8(bios, p); ptr[0] = p + 1
      val = self.card.pll_read(idx)
    elif arg == ATOM_ARG_MC:
      idx = _u8(bios, p); ptr[0] = p + 1
      val = self.card.mc_read(idx)
    if saved is not None:
      saved[0] = val
    val &= ATOM_ARG_MASK[align]
    val >>= ATOM_ARG_SHIFT[align]
    return val

  def _get_dst(self, arg: int, attr: int, ptr: list[int], saved: list[int] | None) -> int:
    full = arg | (ATOM_DST_TO_SRC[(attr >> 3) & 7][(attr >> 6) & 3] << 3)
    return self._get_src_int(full, ptr, saved, True)

  def _get_src(self, attr: int, ptr: list[int]) -> int:
    return self._get_src_int(attr, ptr, None, True)

  def _skip_dst(self, arg: int, attr: int, ptr: list[int]):
    full = arg | (ATOM_DST_TO_SRC[(attr >> 3) & 7][(attr >> 6) & 3] << 3)
    self._skip_src(full, ptr)

  def _op_mul32(self, arg: int, ptr: list[int]):
    attr = _u8(self.ctx.bios, ptr[0]); ptr[0] += 1
    dst = self._get_dst(arg, attr, ptr, None)
    src = self._get_src(attr, ptr)
    prod = (dst * src) & ((1 << 64) - 1)
    self.ctx.divmul[0] = prod & 0xFFFFFFFF
    self.ctx.divmul[1] = (prod >> 32) & 0xFFFFFFFF

  def _op_div32(self, arg: int, ptr: list[int]):
    attr = _u8(self.ctx.bios, ptr[0]); ptr[0] += 1
    dst = self._get_dst(arg, attr, ptr, None)
    src = self._get_src(attr, ptr)
    if src:
      val64 = dst | (self.ctx.divmul[1] << 32)
      q, r = divmod(val64, src)
      self.ctx.divmul[0] = q & 0xFFFFFFFF
      self.ctx.divmul[1] = (q >> 32) & 0xFFFFFFFF
    else:
      self.ctx.divmul[0] = self.ctx.divmul[1] = 0

  def _bump_writes(self, g: AtomContext):
    g.reg_write_count += 1
    if ATOM_MAX_WRITES > 0 and g.reg_write_count > ATOM_MAX_WRITES:
      raise RuntimeError(f"ATOM MMIO write cap exceeded ({ATOM_MAX_WRITES})")

  def _put_dst(self, arg: int, attr: int, ptr: list[int], val: int, saved: int):
    g = self.ctx
    align = ATOM_DST_TO_SRC[(attr >> 3) & 7][(attr >> 6) & 3]
    old = val & (ATOM_ARG_MASK[align] >> ATOM_ARG_SHIFT[align])
    val = ((val << ATOM_ARG_SHIFT[align]) & ATOM_ARG_MASK[align]) | (saved & ~ATOM_ARG_MASK[align])
    p = ptr[0]
    if arg == ATOM_ARG_REG:
      idx = _u16(g.bios, p); ptr[0] = p + 2
      idx += g.reg_block
      if g.io_mode == ATOM_IO_MM:
        self.card.reg_write(idx, val << 2 if idx == 0 else val)
        self._bump_writes(g)
      elif g.io_mode & ATOM_IO_IIO:
        iio = g.iio.get(g.io_mode & 0x7F)
        if not iio:
          raise RuntimeError(f"undefined IIO write {g.io_mode & 0x7F}")
        self._iio_execute(iio, idx, val)
      else:
        raise RuntimeError(f"bad io_mode {g.io_mode}")
    elif arg == ATOM_ARG_WS:
      idx = _u8(g.bios, p); ptr[0] = p + 1
      if idx < len(self.ws):
        self.ws[idx] = val
      elif idx == ATOM_WS_QUOTIENT: g.divmul[0] = val
      elif idx == ATOM_WS_REMAINDER: g.divmul[1] = val
      elif idx == ATOM_WS_DATAPTR: g.data_block = val
      elif idx == ATOM_WS_SHIFT: g.shift = val
      elif idx == ATOM_WS_FB_WINDOW: g.fb_base = val
      elif idx == ATOM_WS_ATTRIBUTES: g.io_attr = val
      elif idx == ATOM_WS_REGPTR: g.reg_block = val & 0xFFFF
    elif arg == ATOM_ARG_PS:
      idx = _u8(g.bios, p); ptr[0] = p + 1
      if idx < self.ps_size:
        self.ps[idx] = val
    elif arg == ATOM_ARG_MC:
      idx = _u8(g.bios, p); ptr[0] = p + 1
      self.card.mc_write(idx, val)
      self._bump_writes(g)
    elif arg == ATOM_ARG_FB:
      idx = _u8(g.bios, p); ptr[0] = p + 1
      off = (g.fb_base // 4) + idx
      if g.scratch and (g.fb_base + idx * 4) <= g.scratch_size_bytes:
        g.scratch[off] = val
    elif arg == ATOM_ARG_PLL:
      idx = _u8(g.bios, p); ptr[0] = p + 1
      self.card.pll_write(idx, val)

  def _execute_locked(self, index: int, ps: list[int], ps_size: int) -> int:
    g, bios = self.ctx, self.ctx.bios
    base = _u16(bios, g.cmd_table + 4 + 2 * index)
    if not base:
      return -1
    if g.execute_depth >= ATOM_EXECUTE_MAX_DEPTH:
      raise RuntimeError("ATOM recursion limit")
    g.execute_depth += 1
    saved_ps, saved_ps_size, saved_ws, saved_base = self.ps, self.ps_size, self.ws, self._base
    ws_size = _u8(bios, base + ATOM_CT_WS_PTR)
    ps_shift = (_u8(bios, base + ATOM_CT_PS_PTR) & ATOM_CT_PS_MASK) // 4
    self.ps = ps
    self.ps_size = ps_size
    self.ws = [0] * ws_size
    self._base = base
    ptr = [base + ATOM_CT_CODE_PTR]
    last_jump, last_jump_t = 0, 0.0
    abort = False
    try:
      while True:
        op = _u8(bios, ptr[0]); ptr[0] += 1
        if os.environ.get("AMD_ATOM_TRACE"):
          print(f"  atom op={op} @{ptr[0]-1:#x}", flush=True)
        if op == 0 or op >= ATOM_OP_CNT:
          break
        if abort:
          raise RuntimeError(f"ATOM abort at {ptr[0]-1:#x}")
        try:
          if 1 <= op <= 6:
            self._op_move(op - 1, ptr)
          elif 7 <= op <= 12:
            self._op_bin(lambda a, b: a & b, op - 7, ptr)
          elif 13 <= op <= 18:
            self._op_bin(lambda a, b: a | b, op - 13, ptr)
          elif 19 <= op <= 24:
            self._op_shift(lambda v, s: v << s, op - 19, ptr)
          elif 25 <= op <= 30:
            self._op_shift(lambda v, s: v >> s, op - 25, ptr)
          elif 31 <= op <= 36:
            self._op_mul(op - 31, ptr)
          elif 37 <= op <= 42:
            self._op_div(op - 37, ptr)
          elif 43 <= op <= 48:
            self._op_bin(lambda a, b: a + b, op - 43, ptr)
          elif 49 <= op <= 54:
            self._op_bin(lambda a, b: a - b, op - 49, ptr)
          elif 92 <= op <= 97:
            self._op_mask(op - 92, ptr)
          elif op == 55:
            port = _u16(bios, ptr[0]); ptr[0] += 2
            g.io_mode = ATOM_IO_MM if port == 0 else (ATOM_IO_IIO | port)
          elif op == 56:
            g.io_mode = ATOM_IO_PCI; ptr[0] += 1
          elif op == 57:
            g.io_mode = ATOM_IO_SYSIO; ptr[0] += 1
          elif op == 58:
            g.reg_block = _u16(bios, ptr[0]); ptr[0] += 2
          elif op == 59:
            attr = _u8(bios, ptr[0]); ptr[0] += 1
            g.fb_base = self._get_src(attr, ptr)
          elif 60 <= op <= 65:
            self._op_compare(op - 60, ptr)
          elif op == 66:
            self._op_switch(ptr)
          elif 67 <= op <= 73:
            target = _u16(bios, ptr[0]); ptr[0] += 2
            abs_t = base + target
            cond = self._jump_cond(ATOM_JUMP_OP_COND[op - 67])
            now = time.monotonic()
            backward = abs_t < ptr[0]
            stuck_t = backward and last_jump == abs_t and now - last_jump_t > ATOM_CMD_TIMEOUT_SEC
            if cond:
              cnt = self.card._jump_counts.get(abs_t, 0) + 1
              self.card._jump_counts[abs_t] = cnt
              stuck_n = backward and cnt > ATOM_JUMP_MAX_ITERS
              if (stuck_t or stuck_n) and os.environ.get("AMD_ATOM_JUMP_BAIL", "1") != "0":
                if self.debug or os.environ.get("AMD_ATOM_TRACE"):
                  why = "iters" if stuck_n else "time"
                  print(f"  atom: bail stuck jump op={op} ({why}) fall through", flush=True)
                self.card._jump_counts[abs_t] = 0
                last_jump = 0
              elif stuck_t and ATOM_CMD_TIMEOUT_SEC > 0:
                if self.debug:
                  print(f"  atom: jump loop timeout target={abs_t:#x}", flush=True)
                raise RuntimeError("ATOM jump loop timeout")
              else:
                if last_jump != abs_t:
                  last_jump, last_jump_t = abs_t, now
                ptr[0] = abs_t
          elif 74 <= op <= 79:
            self._op_test(op - 74, ptr)
          elif op == 80:
            time.sleep(_u8(bios, ptr[0]) / 1e3); ptr[0] += 1
          elif op == 81:
            time.sleep(_u8(bios, ptr[0]) / 1e6); ptr[0] += 1
          elif op == 82:
            idx = _u8(bios, ptr[0]); ptr[0] += 1
            if _u16(bios, g.cmd_table + 4 + 2 * idx):
              off = min(ps_shift, ps_size)
              if self._execute_locked(idx, self.ps[off:], ps_size - off):
                abort = True
          elif op == 83:
            ptr[0] += 1  # REPEAT — unimplemented in linux too
          elif 84 <= op <= 89:
            self._op_clear(op - 84, ptr)
          elif op == 90:
            pass
          elif op == ATOM_OP_EOT:
            break
          elif op == 102:
            idx = _u8(bios, ptr[0]); ptr[0] += 1
            g.data_block = 0 if not idx else (base if idx == 255 else _u16(bios, g.data_table + 4 + 2 * idx))
          elif op == 98:
            ptr[0] += 1  # POSTCARD
          elif op == 99:
            pass  # BEEP
          elif op in (100, 101):
            pass  # SAVEREG/RESTOREREG unimplemented in linux too
          elif 103 <= op <= 108:
            self._op_bin(lambda a, b: a ^ b, op - 103, ptr)
          elif 109 <= op <= 114:
            self._op_shl_shr(op - 109, ptr, shl=True)
          elif 115 <= op <= 120:
            self._op_shl_shr(op - 115, ptr, shl=False)
          elif op == 122:
            skip = _u16(bios, ptr[0]); ptr[0] += skip + 2  # PROCESSDS
          elif op == 121:
            ptr[0] += 1  # DEBUG
          elif op in (123, 124):
            self._op_mul32(op - 122, ptr)  # MUL32 PS/WS
          elif op in (125, 126):
            self._op_div32(op - 124, ptr)  # DIV32 PS/WS
          elif self.debug or os.environ.get("AMD_ATOM_TRACE"):
            print(f"  atom: unhandled op {op} @ {ptr[0]-1:#x}", flush=True)
            raise RuntimeError(f"ATOM unhandled op {op}")
        except RuntimeError:
          raise
    finally:
      self.ps, self.ps_size, self.ws, self._base = saved_ps, saved_ps_size, saved_ws, saved_base
      g.execute_depth -= 1
    return 0

  def _jump_cond(self, cond: int) -> bool:
    g = self.ctx
    return {
      ATOM_COND_ABOVE: g.cs_above,
      ATOM_COND_ABOVEOREQUAL: g.cs_above or g.cs_equal,
      ATOM_COND_ALWAYS: True,
      ATOM_COND_BELOW: not (g.cs_above or g.cs_equal),
      ATOM_COND_BELOWOREQUAL: not g.cs_above,
      ATOM_COND_EQUAL: g.cs_equal,
      ATOM_COND_NOTEQUAL: not g.cs_equal,
    }[cond]

  def _op_move(self, arg: int, ptr: list[int]):
    attr = _u8(self.ctx.bios, ptr[0]); ptr[0] += 1
    saved = [0]
    dptr = ptr[0]
    if ((attr >> 3) & 7) != ATOM_SRC_DWORD:
      self._get_dst(arg, attr, ptr, saved)
    else:
      self._skip_dst(arg, attr, ptr)
      saved[0] = 0xCDCDCDCD
    val = self._get_src(attr, ptr)
    self._put_dst(arg, attr, [dptr], val, saved[0])

  def _op_bin(self, fn, arg: int, ptr: list[int]):
    attr = _u8(self.ctx.bios, ptr[0]); ptr[0] += 1
    saved = [0]
    dptr = ptr[0]
    dst = self._get_dst(arg, attr, ptr, saved)
    src = self._get_src(attr, ptr)
    self._put_dst(arg, attr, [dptr], fn(dst, src), saved[0])

  def _op_mask(self, arg: int, ptr: list[int]):
    attr = _u8(self.ctx.bios, ptr[0]); ptr[0] += 1
    saved = [0]
    dptr = ptr[0]
    dst = self._get_dst(arg, attr, ptr, saved)
    mask = self._get_src(attr, ptr)
    src = self._get_src(attr, ptr)
    self._put_dst(arg, attr, [dptr], (dst & mask) | src, saved[0])

  def _op_shift(self, fn, arg: int, ptr: list[int]):
    attr = _u8(self.ctx.bios, ptr[0]); ptr[0] += 1
    attr = (attr & 0x38) | (ATOM_DEF_DST[(attr >> 3) & 7] << 6)
    saved = [0]
    dptr = ptr[0]
    dst = self._get_dst(arg, attr, ptr, saved)
    shift = self._get_src_direct(ATOM_SRC_BYTE0, ptr)
    self._put_dst(arg, attr, [dptr], fn(dst, shift), saved[0])

  def _op_mul(self, arg: int, ptr: list[int]):
    attr = _u8(self.ctx.bios, ptr[0]); ptr[0] += 1
    dst = self._get_dst(arg, attr, ptr, None)
    src = self._get_src(attr, ptr)
    self.ctx.divmul[0] = (dst * src) & 0xffffffff

  def _op_div(self, arg: int, ptr: list[int]):
    attr = _u8(self.ctx.bios, ptr[0]); ptr[0] += 1
    dst = self._get_dst(arg, attr, ptr, None)
    src = self._get_src(attr, ptr)
    if src:
      self.ctx.divmul[0] = dst // src
      self.ctx.divmul[1] = dst % src
    else:
      self.ctx.divmul = [0, 0]

  def _op_test(self, arg: int, ptr: list[int]):
    attr = _u8(self.ctx.bios, ptr[0]); ptr[0] += 1
    dst = self._get_dst(arg, attr, ptr, None)
    src = self._get_src(attr, ptr)
    self.ctx.cs_equal = ((dst & src) == 0)

  def _op_shl_shr(self, arg: int, ptr: list[int], shl: bool):
    attr = _u8(self.ctx.bios, ptr[0]); ptr[0] += 1
    saved = [0]
    dptr = ptr[0]
    dst_align = ATOM_DST_TO_SRC[(attr >> 3) & 7][(attr >> 6) & 3]
    dst = self._get_dst(arg, attr, ptr, saved)
    full = saved[0]
    shift = self._get_src(attr, ptr)
    if shl:
      full = (full << shift) & ATOM_ARG_MASK[dst_align]
    else:
      full = (full >> shift) & ATOM_ARG_MASK[dst_align]
    full >>= ATOM_ARG_SHIFT[dst_align]
    self._put_dst(arg, attr, [dptr], full, saved[0])

  def _op_compare(self, arg: int, ptr: list[int]):
    attr = _u8(self.ctx.bios, ptr[0]); ptr[0] += 1
    dst = self._get_dst(arg, attr, ptr, None)
    src = self._get_src(attr, ptr)
    self.ctx.cs_equal = dst == src
    self.ctx.cs_above = dst > src

  def _op_clear(self, arg: int, ptr: list[int]):
    attr = _u8(self.ctx.bios, ptr[0]); ptr[0] += 1
    attr = (attr & 0x38) | (ATOM_DEF_DST[(attr >> 3) & 7] << 6)
    saved = [0]
    dptr = ptr[0]
    self._get_dst(arg, attr, ptr, saved)
    self._put_dst(arg, attr, [dptr], 0, saved[0])

  def _op_switch(self, ptr: list[int]):
    attr = _u8(self.ctx.bios, ptr[0]); ptr[0] += 1
    src = self._get_src(attr, ptr)
    while _u16(self.ctx.bios, ptr[0]) != ATOM_CASE_END:
      if _u8(self.ctx.bios, ptr[0]) == ATOM_CASE_MAGIC:
        ptr[0] += 1
        val = self._get_src_int((attr & 0x38) | ATOM_ARG_IMM, ptr, None, True)
        target = _u16(self.ctx.bios, ptr[0]); ptr[0] += 2
        if val == src:
          ptr[0] = self._base + target
          return
      else:
        return  # bad case — stop (matches linux)
    ptr[0] += 2

  def execute_table(self, index: int, ps: list[int] | None = None, ps_size: int = 16) -> int:
    g = self.ctx
    g.data_block = g.reg_block = g.fb_base = 0
    g.io_mode = ATOM_IO_MM
    g.divmul = [0, 0]
    ps = list(ps or [0] * 16)
    return self._execute_locked(index, ps, ps_size)


def atom_asic_init(boot: "PolarisBoot", bios: bytes | None = None, debug: bool = False) -> AtomContext:
  """amdgpu_atom_asic_init — execute ATOM_CMD_INIT."""
  if bios is None:
    bios = read_vbios_rom(boot)
  ctx = parse_atom_context(bios)
  card = AtomCard(boot, debug=debug)
  exe = AtomExecutor(ctx, card)
  hwi = _u16(bios, ctx.data_table + ATOM_DATA_FWI_PTR)
  ps = [0] * 16
  ps[0] = _u32(bios, hwi + ATOM_FWI_DEFSCLK_PTR)
  ps[1] = _u32(bios, hwi + ATOM_FWI_DEFMCLK_PTR)
  if not ps[0] or not ps[1]:
    raise RuntimeError("ATOM firmware info missing def sclk/mclk")
  if not _u16(bios, ctx.cmd_table + 4 + 2 * ATOM_CMD_INIT):
    raise RuntimeError("ATOM_CMD_INIT table missing")
  if debug:
    print(f"atom: executing asic_init ps0={ps[0]:#x} ps1={ps[1]:#x}", flush=True)
  ret = exe.execute_table(ATOM_CMD_INIT, ps, 16)
  if ret:
    raise RuntimeError(f"atom asic_init failed ret={ret}")
  boot.mmio_sync_safe()
  if debug:
    mem = boot.rreg(0x150a) & 0xffff
    misc0 = boot.rreg(0xa80)
    bar0_probe = boot.dev.vram[0:4] if hasattr(boot.dev, 'vram') else b''
    print(f"atom: asic_init done writes={ctx.reg_write_count} MEMSIZE={mem:#x} "
          f"MISC0={misc0:#x} pci={boot.dev.pci.read_config(0,2)&0xffff:#06x}", flush=True)
  return ctx


def replay_trace_json(boot: "PolarisBoot", path: str, debug: bool = False):
  """Path B: replay Linux-captured register sequence."""
  data = json.loads(open(path).read())
  for ent in data.get("regs", []):
    reg, val = int(ent["reg"]), int(ent["val"])
    if debug:
      print(f"  replay WREG {reg:#06x} = {val:#010x}", flush=True)
    boot.wreg(reg, val)
    delay_ms = ent.get("delay_ms", 0)
    if delay_ms:
      time.sleep(delay_ms / 1000)
  boot.mmio_sync_safe()


def run_asic_init_if_needed(boot: "PolarisBoot") -> bool:
  """Run ATOM asic_init or JSON replay when VRAM is not trained."""
  debug = int(os.environ.get("DEBUG", "0"))
  trace = os.environ.get("AMD_BOOT_ATOM_REPLAY")
  if trace:
    if debug:
      print(f"polaris: replaying asic_init trace {trace}", flush=True)
    replay_trace_json(boot, trace, debug=bool(debug))
    return True
  if os.environ.get("AMD_BOOT_ATOM_INIT", "1") == "0":
    return False
  if not need_asic_init(boot):
    if debug:
      print("polaris: skip atom asic_init (VRAM trained)", flush=True)
    return False
  clear_asic_init_scratch(boot)
  dump = os.environ.get("AMD_BOOT_DUMP_VBIOS")
  bios = read_vbios_rom(boot)
  if dump:
    open(dump, "wb").write(bios)
    if debug:
      print(f"polaris: dumped VBIOS {len(bios)} bytes → {dump}", flush=True)
  if debug:
    print(f"polaris: atom info {atom_info(bios)}", flush=True)
  atom_asic_init(boot, bios=bios, debug=bool(debug))
  return True
