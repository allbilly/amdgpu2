"""Polaris10 (RX570 / gfx803) firmware boot for macOS eGPU via TinyGPU.

Ports the VI (gfx8) bring-up path from linux amdgpu:
  vi_common_init → polaris10_start_smu → gmc_v8_0_mc_program → MC ucode →
  smu7_request_smu_load_fw → gmc_v8_0_gart_enable → gfx_v8_0_kcq"""
from __future__ import annotations
import os, struct, math, time, contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from add import PolarisDevice

# smu_7_1_3_d.h (polaris10_smumgr.c) / gmc_8_1_d.h / gfx_8_0_d.h

mmSMC_IND_ACCESS_CNTL = 0x92
mmSMC_MESSAGE_0 = 0x94
mmSMC_RESP_0 = 0x95
mmSMC_MSG_ARG_0 = 0xa4
mmSMC_IND_INDEX_11 = 0x1ac
mmSMC_IND_DATA_11 = 0x1ad
mmMC_SEQ_MISC0 = 0xa80
mmMC_SEQ_IO_DEBUG_INDEX = 0xa29
mmMC_SEQ_IO_DEBUG_DATA = 0xa2a
mmMC_SEQ_SUP_CNTL = 0xa2f
mmMC_SEQ_SUP_PGM = 0xa33
mmSRBM_GFX_CNTL = 0x391
mmGRBM_STATUS = 0x2004
mmSRBM_STATUS = 0x0e50
mmSRBM_STATUS = 0x0e50
mmCP_ME_CNTL = 0x2086
mmCP_MEC_CNTL = 0x208d
mmCP_HQD_ACTIVE = 0x3247
mmCP_PQ_STATUS = 0x2147
mmCP_MEC_DOORBELL_RANGE_LOWER = 0x2149
mmCP_MEC_DOORBELL_RANGE_UPPER = 0x214a
mmRLC_CNTL = 0x21c0
mmRLC_CP_SCHEDULERS = 0x21c1
mmCONFIG_MEMSIZE = 0x150a
mmMC_VM_FB_LOCATION = 0x809
mmMC_VM_SYSTEM_APERTURE_LOW_ADDR = 0x80d
mmMC_VM_SYSTEM_APERTURE_HIGH_ADDR = 0x80e
mmMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR = 0x80f
mmMC_VM_AGP_BASE = 0x810
mmMC_VM_AGP_TOP = 0x811
mmMC_VM_AGP_BOT = 0x812
mmBIF_FB_EN = 0x1524  # bif_5_0_d.h (gmc_v8_0.c), not 0x1024
mmMM_INDEX = 0x0
mmMM_INDEX_HI = 0x6
mmMM_DATA = 0x1
mmHDP_MISC_CNTL = 0x1544
mmHDP_HOST_PATH_CNTL = 0x1555
mmHDP_REG_COHERENCY_FLUSH_CNTL = 0x155a
mmHDP_MEM_COHERENCY_FLUSH_CNTL = 0x1520
mmHDP_DEBUG0 = 0xbcc
mmHDP_NONSURFACE_BASE = 0xb01
mmHDP_NONSURFACE_INFO = 0xb02
mmHDP_NONSURFACE_SIZE = 0xb03
mmMC_VM_MX_L1_TLB_CNTL = 0x518
mmVM_L2_CNTL = 0x500
mmVM_L2_CNTL2 = 0x501
mmVM_L2_CNTL3 = 0x502
mmVM_L2_CNTL4 = 0x503
mmVM_CONTEXT0_PAGE_TABLE_START_ADDR = 0x557
mmVM_CONTEXT0_PAGE_TABLE_END_ADDR = 0x55f
mmVM_CONTEXT0_PAGE_TABLE_BASE_ADDR = 0x54f
mmVM_CONTEXT0_PROTECTION_FAULT_DEFAULT_ADDR = 0x546
mmVM_CONTEXT0_CNTL2 = 0x50c
mmVM_CONTEXT0_CNTL = 0x504
mmVM_CONTEXT1_PAGE_TABLE_START_ADDR = 0x558
mmVM_CONTEXT1_PAGE_TABLE_END_ADDR = 0x560
mmVM_CONTEXT1_PROTECTION_FAULT_DEFAULT_ADDR = 0x547
mmVM_CONTEXT1_CNTL2 = 0x50d
mmVM_CONTEXT1_CNTL = 0x505
mmVM_INVALIDATE_REQUEST = 0x51e
mmCP_MQD_BASE_ADDR = 0x3245
mmCP_HQD_VMID = 0x3248
mmCP_PQ_WPTR_POLL_CNTL = 0x2148

ixSMC_PC_C = 0x80000370
ixFIRMWARE_FLAGS = 0x3f000
ixRCU_UC_EVENTS = 0xc0000004
ixSMU_STATUS = 0xe0003088
ixSMU_FIRMWARE = 0xe00030a4
ixSMU_INPUT_DATA = 0xe00030b8
SMC_SYSCON_RESET_CNTL = 0x80000000
SMC_SYSCON_CLOCK_CNTL_0 = 0x80000004

PPSMC_MSG_DRV_DRAM_ADDR_HI = 0x250
PPSMC_MSG_DRV_DRAM_ADDR_LO = 0x251
PPSMC_MSG_SMU_DRAM_ADDR_HI = 0x252
PPSMC_MSG_SMU_DRAM_ADDR_LO = 0x253
PPSMC_MSG_LoadUcodes = 0x254
PPSMC_MSG_Test = 0x200

# smu_ucode_xfer_vi.h — UCODE_ID values and load masks (not bit position of id)
UCODE_ID_SDMA0 = 1
UCODE_ID_SDMA1 = 2
UCODE_ID_CP_CE = 3
UCODE_ID_CP_PFP = 4
UCODE_ID_CP_ME = 5
UCODE_ID_CP_MEC = 6
UCODE_ID_CP_MEC_JT1 = 7
UCODE_ID_CP_MEC_JT2 = 8
UCODE_ID_RLC_G = 10

UCODE_ID_SDMA0_MASK = 0x00000002
UCODE_ID_SDMA1_MASK = 0x00000004
UCODE_ID_CP_CE_MASK = 0x00000008
UCODE_ID_CP_PFP_MASK = 0x00000010
UCODE_ID_CP_ME_MASK = 0x00000020
UCODE_ID_CP_MEC_MASK = 0x00000040
UCODE_ID_CP_MEC_JT1_MASK = 0x00000080
UCODE_ID_CP_MEC_JT2_MASK = 0x00000100
UCODE_ID_RLC_G_MASK = 0x00000400
FW_TO_LOAD = (UCODE_ID_RLC_G_MASK | UCODE_ID_SDMA0_MASK | UCODE_ID_SDMA1_MASK |
              UCODE_ID_CP_CE_MASK | UCODE_ID_CP_ME_MASK | UCODE_ID_CP_PFP_MASK | UCODE_ID_CP_MEC_MASK)

SMU_FW_BUF_SIZE = 200 * 4096
SMU_HDR_BUF_SIZE = 4096
PAGE_SIZE = 4096

# gmc_v8_0: VALID|SYSTEM|EXECUTABLE|READABLE|WRITEABLE (amdgpu_ttm_tt_pte_flags)
GART_PTE_FLAGS = 0x73

SMU_HDR_SOFT_REGS_OFF = 0x20000 + 48  # offsetof(SMU74_Firmware_Header, SoftRegisters)
ixSMU74_UcodeLoadStatus = 0x6c
mmBIF_DOORBELL_APER_EN = 0x1501
mmBIF_MM_INDACCESS_CNTL = 0x1500
mmBUS_CNTL = 0x1508
ixROM_CNTL = 0xc0600000
ixROM_INDEX = 0xc0600010
ixROM_DATA = 0xc0600014
ROM_SCK_OVERWRITE = 0x2
BUS_BIOS_ROM_DIS = 0x2
BOOT_SEQ_DONE = 0x80
INTERRUPTS_ENABLED = 0x1
RCU_INTERRUPTS_ENABLED = 0x10000
SMU_MODE_PROT = 0x10000

DOORBELL_KIQ = 0x0
DOORBELL_MEC_RING0 = 0x10
GFX8_MEC_HPD_SIZE = 4096
RING_SIZE = 0x10000
PACKET3_MAP_QUEUES = 0xA2
PACKET3_SET_RESOURCES = 0xA1
PKT_TYPE3 = 3

GMC_GOLDEN_REGS = [
  (0x2768, 0x3, 0x0),  # mmMC_ARB_WTM_GRPWT_RD
  (0x2420, 0x0fffffff, 0x0fffffff),  # mmVM_PRT_APERTURE0_LOW_ADDR
  (0x2428, 0x0fffffff, 0x0fffffff),
  (0x2430, 0x0fffffff, 0x0fffffff),
  (0x2438, 0x0fffffff, 0x0fffffff),
]

# gfx_v8_0.c polaris10 golden tables (reg, mask, val)
GOLDEN_REGS = GMC_GOLDEN_REGS + [
  (3284, 790464, 786944),
  (9860, 127951, 29192),
  (9862, 251658240, 251658240),
  (9859, 511, 64),
  (9741, 4027580415, 1024),
  (8956, 4294967295, 536870913),
  (49793, 65295, 0),
  (41172, 4294967295, 369098770),
  (41173, 4294967295, 42),
  (60489, 3, 65596),
  (60573, 4294967295, 65596),
  (8960, 133693440, 119013376),
  (9538, 983055, 720896),
  (11136, 1048576, 4078960511),
  (11013, 1023, 247),
  (11012, 4294967295, 0),
  (8754, 4, 4),
  (49664, 4294967295, 3758096384),
  (9790, 4294967295, 570494979),
  (12764, 4294967295, 2048),
  (12765, 4294967295, 2048),
  (12774, 4294967295, 16744383),
  (12775, 4294967295, 16744367),
  (2529, 3, 0),
  (1324, 268435455, 268435455),
  (1325, 268435455, 268435455),
  (1326, 268435455, 268435455),
  (1327, 268435455, 268435455),
]

def _le32(b: bytes, off: int) -> int:
  return struct.unpack_from('<I', b, off)[0]

def parse_common_fw(blob: bytes) -> tuple[int, int]:
  return _le32(blob, 24), _le32(blob, 20)

def parse_mc_fw(blob: bytes) -> tuple[int, int, int, int]:
  """io_debug_off, io_debug_sz_bytes, ucode_off, ucode_sz."""
  io_dbg_sz = _le32(blob, 32)
  io_dbg_off = _le32(blob, 36)
  ucode_off, ucode_sz = parse_common_fw(blob)
  return io_dbg_off, io_dbg_sz, ucode_off, ucode_sz

def parse_gfx_fw(blob: bytes) -> tuple[int, int, int, int, int]:
  """ucode_off, ucode_sz, ucode_ver, jt_offset_dwords, jt_size_dwords."""
  ucode_off, ucode_sz = parse_common_fw(blob)
  ucode_ver = _le32(blob, 16) & 0xffff
  jt_off = _le32(blob, 36)
  jt_sz = _le32(blob, 40)
  return ucode_off, ucode_sz, ucode_ver, jt_off, jt_sz

def pack_smu_toc_entry(ucode_id: int, version: int, mc_addr: int, data_size: int, flags: int = 0) -> bytes:
  return struct.pack('<HHIIIIIHH', ucode_id, version,
                     (mc_addr >> 32) & 0xffffffff, mc_addr & 0xffffffff,
                     0, 0, data_size, flags, 0)

def round_up(n: int, a: int) -> int:
  return ((n + a - 1) // a) * a

def order_base_2(x: int) -> int:
  return int(math.log2(x))

def pkt3(op: int, count: int, predicate: int = 0) -> int:
  return (PKT_TYPE3 << 30) | ((count & 0x3fff) << 16) | ((op & 0xff) << 8) | (predicate & 1)


class PolarisBoot:
  def __init__(self, dev: 'PolarisDevice'):
    self.dev = dev
    self.mm = dev.mmio
    self._fw_cache: dict[str, bytes] = {}
    self.vram_start = 0
    self.vram_end = 0
    self.vram_size = 0
    self.gart_start = 0
    self.gart_end = 0
    self.gart_size = 256 * 1024 * 1024
    self.agp_start = 0
    self.agp_size = 0
    self.agp_end = 0
    self.gart_base = 0
    self.gart_pte_off = 0
    self.gart_pte_mem: bytearray | None = None
    self.gart_pte_sysmem = None
    self.gart_pte_phys = 0
    self.vram_visible_mc = 0
    self._gart_alloc_off = 0x100000
    self._compute: ComputeQueue | None = None

  def pci_online(self) -> bool:
    try:
      return (self.dev.pci.read_config(0, 2) & 0xffff) == 0x1002
    except Exception:
      return False

  def _check_pci(self, phase: str):
    if not self.pci_online():
      raise RuntimeError(
        f"GPU fell off PCIe during {phase} — replug USB4, then: python3 add.py --reset. "
        f"Try larger AMD_BOOT_SMC_SYNC (fewer SMC reads during upload) or "
        f"AMD_BOOT_SMC_POLL_MS (slower post-upload polling)."
      )

  def _poll_s(self) -> float:
    return max(0.001, int(os.environ.get("AMD_BOOT_SMC_POLL_MS", "25"))) / 1000.0

  def _timeout_s(self, key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))

  def _settle_s(self) -> float:
    return float(os.environ.get("AMD_BOOT_SMC_SETTLE_MS", "250")) / 1000.0

  def wait_for(self, phase: str, predicate, timeout_s: float | None = None) -> bool:
    deadline = time.time() + (timeout_s if timeout_s is not None else self._timeout_s("AMD_BOOT_SMC_TIMEOUT_S", 60.0))
    delay = self._poll_s()
    while time.time() < deadline:
      self._check_pci(phase)
      if predicate():
        return True
      time.sleep(delay)
      delay = min(delay * 1.5, 0.25)
    return False

  def rreg(self, reg: int) -> int:
    # mmio is numpy int32; shifts like (val & 0xffff) << 24 overflow to 0.
    return int(self.mm[reg]) & 0xffffffff

  def wreg(self, reg: int, val: int):
    self.mm[reg] = int(val) & 0xffffffff

  def mmio_sync(self):
    self.smc_rreg(ixSMC_PC_C)
    pause_ms = int(os.environ.get("AMD_BOOT_SMC_PC_PAUSE_MS", "15"))
    if pause_ms > 0:
      time.sleep(pause_ms / 1000.0)

  def mmio_sync_ind_port(self):
    self.rreg(mmSMC_IND_ACCESS_CNTL)
    self.rreg(mmSMC_IND_INDEX_11)

  def mmio_sync_smc_data(self, addr: int = 0x20000):
    self.wreg(mmSMC_IND_ACCESS_CNTL, 0)
    self.wreg(mmSMC_IND_INDEX_11, addr)
    self.rreg(mmSMC_IND_DATA_11)

  def mmio_sync_safe(self):
    with contextlib.suppress(Exception):
      self.dev.pci.drain_mmio(bar=5, reg=mmGRBM_STATUS)
    self.mmio_sync_ind_port()

  def smc_rreg(self, reg: int) -> int:
    self.wreg(mmSMC_IND_INDEX_11, reg)
    return self.rreg(mmSMC_IND_DATA_11)

  def smc_wreg(self, reg: int, val: int):
    self.wreg(mmSMC_IND_INDEX_11, reg)
    self.wreg(mmSMC_IND_DATA_11, val)
    self.mmio_sync()

  def smc_wreg_safe(self, reg: int, val: int):
    self.wreg(mmSMC_IND_INDEX_11, reg)
    self.wreg(mmSMC_IND_DATA_11, val)
    self.mmio_sync_safe()

  def smc_field(self, reg: int, shift: int, width: int, val: int):
    mask = (1 << width) - 1
    old = self.smc_rreg(reg)
    self.smc_wreg(reg, (old & ~(mask << shift)) | ((val & mask) << shift))

  def srbm_select(self, me=0, pipe=0, queue=0, vmid=0):
    self.wreg(mmSRBM_GFX_CNTL, (queue & 7) | ((me & 3) << 4) | ((pipe & 3) << 8) | ((vmid & 0xf) << 16))


  def fw(self, name: str) -> bytes:
    if name not in self._fw_cache:
      import urllib.request, pathlib
      cache_dir = pathlib.Path.home() / ".cache" / "tinygrad" / "fw"
      cache_dir.mkdir(parents=True, exist_ok=True)
      fp = cache_dir / name
      if fp.is_file():
        self._fw_cache[name] = fp.read_bytes()
      else:
        url = f"https://gitlab.com/kernel-firmware/linux-firmware/-/raw/main/amdgpu/{name}"
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "amdgpu-egpu"}), timeout=60) as r:
          self._fw_cache[name] = r.read()
        fp.write_bytes(self._fw_cache[name])
    return self._fw_cache[name]

  def _smc_val_ok(self, val: int) -> bool:
    return val not in (0, 0xffffffff, 0xaaaa5555, 0x5555aaaa)

  def smc_read(self, reg: int) -> int | None:
    val = self.smc_rreg(reg)
    return val if self._smc_val_ok(val) else None

  def smc_fw_name(self) -> str:
    override = os.environ.get("AMD_SMC_FW", "")
    if override:
      return override
    fw = self.smc_read(ixSMU_FIRMWARE)
    sel = ((fw >> 17) & 1) if fw is not None else 0
    base = "polaris10_smc.bin" if sel else "polaris10_smc_sk.bin"
    if os.environ.get("AMD_SMC_FW_K", "0") == "1":
      return base.replace("polaris10_", "polaris10_k_")
    return base

  def smc_running(self) -> bool:
    pc = self.smc_rreg(ixSMC_PC_C)
    if not self._smc_val_ok(pc) or not (0x20100 <= pc < 0xffffff00):
      return False
    clk = self.smc_read(SMC_SYSCON_CLOCK_CNTL_0)
    if clk is None:
      return True
    return (clk & 1) == 0

  def smc_set_reset(self, asserted: bool):
    val = self.smc_read(SMC_SYSCON_RESET_CNTL)
    if val is None:
      val = 0x40000000
    val = (val | 1) if asserted else (val & ~1)
    self.smc_wreg_safe(SMC_SYSCON_RESET_CNTL, val)

  def smc_set_clock(self, enabled: bool):
    val = self.smc_read(SMC_SYSCON_CLOCK_CNTL_0)
    if val is None:
      val = 0x1000000
    val = (val & ~1) if enabled else (val | 1)
    self.smc_wreg_safe(SMC_SYSCON_CLOCK_CNTL_0, val)

  def _upload_barrier(self, dword_idx: int, sync_every: int):
    if os.environ.get("AMD_BOOT_SMC_DRAIN_ALL", "0") == "1":
      self.mmio_sync_safe()
      return
    if sync_every and dword_idx and (dword_idx % sync_every) == 0:
      self.mmio_sync_safe()

  def smc_firmware_ready(self) -> bool:
    if self.smc_running():
      return True
    if os.environ.get("AMD_BOOT_SMC_USE_FLAGS", "0") != "1":
      return False
    flags = self.smc_read(ixFIRMWARE_FLAGS)
    return flags is not None and bool(flags & INTERRUPTS_ENABLED)

  def smc_wait_firmware_ready(self, phase: str, timeout_s: float | None = None) -> bool:
    return self.wait_for(phase, self.smc_firmware_ready, timeout_s=timeout_s)

  def smc_flush_upload(self, addr: int = 0x20000):
    self.mmio_sync_ind_port()
    if os.environ.get("AMD_BOOT_SMC_FLUSH_READ", "0") == "1":
      self.mmio_sync_smc_data(addr)
    with contextlib.suppress(Exception):
      self.dev.pci.drain_mmio(bar=5, reg=mmGRBM_STATUS)

  def smc_set_sram_addr(self, addr: int):
    self.wreg(mmSMC_IND_INDEX_11, addr)
    acc = self.rreg(mmSMC_IND_ACCESS_CNTL)
    self.wreg(mmSMC_IND_ACCESS_CNTL, acc & ~0x800)

  def smc_copy_bytes(self, addr: int, data: bytes):
    for i in range(0, len(data), 4):
      chunk = data[i:i + 4]
      while len(chunk) < 4:
        chunk += b'\0'
      val = (chunk[0] << 24) | (chunk[1] << 16) | (chunk[2] << 8) | chunk[3]
      self.smc_set_sram_addr(addr + i)
      self.wreg(mmSMC_IND_DATA_11, val)
      self.mmio_sync_ind_port()

  def upload_smc_image_bench(self, blob: bytes, sync_every_dwords: int = 32):
    self.wreg(mmSMC_IND_INDEX_11, 0x20000)
    acc = self.rreg(mmSMC_IND_ACCESS_CNTL)
    self.wreg(mmSMC_IND_ACCESS_CNTL, acc | 0x800)
    for i in range(0, len(blob), 4):
      self.wreg(mmSMC_IND_DATA_11, _le32(blob, i))
      self._upload_barrier(i // 4, sync_every_dwords)
    self.wreg(mmSMC_IND_ACCESS_CNTL, acc & ~0x800)
    self.mmio_sync_safe()

  def upload_smc_image_chunked(self, blob: bytes, chunk_dwords: int = 32, sync_every_dwords: int = 64):
    acc = self.rreg(mmSMC_IND_ACCESS_CNTL)
    self.wreg(mmSMC_IND_ACCESS_CNTL, acc | 0x800)
    for i in range(0, len(blob), chunk_dwords * 4):
      chunk = blob[i:i + chunk_dwords * 4]
      self.wreg(mmSMC_IND_INDEX_11, 0x20000 + i)
      for j in range(0, len(chunk), 4):
        self.wreg(mmSMC_IND_DATA_11, _le32(chunk, j))
        if sync_every_dwords and ((j // 4) % sync_every_dwords == 0) and j:
          self.mmio_sync_ind_port()
      self.mmio_sync_ind_port()
    self.wreg(mmSMC_IND_ACCESS_CNTL, acc & ~0x800)
    self.mmio_sync_ind_port()

  def upload_smc_image_per_addr(self, blob: bytes, sync_every_dwords: int = 64):
    sync_every = int(os.environ.get("AMD_BOOT_SMC_SYNC", str(sync_every_dwords)))
    for i in range(0, len(blob) & ~3, 4):
      self.smc_set_sram_addr(0x20000 + i)
      self.wreg(mmSMC_IND_DATA_11, _le32(blob, i))
      if sync_every and (i // 4) % sync_every == 0 and i:
        self.mmio_sync_safe()
    self.mmio_sync_safe()

  def upload_smc_image_pc_sync(self, blob: bytes, sync_every: int = 512):
    self.wreg(mmSMC_IND_INDEX_11, 0x20000)
    acc = self.rreg(mmSMC_IND_ACCESS_CNTL)
    self.wreg(mmSMC_IND_ACCESS_CNTL, acc | 0x800)
    for i in range(0, len(blob) & ~3, 4):
      self.wreg(mmSMC_IND_DATA_11, _le32(blob, i))
      if sync_every and (i // 4) % sync_every == 0 and i:
        self.mmio_sync()
    self.wreg(mmSMC_IND_ACCESS_CNTL, acc & ~0x800)
    self.mmio_sync_safe()

  def upload_smc_segmented(self, blob: bytes, segment_dwords: int = 4096):
    """Upload in segments: burst writes + one SMC PC barrier per segment (fewer risky reads)."""
    acc = self.rreg(mmSMC_IND_ACCESS_CNTL)
    seg_bytes = segment_dwords * 4
    nseg = (len(blob) + seg_bytes - 1) // seg_bytes
    for si, seg_start in enumerate(range(0, len(blob), seg_bytes)):
      seg = blob[seg_start:seg_start + seg_bytes]
      self.wreg(mmSMC_IND_INDEX_11, 0x20000 + seg_start)
      self.wreg(mmSMC_IND_ACCESS_CNTL, acc | 0x800)
      for j in range(0, len(seg) & ~3, 4):
        self.wreg(mmSMC_IND_DATA_11, _le32(seg, j))
      self.wreg(mmSMC_IND_ACCESS_CNTL, acc & ~0x800)
      self.mmio_sync_safe()
      self.mmio_sync()
      self._check_pci(f"SMC upload segment {si + 1}/{nseg}")
      time.sleep(self._settle_s())
    if int(os.environ.get("DEBUG", "0")):
      print(f"polaris: SMC segmented upload done ({nseg} segments x {segment_dwords} dwords)", flush=True)

  def upload_smc_firmware(self, image: bytes):
    mode = os.environ.get("AMD_BOOT_SMC_UPLOAD", "segmented")
    sync = int(os.environ.get("AMD_BOOT_SMC_SYNC", "4096"))
    if mode == "segmented":
      self.upload_smc_segmented(image, segment_dwords=sync)
    elif mode == "pc_sync":
      self.upload_smc_image_pc_sync(image, sync_every=sync)
    elif mode == "linux":
      self.upload_smc_image_bench(image, sync_every_dwords=sync)
    elif mode == "chunked":
      self.upload_smc_image_chunked(image, sync_every_dwords=sync)
    elif mode == "per_addr":
      self.upload_smc_image_per_addr(image, sync_every_dwords=sync)
    elif mode == "hybrid":
      self.upload_smc_image_chunked(image, sync_every_dwords=sync)
      if os.environ.get("AMD_BOOT_SMC_FINAL_PC_SYNC", "0") == "1":
        self.mmio_sync()
    else:
      raise RuntimeError(f"unknown AMD_BOOT_SMC_UPLOAD={mode}")
    self._check_pci("SMC upload finish")
    self.smc_flush_upload(0x20000)
    if os.environ.get("AMD_BOOT_SMC_VERIFY", "0") == "1" and not self.verify_smc_upload(image):
      raise RuntimeError("SMC firmware upload verify failed (readback mismatch)")


  def read_smc_ram(self, addr: int, ndwords: int) -> list[int]:
    out: list[int] = []
    for i in range(ndwords):
      self.wreg(mmSMC_IND_ACCESS_CNTL, 0)
      self.wreg(mmSMC_IND_INDEX_11, addr + i * 4)
      out.append(self.rreg(mmSMC_IND_DATA_11))
    return out

  def verify_smc_upload(self, blob: bytes, addr: int = 0x20000, samples: int = 16) -> bool:
    if self.read_smc_ram(addr, samples) != [_le32(blob, i * 4) for i in range(samples)]:
      return False
    mid = (len(blob) // 2) & ~3
    tail = len(blob) - 4
    for off in (mid, tail):
      if self.read_smc_ram(addr + off, 1)[0] != _le32(blob, off):
        return False
    return True

  def smc_diag(self) -> str:
    return (f"PC={self.smc_rreg(ixSMC_PC_C):#x} "
            f"FLAGS={self.smc_rreg(ixFIRMWARE_FLAGS):#x} "
            f"EVENTS={self.smc_rreg(ixRCU_UC_EVENTS):#x} "
            f"STATUS={self.smc_rreg(ixSMU_STATUS):#x} "
            f"RESP={self.rreg(mmSMC_RESP_0):#x} "
            f"CLK={self.smc_rreg(SMC_SYSCON_CLOCK_CNTL_0):#x} "
            f"RST={self.smc_rreg(SMC_SYSCON_RESET_CNTL):#x} "
            f"FW={self.smc_rreg(ixSMU_FIRMWARE):#x}")

  def smc_send_msg(self, msg: int, arg: int | None = None, *, label: str | None = None):
    self._check_pci("SMC msg prep")
    deadline = time.time() + self._timeout_s("AMD_BOOT_SMC_MSG_TIMEOUT_S", 30.0)
    delay = self._poll_s()
    while time.time() < deadline:
      resp = self.rreg(mmSMC_RESP_0) & 0xffff
      if resp not in (0, 0xffff):
        break
      time.sleep(delay)
      delay = min(delay * 1.5, 0.25)
    resp = self.rreg(mmSMC_RESP_0) & 0xffff
    if resp not in (0, 0xffff):
      self.wreg(mmSMC_RESP_0, 0)
      self.mmio_sync_safe()
    if arg is not None:
      self.wreg(mmSMC_MSG_ARG_0, arg)
    self.wreg(mmSMC_MESSAGE_0, msg)
    self.mmio_sync_safe()
    if msg == PPSMC_MSG_LoadUcodes:
      timeout_s = self._timeout_s("AMD_BOOT_SMC_LOADUCODES_TIMEOUT_S", 120.0)
      deadline = time.time() + timeout_s
      delay = self._poll_s()
      while time.time() < deadline:
        self._check_pci("SMC LoadUcodes wait")
        resp = self.rreg(mmSMC_RESP_0) & 0xffff
        status = self.smc_soft_reg(ixSMU74_UcodeLoadStatus)
        want = arg if arg is not None else FW_TO_LOAD
        if (status & want) == want:
          if int(os.environ.get("DEBUG", "0")):
            print(f"polaris: SMC LoadUcodes done via UcodeLoadStatus={status:#x}", flush=True)
          return 1
        if resp not in (0, 0xffff):
          if int(os.environ.get("DEBUG", "0")):
            print(f"polaris: SMC LoadUcodes resp={resp:#x} status={status}", flush=True)
          if resp not in (1,):
            raise RuntimeError(f"SMC msg {msg:#x} failed resp={resp:#x}")
          return resp
        time.sleep(delay)
        delay = min(delay * 1.5, 0.25)
      raise RuntimeError(
        f"SMC msg {msg:#x} timeout after {timeout_s:.0f}s "
        f"(RESP={self.rreg(mmSMC_RESP_0):#x} pci_online={self.pci_online()})"
      )
    timeout_s = self._timeout_s("AMD_BOOT_SMC_MSG_TIMEOUT_S", 30.0)
    deadline = time.time() + timeout_s
    delay = self._poll_s()
    while time.time() < deadline:
      self._check_pci("SMC msg wait")
      resp = self.rreg(mmSMC_RESP_0) & 0xffff
      if resp in (0, 0xffff):
        time.sleep(delay)
        delay = min(delay * 1.5, 0.25)
        continue
      if int(os.environ.get("DEBUG", "0")):
        tag = label or f"msg={msg:#x}"
        print(f"polaris: SMC {tag} resp={resp:#x}", flush=True)
      if resp not in (1,):
        raise RuntimeError(f"SMC msg {msg:#x} failed resp={resp:#x}")
      return resp
    raise RuntimeError(
      f"SMC msg {msg:#x} timeout after {timeout_s:.0f}s "
      f"(RESP={self.rreg(mmSMC_RESP_0):#x} pci_online={self.pci_online()})"
    )

  def apply_golden_regs(self):
    for reg, mask, val in GOLDEN_REGS:
      self.wreg(reg, (self.rreg(reg) & ~mask) | val)
    self.mmio_sync_safe()

  def vi_common_init(self):
    self.apply_golden_regs()
    tmp = self.rreg(mmBIF_DOORBELL_APER_EN)
    self.wreg(mmBIF_DOORBELL_APER_EN, (tmp & ~1) | 1)
    self.mmio_sync_safe()

  def enable_vbios_rom(self):
    if os.environ.get("AMD_BOOT_ROM_ENABLE", "1") != "1":
      return
    bus = self.rreg(mmBUS_CNTL)
    self.wreg(mmBUS_CNTL, bus & ~BUS_BIOS_ROM_DIS)
    rom = self.smc_read(ixROM_CNTL)
    self.smc_wreg_safe(ixROM_CNTL, (rom | ROM_SCK_OVERWRITE) if rom is not None else ROM_SCK_OVERWRITE)

  def vbios_rom_magic(self) -> int:
    self.wreg(mmSMC_IND_INDEX_11, ixROM_INDEX)
    self.wreg(mmSMC_IND_DATA_11, 0)
    self.wreg(mmSMC_IND_INDEX_11, ixROM_DATA)
    return self.rreg(mmSMC_IND_DATA_11)

  def wait_boot_seq_done(self, timeout_s: float = 5.0) -> bool:
    return self.wait_for("boot_seq_done",
                         lambda: (v := self.smc_read(ixRCU_UC_EVENTS)) is not None and bool(v & BOOT_SEQ_DONE),
                         timeout_s=timeout_s)

  def start_smc_non_protection(self):
    if self.smc_running() and self.smc_msg_iface_ready():
      return
    self.smc_wreg_safe(ixFIRMWARE_FLAGS, 0)
    if os.environ.get("AMD_BOOT_WAIT_BOOT_SEQ_DONE", "1") == "1":
      if not self.wait_boot_seq_done(float(os.environ.get("AMD_BOOT_BOOT_SEQ_TIMEOUT", "30"))):
        if int(os.environ.get("AMD_BOOT_STRICT_BOOT_SEQ", "0")):
          raise RuntimeError("boot_seq_done timeout")
    self.smc_set_reset(True)
    smc_blob = self.fw(self.smc_fw_name())
    ucode_off, ucode_sz = parse_common_fw(smc_blob)
    image = smc_blob[ucode_off:ucode_off + ucode_sz]
    self.upload_smc_firmware(image)
    self.smc_copy_bytes(0, bytes([0xE0, 0x00, 0x80, 0x40]))
    self.smc_flush_upload(0)
    self.smc_set_clock(True)
    self.smc_set_reset(False)
    time.sleep(self._settle_s())
    if self.smc_wait_firmware_ready("SMC firmware init (non-protection)"):
      return
    raise RuntimeError(f"SMC firmware init timeout (non-protection) {self.smc_diag()}")

  def start_smc_protection(self):
    self.smc_set_reset(True)
    smc_blob = self.fw(self.smc_fw_name())
    ucode_off, ucode_sz = parse_common_fw(smc_blob)
    image = smc_blob[ucode_off:ucode_off + ucode_sz]
    self.upload_smc_firmware(image)
    self.smc_wreg_safe(ixSMU_STATUS, 0)
    self.smc_set_clock(True)
    self.smc_set_reset(False)
    time.sleep(self._settle_s())
    if os.environ.get("AMD_BOOT_FIJI_AUTO_START", "0") == "1":
      self.smc_wreg_safe(ixSMU_INPUT_DATA, 0x80000000)
      self.smc_wreg_safe(ixFIRMWARE_FLAGS, 0)
    rcu_ok = self.wait_for("RCU_INTERRUPTS_ENABLED",
                           lambda: (v := self.smc_read(ixRCU_UC_EVENTS)) is not None and bool(v & RCU_INTERRUPTS_ENABLED),
                           timeout_s=self._timeout_s("AMD_BOOT_RCU_TIMEOUT_S", 5.0))
    if not rcu_ok:
      rcu_ok = self.wait_for("boot_seq_done",
                             lambda: (v := self.smc_read(ixRCU_UC_EVENTS)) is not None and bool(v & BOOT_SEQ_DONE),
                             timeout_s=self._timeout_s("AMD_BOOT_BOOT_SEQ_TIMEOUT", 30.0))
    if not rcu_ok and os.environ.get("AMD_BOOT_PROT_SKIP_RCU", "0") == "1":
      if int(os.environ.get("DEBUG", "0")):
        print(f"polaris: RCU_INTERRUPTS_ENABLED timeout, continuing ({self.smc_diag()})", flush=True)
    elif not rcu_ok:
      raise RuntimeError(f"SMC protection-mode RCU interrupt timeout {self.smc_diag()}")
    if int(os.environ.get("DEBUG", "0")):
      print(f"polaris: sending PPSMC_MSG_Test offset=0x20000 ({self.smc_diag()})", flush=True)
    self.smc_send_msg(PPSMC_MSG_Test, 0x20000)
    if not self.wait_for("SMU_DONE",
                         lambda: (v := self.smc_read(ixSMU_STATUS)) is not None and bool(v & 0x1),
                         timeout_s=self._timeout_s("AMD_BOOT_SMU_DONE_TIMEOUT_S", 30.0)):
      raise RuntimeError(f"SMC protection-mode SMU_DONE timeout {self.smc_diag()}")
    status = self.smc_read(ixSMU_STATUS)
    if status is None or not (status & 0x2):
      raise RuntimeError("SMC protection-mode test failed")
    self.smc_wreg_safe(ixFIRMWARE_FLAGS, 0)
    self.smc_set_reset(True)
    self.smc_set_reset(False)
    time.sleep(self._settle_s())
    if self.smc_wait_firmware_ready("SMC protection phase2"):
      return
    raise RuntimeError(f"SMC protection-mode phase2 timeout {self.smc_diag()}")

  def smc_msg_iface_ready(self) -> bool:
    resp = self.rreg(mmSMC_RESP_0) & 0xffff
    return resp not in (0, 0xffff)

  def start_smc(self):
    if self.smc_running() and self.smc_msg_iface_ready():
      return
    if self.smc_running() and int(os.environ.get("DEBUG", "0")):
      print(f"polaris: SMC PC active but msg iface stale ({self.smc_diag()}), re-booting", flush=True)
    if os.environ.get("AMD_BOOT_GOLDEN", "1") == "1":
      self.vi_common_init()
    self.enable_vbios_rom()
    if int(os.environ.get("DEBUG", "0")):
      magic = self.vbios_rom_magic()
      events = self.smc_read(ixRCU_UC_EVENTS)
      ev = f"{events:#x}" if events is not None else "garbage"
      print(f"polaris: VBIOS ROM[0]={magic:#x} EVENTS={ev}", flush=True)
    prot = os.environ.get("AMD_BOOT_SMC_PROT", "auto")
    smu_fw = self.smc_read(ixSMU_FIRMWARE)
    if prot == "auto":
      use_prot = bool(smu_fw & SMU_MODE_PROT) if smu_fw is not None else False
      if not use_prot and os.environ.get("AMD_BOOT_SMC_PREFER_PROT", "1") == "1":
        use_prot = True
    else:
      use_prot = prot == "1"
    if int(os.environ.get("DEBUG", "0")):
      fw_s = f"{smu_fw:#x}" if smu_fw is not None else "garbage"
      print(f"polaris: SMC mode={'protection' if use_prot else 'non-protection'} "
            f"SMU_FIRMWARE={fw_s} fw={self.smc_fw_name()}", flush=True)
    if use_prot:
      try:
        self.start_smc_protection()
      except RuntimeError:
        if self.smc_running() and self.smc_msg_iface_ready():
          return
        if prot == "auto" and os.environ.get("AMD_BOOT_PROT_FALLBACK", "1") == "1":
          if int(os.environ.get("DEBUG", "0")):
            print("polaris: protection-mode SMC boot failed, trying non-protection", flush=True)
          self.smc_set_reset(True)
          time.sleep(self._settle_s())
          self.start_smc_non_protection()
          return
        raise
      return
    try:
      self.start_smc_non_protection()
    except RuntimeError:
      if self.smc_running() and self.smc_msg_iface_ready():
        return
      if os.environ.get("AMD_BOOT_NO_PROT_FALLBACK", "0") == "1":
        raise
      if int(os.environ.get("DEBUG", "0")):
        print("polaris: non-protection SMC boot failed, trying protection", flush=True)
      self.start_smc_protection()


  def mc_io_debug_up13(self) -> int:
    self.wreg(mmMC_SEQ_IO_DEBUG_INDEX, 0xd)  # ixMC_IO_DEBUG_UP_13
    return self.rreg(mmMC_SEQ_IO_DEBUG_DATA)

  def config_memsize_mb(self) -> int:
    return self.rreg(mmCONFIG_MEMSIZE) & 0xffff

  def vram_trained(self) -> bool:
    """True when VBIOS/asic_init or MC ucode left VRAM in a usable state."""
    misc0 = self.rreg(mmMC_SEQ_MISC0)
    mem_mb = self.config_memsize_mb()
    if mem_mb not in (0, 0xffff) and (misc0 & 0x80):
      return True
    # bit 23 alone means MC ucode ran once, not that asic_init completed
    return False

  def mc_vbios_trained(self) -> bool:
    """VBIOS MC ucode loaded (smu7_check_mc_firmware bit 23)."""
    return bool(self.mc_io_debug_up13() & (1 << 23))

  def vram_mc_offset(self, mc_addr: int) -> int:
    """Byte offset from vram_start for MM_INDEX access (amdgpu_device_mm_access)."""
    return (mc_addr - self.vram_start) & 0xffffffffffffffff

  def vram_mm_write(self, mc_addr: int, data: bytes):
    """Write VRAM via mmMM_INDEX/mmMM_DATA when BAR0 aperture is dead."""
    pos = self.vram_mc_offset(mc_addr)
    if pos % 4 or len(data) % 4:
      raise ValueError(f"vram_mm_write needs 4-byte alignment pos={pos:#x} len={len(data)}")
    # Ensure MM indirect VRAM access is enabled (BIF_MM_INDACCESS_CNTL).
    self.wreg(mmBIF_MM_INDACCESS_CNTL, 0)
    hi = None
    for i in range(0, len(data), 4):
      p = pos + i
      tmp = p >> 31
      self.wreg(mmMM_INDEX, (p & 0x7fffffff) | 0x80000000)
      if tmp != hi:
        self.wreg(mmMM_INDEX_HI, tmp)
        hi = tmp
      self.wreg(mmMM_DATA, struct.unpack_from('<I', data, i)[0])
    self.vram_flush()

  def vram_mm_read(self, mc_addr: int, size: int) -> bytes:
    pos = self.vram_mc_offset(mc_addr)
    if pos % 4 or size % 4:
      raise ValueError(f"vram_mm_read needs 4-byte alignment pos={pos:#x} size={size}")
    self.hdp_invalidate()
    self.wreg(mmBIF_MM_INDACCESS_CNTL, 0)
    out = bytearray(size)
    hi = None
    for i in range(0, size, 4):
      p = pos + i
      tmp = p >> 31
      self.wreg(mmMM_INDEX, (p & 0x7fffffff) | 0x80000000)
      if tmp != hi:
        self.wreg(mmMM_INDEX_HI, tmp)
        hi = tmp
      struct.pack_into('<I', out, i, self.rreg(mmMM_DATA))
    return bytes(out)

  def probe_bar0_writes(self) -> bool:
    """Return True if BAR0 framebuffer writes are visible (needed for VRAM path)."""
    pat = 0xA5A5A5A5
    off = 0x2000
    try:
      self.dev.vram[off:off + 4] = struct.pack('<I', pat)
      got = struct.unpack('<I', bytes(self.dev.vram[off:off + 4]))[0]
      return got == pat
    except Exception:
      return False

  def probe_vram_mm_writes(self) -> bool:
    """Return True if MM_INDEX VRAM writes work (Linux fallback when BAR0 is dead)."""
    pat = 0xA5A5A5A5
    mc = self.vram_visible_mc + 0x3000
    try:
      self.vram_mm_write(mc, struct.pack('<I', pat))
      got = struct.unpack('<I', self.vram_mm_read(mc, 4))[0]
      ok = got == pat
      if int(os.environ.get("DEBUG", "0")):
        print(f"polaris: probe_vram_mm mc={mc:#x} wrote={pat:#x} read={got:#x} ok={ok}", flush=True)
      return ok
    except Exception as e:
      if int(os.environ.get("DEBUG", "0")):
        print(f"polaris: probe_vram_mm failed: {e}", flush=True)
      return False

  def mc_init_locations(self):
    mem_mb = self.rreg(mmCONFIG_MEMSIZE) & 0xffff
    if mem_mb in (0, 0xffff):
      mem_mb = int(os.environ.get("AMD_VRAM_MB", "4096"))
    self.vram_size = mem_mb * 1024 * 1024
    fb_loc = self.rreg(mmMC_VM_FB_LOCATION)
    base = int(fb_loc & 0xffff) << 24
    if base == 0 and self.vram_size and self.vram_size < (1 << 32):
      base = ((1 << 32) - self.vram_size) & 0xffffffff
    self.vram_start = base & 0xffffffff
    self.vram_end = (self.vram_start + self.vram_size - 1) & 0xffffffff
    bar_bytes = self.dev.bar0_size
    if self.vram_size > bar_bytes:
      self.vram_visible_mc = (self.vram_end - bar_bytes + 1) & 0xffffffff
    else:
      self.vram_visible_mc = self.vram_start
    self.gart_size = 256 * 1024 * 1024
    max_mc = (1 << 40) - 1
    four_gb = 1 << 32
    size_bf = self.vram_start
    size_af = max_mc + 1 - round_up(self.vram_end + 1, four_gb)
    if size_bf >= self.gart_size and (size_bf < size_af or size_af < self.gart_size):
      self.gart_start = 0
    else:
      self.gart_start = (max_mc - self.gart_size + 1) & ~(four_gb - 1)
    self.gart_end = self.gart_start + self.gart_size - 1
    # AGP aperture for GTT fw_buf (amdgpu_gmc_agp_location after full VRAM)
    self.agp_start = round_up(self.vram_end + 1, four_gb)
    if self.agp_start <= self.vram_end:
      self.agp_start = four_gb
    self.agp_size = min(max_mc + 1 - self.agp_start, 512 * 1024 * 1024)
    self.agp_end = self.agp_start + self.agp_size - 1

  def mc_program_fb_location(self):
    if (self.rreg(mmMC_VM_FB_LOCATION) & 0xffff) == 0 and self.vram_size:
      tmp = ((self.vram_end >> 24) & 0xffff) << 16 | ((self.vram_start >> 24) & 0xffff)
      self.wreg(mmMC_VM_FB_LOCATION, tmp)

  def gmc_sw_init(self):
    """gmc_v8_0_mc_init + vram_gtt_location (sw_init only, before SMC fw load)."""
    self.mc_init_locations()
    self.dev._vram_start = self.vram_visible_mc
    if int(os.environ.get("DEBUG", "0")):
      print(f"polaris: gmc_sw_init vram={self.vram_start:#x}-{self.vram_end:#x} "
            f"visible_mc={self.vram_visible_mc:#x} gart={self.gart_start:#x}-{self.gart_end:#x} "
            f"agp={self.agp_start:#x}", flush=True)

  def mc_wait_idle(self, timeout_s: float = 1.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
      if not (self.rreg(mmSRBM_STATUS) & 0x3f0000):
        return
      time.sleep(0.001)

  def mc_program(self):
    self.mc_wait_idle()
    self.mc_init_locations()
    self.dev._vram_start = self.vram_visible_mc
    for reg, mask, val in GMC_GOLDEN_REGS:
      self.wreg(reg, (self.rreg(reg) & ~mask) | val)
    for i in range(32):
      j = i * 6
      for off in (0xb05 + j, 0xb06 + j, 0xb07 + j, 0xb08 + j, 0xb09 + j):
        self.wreg(off, 0)
    self.wreg(mmHDP_REG_COHERENCY_FLUSH_CNTL, 0)
    scratch = self.vram_visible_mc or self.vram_start
    self.wreg(mmMC_VM_SYSTEM_APERTURE_LOW_ADDR, self.vram_start >> 12)
    self.wreg(mmMC_VM_SYSTEM_APERTURE_HIGH_ADDR, self.vram_end >> 12)
    self.wreg(mmMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR, scratch >> 12)
    self.wreg(mmMC_VM_AGP_BASE, 0)
    self.wreg(mmMC_VM_AGP_TOP, self.agp_end >> 22)
    self.wreg(mmMC_VM_AGP_BOT, self.agp_start >> 22)
    self.wreg(mmBIF_FB_EN, 0x3)
    mem_mb = self.rreg(mmCONFIG_MEMSIZE) & 0xffff
    if mem_mb in (0, 0xffff) and self.vram_size:
      self.wreg(mmCONFIG_MEMSIZE, self.vram_size // (1024 * 1024))
    if os.environ.get("AMD_BOOT_HDP_NONSURFACE", "1") == "1":
      self.wreg(mmHDP_NONSURFACE_BASE, self.vram_start >> 8)
      self.wreg(mmHDP_NONSURFACE_INFO, (2 << 7) | (1 << 30))
      self.wreg(mmHDP_NONSURFACE_SIZE, 0x3fffffff)
    tmp = self.rreg(mmHDP_MISC_CNTL)
    self.wreg(mmHDP_MISC_CNTL, tmp & ~(1 << 0))
    self.wreg(mmHDP_HOST_PATH_CNTL, self.rreg(mmHDP_HOST_PATH_CNTL))
    self.mc_program_fb_location()
    self.mmio_sync_safe()

  def load_mc_firmware(self):
    misc0 = self.rreg(mmMC_SEQ_MISC0)
    if self.vram_trained():
      if int(os.environ.get("DEBUG", "0")):
        up13 = self.mc_io_debug_up13()
        print(f"polaris: MC skip load MISC0={misc0:#x} UP_13={up13:#x} MEMSIZE={self.config_memsize_mb()}", flush=True)
      self.mc_init_locations()
      self.dev._vram_start = self.vram_visible_mc
      self.mc_program_fb_location()
      return
    blob = self.fw("polaris10_mc.bin")
    io_dbg_off, io_dbg_sz, ucode_off, ucode_sz = parse_mc_fw(blob)
    io_regs = blob[io_dbg_off:io_dbg_off + io_dbg_sz]
    ucode = blob[ucode_off:ucode_off + ucode_sz]
    regs_cnt = io_dbg_sz // 8
    data = self.rreg(mmMC_SEQ_MISC0) & ~0x40
    self.wreg(mmMC_SEQ_MISC0, data)
    for i in range(regs_cnt):
      self.wreg(mmMC_SEQ_IO_DEBUG_INDEX, _le32(io_regs, i * 8))
      self.wreg(mmMC_SEQ_IO_DEBUG_DATA, _le32(io_regs, i * 8 + 4))
    self.wreg(mmMC_SEQ_SUP_CNTL, 8)
    self.wreg(mmMC_SEQ_SUP_CNTL, 16)
    for i in range(0, len(ucode), 4):
      self.wreg(mmMC_SEQ_SUP_PGM, _le32(ucode, i))
      if i and (i // 4) % 64 == 0:
        self.mmio_sync_safe()
    self.wreg(mmMC_SEQ_SUP_CNTL, 8)
    self.wreg(mmMC_SEQ_SUP_CNTL, 4)
    self.wreg(mmMC_SEQ_SUP_CNTL, 1)
    timeout_s = self._timeout_s("AMD_BOOT_MC_TIMEOUT_S", 5.0)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
      misc0 = self.rreg(mmMC_SEQ_MISC0)
      if misc0 & 0x80:
        self.mc_init_locations()
        if int(os.environ.get("DEBUG", "0")):
          mem_mb = self.rreg(mmCONFIG_MEMSIZE) & 0xffff
          print(f"polaris: MC training done MISC0={misc0:#x} MEMSIZE={mem_mb} vram_start={self.vram_start:#x}", flush=True)
        self.dev._vram_start = self.vram_visible_mc
        self.mc_program_fb_location()
        return
      if int(os.environ.get("DEBUG", "0")):
        print(f"polaris: MC training wait MISC0={misc0:#x}", flush=True)
      time.sleep(0.01)
    raise RuntimeError(f"MC ucode training timeout MISC0={self.rreg(mmMC_SEQ_MISC0):#x}")

  def hdp_flush(self):
    """vi_flush_hdp: flush HDP write cache after VRAM writes."""
    self.wreg(mmHDP_MEM_COHERENCY_FLUSH_CNTL, 1)
    self.rreg(mmHDP_MEM_COHERENCY_FLUSH_CNTL)

  def hdp_invalidate(self):
    """vi_invalidate_hdp: invalidate HDP read cache before VRAM reads."""
    self.wreg(mmHDP_DEBUG0, 1)
    self.rreg(mmHDP_DEBUG0)

  def vram_flush(self):
    self.wreg(mmHDP_REG_COHERENCY_FLUSH_CNTL, 1)
    self.wreg(mmHDP_REG_COHERENCY_FLUSH_CNTL, 0)
    self.hdp_flush()
    self.mmio_sync_safe()

  def _gart_program_vm(self, pte_base_addr: int, pte_physical: bool):
    self.wreg(mmMC_VM_MX_L1_TLB_CNTL, 0x98000b)
    self.wreg(mmVM_L2_CNTL, 0x30103)
    self.wreg(mmVM_L2_CNTL2, 0x30003)
    self.wreg(mmVM_L2_CNTL3, 0x24100003)
    l2c4 = 0
    if pte_physical:
      l2c4 |= 1 << 9  # VMC_TAP_CONTEXT0_PTE_REQUEST_PHYSICAL
    self.wreg(mmVM_L2_CNTL4, l2c4)
    gart_start = self.gart_start
    gart_end = self.gart_end
    self.wreg(mmVM_CONTEXT0_PAGE_TABLE_START_ADDR, gart_start >> 12)
    self.wreg(mmVM_CONTEXT0_PAGE_TABLE_END_ADDR, gart_end >> 12)
    self.wreg(mmVM_CONTEXT0_PAGE_TABLE_BASE_ADDR, pte_base_addr >> 12)
    self.wreg(mmVM_CONTEXT0_PROTECTION_FAULT_DEFAULT_ADDR, 0)
    self.wreg(mmVM_CONTEXT0_CNTL2, 0)
    # ENABLE_CONTEXT | PAGE_TABLE_DEPTH=0 | RANGE_PROTECTION_FAULT_ENABLE_DEFAULT
    self.wreg(mmVM_CONTEXT0_CNTL, 0x11)
    self.wreg(mmVM_CONTEXT1_PAGE_TABLE_START_ADDR, 0)
    self.wreg(mmVM_CONTEXT1_PAGE_TABLE_END_ADDR, (1 << 28) - 1)
    self.wreg(mmVM_CONTEXT1_PROTECTION_FAULT_DEFAULT_ADDR, 0)
    self.wreg(mmVM_CONTEXT1_CNTL2, 4)
    self.wreg(mmVM_CONTEXT1_CNTL, 0x3000007)
    self.wreg(mmVM_INVALIDATE_REQUEST, 1)
    self.mmio_sync_safe()

  def gart_enable(self):
    gart_words = 256 * 1024 // 4
    gart_bytes = gart_words * 4
    use_sysmem = os.environ.get("AMD_BOOT_GART_SYSMEM", "auto")
    if use_sysmem == "auto":
      use_sysmem = "0" if self.probe_bar0_writes() else "1"
    self.gart_pte_mem = bytearray(gart_bytes)
    invalid_pte = struct.pack('<I', 0x10)
    for i in range(gart_words):
      self.gart_pte_mem[i * 4:i * 4 + 4] = invalid_pte
    if use_sysmem == "1":
      mem, paddrs, _ = self.alloc_sysmem_buffer(gart_bytes)
      self.gart_pte_sysmem = mem
      npages_table = (gart_bytes + PAGE_SIZE - 1) // PAGE_SIZE
      # Self-map: first GART slots point at this PTE table in host memory.
      table_gpu_base = self.gart_start
      self.gart_base = table_gpu_base
      for i, paddr in enumerate(paddrs[:npages_table]):
        off = i * 4
        pte = ((paddr >> 12) << 12) | GART_PTE_FLAGS
        struct.pack_into('<I', self.gart_pte_mem, off, pte)
        mem[off:off + 4] = self.gart_pte_mem[off:off + 4]
      invalid = struct.pack('<I', 0x10)
      for i in range(npages_table, gart_words):
        off = i * 4
        self.gart_pte_mem[off:off + 4] = invalid
        mem[off:off + 4] = invalid
      self._gart_program_vm(table_gpu_base, pte_physical=False)
      if int(os.environ.get("DEBUG", "0")):
        print(f"polaris: GART PTE self-map at {table_gpu_base:#x} "
              f"phys={[hex(p) for p in paddrs[:2]]}", flush=True)
    else:
      self.gart_pte_off = self.dev.alloc_vram(gart_bytes, align=PAGE_SIZE)
      self.dev.upload(self.gart_pte_off, bytes(self.gart_pte_mem))
      self.gart_base = self.vram_visible_mc + self.gart_pte_off
      self._gart_program_vm(self.gart_base, pte_physical=False)
      if int(os.environ.get("DEBUG", "0")):
        print(f"polaris: GART PTE table in VRAM mc={self.gart_base:#x}", flush=True)

  def gart_flush_tlb(self):
    self.wreg(mmVM_INVALIDATE_REQUEST, 1)
    self.mmio_sync_safe()

  def alloc_sysmem_buffer(self, size: int, contiguous: bool = False) -> tuple[object, list[int], int]:
    mem, paddrs = self.dev.pci.alloc_sysmem(size, contiguous=contiguous)
    return mem, paddrs, size

  def _gart_write_pte(self, pte_off: int, pte_val: int):
    if self.gart_pte_mem is None:
      return
    chunk = struct.pack('<I', pte_val)
    self.gart_pte_mem[pte_off:pte_off + 4] = chunk
    if self.gart_pte_sysmem is not None:
      self.gart_pte_sysmem[pte_off:pte_off + 4] = chunk
    elif self.gart_pte_off:
      self.dev.upload(self.gart_pte_off + pte_off, chunk)

  def map_sysmem_gpu(self, paddrs: list[int], size: int, gpu_va: int | None = None) -> int:
    if gpu_va is None:
      gpu_va = self._next_gart_va(size)
    if self.gart_pte_mem is not None:
      npages = (size + 0xfff) // 0x1000
      base_pfn = (gpu_va - self.gart_start) >> 12
      for i, paddr in enumerate(paddrs[:npages]):
        off = (base_pfn + i) * 4
        if off + 4 <= len(self.gart_pte_mem):
          self._gart_write_pte(off, (paddr >> 12) << 12 | GART_PTE_FLAGS)
    self.gart_flush_tlb()
    return gpu_va

  def _next_gart_va(self, size: int) -> int:
    off = round_up(self._gart_alloc_off, PAGE_SIZE)
    self._gart_alloc_off = off + round_up(size, PAGE_SIZE)
    return self.gart_start + off

  def alloc_fw_buffer(self, size: int) -> tuple[int, object, int, list[int]]:
    # Linux amdgpu_gmc_agp_addr needs single contiguous segment; VI uses GART PTEs.
    mem, paddrs, nbytes = self.alloc_sysmem_buffer(size, contiguous=True)
    gpu_addr = self.map_sysmem_gpu(paddrs, nbytes) if paddrs else 0
    return gpu_addr, mem, nbytes, paddrs

  def build_fw_images(self, gpu_base: int, writer, addr_fixup=None) -> tuple[bytes, list[tuple]]:
    """Stage IP firmware; addr_fixup(gpu_mc_addr, image) may rewrite TOC MC address."""
    entries: list[tuple] = []
    off = 0

    def place_image(image: bytes) -> int:
      nonlocal off
      aligned = round_up(off, PAGE_SIZE)
      writer(aligned, image)
      addr = gpu_base + aligned
      if addr_fixup:
        addr = addr_fixup(addr, image)
      off = aligned + round_up(len(image), PAGE_SIZE)
      return addr

    def add_common(name: str, ucode_id: int, flags: int = 0):
      blob = self.fw(name)
      ucode_off, ucode_sz = parse_common_fw(blob)
      version = _le32(blob, 16) & 0xffff
      image = blob[ucode_off:ucode_off + ucode_sz]
      addr = place_image(image)
      entries.append((ucode_id, version, addr, ucode_sz, flags))

    add_common("polaris10_rlc.bin", UCODE_ID_RLC_G, flags=1)
    add_common("polaris10_ce.bin", UCODE_ID_CP_CE)
    add_common("polaris10_pfp.bin", UCODE_ID_CP_PFP)
    add_common("polaris10_me.bin", UCODE_ID_CP_ME)

    mec_blob = self.fw("polaris10_mec.bin")
    ucode_off, _ucode_sz, version, jt_off, jt_sz = parse_gfx_fw(mec_blob)
    mec_bytes = jt_off * 4
    mec_image = mec_blob[ucode_off:ucode_off + mec_bytes]
    mec_addr = place_image(mec_image)
    entries.append((UCODE_ID_CP_MEC, version, mec_addr, mec_bytes, 1))
    jt_bytes = jt_sz * 4
    jt_image = mec_blob[ucode_off + mec_bytes:ucode_off + mec_bytes + jt_bytes]
    jt_addr = place_image(jt_image)
    entries.append((UCODE_ID_CP_MEC_JT1, version, jt_addr, jt_bytes, 0))
    entries.append((UCODE_ID_CP_MEC_JT2, version, jt_addr, jt_bytes, 0))

    add_common("polaris10_sdma.bin", UCODE_ID_SDMA0)
    add_common("polaris10_sdma1.bin", UCODE_ID_SDMA1)

    toc = struct.pack('<II', 1, len(entries))
    for ucode_id, ver, addr, sz, flags in entries:
      toc += pack_smu_toc_entry(ucode_id, ver, addr, sz, flags)
    return toc, entries

  def agp_mc_addr(self, paddr: int) -> int:
    """amdgpu_gmc_agp_addr: agp_start + dma_address (full phys, VI rarely uses AGP)."""
    if paddr + PAGE_SIZE >= self.agp_size:
      raise ValueError(f"paddr {paddr:#x} outside AGP aperture size {self.agp_size:#x}")
    return self.agp_start + paddr

  def ensure_gart_ready(self):
    """Linux amdgpu_ttm_alloc_gart binds PTEs at sw_init; enable GART before LoadUcodes on eGPU."""
    if self.gart_pte_mem is None:
      self.gart_enable()

  def _flush_fw_sysmem(self, layout: str, fw_mem):
    """ARM/M1: CPU cache may hide sysmem writes from eGPU DMA (rpi-pcie #756)."""
    if layout not in ("hybrid", "agp", "gtt") or fw_mem is None:
      return
    from add import sysmem_dma_flush
    sysmem_dma_flush(fw_mem, SMU_FW_BUF_SIZE)
    if int(os.environ.get("DEBUG", "0")):
      print("polaris: sysmem_dma_flush fw_buf", flush=True)

  def load_ip_firmware(self):
    """Linux smu7_request_smu_load_fw — runs before gmc_v8_0_gart_enable on VI."""
    if not self.smc_running():
      raise RuntimeError("SMC not running before load_ip_firmware")
    fw_mask = int(os.environ.get("AMD_BOOT_FW_MASK", str(FW_TO_LOAD)), 0)
    bar0_ok = self.probe_bar0_writes()
    mm_ok = self.probe_vram_mm_writes() if not bar0_ok else False
    layout = os.environ.get("AMD_BOOT_FW_LAYOUT", "auto")
    if layout == "auto":
      if bar0_ok:
        layout = "vram"
      elif os.environ.get("AMD_BOOT_FW_GTT", "0") == "1":
        layout = "gtt"
      else:
        layout = "hybrid"  # VRAM MC header/smu (MM_INDEX) + AGP fw images
    use_gtt = layout == "gtt"
    use_phys = layout in ("agp",) or os.environ.get("AMD_BOOT_FW_PHYS_ADDR", "0") == "1"

    if layout == "hybrid":
      self.ensure_gart_ready()
      smu_dram_off = round_up(0x100000, PAGE_SIZE)
      hdr_off = round_up(smu_dram_off + SMU_FW_BUF_SIZE, PAGE_SIZE)
      smu_dram_gpu = self.vram_visible_mc + smu_dram_off
      hdr_gpu = self.vram_visible_mc + hdr_off
      fw_gpu_base, fw_mem, _, fw_paddrs = self.alloc_fw_buffer(SMU_FW_BUF_SIZE)
      if not fw_gpu_base or not fw_paddrs:
        raise RuntimeError("hybrid layout needs GART-mapped contiguous fw_buf")

      def writer(off: int, image: bytes):
        fw_mem[off:off + len(image)] = image

      layout_tag = f"hybrid-gart vram_start={self.vram_start:#x} mm={'1' if mm_ok else '0'}"
      addr_fixup_fn = None
      vram_writer = self.vram_mm_write
    elif layout == "agp":
      total = SMU_FW_BUF_SIZE + SMU_HDR_BUF_SIZE + SMU_FW_BUF_SIZE
      mem, paddrs, _ = self.alloc_sysmem_buffer(total, contiguous=True)
      if not paddrs:
        raise RuntimeError("agp firmware layout needs alloc_sysmem paddrs")
      base_paddr = paddrs[0] & ~0xfff
      smu_off, hdr_off, fw_off = 0, SMU_FW_BUF_SIZE, SMU_FW_BUF_SIZE + SMU_HDR_BUF_SIZE
      use_raw_phys = os.environ.get("AMD_BOOT_AGP_RAW_PHYS", "0") == "1"
      if use_raw_phys:
        smu_dram_gpu = base_paddr + smu_off
        hdr_gpu = base_paddr + hdr_off
        fw_gpu_base = base_paddr + fw_off
        layout_tag = f"phys raw_base={base_paddr:#x}"
      else:
        smu_dram_gpu = self.agp_mc_addr(base_paddr + smu_off)
        hdr_gpu = self.agp_mc_addr(base_paddr + hdr_off)
        fw_gpu_base = self.agp_mc_addr(base_paddr + fw_off)
        layout_tag = f"agp phys_base={base_paddr:#x}"

      def writer(off: int, image: bytes):
        mem[fw_off + off:fw_off + off + len(image)] = image

      addr_fixup_fn = None
      vram_writer = None
      fw_mem = mem
      hdr_off_vram = hdr_off
    elif use_gtt:
      if self.gart_pte_mem is None and os.environ.get("AMD_BOOT_FW_GTT", "0") != "1":
        raise RuntimeError("GART not enabled; call gart_enable before GTT load_ip_firmware")
      smu_dram_gpu, smu_dram_mem, _, smu_paddrs = self.alloc_fw_buffer(SMU_FW_BUF_SIZE)
      hdr_gpu, hdr_mem, _, hdr_paddrs = self.alloc_fw_buffer(SMU_HDR_BUF_SIZE)
      fw_gpu_base, fw_mem, _, fw_paddrs = self.alloc_fw_buffer(SMU_FW_BUF_SIZE)
      if not smu_dram_gpu or not hdr_gpu or not fw_gpu_base:
        raise RuntimeError("failed to map firmware buffers in GART")
      fw_paddr_base = fw_paddrs[0] & ~0xfff if fw_paddrs else 0
      mem = hdr_mem
      if use_phys and smu_paddrs:
        smu_dram_gpu = smu_paddrs[0] & ~0xfff
      if use_phys and hdr_paddrs:
        hdr_gpu = hdr_paddrs[0] & ~0xfff

      def writer(off: int, image: bytes):
        fw_mem[off:off + len(image)] = image

      def addr_fixup(mc_addr: int, _image: bytes) -> int:
        if not use_phys or not fw_paddr_base:
          return mc_addr
        return fw_paddr_base + (mc_addr - fw_gpu_base)

      layout_tag = "gtt" + ("-phys" if use_phys else "")
      addr_fixup_fn = addr_fixup
      vram_writer = None
    else:
      smu_dram_off = self.dev.alloc_vram(SMU_FW_BUF_SIZE, align=PAGE_SIZE)
      hdr_off = self.dev.alloc_vram(SMU_HDR_BUF_SIZE, align=PAGE_SIZE)
      fw_off = self.dev.alloc_vram(SMU_FW_BUF_SIZE, align=PAGE_SIZE)
      smu_dram_gpu = self.vram_visible_mc + smu_dram_off
      hdr_gpu = self.vram_visible_mc + hdr_off
      fw_gpu_base = self.vram_visible_mc + fw_off
      self.dev.upload(smu_dram_off, bytes(SMU_FW_BUF_SIZE))
      addr_fixup_fn = None
      vram_writer = None

      def writer(off: int, image: bytes):
        self.dev.upload(fw_off + off, image)

      layout_tag = "vram"

    toc, entries = self.build_fw_images(
      fw_gpu_base, writer,
      addr_fixup=addr_fixup_fn if layout in ("hybrid", "gtt") or use_phys else None)
    if os.environ.get("AMD_BOOT_FW_MINIMAL", "0") == "1":
      id_to_mask = {
        UCODE_ID_SDMA0: UCODE_ID_SDMA0_MASK, UCODE_ID_SDMA1: UCODE_ID_SDMA1_MASK,
        UCODE_ID_CP_CE: UCODE_ID_CP_CE_MASK, UCODE_ID_CP_PFP: UCODE_ID_CP_PFP_MASK,
        UCODE_ID_CP_ME: UCODE_ID_CP_ME_MASK, UCODE_ID_CP_MEC: UCODE_ID_CP_MEC_MASK,
        UCODE_ID_CP_MEC_JT1: UCODE_ID_CP_MEC_JT1_MASK, UCODE_ID_CP_MEC_JT2: UCODE_ID_CP_MEC_JT2_MASK,
        UCODE_ID_RLC_G: UCODE_ID_RLC_G_MASK,
      }
      entries = [e for e in entries if id_to_mask.get(e[0], 0) & fw_mask]
      toc = struct.pack('<II', 1, len(entries))
      for ucode_id, ver, addr, sz, flags in entries:
        toc += pack_smu_toc_entry(ucode_id, ver, addr, sz, flags)
    if layout == "hybrid":
      if not mm_ok and os.environ.get("AMD_BOOT_FORCE_HYBRID", "1") != "1":
        raise RuntimeError("hybrid layout needs MM_INDEX VRAM writes (BAR0 dead and MM_INDEX probe failed; set AMD_BOOT_FORCE_HYBRID=1 to try anyway)")
      if not mm_ok and int(os.environ.get("DEBUG", "0")):
        print("polaris: WARNING hybrid layout with failed MM_INDEX probe", flush=True)
      vram_writer(smu_dram_gpu, bytes(SMU_FW_BUF_SIZE))
      vram_writer(hdr_gpu, toc)
    elif layout == "agp":
      mem[hdr_off_vram:hdr_off_vram + len(toc)] = toc
    elif use_gtt:
      mem[0:len(toc)] = toc
    else:
      if bar0_ok:
        self.dev.upload(hdr_off, toc)
      else:
        self.vram_mm_write(hdr_gpu, toc)
    self._flush_fw_sysmem(layout, fw_mem if layout in ("hybrid", "agp", "gtt") else None)
    self.vram_flush()
    if int(os.environ.get("DEBUG", "0")):
      soft = self.read_soft_regs_start()
      print(f"polaris: load_ip_firmware {layout_tag} soft_regs={soft:#x} "
            f"smu_dram={smu_dram_gpu:#x} hdr={hdr_gpu:#x} fw_buf={fw_gpu_base:#x} mask={fw_mask:#x}", flush=True)
      for ucode_id, ver, addr, sz, flags in entries:
        print(f"  toc id={ucode_id} ver={ver:#x} addr={addr:#x} sz={sz:#x} flags={flags}", flush=True)
    self.clear_ucode_load_status()
    if int(os.environ.get("DEBUG", "0")):
      pre = self.smc_soft_reg(ixSMU74_UcodeLoadStatus)
      print(f"polaris: UcodeLoadStatus before LoadUcodes={pre}", flush=True)
    settle = self._settle_s()
    if settle > 0:
      time.sleep(settle)
    self.dev.pci.drain_mmio(bar=5, reg=0x2004)
    self._check_pci("LoadUcodes prep")
    self.smc_send_msg(PPSMC_MSG_SMU_DRAM_ADDR_HI, smu_dram_gpu >> 32, label="SMU_DRAM_HI")
    self.smc_send_msg(PPSMC_MSG_SMU_DRAM_ADDR_LO, smu_dram_gpu & 0xffffffff, label="SMU_DRAM_LO")
    self.smc_send_msg(PPSMC_MSG_DRV_DRAM_ADDR_HI, hdr_gpu >> 32, label="DRV_DRAM_HI")
    self.smc_send_msg(PPSMC_MSG_DRV_DRAM_ADDR_LO, hdr_gpu & 0xffffffff, label="DRV_DRAM_LO")
    self.smc_send_msg(PPSMC_MSG_LoadUcodes, fw_mask, label="LoadUcodes")
    if not self.wait_ucode_load(fw_mask, timeout_s=self._timeout_s("AMD_BOOT_UCODE_LOAD_TIMEOUT_S", 30.0)):
      resp = self.rreg(mmSMC_RESP_0) & 0xffff
      status = self.smc_soft_reg(ixSMU74_UcodeLoadStatus)
      status_s = f"{status:#x}" if status is not None else "garbage"
      raise RuntimeError(
        f"IP firmware load timeout (RESP={resp:#x} UcodeLoadStatus={status_s} want {fw_mask:#x}) {self.smc_diag()}")

  def enable_compute(self):
    self.wreg(mmCP_MEC_CNTL, 0)
    self.wreg(mmCP_MEC_DOORBELL_RANGE_LOWER, DOORBELL_KIQ << 2)
    self.wreg(mmCP_MEC_DOORBELL_RANGE_UPPER, (DOORBELL_MEC_RING0 + 8) << 2)
    pq = self.rreg(mmCP_PQ_STATUS)
    self.wreg(mmCP_PQ_STATUS, pq | (1 << 28))
    self.mmio_sync_safe()

  def init_compute_queue(self) -> 'ComputeQueue':
    if self._compute is None:
      self._compute = ComputeQueue(self)
      self._compute.init()
      self._compute.setup_with_kiq()
    return self._compute

  def read_soft_regs_start(self) -> int:
    val = self.smc_read(SMU_HDR_SOFT_REGS_OFF)
    return val if val is not None else 0

  def smc_soft_reg(self, offset: int) -> int:
    base = self.read_soft_regs_start()
    if not base:
      return 0
    return self.smc_rreg(base + offset) & 0xffffffff

  def smc_soft_wreg(self, offset: int, val: int):
    base = self.read_soft_regs_start()
    if base:
      self.smc_wreg_safe(base + offset, val)

  def clear_ucode_load_status(self):
    self.smc_soft_wreg(ixSMU74_UcodeLoadStatus, 0)

  def wait_ucode_load(self, fw_mask: int, timeout_s: float = 60.0) -> bool:
    deadline = time.time() + timeout_s
    last_status = -1
    while time.time() < deadline:
      status = self.smc_soft_reg(ixSMU74_UcodeLoadStatus)
      if status != last_status and int(os.environ.get("DEBUG", "0")):
        print(f"polaris: UcodeLoadStatus={status:#x} (want {fw_mask:#x})", flush=True)
        last_status = status
      if (status & fw_mask) == fw_mask:
        return True
      time.sleep(0.01)
    return False

  def boot(self):
    if self.dev.gpu_ready():
      return
    # Linux: amdgpu_atom_asic_init before ip_init (SMC / mc_program / LoadUcodes)
    self.vi_common_init()
    self.enable_vbios_rom()
    from atom_replay import run_asic_init_if_needed
    run_asic_init_if_needed(self)
    self.gmc_sw_init()
    self.start_smc()
    self.mc_program()
    try:
      self.load_mc_firmware()
      if int(os.environ.get("DEBUG", "0")):
        mem_mb = self.rreg(mmCONFIG_MEMSIZE) & 0xffff
        print(f"polaris: MC training ok MEMSIZE={mem_mb:#x}", flush=True)
    except RuntimeError as e:
      if int(os.environ.get("AMD_BOOT_STRICT_MC", "0")):
        raise
      if int(os.environ.get("DEBUG", "0")):
        print(f"polaris: MC load skipped ({e})", flush=True)
    # GART PTE bind (Linux sw_init) must be live before LoadUcodes on eGPU.
    self.gart_enable()
    if self.smc_running():
      self.load_ip_firmware()
    # gart_enable is idempotent if already called above
    self.gart_enable()
    self.enable_compute()
    self.init_compute_queue()
    if not self.dev.gpu_ready() and self.rreg(mmCP_MEC_CNTL) == 0x50000000:
      raise RuntimeError(
        f"Polaris boot incomplete: SMC={self.smc_running()} "
        f"CP_HQD_ACTIVE={self.rreg(mmCP_HQD_ACTIVE):#x}"
      )


mmCP_HQD_PQ_BASE_LO = 0x324d
mmCP_HQD_PQ_BASE_HI = 0x324e
mmCP_HQD_PQ_RPTR_REPORT_ADDR_LO = 0x3250
mmCP_HQD_PQ_RPTR_REPORT_ADDR_HI = 0x3251
mmCP_HQD_PQ_WPTR_POLL_ADDR_LO = 0x3252
mmCP_HQD_PQ_WPTR_POLL_ADDR_HI = 0x3253
mmCP_HQD_PQ_DOORBELL_CONTROL = 0x3254
mmCP_HQD_PQ_WPTR = 0x3255
mmCP_HQD_PQ_CONTROL = 0x3256
mmCP_HQD_IB_CONTROL = 0x325a
mmCP_HQD_IQ_TIMER = 0x325c
mmCP_HQD_CTX_SAVE_CONTROL = 0x3260
mmCP_HQD_EOP_BASE_ADDR_LO = 0x3264
mmCP_HQD_EOP_BASE_ADDR_HI = 0x3265
mmCP_HQD_EOP_CONTROL = 0x3266
mmCP_MQD_CONTROL = 0x3267
mmCP_HQD_PERSISTENT_STATE = 0x3249


class ComputeQueue:
  """gfx_v8_0 KIQ + KCQ setup for MEC compute ring 0."""

  def __init__(self, boot: PolarisBoot, me=1, pipe=0, queue=0, doorbell_index=DOORBELL_MEC_RING0):
    self.boot = boot
    self.dev = boot.dev
    self.me, self.pipe, self.queue = me, pipe, queue
    self.doorbell_index = doorbell_index
    self.ring_off = self.mqd_off = self.eop_off = self.wptr_off = 0
    self.ring_gpu = self.mqd_gpu = self.eop_gpu = self.wptr_gpu = 0
    self.wptr = 0

  def _alloc_vram(self, size: int, align=0x1000) -> int:
    return self.dev.alloc_vram(size, align)

  def _write_ring(self, words: list[int]):
    data = struct.pack('<' + 'I' * len(words), *words)
    self.dev.upload(self.ring_off, data)

  def _mqd_regs(self, ring_gpu: int, ring_size: int, mqd_gpu: int, eop_gpu: int, wptr_gpu: int,
                doorbell_index: int, active: bool) -> dict[int, int]:
    qsize = order_base_2(ring_size // 4) - 1
    rptr_blk = order_base_2(1024) - 1
    eop_size = order_base_2(GFX8_MEC_HPD_SIZE // 4) - 1
    pq_ctl = (qsize << 0) | (rptr_blk << 8) | 0x80000 | 0x100000
    ib_ctl = 0x30003
    iq_timer = 0x30000
    ctx_save = 0x30000
    eop_ctl = eop_size << 12
    mqd = {
      mmCP_MQD_BASE_ADDR: mqd_gpu & 0xfffffffc,
      mmCP_MQD_BASE_ADDR + 1: (mqd_gpu >> 32) & 0xffffffff,
      mmCP_HQD_VMID: 0,
      mmCP_HQD_PERSISTENT_STATE: 0x53,
      mmCP_HQD_PQ_BASE_LO: (ring_gpu >> 8) & 0xffffffff,
      mmCP_HQD_PQ_BASE_HI: ring_gpu >> 8 >> 32,
      mmCP_HQD_PQ_RPTR_REPORT_ADDR_LO: wptr_gpu & 0xfffffffc,
      mmCP_HQD_PQ_RPTR_REPORT_ADDR_HI: (wptr_gpu >> 32) & 0xffff,
      mmCP_HQD_PQ_WPTR_POLL_ADDR_LO: wptr_gpu & 0xffffffff,
      mmCP_HQD_PQ_WPTR_POLL_ADDR_HI: (wptr_gpu >> 32) & 0xffffffff,
      mmCP_HQD_PQ_DOORBELL_CONTROL: (doorbell_index << 28) | 0x10000,
      mmCP_HQD_PQ_WPTR: 0,
      mmCP_HQD_PQ_CONTROL: pq_ctl,
      mmCP_HQD_IB_CONTROL: ib_ctl,
      mmCP_HQD_IQ_TIMER: iq_timer,
      mmCP_HQD_CTX_SAVE_CONTROL: ctx_save,
      mmCP_HQD_EOP_BASE_ADDR_LO: eop_gpu & 0xffffffff,
      mmCP_HQD_EOP_BASE_ADDR_HI: (eop_gpu >> 32) & 0xffffffff,
      mmCP_HQD_EOP_CONTROL: eop_ctl,
      mmCP_MQD_CONTROL: 0x1 if active else 0,
    }
    return mqd

  def _mqd_commit(self, mqd: dict[int, int]):
    self.boot.srbm_select(self.me, self.pipe, self.queue, 0)
    self.boot.wreg(mmCP_PQ_WPTR_POLL_CNTL, 0)
    for reg in range(mmCP_HQD_VMID, mmCP_HQD_EOP_CONTROL + 1):
      if reg in mqd:
        self.boot.wreg(reg, mqd[reg])
    for reg in range(mmCP_HQD_EOP_CONTROL + 1, mmCP_HQD_ACTIVE + 1):
      if reg in mqd:
        self.boot.wreg(reg, mqd[reg])
    self.boot.srbm_select(0, 0, 0, 0)
    self.boot.mmio_sync()

  def _map_queues_pkt(self, target: 'ComputeQueue') -> list[int]:
    w: list[int] = []
    w.append(pkt3(PACKET3_SET_RESOURCES, 6))
    w.extend([0, 1, 0, 0, 0, 0, 0])
    w.append(pkt3(PACKET3_MAP_QUEUES, 5))
    w.append(0x20000000)
    doorbell_me = 0 if target.me == 1 else 1
    w.append((target.doorbell_index << 2) | (target.queue << 26) | (target.pipe << 29) | (doorbell_me << 31))
    w.append(target.mqd_gpu & 0xffffffff)
    w.append((target.mqd_gpu >> 32) & 0xffffffff)
    w.append(target.wptr_gpu & 0xffffffff)
    w.append((target.wptr_gpu >> 32) & 0xffffffff)
    return w

  def init(self):
    self.ring_off = self._alloc_vram(RING_SIZE)
    self.mqd_off = self._alloc_vram(4096)
    self.eop_off = self._alloc_vram(GFX8_MEC_HPD_SIZE)
    self.wptr_off = self._alloc_vram(4096)
    self.ring_gpu = self.dev.vram_gpu_addr(self.ring_off)
    self.mqd_gpu = self.dev.vram_gpu_addr(self.mqd_off)
    self.eop_gpu = self.dev.vram_gpu_addr(self.eop_off)
    self.wptr_gpu = self.dev.vram_gpu_addr(self.wptr_off)

  def submit_ib(self, ib_words: list[int]):
    pkt = [pkt3(0x10, len(ib_words) + 2, 0), 0, len(ib_words) * 4] + ib_words
    self._write_ring(pkt)
    self.wptr = (self.wptr + len(pkt)) % (RING_SIZE // 4)
    self.dev.ring_doorbell(self.doorbell_index, self.wptr)

  def setup_with_kiq(self):
    mqd = self._mqd_regs(self.ring_gpu, RING_SIZE, self.mqd_gpu, self.eop_gpu, self.wptr_gpu,
                         self.doorbell_index, active=True)
    self._mqd_commit(mqd)
    kiq = ComputeQueue(self.boot, me=2, pipe=0, queue=0, doorbell_index=DOORBELL_KIQ)
    kiq.init()
    kiq_pkt = kiq._map_queues_pkt(self)
    kiq._write_ring(kiq_pkt)
    kiq.wptr = len(kiq_pkt)
    self.dev.ring_doorbell(DOORBELL_KIQ, kiq.wptr)
    time.sleep(0.05)

