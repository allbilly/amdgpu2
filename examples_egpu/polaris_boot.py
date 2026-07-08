"""Polaris10 (RX570 / gfx803) firmware boot for macOS eGPU via TinyGPU.

Ports the VI (gfx8) bring-up path from linux amdgpu:
  vi_common_init → asic_init → start_smc → gmc_v8_0_mc_program → MC ucode →
  gmc_v8_0_gart_enable → smu7_request_smu_load_fw → gfx_v8_0_kcq"""
from __future__ import annotations
import os, struct, math, sys, time, contextlib
from typing import TYPE_CHECKING


def _darwin_egpu() -> bool:
  return sys.platform == "darwin"


def boot_no_doorbell() -> bool:
  """macOS USB4: BAR2 doorbell writes trigger APCIE MSI → kernel panic."""
  default = "1" if _darwin_egpu() else "0"
  return os.environ.get("AMD_BOOT_NO_DOORBELL", default) == "1"


def boot_use_mmio_wptr() -> bool:
  """TrustOS bare-metal path when doorbells are disabled."""
  if os.environ.get("AMD_BOOT_MMIO_WPTR") == "1":
    return True
  if os.environ.get("AMD_BOOT_MMIO_WPTR") == "0":
    return False
  return boot_no_doorbell()


def boot_allow_hqd_activation() -> bool:
  """Whether it is safe to make a compute HQD live (CP_HQD_ACTIVE + PRELOAD_REQ).

  Activating a KCQ whose MQD/ring/rptr/wptr live in GART **host sysmem** forces the
  MEC to DMA-read those addresses at preload. On the M1/USB4 (AppleT8103 PCIe)
  transport that device→host read is not serviceable and the bridge raises the
  `apciec unhandled interrupts (0x200000)` error → macOS kernel panic. Keep queues
  MQD-in-memory only (no activation) unless the operator explicitly opts in after a
  proven device-DMA path. `--boot-stage=kcq-ring-test`/`add` set their own gates."""
  return any(os.environ.get(k, "0") == "1" for k in (
    "AMD_BOOT_KCQ_ACTIVATE", "AMD_BOOT_RING_TEST", "AMD_BOOT_ADD", "AMD_BOOT_FULL"))

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
mmCP_ME_CNTL = 0x21b6
mmCP_MEC_CNTL = 0x208d
mmCP_HQD_ACTIVE = 0x3247
mmCP_PQ_STATUS = 0x2147
mmCP_MEC_DOORBELL_RANGE_LOWER = 0x2149
mmCP_MEC_DOORBELL_RANGE_UPPER = 0x214a
mmRLC_CNTL = 0xec00
mmRLC_CP_SCHEDULERS = 0xecaa
# Interrupt handler (IH) block — oss_3_0_d.h / bif_5_1_d.h / gfx_8_0_d.h
mmIH_RB_CNTL = 0xe30
mmIH_RB_RPTR = 0xe32
mmIH_RB_WPTR = 0xe33
mmIH_CNTL = 0xe36
mmIH_DOORBELL_RPTR = 0xe42
mmINTERRUPT_CNTL = 0x151a
mmINTERRUPT_CNTL2 = 0x151b
mmCP_INT_CNTL_RING0 = 0x306a
mmCPC_INT_CNTL = 0x30b4
IH_RB_CNTL__RB_ENABLE_MASK = 0x00000001
IH_RB_CNTL__ENABLE_INTR_MASK = 0x00020000
IH_DOORBELL_RPTR__ENABLE_MASK = 0x10000000
mmGRBM_SOFT_RESET = 0x2008
mmCP_PFP_UCODE_ADDR = 0xf814
mmCP_PFP_UCODE_DATA = 0xf815
mmCP_ME_RAM_WADDR = 0xf816
mmCP_ME_RAM_DATA = 0xf817
mmCP_CE_UCODE_ADDR = 0xf818
mmCP_CE_UCODE_DATA = 0xf819
mmCP_MEC_ME1_UCODE_ADDR = 0xf81a
mmCP_MEC_ME1_UCODE_DATA = 0xf81b
mmCP_MEC_ME2_UCODE_ADDR = 0xf81c
mmCP_MEC_ME2_UCODE_DATA = 0xf81d
mmRLC_GPM_UCODE_ADDR = 0xf83c
mmRLC_GPM_UCODE_DATA = 0xf83d
mmSDMA0_UCODE_ADDR = 0x3400
mmSDMA0_UCODE_DATA = 0x3401
mmSDMA0_F32_CNTL = 0x3412
mmSDMA1_UCODE_ADDR = 0x3600
mmSDMA1_UCODE_DATA = 0x3601
mmSDMA1_F32_CNTL = 0x3612
SDMA1_REG_OFFSET = 0x200

CP_ME_CNTL_HALT = 0x01000000 | 0x04000000 | 0x10000000  # CE|PFP|ME
CP_MEC_CNTL_HALT = 0x40000000 | 0x10000000  # ME1|ME2
CP_MEC_ME1_HALT = 0x40000000
CP_MEC_ME2_HALT = 0x10000000
CP_MEC_ME1_ONLY = CP_MEC_ME2_HALT  # ME1 run, ME2 stay halted (TrustOS: MEC2 VM faults)
GRBM_SOFT_RESET_RLC = 0x4
RLC_CNTL_ENABLE = 0x1
SDMA_F32_CNTL_HALT = 0x1
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
mmVM_CONTEXT1_PAGE_TABLE_BASE_ADDR = 0x550
mmVM_CONTEXT8_PAGE_TABLE_BASE_ADDR = 0x50e
mmVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR = 0x575
mmVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR = 0x576
mmVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET = 0x577
mmSCRATCH_REG0 = 0xc040
AMDGPU_NUM_VMID = 16
PACKET3_SET_UCONFIG_REG = 0x79
PACKET3_SET_UCONFIG_REG_START = 0xc000
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
FW_RLC_ONLY = UCODE_ID_RLC_G_MASK
FW_CP_GFX_MASK = (UCODE_ID_CP_CE_MASK | UCODE_ID_CP_PFP_MASK | UCODE_ID_CP_ME_MASK)
FW_COMPUTE_MIN = (FW_RLC_ONLY | FW_CP_GFX_MASK | UCODE_ID_CP_MEC_MASK)
FW_TO_LOAD = (FW_COMPUTE_MIN | UCODE_ID_SDMA0_MASK | UCODE_ID_SDMA1_MASK)

SMU_FW_BUF_SIZE = 200 * 4096
SMU_HDR_BUF_SIZE = 4096
PAGE_SIZE = 4096

# gmc_v8_0 GTT PTE: VALID|SYSTEM|SNOOPED|EXECUTABLE|READABLE|WRITEABLE (amdgpu_vm.h)
GART_PTE_FLAGS = 0x77

SMU_HDR_SOFT_REGS_OFF = 0x20000 + 48  # offsetof(SMU74_Firmware_Header, SoftRegisters)
SMU7_FIRMWARE_HEADER_LOCATION = 0x20000
SMU74_FIRMWARE_HDR_SOFTREGS = 48
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
DOORBELL_MEC_RING7 = 0x17  # amdgpu_doorbell.h — RANGE_UPPER uses ring7, not ring0+8
# KIQ shares MEC1 with KCQ but uses pipe=1 (KCQ uses pipe=0) — amdgpu_gfx_kiq_acquire
KIQ_ME, KIQ_PIPE, KIQ_QUEUE = 1, 1, 0
GFX8_MEC_HPD_SIZE = 4096
RING_SIZE = 0x10000
# gfx_v8_0_ring_funcs_kiq/compute: align_mask=0xff (256-dword ring commit)
VI_RING_ALIGN_MASK = 0xff
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

# gfx_v8_0_ring_funcs_kiq.nop
VI_PKT3_NOP = pkt3(0x10, 0x3FFF)


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
    self._soft_regs_start = 0

  def pci_online(self) -> bool:
    retries = int(os.environ.get("AMD_PCI_ONLINE_RETRIES", "5"))
    for _ in range(retries):
      try:
        if (self.dev.pci.read_config(0, 2) & 0xffff) == 0x1002:
          return True
      except Exception:
        pass
      time.sleep(0.05)
    return False

  def _check_pci(self, phase: str):
    if self.pci_online():
      return
    time.sleep(0.25)
    self.dev.pci.drain_mmio(bar=5, reg=0x2004)
    if self.pci_online():
      return
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

  def mmio_settle(self, label: str = "settle", heavy: bool = False):
    """USB4/TinyGPU: MMIO writes are queued; wait for backlog before unhalt."""
    if heavy:
      rounds = int(os.environ.get("AMD_MMIO_SETTLE_ROUNDS", "30"))
      pause_ms = int(os.environ.get("AMD_MMIO_SETTLE_MS", "100"))
    else:
      rounds = int(os.environ.get("AMD_MMIO_SETTLE_ROUNDS_LIGHT", "5"))
      pause_ms = int(os.environ.get("AMD_MMIO_SETTLE_MS_LIGHT", "50"))
    for i in range(rounds):
      self.mmio_sync_safe()
      if i % 5 == 0:
        self._check_pci(f"{label} {i}/{rounds}")
      if pause_ms:
        time.sleep(pause_ms / 1000.0)
    if int(os.environ.get("DEBUG", "0")):
      print(f"polaris: mmio_settle {label} rounds={rounds} pause_ms={pause_ms}", flush=True)

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
    """vi_srbm_select — oss_3_0_sh_mask.h SRBM_GFX_CNTL field layout."""
    val = ((pipe & 3) << 0) | ((me & 3) << 2) | ((vmid & 0xf) << 4) | ((queue & 7) << 8)
    self.wreg(mmSRBM_GFX_CNTL, val)

  def read_hqd_active(self, me=0, pipe=0, queue=0, vmid=0) -> int:
    self.srbm_select(me, pipe, queue, vmid)
    val = self.rreg(mmCP_HQD_ACTIVE)
    self.srbm_select(0, 0, 0, 0)
    return val

  def deactivate_hqd(self, me=0, pipe=0, queue=0, req: int = 1, timeout_s: float = 1.0):
    """gfx_v8_0_deactivate_hqd — drain queue before MQD reprogram."""
    self.srbm_select(me, pipe, queue, 0)
    if self.rreg(mmCP_HQD_ACTIVE) & 1:
      self.wreg(mmCP_HQD_DEQUEUE_REQUEST, req)
      deadline = time.time() + timeout_s
      while time.time() < deadline:
        if not (self.rreg(mmCP_HQD_ACTIVE) & 1):
          break
        time.sleep(0.001)
    self.wreg(mmCP_HQD_DEQUEUE_REQUEST, 0)
    self.wreg(mmCP_HQD_PQ_RPTR, 0)
    self.wreg(mmCP_HQD_PQ_WPTR, 0)
    self.srbm_select(0, 0, 0, 0)
    self.mmio_sync_safe()

  def boot_through_fw_direct(self, fw_mask: int | None = None, unhalt: bool | None = None):
    """ATOM → SMC → MC → GART → direct MMIO firmware (no compute/KIQ)."""
    from atom_replay import run_asic_init_if_needed, vram_training_ok
    if fw_mask is None:
      fw_mask = int(os.environ.get("AMD_BOOT_FW_MASK", str(FW_RLC_ONLY)), 0)
    self.vi_common_init()
    self.enable_vbios_rom()
    run_asic_init_if_needed(self)
    if not vram_training_ok(self):
      self.mc_program_light()
      with contextlib.suppress(RuntimeError):
        self.load_mc_firmware()
    self.gmc_sw_init()
    self.start_smc()
    self.process_smc_firmware_header()
    self.mc_program()
    with contextlib.suppress(RuntimeError):
      self.load_mc_firmware()
    if self.gart_pte_mem is None:
      self.gart_enable()
    if self.smc_running():
      self.load_ip_firmware_direct(fw_mask, unhalt=unhalt)
    else:
      raise RuntimeError("boot_through_fw_direct: SMC not running")


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
    if msg == PPSMC_MSG_LoadUcodes:
      self._check_pci("SMC LoadUcodes start")
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
    self.disable_gpu_interrupts("vi_common_init")
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
    """MM_INDEX pos: byte offset from vram_start (Linux amdgpu_device_mm_access)."""
    full_off = (mc_addr - self.vram_start) & 0xffffffffffffffff
    if os.environ.get("AMD_BOOT_MM_OFFSET", "full") == "visible":
      return (mc_addr - self.vram_visible_mc) & 0xffffffffffffffff
    return full_off

  def vram_mc_addr(self, byte_off: int) -> int:
    return (self.vram_start + byte_off) & 0xffffffffffffffff

  def vram_mm_write(self, mc_addr: int, data: bytes):
    """Write VRAM via mmMM_INDEX/mmMM_DATA when BAR0 aperture is dead."""
    pos = self.vram_mc_offset(mc_addr)
    if pos % 4 or len(data) % 4:
      raise ValueError(f"vram_mm_write needs 4-byte alignment pos={pos:#x} len={len(data)}")
    self.wreg(mmBIF_MM_INDACCESS_CNTL, 0)
    drain_every = int(os.environ.get("AMD_MMIO_DRAIN_EVERY", "128"))
    hi = None
    for i in range(0, len(data), 4):
      p = pos + i
      tmp = p >> 31
      self.wreg(mmMM_INDEX, (p & 0x7fffffff) | 0x80000000)
      if tmp != hi:
        self.wreg(mmMM_INDEX_HI, tmp)
        hi = tmp
      self.wreg(mmMM_DATA, struct.unpack_from('<I', data, i)[0])
      if drain_every and (i // 4) % drain_every == 0 and i:
        self.dev.pci.drain_mmio(bar=5, reg=0x2004)
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
    offs = [0x3000, 0x10000, self.vram_visible_mc - self.vram_start + 0x3000]
    for off in offs:
      off &= 0xffffffff
      mc = self.vram_mc_addr(off)
      try:
        self.vram_mm_write(mc, struct.pack('<I', pat))
        got = struct.unpack('<I', self.vram_mm_read(mc, 4))[0]
        ok = got == pat
        if int(os.environ.get("DEBUG", "0")):
          print(f"polaris: probe_vram_mm off={off:#x} mc={mc:#x} wrote={pat:#x} read={got:#x} ok={ok}", flush=True)
        if ok:
          return True
      except Exception as e:
        if int(os.environ.get("DEBUG", "0")):
          print(f"polaris: probe_vram_mm off={off:#x} failed: {e}", flush=True)
    return False

  def mc_init_locations(self):
    mem_mb = self.rreg(mmCONFIG_MEMSIZE) & 0xffff
    fb_loc = self.rreg(mmMC_VM_FB_LOCATION)
    if fb_loc not in (0, 0xffffffff):
      self.vram_start = ((fb_loc & 0xffff) << 24) & 0xffffffff
    else:
      self.vram_start = 0
    if mem_mb in (0, 0xffff) or mem_mb < 128:
      mem_mb = int(os.environ.get("AMD_VRAM_MB", "4096"))
    self.vram_size = mem_mb * 1024 * 1024
    self.vram_end = (self.vram_start + self.vram_size - 1) & 0xffffffff
    # VBIOS partial FB (e.g. 0xf400f400) + large override would wrap — normalize at mc_program_light
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

  def mc_program_light(self):
    """Minimal MC when VRAM not fully trained — avoid clobbering VBIOS apertures."""
    self.mc_init_locations()
    self.dev._vram_start = self.vram_visible_mc
    mem_mb = self.rreg(mmCONFIG_MEMSIZE) & 0xffff
    want_mb = int(os.environ.get("AMD_VRAM_MB", "4096"))
    if mem_mb < 128 and want_mb >= 128:
      self.vram_start = 0
      self.vram_size = want_mb * 1024 * 1024
      self.vram_end = (self.vram_size - 1) & 0xffffffff
      bar_bytes = self.dev.bar0_size
      self.vram_visible_mc = (self.vram_end - bar_bytes + 1) & 0xffffffff if self.vram_size > bar_bytes else self.vram_start
      self.dev._vram_start = self.vram_visible_mc
      self.wreg(mmCONFIG_MEMSIZE, want_mb)
      tmp = ((self.vram_end >> 24) & 0xffff) << 16 | ((self.vram_start >> 24) & 0xffff)
      self.wreg(mmMC_VM_FB_LOCATION, tmp)
    self.wreg(mmBIF_FB_EN, 0x3)
    self.mmio_sync_safe()
    if int(os.environ.get("DEBUG", "0")):
      print(f"polaris: mc_program_light FB_LOC={self.rreg(mmMC_VM_FB_LOCATION):#x} "
            f"MEMSIZE={self.config_memsize_mb()} vram={self.vram_start:#x}-{self.vram_end:#x} "
            f"visible={self.vram_visible_mc:#x}", flush=True)

  def mc_program(self):
    mode = os.environ.get("AMD_BOOT_MC_PROGRAM", "auto")
    if mode == "light" or (mode == "auto" and not self.vram_trained()):
      self.mc_program_light()
      return
    if mode == "skip":
      return
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
    if os.environ.get("AMD_BOOT_HDP_NONSURFACE", "0") == "1":
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
    wait_logged = False
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
      if int(os.environ.get("DEBUG", "0")) and not wait_logged:
        print(f"polaris: MC training wait MISC0={misc0:#x}", flush=True)
        wait_logged = True
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
    pte_base = pte_base_addr >> 12
    self.wreg(mmVM_CONTEXT0_PAGE_TABLE_BASE_ADDR, pte_base)
    self.wreg(mmVM_CONTEXT0_PROTECTION_FAULT_DEFAULT_ADDR, 0)
    self.wreg(mmVM_CONTEXT0_CNTL2, 0)
    # ENABLE_CONTEXT | PAGE_TABLE_DEPTH=0 | RANGE_PROTECTION_FAULT_ENABLE_DEFAULT
    self.wreg(mmVM_CONTEXT0_CNTL, 0x11)
    self.wreg(mmVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR, 0)
    self.wreg(mmVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR, 0)
    self.wreg(mmVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET, 0)
    self.wreg(mmVM_CONTEXT1_PAGE_TABLE_START_ADDR, 0)
    self.wreg(mmVM_CONTEXT1_PAGE_TABLE_END_ADDR, (1 << 28) - 1)
    for i in range(1, AMDGPU_NUM_VMID):
      if i < 8:
        self.wreg(mmVM_CONTEXT0_PAGE_TABLE_BASE_ADDR + i, pte_base)
      else:
        self.wreg(mmVM_CONTEXT8_PAGE_TABLE_BASE_ADDR + (i - 8), pte_base)
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
      self._gart_pte_flush()
      self.hdp_flush()
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

  def _gart_pte_flush(self):
    # Flush host PTE table so eGPU DMA sees GART updates (M1 lacks IO coherency).
    if self.gart_pte_sysmem is None:
      return
    from add import sysmem_dma_flush
    sysmem_dma_flush(self.gart_pte_sysmem, len(self.gart_pte_mem))

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
      self._gart_pte_flush()
      self.hdp_flush()
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

  def _flush_fw_sysmem(self, layout: str, fw_mem, extra=None):
    """ARM/M1: CPU cache may hide sysmem writes from eGPU DMA (rpi-pcie #756)."""
    if layout not in ("hybrid", "agp", "gtt"):
      return
    from add import sysmem_dma_flush
    for m, sz in [(fw_mem, SMU_FW_BUF_SIZE)] + (extra or []):
      if m is not None:
        sysmem_dma_flush(m, sz)
    if int(os.environ.get("DEBUG", "0")):
      print("polaris: sysmem_dma_flush fw_buf", flush=True)

  def alloc_gtt_buffer(self, size: int, align: int = PAGE_SIZE) -> tuple[int, object, list[int]]:
    """Allocate host sysmem, map into GART, return (gpu_va, mem, paddrs)."""
    nbytes = round_up(size, align)
    mem, paddrs, _ = self.alloc_sysmem_buffer(nbytes, contiguous=True)
    if not paddrs:
      raise RuntimeError("alloc_gtt_buffer: no paddrs from alloc_sysmem")
    gpu_va = self.map_sysmem_gpu(paddrs, nbytes)
    from add import sysmem_dma_flush
    sysmem_dma_flush(mem, nbytes)
    return gpu_va, mem, paddrs

  def probe_gart_dma(self) -> dict:
    """Validate GART PTE self-map + sysmem page mapping (CPU-side; run before kiq-map)."""
    from add import sysmem_dma_flush
    self.gmc_sw_init()
    if self.gart_pte_mem is None:
      os.environ["AMD_BOOT_GART_SYSMEM"] = "1"
      self.gart_enable()
    pat = 0xCAFEBABE
    src_va, src_mem, paddrs = self.alloc_gtt_buffer(PAGE_SIZE)
    src_mem[0:4] = struct.pack('<I', pat)
    sysmem_dma_flush(src_mem, PAGE_SIZE)
    base_pfn = (src_va - self.gart_start) >> 12
    pte_off = base_pfn * 4
    pte_cpu = struct.unpack_from('<I', self.gart_pte_mem, pte_off)[0]
    expected_pte = ((paddrs[0] >> 12) << 12) | GART_PTE_FLAGS
    pte_host = 0
    if self.gart_pte_sysmem is not None:
      pte_host = struct.unpack_from('<I', bytes(self.gart_pte_sysmem[pte_off:pte_off + 4]), 0)[0]
    self_map = struct.unpack_from('<I', self.gart_pte_mem, 0)[0]
    ok = (pte_cpu == expected_pte) and (not self.gart_pte_sysmem or pte_host == expected_pte)
    result = {
      "gart_start": self.gart_start,
      "src_va": src_va,
      "paddr": paddrs[0],
      "pte_cpu": pte_cpu,
      "pte_host": pte_host,
      "pte_expected": expected_pte,
      "pte_ok": ok,
      "self_map_pte": self_map,
      "self_map_valid": bool(self_map & 0x1),
      "pattern": pat,
    }
    print(f"gart_probe pte_ok={ok} src_va={src_va:#x} paddr={paddrs[0]:#x} "
          f"pte={pte_cpu:#x} expected={expected_pte:#x} self_map={self_map:#x}")
    if not ok:
      raise RuntimeError(f"GART PTE mismatch: got {pte_cpu:#x} expected {expected_pte:#x}")
    return result

  def cp_gfx_enable(self, enable: bool):
    tmp = self.rreg(mmCP_ME_CNTL)
    if enable:
      tmp &= ~CP_ME_CNTL_HALT
    else:
      tmp |= CP_ME_CNTL_HALT
    self.wreg(mmCP_ME_CNTL, tmp)
    time.sleep(0.05)

  def cp_compute_enable(self, enable: bool, me1_only: bool | None = None):
    if me1_only is None:
      me1_only = os.environ.get("AMD_BOOT_MEC2_HALT", "1") == "1"
    if enable:
      self.mmio_settle("pre-mec-unhalt", heavy=True)
      self.wreg(mmCP_MEC_CNTL, CP_MEC_ME1_ONLY if me1_only else 0)
    else:
      self.wreg(mmCP_MEC_CNTL, CP_MEC_CNTL_HALT)
    time.sleep(0.05)

  def interrupts_masked(self) -> bool:
    default = "1" if _darwin_egpu() else "0"
    return os.environ.get("AMD_BOOT_MASK_INTERRUPTS", default) == "1"

  def disable_gpu_interrupts(self, label: str = ""):
    """Quiesce every GPU interrupt source so the eGPU never asserts an IRQ.

    macOS/TinyGPU has no handler for eGPU interrupts; once CP/MEC firmware runs it
    raises MSIs the USB4 bridge cannot route → 'apciec unhandled interrupts' kernel
    panic. Mirrors tonga_ih_disable_interrupts and additionally clears the CP /
    compute-pipe EOP/priv/error interrupt enables. Complements PolarisDevice's
    PCI-level MSI mask (add.py mask_msi); run before unhalting firmware."""
    if not self.interrupts_masked():
      return
    with contextlib.suppress(Exception):
      rb = self.rreg(mmIH_RB_CNTL)
      self.wreg(mmIH_RB_CNTL, rb & ~(IH_RB_CNTL__RB_ENABLE_MASK | IH_RB_CNTL__ENABLE_INTR_MASK))
      self.wreg(mmIH_RB_RPTR, 0)
      self.wreg(mmIH_RB_WPTR, 0)
      dbell = self.rreg(mmIH_DOORBELL_RPTR)
      self.wreg(mmIH_DOORBELL_RPTR, dbell & ~IH_DOORBELL_RPTR__ENABLE_MASK)
    with contextlib.suppress(Exception):
      self.wreg(mmCP_INT_CNTL_RING0, 0)
      self.wreg(mmCPC_INT_CNTL, 0)
    self.mmio_sync_safe()
    if int(os.environ.get("DEBUG", "0")):
      with contextlib.suppress(Exception):
        print(f"polaris: GPU interrupts masked {label} "
              f"IH_RB_CNTL={self.rreg(mmIH_RB_CNTL):#x} "
              f"CP_INT_CNTL_RING0={self.rreg(mmCP_INT_CNTL_RING0):#x} "
              f"CPC_INT_CNTL={self.rreg(mmCPC_INT_CNTL):#x}", flush=True)

  def rlc_stop(self):
    tmp = self.rreg(mmRLC_CNTL) & ~RLC_CNTL_ENABLE
    self.wreg(mmRLC_CNTL, tmp)
    time.sleep(0.05)

  def rlc_start(self):
    tmp = self.rreg(mmRLC_CNTL) | RLC_CNTL_ENABLE
    self.wreg(mmRLC_CNTL, tmp)
    time.sleep(0.05)

  def rlc_reset(self):
    tmp = self.rreg(mmGRBM_SOFT_RESET) | GRBM_SOFT_RESET_RLC
    self.wreg(mmGRBM_SOFT_RESET, tmp)
    time.sleep(0.05)
    tmp = self.rreg(mmGRBM_SOFT_RESET) & ~GRBM_SOFT_RESET_RLC
    self.wreg(mmGRBM_SOFT_RESET, tmp)
    time.sleep(0.05)

  def sdma_enable(self, enable: bool):
    for off in (0, SDMA1_REG_OFFSET):
      reg = mmSDMA0_F32_CNTL + off
      tmp = self.rreg(reg)
      if enable:
        tmp &= ~SDMA_F32_CNTL_HALT
      else:
        tmp |= SDMA_F32_CNTL_HALT
      self.wreg(reg, tmp)
    time.sleep(0.05)

  def _fw_ucode_words(self, blob: bytes) -> tuple[list[int], int]:
    ucode_off, ucode_sz = parse_common_fw(blob)
    version = _le32(blob, 16)  # amdgpu_firmware_header::ucode_version (gfx_v7_0.c)
    words = [_le32(blob, ucode_off + i) for i in range(0, ucode_sz, 4)]
    return words, version

  def _mmio_drain_every(self, nwords: int) -> int:
    """USB4 TinyGPU: fire-and-forget MMIO needs periodic drain; large blobs need tighter spacing."""
    env = os.environ.get("AMD_MMIO_DRAIN_EVERY")
    if env is not None:
      return max(0, int(env))
    if nwords > 32000:
      return 32
    if nwords > 8000:
      return 64
    return 128

  def _mmio_load_ucode(self, addr_reg: int, data_reg: int, words: list[int], version: int,
                       final_addr: int | None = None, label: str = "ucode"):
    drain = self._mmio_drain_every(len(words))
    pci_every = max(0, int(os.environ.get("AMD_BOOT_FW_PCI_EVERY", "4096")))
    large = len(words) > 32000
    pause_ms = max(0, int(os.environ.get("AMD_BOOT_FW_WRITE_PAUSE_MS", "8" if large else "0")))
    self.wreg(addr_reg, 0)
    self.mmio_sync_safe()
    for i, w in enumerate(words):
      self.wreg(data_reg, w)
      if drain and i and (i % drain) == 0:
        self.mmio_sync_safe()
        if pause_ms:
          time.sleep(pause_ms / 1000.0)
      if pci_every and i and (i % pci_every) == 0:
        self._check_pci(f"direct {label} {i}/{len(words)}")
    done_addr = version if final_addr is None else final_addr
    self.wreg(addr_reg, done_addr)
    self.mmio_sync_safe()
    self.mmio_settle(f"post-{label}", heavy=large)
    self._check_pci(f"direct {label} done")

  def load_ip_firmware_direct(self, fw_mask: int | None = None, unhalt: bool | None = None):
    """gfx_v7_0/cik_sdma direct MMIO ucode upload — bypasses SMC LoadUcodes.

    Linux refs: gfx_v7_0_cp_gfx_load_microcode, gfx_v7_0_cp_compute_load_microcode
    (MEC final ADDR=0), cik_sdma_load_microcode (SDMA ADDR=version).
    """
    if fw_mask is None:
      fw_mask = int(os.environ.get("AMD_BOOT_FW_MASK", str(FW_RLC_ONLY)), 0)
    if unhalt is None:
      unhalt = os.environ.get("AMD_BOOT_FW_UNHALT", "1") == "1"
    pause_ms = max(0, int(os.environ.get("AMD_BOOT_FW_PAUSE_MS", "50")))
    want = {
      UCODE_ID_RLC_G: ("polaris10_rlc.bin", mmRLC_GPM_UCODE_ADDR, mmRLC_GPM_UCODE_DATA,
                       UCODE_ID_RLC_G_MASK, None),
      UCODE_ID_CP_PFP: ("polaris10_pfp.bin", mmCP_PFP_UCODE_ADDR, mmCP_PFP_UCODE_DATA,
                        UCODE_ID_CP_PFP_MASK, None),
      UCODE_ID_CP_CE: ("polaris10_ce.bin", mmCP_CE_UCODE_ADDR, mmCP_CE_UCODE_DATA,
                       UCODE_ID_CP_CE_MASK, None),
      UCODE_ID_CP_ME: ("polaris10_me.bin", mmCP_ME_RAM_WADDR, mmCP_ME_RAM_DATA,
                       UCODE_ID_CP_ME_MASK, None),
      UCODE_ID_CP_MEC: ("polaris10_mec.bin", mmCP_MEC_ME1_UCODE_ADDR, mmCP_MEC_ME1_UCODE_DATA,
                        UCODE_ID_CP_MEC_MASK, 0),  # gfx_v7_0_cp_compute_load_microcode
      UCODE_ID_SDMA0: ("polaris10_sdma.bin", mmSDMA0_UCODE_ADDR, mmSDMA0_UCODE_DATA,
                       UCODE_ID_SDMA0_MASK, None),
      UCODE_ID_SDMA1: ("polaris10_sdma1.bin", mmSDMA1_UCODE_ADDR, mmSDMA1_UCODE_DATA,
                       UCODE_ID_SDMA1_MASK, None),
    }
    self.rlc_stop()
    self.cp_gfx_enable(False)
    self.cp_compute_enable(False)
    self.sdma_enable(False)
    self.rlc_reset()
    loaded = []
    for ucode_id, (name, addr_reg, data_reg, bit, final_addr) in want.items():
      if not (fw_mask & bit):
        continue
      blob = self.fw(name)
      words, version = self._fw_ucode_words(blob)
      if int(os.environ.get("DEBUG", "0")):
        print(f"polaris: direct ucode {name} words={len(words)} ver={version:#x} "
              f"final_addr={(final_addr if final_addr is not None else version):#x}", flush=True)
      t0 = time.time()
      self._mmio_load_ucode(addr_reg, data_reg, words, version, final_addr=final_addr, label=name)
      loaded.append(name)
      if int(os.environ.get("DEBUG", "0")):
        print(f"polaris: direct {name} ok in {time.time()-t0:.1f}s", flush=True)
      if pause_ms:
        time.sleep(pause_ms / 1000.0)
    if fw_mask & UCODE_ID_RLC_G_MASK:
      self.rlc_start()
    if unhalt:
      self.disable_gpu_interrupts("pre-unhalt-direct")
      if fw_mask & (FW_CP_GFX_MASK | UCODE_ID_CP_MEC_MASK):
        self.cp_gfx_enable(True)
      if fw_mask & UCODE_ID_CP_MEC_MASK:
        self.cp_compute_enable(True)
      if fw_mask & (UCODE_ID_SDMA0_MASK | UCODE_ID_SDMA1_MASK):
        self.sdma_enable(True)
    print(f"polaris: direct MMIO firmware loaded mask={fw_mask:#x} "
          f"unhalt={unhalt} ({', '.join(loaded) or 'none'})", flush=True)

  def unhalt_loaded_firmware(self, fw_mask: int | None = None):
    """Unhalt CP/SDMA after upload+settle (separate from upload for USB4 safety)."""
    if fw_mask is None:
      fw_mask = int(os.environ.get("AMD_BOOT_FW_MASK", str(FW_COMPUTE_MIN)), 0)
    # Mask GPU interrupts before firmware goes live so it cannot raise MSIs that
    # kernel-panic the macOS USB4 bridge.
    self.disable_gpu_interrupts("pre-unhalt")
    if fw_mask & (FW_CP_GFX_MASK | UCODE_ID_CP_MEC_MASK):
      self.cp_gfx_enable(True)
    if fw_mask & UCODE_ID_CP_MEC_MASK:
      self.cp_compute_enable(True)
    if fw_mask & (UCODE_ID_SDMA0_MASK | UCODE_ID_SDMA1_MASK):
      self.sdma_enable(True)
    print(f"polaris: firmware unhalt mask={fw_mask:#x} "
          f"CP_MEC_CNTL={self.rreg(mmCP_MEC_CNTL):#x}", flush=True)

  def load_ip_firmware_prereqs(self) -> tuple[bool, str, bool, bool]:
    """Whether LoadUcodes is safe: Linux needs a CPU-writable VRAM path (BAR0 or
    MM_INDEX) for the SMC TOC/header/scratch. On this TinyGPU/USB4 eGPU the VRAM
    aperture is dead even after ATOM training completes, so trained registers
    alone are NOT sufficient — SMC DMA of the TOC would hang and drop USB4.
    Require an actually-verified write path, or a GART-sysmem DMA layout."""
    bar0_ok = self.probe_bar0_writes()
    mm_ok = self.probe_vram_mm_writes() if not bar0_ok else False
    trained = self.vram_trained()
    if bar0_ok or mm_ok:
      return True, f"trained={trained} bar0={bar0_ok} mm_index={mm_ok}", bar0_ok, mm_ok
    return False, (
      f"VRAM trained={trained} but no CPU-visible VRAM data path (BAR0+MM_INDEX both "
      f"dead on this TinyGPU/USB4 transport) — SMC cannot DMA the firmware TOC/header; "
      f"LoadUcodes will hang and drop the USB4 link. Need a working BAR0 aperture or a "
      f"proven GART-sysmem DMA path (set AMD_BOOT_LOADUCODES_UNTRAINED=1 to force — unsafe)."
    ), bar0_ok, mm_ok

  def load_ip_firmware(self):
    """Linux smu7_request_smu_load_fw — after gmc_v8_0_hw_init (mc_program + gart_enable)."""
    if not self.smc_running():
      raise RuntimeError("SMC not running before load_ip_firmware")
    self.mc_init_locations()
    self.dev._vram_start = self.vram_visible_mc
    fw_mask = int(os.environ.get("AMD_BOOT_FW_MASK", str(FW_TO_LOAD)), 0)
    allowed, reason, bar0_ok, mm_ok = self.load_ip_firmware_prereqs()
    if not allowed and os.environ.get("AMD_BOOT_LOADUCODES_UNTRAINED", "0") != "1":
      raise RuntimeError(f"LoadUcodes refused: {reason}")
    if not allowed:
      print(f"polaris: WARNING — forced LoadUcodes on untrained VRAM (high PCIe drop risk)", flush=True)
    layout = os.environ.get("AMD_BOOT_FW_LAYOUT", "auto")
    if layout == "auto":
      if bar0_ok:
        layout = "vram"
      elif mm_ok:
        layout = "hybrid"  # Linux: VRAM header/smu + GART fw_buf
      else:
        layout = "gtt"
        if int(os.environ.get("DEBUG", "0")):
          print(f"polaris: auto layout gtt ({reason})", flush=True)
    if layout == "gtt" and not allowed:
      raise RuntimeError(
        "GTT-only firmware layout unusable without VRAM path — fix ATOM/MC training first")
    use_gtt = layout == "gtt"
    use_phys = layout in ("agp",) or os.environ.get("AMD_BOOT_FW_PHYS_ADDR", "0") == "1"

    if layout == "hybrid":
      self.ensure_gart_ready()
      smu_dram_off = round_up(0x100000, PAGE_SIZE)
      hdr_off = round_up(smu_dram_off + SMU_FW_BUF_SIZE, PAGE_SIZE)
      smu_dram_gpu = self.vram_mc_addr(smu_dram_off)
      hdr_gpu = self.vram_mc_addr(hdr_off)
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
      self.ensure_gart_ready()
      if self.gart_pte_mem is None:
        raise RuntimeError("GART not enabled; call gart_enable before GTT load_ip_firmware")
      total = SMU_FW_BUF_SIZE + SMU_HDR_BUF_SIZE + SMU_FW_BUF_SIZE
      mem, paddrs, nbytes = self.alloc_sysmem_buffer(total, contiguous=True)
      if not paddrs:
        raise RuntimeError("gtt layout needs contiguous sysmem")
      base_va = self.map_sysmem_gpu(paddrs, nbytes)
      smu_off = 0
      hdr_off = SMU_FW_BUF_SIZE
      fw_off = SMU_FW_BUF_SIZE + SMU_HDR_BUF_SIZE
      smu_dram_gpu = base_va + smu_off
      hdr_gpu = base_va + hdr_off
      fw_gpu_base = base_va + fw_off
      smu_dram_mem = mem
      hdr_mem = mem
      fw_paddr_base = paddrs[0] & ~0xfff
      if use_phys:
        smu_dram_gpu = paddrs[0] & ~0xfff
        hdr_gpu = (paddrs[0] & ~0xfff) + hdr_off
        fw_gpu_base = (paddrs[0] & ~0xfff) + fw_off

      def writer(off: int, image: bytes):
        mem[fw_off + off:fw_off + off + len(image)] = image

      def addr_fixup(mc_addr: int, _image: bytes) -> int:
        if not use_phys:
          return mc_addr
        return fw_paddr_base + (mc_addr - fw_gpu_base)

      layout_tag = "gtt-contig" + ("-phys" if use_phys else "")
      addr_fixup_fn = addr_fixup
      vram_writer = None
      fw_mem = mem
      self.gart_flush_tlb()
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
      if os.environ.get("AMD_BOOT_SMU_SCRATCH_WRITE", "0") == "1":
        vram_writer(smu_dram_gpu, bytes(SMU_FW_BUF_SIZE))
      vram_writer(hdr_gpu, toc)
    elif layout == "agp":
      mem[hdr_off_vram:hdr_off_vram + len(toc)] = toc
    elif use_gtt:
      mem[hdr_off:hdr_off + len(toc)] = toc
    else:
      if bar0_ok:
        self.dev.upload(hdr_off, toc)
      else:
        self.vram_mm_write(hdr_gpu, toc)
    flush_extra = []
    if layout == "gtt":
      flush_extra = [(mem, min(total, nbytes))]
    self._flush_fw_sysmem(layout, fw_mem if layout in ("hybrid", "agp", "gtt") else None, flush_extra)
    if layout != "agp":
      self.vram_flush()
    if int(os.environ.get("DEBUG", "0")):
      soft = self.read_soft_regs_start()
      print(f"polaris: load_ip_firmware {layout_tag} soft_regs={soft:#x} "
            f"smu_dram={smu_dram_gpu:#x} hdr={hdr_gpu:#x} fw_buf={fw_gpu_base:#x} mask={fw_mask:#x}", flush=True)
      for ucode_id, ver, addr, sz, flags in entries:
        print(f"  toc id={ucode_id} ver={ver:#x} addr={addr:#x} sz={sz:#x} flags={flags}", flush=True)
    if os.environ.get("AMD_BOOT_SKIP_UCODE_CLEAR", "0") != "1":
      self.clear_ucode_load_status()
    settle = min(self._settle_s(), float(os.environ.get("AMD_BOOT_LOADUCODES_SETTLE_MS", "200")) / 1000.0)
    if settle > 0:
      time.sleep(settle)
    self.dev.pci.drain_mmio(bar=5, reg=0x2004)
    skip_smu_dram = os.environ.get("AMD_BOOT_SMC_SKIP_DRAM", "0") == "1"
    msgs = []
    if not skip_smu_dram:
      msgs += [
        (PPSMC_MSG_SMU_DRAM_ADDR_HI, smu_dram_gpu >> 32, "SMU_DRAM_HI"),
        (PPSMC_MSG_SMU_DRAM_ADDR_LO, smu_dram_gpu & 0xffffffff, "SMU_DRAM_LO"),
      ]
    msgs += [
      (PPSMC_MSG_DRV_DRAM_ADDR_HI, hdr_gpu >> 32, "DRV_DRAM_HI"),
      (PPSMC_MSG_DRV_DRAM_ADDR_LO, hdr_gpu & 0xffffffff, "DRV_DRAM_LO"),
      (PPSMC_MSG_LoadUcodes, fw_mask, "LoadUcodes"),
    ]
    for msg, arg, label in msgs:
      if settle > 0:
        time.sleep(settle / 3)
      if int(os.environ.get("DEBUG", "0")):
        print(f"polaris: SMC send {label} arg={arg:#x}", flush=True)
      self.smc_send_msg(msg, arg, label=label)
    if not self.wait_ucode_load(fw_mask, timeout_s=self._timeout_s("AMD_BOOT_UCODE_LOAD_TIMEOUT_S", 20.0)):
      resp = self.rreg(mmSMC_RESP_0) & 0xffff
      status = self.smc_soft_reg(ixSMU74_UcodeLoadStatus)
      status_s = f"{status:#x}" if status is not None else "garbage"
      raise RuntimeError(
        f"IP firmware load timeout (RESP={resp:#x} UcodeLoadStatus={status_s} want {fw_mask:#x}) {self.smc_diag()}")

  def kiq_setting(self, me: int, pipe: int, queue: int):
    """gfx_v8_0_kiq_setting — tell RLC which queue is KIQ."""
    self.rlc_exit_safe_mode()
    tmp = self.rreg(mmRLC_CP_SCHEDULERS) & 0xffffff00
    tmp |= (me << 5) | (pipe << 3) | queue
    self.wreg(mmRLC_CP_SCHEDULERS, tmp | 0x80)

  def rlc_exit_safe_mode(self):
    """gfx_v8_0_unset_safe_mode — MEC/KIQ need RLC out of safe mode."""
    data = self.rreg(mmRLC_CNTL)
    self.wreg(mmRLC_SAFE_MODE, (data | 0x1) & ~0x1e)
    deadline = time.time() + 0.05
    while time.time() < deadline:
      if (self.rreg(mmRLC_SAFE_MODE) & 0x1) == 0:
        return
      time.sleep(0.001)

  def set_mec_doorbell_range(self):
    """gfx_v8_0_set_mec_doorbell_range (Polaris10 > Tonga)."""
    self.wreg(mmCP_MEC_DOORBELL_RANGE_LOWER, DOORBELL_KIQ << 2)
    self.wreg(mmCP_MEC_DOORBELL_RANGE_UPPER, DOORBELL_MEC_RING7 << 2)
    if not boot_no_doorbell():
      self.wreg(mmCP_PQ_STATUS, self.rreg(mmCP_PQ_STATUS) | CP_PQ_STATUS_DOORBELL_ENABLE_MASK)
    self.mmio_sync_safe()

  def compute_fw_loaded(self) -> bool:
    """ME1 running with SMC up — skip re-upload on kiq-map if prior stage left GPU hot."""
    if not self.smc_running():
      return False
    mec = self.rreg(mmCP_MEC_CNTL)
    return (mec & CP_MEC_ME1_HALT) == 0

  def boot_minimal_for_compute(self):
    """GART + doorbells when firmware already resident (GPU state persists across CLI invocations)."""
    self.gmc_sw_init()
    # TinyGPU eGPU: host-backed GART PTE table (GPU must DMA sysmem rings/MQDs).
    self.gart_pte_mem = None
    self.gart_pte_sysmem = None
    os.environ["AMD_BOOT_GART_SYSMEM"] = "1"
    self.gart_enable()
    if not self.compute_fw_loaded():
      raise RuntimeError(
        "compute firmware not loaded — run --boot-stage=fw-mec && --boot-stage=fw-start first")
    self.rlc_exit_safe_mode()
    self.enable_compute()
    self.set_mec_doorbell_range()

  def enable_compute(self):
    self.disable_gpu_interrupts("pre-enable-compute")
    self.cp_compute_enable(True)
    self.set_mec_doorbell_range()

  def init_compute_queue(self) -> 'ComputeQueue':
    if self._compute is None:
      self._compute = ComputeQueue(self)
      self._compute.init()
      self._compute.setup_with_kiq()
    return self._compute

  def process_smc_firmware_header(self):
    """polaris10_process_firmware_header — read SoftRegisters ptr from SMC FW header."""
    vals = self.read_smc_ram(SMU7_FIRMWARE_HEADER_LOCATION + SMU74_FIRMWARE_HDR_SOFTREGS, 1)
    val = vals[0] if vals else 0
    if self._smc_val_ok(val):
      self._soft_regs_start = val
      if int(os.environ.get("DEBUG", "0")):
        print(f"polaris: SMC soft_regs_start={val:#x}", flush=True)
    elif int(os.environ.get("DEBUG", "0")):
      print(f"polaris: SMC firmware header SoftRegisters invalid ({val:#x})", flush=True)

  def read_soft_regs_start(self) -> int:
    if not self._soft_regs_start:
      self.process_smc_firmware_header()
    return self._soft_regs_start

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
    poll = max(0.05, float(os.environ.get("AMD_BOOT_UCODE_POLL_MS", "100")) / 1000.0)
    n = 0
    while time.time() < deadline:
      n += 1
      if n % 5 == 0:
        with contextlib.suppress(RuntimeError):
          self._check_pci("ucode load poll")
      status = self.smc_soft_reg(ixSMU74_UcodeLoadStatus)
      if status != last_status and int(os.environ.get("DEBUG", "0")):
        print(f"polaris: UcodeLoadStatus={status:#x} (want {fw_mask:#x})", flush=True)
        last_status = status
      if (status & fw_mask) == fw_mask:
        return True
      time.sleep(poll)
    return False

  def post_atom_sync(self):
    """HDP flush/invalidate after ATOM VRAM/MM writes (vi_flush_hdp path)."""
    self.hdp_flush()
    self.hdp_invalidate()
    self.mmio_sync_safe()

  def boot(self):
    if self.dev.gpu_ready():
      return
    from atom_replay import run_asic_init_if_needed, vram_training_ok
    self.vi_common_init()
    self.enable_vbios_rom()
    run_asic_init_if_needed(self)
    if not vram_training_ok(self):
      self.mc_program_light()
      try:
        self.load_mc_firmware()
        if int(os.environ.get("DEBUG", "0")):
          print(f"polaris: post-atom MC MISC0={self.rreg(mmMC_SEQ_MISC0):#x} "
                f"MEMSIZE={self.config_memsize_mb()}", flush=True)
      except RuntimeError as e:
        if int(os.environ.get("DEBUG", "0")):
          print(f"polaris: post-atom MC load ({e})", flush=True)
    self.gmc_sw_init()
    self.start_smc()
    self.process_smc_firmware_header()
    # Linux: gmc_v8_0_hw_init (mc_program + MC ucode) before amdgpu_device_fw_loading
    self.mc_program()
    try:
      self.load_mc_firmware()
      if int(os.environ.get("DEBUG", "0")):
        mem_mb = self.rreg(mmCONFIG_MEMSIZE) & 0xffff
        misc0 = self.rreg(mmMC_SEQ_MISC0)
        print(f"polaris: pre-LoadUcodes MC MISC0={misc0:#x} MEMSIZE={mem_mb:#x}", flush=True)
    except RuntimeError as e:
      if int(os.environ.get("AMD_BOOT_STRICT_MC", "0")):
        raise
      if int(os.environ.get("DEBUG", "0")):
        print(f"polaris: pre-LoadUcodes MC load ({e})", flush=True)
    fw_allowed, fw_reason, bar0_ok, mm_ok = self.load_ip_firmware_prereqs()
    if int(os.environ.get("DEBUG", "0")):
      print(f"polaris: FW probe bar0={bar0_ok} mm_index={mm_ok} allowed={fw_allowed} "
            f"MEMSIZE={self.config_memsize_mb()} MISC0={self.rreg(mmMC_SEQ_MISC0):#x}", flush=True)
      if not fw_allowed:
        print(f"polaris: {fw_reason}", flush=True)
    fw_layout = os.environ.get("AMD_BOOT_FW_LAYOUT", "auto")
    need_gart = fw_layout in ("gtt", "hybrid", "agp") or (fw_layout == "auto" and not bar0_ok)
    if need_gart:
      self.gart_enable()
    fw_mask = int(os.environ.get("AMD_BOOT_FW_MASK", str(FW_TO_LOAD)), 0)
    fw_direct = os.environ.get("AMD_BOOT_FW_DIRECT", "auto")
    if self.smc_running():
      force = os.environ.get("AMD_BOOT_LOADUCODES_UNTRAINED", "0") == "1"
      if fw_allowed:
        self.load_ip_firmware()
      elif force:
        print("polaris: WARNING — AMD_BOOT_LOADUCODES_UNTRAINED=1 (crash risk)", flush=True)
        self.load_ip_firmware()
      elif fw_direct != "0":
        print(f"polaris: LoadUcodes skipped — direct MMIO upload ({fw_reason})", flush=True)
        self.load_ip_firmware_direct(fw_mask)
      else:
        print(f"polaris: skip LoadUcodes ({fw_reason})", flush=True)
    try:
      self.load_mc_firmware()
      if int(os.environ.get("DEBUG", "0")):
        mem_mb = self.rreg(mmCONFIG_MEMSIZE) & 0xffff
        misc0 = self.rreg(mmMC_SEQ_MISC0)
        print(f"polaris: MC training MISC0={misc0:#x} MEMSIZE={mem_mb:#x}", flush=True)
    except RuntimeError as e:
      if int(os.environ.get("AMD_BOOT_STRICT_MC", "0")):
        raise
      if int(os.environ.get("DEBUG", "0")):
        print(f"polaris: MC load skipped ({e})", flush=True)
    if self.gart_pte_mem is None:
      self.gart_enable()
    self.enable_compute()
    self.init_compute_queue()
    if not self.dev.gpu_ready() and self.rreg(mmCP_MEC_CNTL) == CP_MEC_CNTL_HALT:
      if not vram_training_ok(self):
        raise RuntimeError(
          "Polaris boot stopped safely: VRAM not trained — firmware not loaded. "
          "Need ATOM training (MEMSIZE>=128, MISC0|0x80) or AMD_BOOT_FW_DIRECT=1.")
      raise RuntimeError(
        f"Polaris boot incomplete: SMC={self.smc_running()} "
        f"CP_MEC_CNTL={self.rreg(mmCP_MEC_CNTL):#x} CP_HQD_ACTIVE={self.rreg(mmCP_HQD_ACTIVE):#x}")


mmCP_PQ_WPTR_POLL_CNTL = 0x3083
mmCP_HQD_PQ_RPTR = 0x324f
mmCP_HQD_IB_BASE_ADDR_LO = 0x3257
mmCP_HQD_QUANTUM = 0x324c
mmCP_HQD_EOP_RPTR = 0x326d
mmCP_HQD_EOP_WPTR = 0x326e
mmCP_HQD_EOP_EVENTS = 0x326f
mmRLC_SAFE_MODE = 0xec05
mmRLC_GPM_STAT = 0xec10
mmCP_HQD_ERROR = 0x3278
mmCP_HQD_EOP_WPTR_MEM = 0x3279
mmCP_HQD_EOP_DONES = 0x327a

# gfx_8_0_sh_mask.h field helpers
def _reg_field(val: int, mask: int, shift: int, fval: int) -> int:
  return (val & ~mask) | ((fval << shift) & mask)

CP_HQD_PQ_CONTROL_QUEUE_SIZE_MASK = 0x3f
CP_HQD_PQ_CONTROL_RPTR_BLOCK_SIZE_MASK = 0x3f00
CP_HQD_PQ_CONTROL_UNORD_DISPATCH_MASK = 0x10000000
CP_HQD_PQ_CONTROL_ROQ_PQ_IB_FLIP_MASK = 0x20000000
CP_HQD_PQ_CONTROL_PRIV_STATE_MASK = 0x40000000
CP_HQD_PQ_CONTROL_KMD_QUEUE_MASK = 0x80000000
CP_HQD_PQ_DOORBELL_OFFSET_MASK = 0x7ffffc
CP_HQD_PQ_DOORBELL_EN_MASK = 0x40000000
CP_HQD_IB_CONTROL_MIN_IB_AVAIL_SIZE_MASK = 0x300000
CP_HQD_IB_CONTROL_MTYPE_MASK = 0xc0000
CP_HQD_IQ_TIMER_MTYPE_MASK = 0x3000000
CP_HQD_CTX_SAVE_CONTROL_MTYPE_MASK = 0x3000000
CP_HQD_PERSISTENT_STATE_PRELOAD_SIZE_MASK = 0xff
CP_HQD_PERSISTENT_STATE_PRELOAD_REQ_MASK = 0x1
mmCP_HQD_DEQUEUE_REQUEST = 0x325d
CP_HQD_EOP_CONTROL_EOP_SIZE_MASK = 0xf000
CP_HQD_QUANTUM_QUANTUM_EN_MASK = 0x1
CP_HQD_QUANTUM_QUANTUM_SCALE_MASK = 0x6
CP_HQD_QUANTUM_QUANTUM_DURATION_MASK = 0xfffffff0
CP_MQD_CONTROL_VMID_MASK = 0xf
CP_PQ_STATUS_DOORBELL_ENABLE_MASK = 0x2

# vi_structs.h — vi_mqd_allocation dword count
VI_MQD_ALLOC_DWORDS = 261
MQD_HQD_WORD = 128  # cp_mqd_base_addr_lo

# PACKET3_MAP_QUEUES field builders (amdgpu/vid.h, VI)
def _map_queues_num_q(n: int) -> int: return n << 29
def _map_queues_dbell(off: int) -> int: return off << 2
def _map_queues_queue(q: int) -> int: return q << 26
def _map_queues_pipe(p: int) -> int: return p << 29
def _map_queues_me(m: int) -> int: return m << 31


class ViMqd:
  """struct vi_mqd_allocation in GPU memory (ref/linux vi_structs.h)."""

  def __init__(self):
    self.w = [0] * VI_MQD_ALLOC_DWORDS
    self.w[257 + 2] = 0xffffffff  # dynamic_cu_mask
    self.w[257 + 3] = 0xffffffff  # dynamic_rb_mask

  def hqd(self, reg: int) -> int:
    return self.w[MQD_HQD_WORD + (reg - mmCP_MQD_BASE_ADDR)]

  def set_hqd(self, reg: int, val: int):
    self.w[MQD_HQD_WORD + (reg - mmCP_MQD_BASE_ADDR)] = val & 0xffffffff

  def to_bytes(self) -> bytes:
    return struct.pack('<' + 'I' * VI_MQD_ALLOC_DWORDS, *self.w)


def mqd_init_vi(boot: PolarisBoot, cq: 'ComputeQueue', is_kiq: bool, activate: bool = False) -> ViMqd:
  """Port of gfx_v8_0_mqd_init (ref/linux gfx_v8_0.c)."""
  m = ViMqd()
  m.w[0] = 0xC0310800
  m.w[11] = 1  # compute_pipelinestat_enable
  for i in (23, 24, 26, 27):
    m.w[i] = 0xffffffff  # static thread mgmt SE0-3
  m.w[32] = 3  # compute_misc_reserved
  cu_addr = cq.mqd_gpu + (257 + 2) * 4
  m.w[126] = cu_addr & 0xffffffff
  m.w[127] = (cu_addr >> 32) & 0xffffffff

  boot.srbm_select(cq.me, cq.pipe, cq.queue, 0)
  eop_base = cq.eop_gpu >> 8
  m.set_hqd(mmCP_HQD_EOP_BASE_ADDR_LO, eop_base & 0xffffffff)
  m.set_hqd(mmCP_HQD_EOP_BASE_ADDR_HI, (eop_base >> 32) & 0xffffffff)
  eop_sz = order_base_2(GFX8_MEC_HPD_SIZE // 4) - 1
  tmp = boot.rreg(mmCP_HQD_EOP_CONTROL)
  tmp = _reg_field(tmp, CP_HQD_EOP_CONTROL_EOP_SIZE_MASK, 12, eop_sz)
  m.set_hqd(mmCP_HQD_EOP_CONTROL, tmp)

  dbell = _reg_field(0, CP_HQD_PQ_DOORBELL_OFFSET_MASK, 2, cq.doorbell_index)
  dbell = _reg_field(dbell, CP_HQD_PQ_DOORBELL_EN_MASK, 30, 0 if boot_no_doorbell() else 1)
  m.set_hqd(mmCP_HQD_PQ_DOORBELL_CONTROL, dbell)

  m.set_hqd(mmCP_MQD_BASE_ADDR, cq.mqd_gpu & 0xfffffffc)
  m.set_hqd(mmCP_MQD_BASE_ADDR + 1, (cq.mqd_gpu >> 32) & 0xffffffff)
  mqd_ctl = _reg_field(boot.rreg(mmCP_MQD_CONTROL), CP_MQD_CONTROL_VMID_MASK, 0, 0)
  m.set_hqd(mmCP_MQD_CONTROL, mqd_ctl)

  hqd_base = cq.ring_gpu >> 8
  m.set_hqd(mmCP_HQD_PQ_BASE_LO, hqd_base & 0xffffffff)
  m.set_hqd(mmCP_HQD_PQ_BASE_HI, (hqd_base >> 32) & 0xffffffff)

  qsize = order_base_2(RING_SIZE // 4) - 1
  rptr_blk = order_base_2(PAGE_SIZE // 4) - 1
  tmp = boot.rreg(mmCP_HQD_PQ_CONTROL)
  tmp = _reg_field(tmp, CP_HQD_PQ_CONTROL_QUEUE_SIZE_MASK, 0, qsize)
  tmp = _reg_field(tmp, CP_HQD_PQ_CONTROL_RPTR_BLOCK_SIZE_MASK, 8, rptr_blk)
  tmp = _reg_field(tmp, CP_HQD_PQ_CONTROL_UNORD_DISPATCH_MASK, 28, 0)
  tmp = _reg_field(tmp, CP_HQD_PQ_CONTROL_ROQ_PQ_IB_FLIP_MASK, 29, 0)
  tmp = _reg_field(tmp, CP_HQD_PQ_CONTROL_PRIV_STATE_MASK, 30, 1)
  tmp = _reg_field(tmp, CP_HQD_PQ_CONTROL_KMD_QUEUE_MASK, 31, 1)
  m.set_hqd(mmCP_HQD_PQ_CONTROL, tmp)

  m.set_hqd(mmCP_HQD_PQ_RPTR_REPORT_ADDR_LO, cq.rptr_gpu & 0xfffffffc)
  m.set_hqd(mmCP_HQD_PQ_RPTR_REPORT_ADDR_HI, (cq.rptr_gpu >> 32) & 0xffff)
  m.set_hqd(mmCP_HQD_PQ_WPTR_POLL_ADDR_LO, cq.wptr_gpu & 0xfffffffc)
  m.set_hqd(mmCP_HQD_PQ_WPTR_POLL_ADDR_HI, (cq.wptr_gpu >> 32) & 0xffff)
  m.set_hqd(mmCP_HQD_PQ_WPTR, 0)
  m.set_hqd(mmCP_HQD_PQ_RPTR, boot.rreg(mmCP_HQD_PQ_RPTR))
  m.set_hqd(mmCP_HQD_VMID, 0)
  ps = boot.rreg(mmCP_HQD_PERSISTENT_STATE)
  ps = _reg_field(ps, CP_HQD_PERSISTENT_STATE_PRELOAD_SIZE_MASK, 0, 0x53)
  if activate:
    ps |= CP_HQD_PERSISTENT_STATE_PRELOAD_REQ_MASK
  m.set_hqd(mmCP_HQD_PERSISTENT_STATE, ps)
  tmp = boot.rreg(mmCP_HQD_IB_CONTROL)
  tmp = _reg_field(tmp, CP_HQD_IB_CONTROL_MIN_IB_AVAIL_SIZE_MASK, 20, 3)
  tmp = _reg_field(tmp, CP_HQD_IB_CONTROL_MTYPE_MASK, 16, 3)
  m.set_hqd(mmCP_HQD_IB_CONTROL, tmp)
  tmp = _reg_field(boot.rreg(mmCP_HQD_IQ_TIMER), CP_HQD_IQ_TIMER_MTYPE_MASK, 24, 3)
  m.set_hqd(mmCP_HQD_IQ_TIMER, tmp)
  tmp = _reg_field(boot.rreg(mmCP_HQD_CTX_SAVE_CONTROL), CP_HQD_CTX_SAVE_CONTROL_MTYPE_MASK, 24, 3)
  m.set_hqd(mmCP_HQD_CTX_SAVE_CONTROL, tmp)
  for reg in (mmCP_HQD_EOP_RPTR, mmCP_HQD_EOP_WPTR, mmCP_HQD_EOP_WPTR_MEM, mmCP_HQD_EOP_DONES):
    m.set_hqd(reg, boot.rreg(reg))
  for reg in (mmCP_HQD_EOP_EVENTS, mmCP_HQD_ERROR):
    m.set_hqd(reg, boot.rreg(reg))
  tmp = boot.rreg(mmCP_HQD_QUANTUM)
  tmp = _reg_field(tmp, CP_HQD_QUANTUM_QUANTUM_EN_MASK, 0, 1)
  tmp = _reg_field(tmp, CP_HQD_QUANTUM_QUANTUM_SCALE_MASK, 1, 1)
  tmp = _reg_field(tmp, CP_HQD_QUANTUM_QUANTUM_DURATION_MASK, 4, 10)
  m.set_hqd(mmCP_HQD_QUANTUM, tmp)
  if is_kiq or activate:
    m.set_hqd(mmCP_HQD_ACTIVE, 1)
  boot.srbm_select(0, 0, 0, 0)
  return m


def mqd_commit_vi(boot: PolarisBoot, cq: 'ComputeQueue', mqd: ViMqd, deactivate: bool = False):
  """Port of gfx_v8_0_mqd_commit (ref/linux gfx_v8_0.c)."""
  if deactivate:
    boot.deactivate_hqd(cq.me, cq.pipe, cq.queue)
  boot.srbm_select(cq.me, cq.pipe, cq.queue, 0)
  boot.wreg(mmCP_PQ_WPTR_POLL_CNTL, boot.rreg(mmCP_PQ_WPTR_POLL_CNTL) & ~1)
  for reg in range(mmCP_HQD_VMID, mmCP_HQD_EOP_CONTROL + 1):
    boot.wreg(reg, mqd.hqd(reg))
  boot.wreg(mmCP_HQD_EOP_RPTR, mqd.hqd(mmCP_HQD_EOP_RPTR))
  boot.wreg(mmCP_HQD_EOP_WPTR, mqd.hqd(mmCP_HQD_EOP_WPTR))
  boot.wreg(mmCP_HQD_EOP_WPTR_MEM, mqd.hqd(mmCP_HQD_EOP_WPTR_MEM))
  for reg in range(mmCP_HQD_EOP_EVENTS, mmCP_HQD_ERROR + 1):
    boot.wreg(reg, mqd.hqd(reg))
  for reg in range(mmCP_MQD_BASE_ADDR, mmCP_HQD_ACTIVE + 1):
    boot.wreg(reg, mqd.hqd(reg))
  boot.srbm_select(0, 0, 0, 0)
  boot.mmio_sync_safe()


mmCP_HQD_PQ_BASE_LO = 0x324d
mmCP_HQD_PQ_BASE_HI = 0x324e
mmCP_HQD_PQ_RPTR_REPORT_ADDR_LO = 0x3250
mmCP_HQD_PQ_RPTR_REPORT_ADDR_HI = 0x3251
mmCP_HQD_PQ_WPTR_POLL_ADDR_LO = 0x3252
mmCP_HQD_PQ_WPTR_POLL_ADDR_HI = 0x3253
mmCP_HQD_PQ_DOORBELL_CONTROL = 0x3254
mmCP_HQD_PQ_WPTR = 0x3255
mmCP_HQD_PQ_CONTROL = 0x3256
mmCP_HQD_IB_BASE_ADDR_LO = 0x3257
mmCP_HQD_IB_CONTROL = 0x325a
mmCP_HQD_IQ_TIMER = 0x325b
mmCP_HQD_CTX_SAVE_CONTROL = 0x3272
mmCP_HQD_EOP_BASE_ADDR_LO = 0x326a
mmCP_HQD_EOP_BASE_ADDR_HI = 0x326b
mmCP_HQD_EOP_CONTROL = 0x326c
mmCP_MQD_CONTROL = 0x3267
mmCP_HQD_PERSISTENT_STATE = 0x3249


class ComputeQueue:
  """gfx_v8_0 KIQ + KCQ setup for MEC compute ring 0."""

  def __init__(self, boot: PolarisBoot, me=1, pipe=0, queue=0, doorbell_index=DOORBELL_MEC_RING0):
    self.boot = boot
    self.dev = boot.dev
    self.me, self.pipe, self.queue = me, pipe, queue
    self.doorbell_index = doorbell_index
    # Use GART/sysmem when PTE table is in host memory (TinyGPU eGPU path).
    self._gtt = boot.gart_pte_sysmem is not None or not boot.probe_bar0_writes()
    self.ring_off = self.mqd_off = self.eop_off = self.wptr_off = 0
    self.ring_gpu = self.mqd_gpu = self.eop_gpu = self.wptr_gpu = self.rptr_gpu = 0
    self.ring_mem = self.mqd_mem = self.eop_mem = self.wptr_mem = None
    self.wptr = 0

  def _alloc_buf(self, size: int, align=0x1000) -> tuple[int, object | None, int]:
    if self._gtt:
      gpu_va, mem, _ = self.boot.alloc_gtt_buffer(size, align)
      return gpu_va, mem, 0
    off = self.dev.alloc_vram(size, align)
    return self.dev.vram_gpu_addr(off), None, off

  def _write_bytes(self, off_or_mem, data: bytes, mem=None):
    if mem is not None:
      mem[0:len(data)] = data
      from add import sysmem_dma_flush
      sysmem_dma_flush(mem, len(data))
    else:
      self.dev.upload(off_or_mem, data)

  def _write_ring(self, words: list[int], offset_dwords: int = 0):
    data = struct.pack('<' + 'I' * len(words), *words)
    byte_off = offset_dwords * 4
    if self.ring_mem is not None:
      self.ring_mem[byte_off:byte_off + len(data)] = data
      from add import sysmem_dma_flush
      sysmem_dma_flush(self.ring_mem, byte_off + len(data))
    else:
      self.dev.upload(self.ring_off + byte_off, data)

  def _publish_wptr(self, wptr: int):
    """CPU shadow of wptr for CP poll (gfx_v8_0_ring_set_wptr_compute)."""
    if self.wptr_mem is None:
      return
    self.wptr_mem[64:68] = struct.pack('<I', wptr & 0xffffffff)
    from add import sysmem_dma_flush
    sysmem_dma_flush(self.wptr_mem, 128)

  def _signal_wptr(self, wptr: int, doorbell_index: int):
    """Linux: doorbell + wptr shadow. macOS/TinyGPU: MMIO PQ_WPTR only (no BAR2 MSI)."""
    self._publish_wptr(wptr)
    boot = self.boot
    if not boot_no_doorbell():
      self.dev.ring_doorbell(doorbell_index, wptr)
    if boot_use_mmio_wptr():
      boot.srbm_select(self.me, self.pipe, self.queue, 0)
      boot.wreg(mmCP_HQD_PQ_WPTR, wptr & 0xffffffff)
      boot.srbm_select(0, 0, 0, 0)
      boot.mmio_sync_safe()

  def _upload_mqd(self, mqd: ViMqd):
    data = mqd.to_bytes()
    if self.mqd_mem is not None:
      self._write_bytes(0, data, self.mqd_mem)
    else:
      self.dev.upload(self.mqd_off, data)

  def _kiq_map_queues_pkt(self, target: 'ComputeQueue') -> list[int]:
    """gfx_v8_0_kiq_kcq_enable PM4 (ref/linux gfx_v8_0.c)."""
    w: list[int] = []
    w.append(pkt3(PACKET3_SET_RESOURCES, 6))
    w.extend([0, 1, 0, 0, 0, 0, 0])  # queue_mask bit0 = KCQ ring0
    w.append(pkt3(PACKET3_MAP_QUEUES, 5))
    w.append(_map_queues_num_q(1))
    me_bit = 0 if target.me == 1 else 1
    w.append(_map_queues_dbell(target.doorbell_index) | _map_queues_queue(target.queue) |
             _map_queues_pipe(target.pipe) | _map_queues_me(me_bit))
    w.append(target.mqd_gpu & 0xffffffff)
    w.append((target.mqd_gpu >> 32) & 0xffffffff)
    w.append(target.wptr_gpu & 0xffffffff)
    w.append((target.wptr_gpu >> 32) & 0xffffffff)
    return w

  def _ring_commit(self, words: list[int], doorbell_index: int,
                   align_mask: int = VI_RING_ALIGN_MASK):
    """amdgpu_ring_commit: VI KIQ/KCQ rings pad to 256-dword boundary."""
    w = list(words)
    base = self.wptr
    new_wptr = base + len(w)
    pad = (align_mask + 1) - (new_wptr & align_mask)
    pad &= align_mask
    if pad:
      w.extend([VI_PKT3_NOP] * pad)
    self._write_ring(w, offset_dwords=base)
    new_wptr = base + len(w)
    boot = self.boot
    boot.hdp_flush()
    boot.hdp_invalidate()
    for mem in (self.ring_mem, self.mqd_mem, self.eop_mem, self.wptr_mem):
      if mem is not None:
        from add import sysmem_dma_flush
        sysmem_dma_flush(mem, len(mem))
    boot.mmio_settle("pre-doorbell", heavy=False)
    self.wptr = new_wptr % (RING_SIZE // 4)
    self._signal_wptr(self.wptr, doorbell_index)
    boot.mmio_sync_safe()
    default_settle = "10" if boot_no_doorbell() else "50"
    settle_ms = int(os.environ.get("AMD_BOOT_DOORBELL_SETTLE_MS", default_settle))
    if settle_ms:
      time.sleep(settle_ms / 1000.0)
    boot._check_pci("post-doorbell")

  def init(self):
    self.ring_gpu, self.ring_mem, self.ring_off = self._alloc_buf(RING_SIZE)
    self.mqd_gpu, self.mqd_mem, self.mqd_off = self._alloc_buf(4096)
    self.eop_gpu, self.eop_mem, self.eop_off = self._alloc_buf(GFX8_MEC_HPD_SIZE)
    wb_gpu, self.wptr_mem, self.wptr_off = self._alloc_buf(4096)
    self.rptr_gpu = wb_gpu
    self.wptr_gpu = wb_gpu + 64
    if self.wptr_mem is not None:
      self.wptr_mem[0:128] = bytes(128)
      from add import sysmem_dma_flush
      sysmem_dma_flush(self.wptr_mem, 128)

  def ring_test_scratch(self, timeout_s: float = 2.0) -> bool:
    """gfx_v8_0_ring_test_ring: SET_UCONFIG_REG on SCRATCH_REG0 (ref/linux gfx_v8_0.c)."""
    boot = self.boot
    boot.wreg(mmSCRATCH_REG0, 0xCAFEDEAD)
    scratch_idx = (mmSCRATCH_REG0 - PACKET3_SET_UCONFIG_REG_START) >> 2
    pkt = [pkt3(PACKET3_SET_UCONFIG_REG, 1), scratch_idx, 0xDEADBEEF]
    self._ring_commit(pkt, self.doorbell_index)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
      if boot.rreg(mmSCRATCH_REG0) == 0xDEADBEEF:
        return True
      time.sleep(0.001)
    return False

  def submit_ib(self, ib_words: list[int]):
    pkt = [pkt3(0x10, len(ib_words) + 2, 0), 0, len(ib_words) * 4] + ib_words
    base = self.wptr
    new_wptr = base + len(pkt)
    pad = (VI_RING_ALIGN_MASK + 1) - (new_wptr & VI_RING_ALIGN_MASK)
    pad &= VI_RING_ALIGN_MASK
    if pad:
      pkt = pkt + [VI_PKT3_NOP] * pad
    self._write_ring(pkt, offset_dwords=base)
    self.wptr = (base + len(pkt)) % (RING_SIZE // 4)
    self.boot.hdp_flush()
    self.boot.hdp_invalidate()
    if self.ring_mem is not None:
      from add import sysmem_dma_flush
      sysmem_dma_flush(self.ring_mem, min(len(self.ring_mem), (base + len(pkt)) * 4))
    self._publish_wptr(self.wptr)
    self._signal_wptr(self.wptr, self.doorbell_index)
    if int(os.environ.get("DEBUG", "0")):
      boot = self.boot
      boot.srbm_select(self.me, self.pipe, self.queue, 0)
      rptr = boot.rreg(mmCP_HQD_PQ_RPTR)
      wptr = boot.rreg(mmCP_HQD_PQ_WPTR)
      boot.srbm_select(0, 0, 0, 0)
      print(f"polaris: KCQ after submit PQ_WPTR={wptr:#x} PQ_RPTR={rptr:#x}", flush=True)

  def setup_with_kiq(self, map_queues: bool | None = None):
    """gfx_v8_0_kiq_resume + kcq_resume (ref/linux gfx_v8_0.c)."""
    if map_queues is None:
      map_queues = os.environ.get("AMD_BOOT_KIQ_MAP", "1") == "1"
    boot = self.boot
    boot.deactivate_hqd(self.me, self.pipe, self.queue)
    # 1) KIQ: RLC scheduler + MQD init + commit (activates KIQ HQD)
    kiq = ComputeQueue(boot, me=KIQ_ME, pipe=KIQ_PIPE, queue=KIQ_QUEUE, doorbell_index=DOORBELL_KIQ)
    kiq.init()
    boot.kiq_setting(kiq.me, kiq.pipe, kiq.queue)
    kiq_mqd = mqd_init_vi(boot, kiq, is_kiq=True)
    mqd_commit_vi(boot, kiq, kiq_mqd)
    kiq._upload_mqd(kiq_mqd)
    kiq_active = boot.read_hqd_active(kiq.me, kiq.pipe, kiq.queue)
    if int(os.environ.get("DEBUG", "0")):
      print(f"polaris: KIQ CP_HQD_ACTIVE={kiq_active:#x}", flush=True)

    if os.environ.get("AMD_BOOT_KIQ_NOP_TEST", "0") == "1":
      kiq._ring_commit([VI_PKT3_NOP], DOORBELL_KIQ)
      boot.srbm_select(kiq.me, kiq.pipe, kiq.queue, 0)
      krptr = boot.rreg(mmCP_HQD_PQ_RPTR)
      kwptr = boot.rreg(mmCP_HQD_PQ_WPTR)
      boot.srbm_select(0, 0, 0, 0)
      print(f"polaris: KIQ NOP test PQ_WPTR={kwptr:#x} PQ_RPTR={krptr:#x}", flush=True)

    kcq_mode = os.environ.get("AMD_BOOT_KCQ_DIRECT", "auto")
    direct_kcq = kcq_mode == "1" or (kcq_mode == "auto" and not map_queues)
    # HQD activation DMA-reads host sysmem at MEC preload → APCIE panic on USB4.
    # Keep MQD-in-memory only unless explicitly opted in (see boot_allow_hqd_activation).
    activate = direct_kcq and boot_allow_hqd_activation()
    if direct_kcq and not activate:
      kcq_mqd = mqd_init_vi(boot, self, is_kiq=False, activate=False)
      self._upload_mqd(kcq_mqd)
      boot.set_mec_doorbell_range()
      print("polaris: KCQ MQD staged in memory; HQD activation gated "
            "(set AMD_BOOT_KCQ_ACTIVATE=1 to make it live — DMA/panic risk)", flush=True)
      return

    # 2) KCQ MQD — MAP_QUEUES activates HQD, or direct commit (TrustOS fallback)
    kcq_mqd = mqd_init_vi(boot, self, is_kiq=False, activate=activate)
    self._upload_mqd(kcq_mqd)
    if activate:
      boot.deactivate_hqd(self.me, self.pipe, self.queue)
      mqd_commit_vi(boot, self, kcq_mqd, deactivate=False)
      boot.set_mec_doorbell_range()
      kcq_active = boot.read_hqd_active(self.me, self.pipe, self.queue)
      print(f"polaris: KCQ direct HQD commit KCQ_HQD_ACTIVE={kcq_active:#x}", flush=True)
      return

    if not map_queues:
      if int(os.environ.get("DEBUG", "0")):
        print("polaris: KIQ MAP_QUEUES skipped (AMD_BOOT_KIQ_MAP=0)", flush=True)
      return

    # 3) MAP_QUEUES via KIQ ring — linux: set_mec_doorbell_range then amdgpu_ring_commit
    boot.set_mec_doorbell_range()
    pkt = kiq._kiq_map_queues_pkt(self)
    if int(os.environ.get("DEBUG", "0")):
      print(f"polaris: KIQ MAP_QUEUES pkt_words={len(pkt)} "
            f"kcq_mqd={self.mqd_gpu:#x} kcq_wptr={self.wptr_gpu:#x}", flush=True)
    kiq._ring_commit(pkt, DOORBELL_KIQ)

    # 4) Wait for KCQ activation (poll, no extra doorbells)
    deadline = time.time() + float(os.environ.get("AMD_BOOT_KCQ_ACTIVE_TIMEOUT_S", "5"))
    kcq_active = 0
    while time.time() < deadline:
      kcq_active = boot.read_hqd_active(self.me, self.pipe, self.queue)
      if kcq_active & 1:
        break
      time.sleep(0.01)
    kcq_active = boot.read_hqd_active(self.me, self.pipe, self.queue)
    if (not (kcq_active & 1) and os.environ.get("AMD_BOOT_KCQ_DIRECT", "auto") == "auto"
        and boot_allow_hqd_activation()):
      print("polaris: MAP_QUEUES did not activate KCQ — trying direct HQD commit", flush=True)
      boot.deactivate_hqd(self.me, self.pipe, self.queue)
      kcq_mqd = mqd_init_vi(boot, self, is_kiq=False, activate=True)
      self._upload_mqd(kcq_mqd)
      mqd_commit_vi(boot, self, kcq_mqd, deactivate=False)
      boot.set_mec_doorbell_range()
      kcq_active = boot.read_hqd_active(self.me, self.pipe, self.queue)
      print(f"polaris: KCQ after direct commit KCQ_HQD_ACTIVE={kcq_active:#x}", flush=True)
    elif not (kcq_active & 1):
      print("polaris: KCQ still inactive — set AMD_BOOT_KCQ_DIRECT=1", flush=True)
    if int(os.environ.get("DEBUG", "0")):
      boot.srbm_select(kiq.me, kiq.pipe, kiq.queue, 0)
      kiq_wptr = boot.rreg(mmCP_HQD_PQ_WPTR)
      kiq_rptr = boot.rreg(mmCP_HQD_PQ_RPTR)
      kiq_err = boot.rreg(mmCP_HQD_ERROR)
      boot.srbm_select(0, 0, 0, 0)
      print(f"polaris: KIQ after doorbell PQ_WPTR={kiq_wptr:#x} PQ_RPTR={kiq_rptr:#x} "
            f"CP_HQD_ERROR={kiq_err:#x}", flush=True)

