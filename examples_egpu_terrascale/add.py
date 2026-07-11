#!/usr/bin/env python3
# NO CPU OFFLOAD: never label a CPU-calculated payload as a GPU add.  The
# --cp-mem-write-test diagnostic only proves CP DMA; true GPU add must execute
# an RV770 graphics/ALU shader and fails loudly until that path exists.
"""Standalone Terascale eGPU vector-add scaffold (HD 5570 / HD 4850) over TinyGPU.

Hardware not required yet - `--selftest` / `--dry-run` work offline. When the
card is attached via TinyGPU, `--probe` enumerates PCI and MMIO.

Targets (linux `drivers/gpu/drm/radeon`):
  HD 5570 - Redwood / Evergreen (TeraScale 2), PCI 1002:68D9 (also 68D8/68DA...)
  HD 4850 - RV770 / R700     (TeraScale 1), PCI 1002:9442

Evergreen has a real compute path (Mesa `evergreen_compute.c` / r600g OpenCL):
  SQ_PGM_START_LS + SPI_COMPUTE_NUM_THREAD_* + PKT3_DISPATCH_DIRECT (compute bit).
RV770 shares the R600 CP ring (`r600_cp_resume`) but **no LS compute**; GFX/ALU
path is stubbed until HW bring-up.

Refs: `ref/linux/.../radeon/{evergreen,r600,rv770}.c`, `evergreend.h`, `r600d.h`.

Usage:
  python3 examples_egpu_terrascale/add.py --selftest
  python3 examples_egpu_terrascale/add.py --chip=hd5570 --dry-run
  python3 examples_egpu_terrascale/add.py --probe          # needs TinyGPU + card
  python3 examples_egpu_terrascale/add.py --dump-rom       # dump onboard VBIOS
  python3 examples_egpu_terrascale/add.py --clock-probe    # SPLL_CHG / MPLL CLKF
  python3 examples_egpu_terrascale/add.py --atom           # ATOM asic_init (needs real SPLL_CHG)
  python3 examples_egpu_terrascale/add.py --cp-mem-write-test
                                                        # CP/AGP payload-write diagnostic, not add
  python3 examples_egpu_terrascale/add.py --gpu-add-preflight
                                                        # allocates real VS/PS/input/target, no draw
  python3 examples_egpu_terrascale/add.py                  # true GPU add; fails until RV770 ALU path lands

HD 4850: AGP-first (MEMSIZE=stub, FB@0xE0..., BIF off). Cold CHG -> --atom for MPLL;
warm re-runs work with AMD_BOOT_ATOM=0. Default add never maps or accesses BAR0/VRAM.
Never AMD_ATOM_SYNTH_SPLL_CHG / BIF+BAR0 poke.
"""
from __future__ import annotations
import os, sys, ctypes, ctypes.util, time, mmap, struct, array, socket, subprocess, shutil
import contextlib, functools, enum, urllib.request, hashlib
import tempfile, pathlib, math, json
from dataclasses import dataclass, field

# =============================================================================
# TinyGPU transport + PM4 helpers (from examples_egpu/add.py)
# =============================================================================
DEBUG = int(os.environ.get("DEBUG", "0"))
OSX = sys.platform == "darwin"

def getenv(k: str, default=0):
  v = os.environ.get(k)
  if v is None: return default
  try: return int(v)
  except: return v

def round_up(n: int, a: int) -> int: return ((n + a - 1) // a) * a
def ceildiv(n: int, a: int) -> int: return -(n // -a)
def lo32(x: int) -> int: return x & 0xFFFFFFFF
def hi32(x: int) -> int: return x >> 32
def data64_le(x: int) -> tuple: return (x & 0xFFFFFFFF, (x >> 32) & 0xFFFFFFFF)
def temp(name: str) -> str: return os.path.join(tempfile.gettempdir(), name)
def unwrap(x): return x

def wait_cond(cb, *args, value=True, timeout_ms=10000, msg=""):
  start = int(time.perf_counter() * 1000)
  while int(time.perf_counter() * 1000) - start < timeout_ms:
    if (val := cb(*args)) == value: return val
  raise TimeoutError(f"{msg}. Timed out after {timeout_ms} ms, last={val!r} expected={value!r}")

def _ensure_downloads_dir() -> pathlib.Path:
  d = pathlib.Path(os.path.expanduser("~")) / ".cache" / "tinygrad"
  d.mkdir(parents=True, exist_ok=True)
  return d

def fetch_fw(path: str, name: str, sha256: str | None = None) -> bytes:
  cache_dir = _ensure_downloads_dir() / "fw"
  cache_dir.mkdir(parents=True, exist_ok=True)
  fp = cache_dir / name
  if fp.is_file() and (sha256 is None or hashlib.sha256(fp.read_bytes()).hexdigest() == sha256):
    return fp.read_bytes()
  url = f"https://gitlab.com/kernel-firmware/linux-firmware/-/raw/main/{path}/{name}"
  with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "amdgpu-egpu"}), timeout=30) as r:
    data = r.read()
  if sha256 and hashlib.sha256(data).hexdigest() != sha256:
    raise RuntimeError(f"fetch_fw sha mismatch for {name}")
  fp.write_bytes(data)
  return data

# ============================================================================
# Remote PCI transport (vendored from nvgpu examples/add.py / tinygrad system.py)
# ============================================================================
class MMIOInterface:
  def __init__(self, addr, nbytes, fmt='B'):
    self.mv = (ctypes.c_uint8 * nbytes).from_address(addr)
    self.addr, self.nbytes, self.fmt = addr, nbytes, fmt
    if fmt != 'B':
      self._arr = array.array(fmt)
      self._arr.frombytes(bytes(self.mv))
  def __len__(self): return self.nbytes // (1 if self.fmt == 'B' else struct.calcsize(self.fmt))
  def __getitem__(self, k):
    if self.fmt == 'B':
      sl = k if isinstance(k, slice) else slice(k, k+1)
      return bytes(self.mv[sl.start:sl.stop])
    return self._arr[k]
  def __setitem__(self, k, v):
    if self.fmt == 'B':
      if isinstance(k, slice):
        self.mv[k.start:k.stop] = v if isinstance(v, (bytes, bytearray)) else bytes(v)
      else:
        self.mv[k] = v if isinstance(v, int) else v[0]
    else:
      self._arr[k] = v
      ctypes.memmove(self.addr, self._arr.tobytes(), self.nbytes)
  def view(self, offset=0, size=None, fmt=None):
    sz = (self.nbytes - offset) if size is None else size
    return MMIOInterface(self.addr + offset, sz, fmt=fmt or self.fmt)

def _sysmem_msync(mem, size: int) -> None:
  """Best-effort msync over the mapping (CPU cache vs device DMA coherency)."""
  if os.environ.get("AMD_BOOT_SYSMEM_FLUSH", "1") == "0":
    return
  if not hasattr(mem, "addr") or not size:
    return
  libc = ctypes.CDLL(ctypes.util.find_library("c"))
  MS_SYNC = 0x10
  if libc.msync(ctypes.c_void_p(mem.addr), size, MS_SYNC) != 0:
    with contextlib.suppress(Exception):
      libc.sync()

def sysmem_sync_for_device(mem, size: int) -> None:
  """Publish CPU-written sysmem so the eGPU DMA engine sees it.

  Used for shader binaries, vertex data, ring contents, cleared fence and
  initialized canary color pages before a submit.
  """
  _sysmem_msync(mem, size)

def sysmem_sync_for_cpu(mem, size: int) -> None:
  """Make device-written sysmem visible to the CPU before a readback.

  The CP MEM_WRITE test already proved GPU writes can reach the mapping, so on
  most hosts this is a no-op; we still centralize it so the graphics result is
  never read through stale CPU caches.
  """
  _sysmem_msync(mem, size)

def sysmem_dma_flush(mem, size: int):
  """Deprecated: use sysmem_sync_for_device / sysmem_sync_for_cpu."""
  sysmem_sync_for_device(mem, size)

class FileIOInterface:
  def __init__(self, fd=None):
    self.fd = fd
  def mmap(self, start, sz, prot, flags, offset):
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    libc.mmap.restype = ctypes.c_void_p
    addr = libc.mmap(start or None, sz, prot, flags, self.fd, offset)
    if not addr or addr == ctypes.c_void_p(-1).value:
      raise OSError("mmap failed")
    return addr

class RemoteCmd(enum.IntEnum):
  PROBE, MAP_BAR, MAP_SYSMEM_FD, CFG_READ, CFG_WRITE, RESET, MMIO_READ, MMIO_WRITE, MAP_SYSMEM, SYSMEM_READ, SYSMEM_WRITE, RESIZE_BAR, PING = range(13)

class RemoteMMIOInterface:
  def __init__(self, dev, residx, nbytes, fmt='B', off=0):
    self.dev, self.residx, self.nbytes, self.fmt, self.off = dev, residx, nbytes, fmt, off
    self.el_sz = struct.calcsize(fmt)
  def __len__(self): return self.nbytes // self.el_sz
  def __getitem__(self, index):
    sl = index if isinstance(index, slice) else slice(index, index + 1)
    start, stop = (sl.start or 0) * self.el_sz, (sl.stop or len(self)) * self.el_sz
    data = self.dev._bulk_read(RemoteCmd.MMIO_READ, self.residx, self.off + start, stop - start)
    if self.fmt == 'B': return data if isinstance(index, slice) else data[0]
    vals = struct.unpack(f'<{(stop-start)//self.el_sz}{self.fmt}', data)
    return vals if isinstance(index, slice) else vals[0]
  def __setitem__(self, index, val):
    start = (index.start or 0) * self.el_sz if isinstance(index, slice) else index * self.el_sz
    if self.fmt == 'B':
      data = bytes(val) if isinstance(val, (bytes, bytearray, memoryview)) else bytes([val])
    elif isinstance(index, slice):
      data = struct.pack(f'<{len(val)}{self.fmt}', *val)
    else:
      data = struct.pack(f'<{self.fmt}', val)
    self.dev._bulk_write(RemoteCmd.MMIO_WRITE, self.residx, self.off + start, data)
  def view(self, offset=0, size=None, fmt=None):
    return RemoteMMIOInterface(self.dev, self.residx, size or (self.nbytes - offset), fmt or self.fmt, self.off + offset)

class RemotePCIDevice:
  def __init__(self, devpref, pcibus, sock):
    self.sock, self.pcibus, self.dev_id = sock, pcibus, 0
    for buft in [socket.SO_SNDBUF, socket.SO_RCVBUF]: self.sock.setsockopt(socket.SOL_SOCKET, buft, 64 << 20)
    self._lock_fd = self._flock_acquire(f"{devpref.lower()}_{pcibus.lower()}.lock")
    self._mmio_writes_since_drain = 0
    self._mmio_drain_every = max(1, int(os.environ.get("AMD_MMIO_DRAIN_EVERY", "128")))

  @staticmethod
  def _flock_acquire(name):
    import fcntl
    lock_name = temp(name)
    lock_fd = os.open(lock_name, os.O_RDWR | os.O_CREAT, 0o666)
    try: fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError: raise RuntimeError(f"eGPU lock held: {lock_name} (only one TinyGPU client at a time)")
    return lock_fd

  @staticmethod
  def _recvall(sock, n):
    data = b''
    while len(data) < n and (chunk := sock.recv(n - len(data))): data += chunk
    if len(data) < n: raise RuntimeError("TinyGPU connection closed")
    return data

  @staticmethod
  def _rpc(sock, dev_id, cmd, *args, bar=0, readout_size=0, payload=b'', has_fd=False):
    sock.sendall(struct.pack('<BIIQQQ', cmd, dev_id, bar, *(*args, 0, 0, 0)[:3]) + payload)
    if has_fd:
      msg, anc, _, _ = sock.recvmsg(17, socket.CMSG_LEN(4))
      fd = struct.unpack('<i', anc[0][2][:4])[0]
    else:
      msg, fd = RemotePCIDevice._recvall(sock, 17), None
    resp = struct.unpack('<BQQ', msg)
    if resp[0] != 0:
      raise RuntimeError(f"TinyGPU RPC failed: {RemotePCIDevice._recvall(sock, resp[1]).decode('utf-8') if resp[1] > 0 else 'unknown error'}")
    return (resp[1], resp[2]) + ((RemotePCIDevice._recvall(sock, readout_size) if readout_size > 0 else None),) + (fd,)

  def _bulk_read(self, cmd, idx, offset, size):
    return unwrap(self._rpc(self.sock, self.dev_id, cmd, offset, size, bar=idx, readout_size=size)[2])
  def drain_mmio(self, bar: int = 5, reg: int = 0x2004):
    """MMIO read round-trip so TinyGPU processes prior fire-and-forget writes."""
    self._bulk_read(RemoteCmd.MMIO_READ, bar, reg * 4, 4)
    self._mmio_writes_since_drain = 0
  def _bulk_write(self, cmd, idx, offset, data):
    self.sock.sendall(struct.pack('<BIIQQQ', cmd, self.dev_id, idx, offset, len(data), 0) + data)
    if cmd != RemoteCmd.MMIO_WRITE:
      return
    self._mmio_writes_since_drain += 1
    if self._mmio_writes_since_drain >= self._mmio_drain_every:
      self.drain_mmio(bar=idx)

  def alloc_sysmem(self, size, contiguous=False):
    mapped_size, _, _, fd = self._rpc(self.sock, self.dev_id, RemoteCmd.MAP_SYSMEM_FD, size, int(contiguous), has_fd=True)
    mem = MMIOInterface(FileIOInterface(fd).mmap(0, mapped_size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, 0), mapped_size, fmt='B')
    raw = bytes(mem[0:min(mapped_size, 0x10000)])
    pairs, off = [], 0
    while off + 16 <= len(raw):
      p, sz = struct.unpack_from('<QQ', raw, off)
      if sz == 0: break
      pairs.append((p, sz)); off += 16
    page_list = [p + i for p, sz in pairs for i in range(0, sz, 0x1000)][:ceildiv(size, 0x1000)]
    return mem, page_list

  def reset(self):
    self._rpc(self.sock, self.dev_id, RemoteCmd.RESET)

  def write_config(self, offset: int, size: int, val: int):
    self._rpc(self.sock, self.dev_id, RemoteCmd.CFG_WRITE, offset, size, val)

  def mask_msi(self) -> list[str]:
    """Stop the eGPU from asserting IRQs to the macOS USB4 bridge.

    Root cause of the recurring `apciec unhandled interrupts (0x200000)` kernel
    panic: TinyGPU.app passes raw MMIO/BAR but installs no interrupt handler, so
    once CP/MEC firmware runs the GPU sends MSIs the AppleT8103PCIe bridge cannot
    route. We keep the device polling-only: disable legacy INTx (PCI command bit
    10) and clear the MSI/MSI-X enable bits in config space. Bus-master (DMA for
    GART sysmem) is left untouched."""
    cleared: list[str] = []
    with contextlib.suppress(Exception):
      cmd = self.read_config(0x04, 2)
      if not (cmd & (1 << 10)):
        self.write_config(0x04, 2, (cmd | (1 << 10)) & 0xffff)
      cleared.append("intx")
    with contextlib.suppress(Exception):
      status = self.read_config(0x06, 2)
      if not (status & (1 << 4)):
        return cleared  # no PCI capability list
      cap = self.read_config(0x34, 1) & 0xfc
      seen = 0
      while cap and cap != 0xfc and seen < 48:
        seen += 1
        cap_id = self.read_config(cap, 1) & 0xff
        nxt = self.read_config(cap + 1, 1) & 0xfc
        if cap_id == 0x05:  # MSI capability - clear MSI Enable (bit 0 of Message Control)
          mc = self.read_config(cap + 2, 2)
          if mc & 0x1:
            self.write_config(cap + 2, 2, mc & ~0x1)
          cleared.append(f"msi@{cap:#x}")
        elif cap_id == 0x11:  # MSI-X - clear MSI-X Enable (bit 15), set Function Mask (bit 14)
          mc = self.read_config(cap + 2, 2)
          self.write_config(cap + 2, 2, (mc & ~(1 << 15)) | (1 << 14))
          cleared.append(f"msix@{cap:#x}")
        cap = nxt
    return cleared

  def amd_cfg_reset(self):
    """Linux vi_asic_pci_config_reset: write AMDGPU_ASIC_RESET_DATA to cfg 0x7c."""
    self.write_config(AMDGPU_ASIC_RESET_CFG_OFF, 4, AMDGPU_ASIC_RESET_DATA)

  def poll_config(self, wait_s: float = 5.0) -> tuple[int, int]:
    deadline = time.time() + wait_s
    vid = did = 0xffff
    while time.time() < deadline:
      with contextlib.suppress(Exception):
        vid = self.read_config(0, 2) & 0xffff
        did = self.read_config(2, 2) & 0xffff
        if vid == 0x1002:
          return vid, did
      time.sleep(0.05)
    return vid, did

  def poll_memsize_ready(self, mmio, wait_s: float = 2.0) -> bool:
    """Linux vi_asic_pci_config_reset: wait until mmCONFIG_MEMSIZE != 0xffffffff."""
    deadline = time.time() + wait_s
    while time.time() < deadline:
      with contextlib.suppress(Exception):
        if int(mmio[REG_CONFIG_MEMSIZE]) != 0xffffffff:
          return True
      time.sleep(0.001)
    return False

  def software_reset(self, wait_s: float = 5.0, mode: str = "auto") -> tuple[int, int]:
    """Reset eGPU using one of several strategies (AMD_RESET_MODE)."""
    mode = mode or os.environ.get("AMD_RESET_MODE", "auto")
    wait_s = float(os.environ.get("AMD_EGPU_RESET_WAIT_S", wait_s))
    vid = self.read_config(0, 2) & 0xffff
    if mode == "mmio":
      return vid, self.read_config(2, 2) & 0xffff

    def do_pci():
      self.reset()
      self.bar_info.cache_clear()

    def try_amd_cfg():
      with contextlib.suppress(Exception):
        if (self.read_config(0, 2) & 0xffff) == 0xffff:
          return
        self.amd_cfg_reset()
        time.sleep(0.1)

    strategies = {
      "pci": [do_pci],
      "amd_cfg": [try_amd_cfg],
      "gentle": [try_amd_cfg, do_pci],
      "full": [try_amd_cfg, do_pci, try_amd_cfg],
      "aggressive": [do_pci, try_amd_cfg, do_pci],
    }
    steps = strategies.get(mode, strategies["gentle"] if vid == 0x1002 else strategies["aggressive"])

    if mode == "auto":
      if vid == 0xffff:
        steps = []  # never hot-reset a missing device
      elif vid == 0x1002:
        steps = [try_amd_cfg]
      else:
        steps = [do_pci]

    attempts = max(1, int(os.environ.get("AMD_EGPU_RESET_ATTEMPTS", 3)))
    per_try = wait_s / attempts if vid != 0xffff else wait_s
    for i in range(attempts):
      if i and steps:
        print(f"polaris: reset retry {i + 1}/{attempts} mode={mode}", flush=True)
        time.sleep(min(0.5 * (2 ** i), 2.0))
      for step in steps:
        step()
      vid, did = self.poll_config(wait_s=per_try)
      if vid == 0x1002:
        time.sleep(float(os.environ.get("AMD_BOOT_SMC_SETTLE_MS", 250)) / 1000.0)
        return vid, did
    return vid, did

  def read_config(self, offset, size): return self._rpc(self.sock, self.dev_id, RemoteCmd.CFG_READ, offset, size)[0]
  @functools.cache
  def bar_info(self, bar_idx): return self._rpc(self.sock, self.dev_id, RemoteCmd.MAP_BAR, bar=bar_idx)[:2]
  def map_bar(self, bar, off=0, size=None, fmt='B'):
    return RemoteMMIOInterface(self, bar, size or self.bar_info(bar)[1], fmt).view(off, size, fmt)

class APLRemotePCIDevice(RemotePCIDevice):
  APP_PATH = "/Applications/TinyGPU.app/Contents/MacOS/TinyGPU"

  def __init__(self, devpref="AMD", pcibus="usb4"):
    self._sock_path = os.environ.get("APL_REMOTE_SOCK", temp("tinygpu.sock"))
    self._server_proc = None
    self.sock = self._connect()
    super().__init__(devpref, pcibus, self.sock)

  def _connect(self) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    for i in range(100):
      with contextlib.suppress(ConnectionRefusedError, FileNotFoundError):
        sock.connect(self._sock_path)
        return sock
      if i == 0:
        self._server_proc = subprocess.Popen(
          [self.APP_PATH, "server", self._sock_path],
          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      time.sleep(0.05)
    raise RuntimeError(f"Failed to connect to TinyGPU at {self._sock_path}. Run: curl -fsSL https://raw.githubusercontent.com/tinygrad/tinygrad/master/extra/setup_tinygpu_osx.sh | sh")

  def restart_server(self):
    """Restart TinyGPU server subprocess (may help recover vid=0xffff)."""
    with contextlib.suppress(Exception):
      self.sock.close()
    if self._server_proc and self._server_proc.poll() is None:
      self._server_proc.terminate()
      with contextlib.suppress(Exception):
        self._server_proc.wait(timeout=2)
    elif self._server_proc is None:
      with contextlib.suppress(Exception):
        subprocess.run(["pkill", "-f", f"TinyGPU server {self._sock_path}"], timeout=2)
    time.sleep(0.5)
    self.sock = self._connect()

# =============================================================================
# Chip table (PCI IDs from pci-ids / linux-hardware; family from radeon_family.h)
# =============================================================================
CHIP_RV770 = "rv770"       # radeon_family.h CHIP_RV770 - HD 4850
CHIP_REDWOOD = "redwood"   # radeon_family.h CHIP_REDWOOD - HD 5570

@dataclass(frozen=True)
class ChipInfo:
  name: str
  family: str                 # CHIP_* string
  pci_ids: tuple[int, ...]    # device IDs (vendor always 0x1002)
  terrascale: int             # 1 = R700, 2 = Evergreen
  has_ls_compute: bool        # Evergreen+ LS compute (Mesa evergreen_compute)
  llvm_mcpu: str
  note: str

CHIPS: dict[str, ChipInfo] = {
  "hd5570": ChipInfo(
    name="Radeon HD 5570",
    family=CHIP_REDWOOD,
    pci_ids=(0x68D8, 0x68D9, 0x68DA, 0x68C1, 0x68C7, 0x68C8, 0x68C9),
    terrascale=2,
    has_ls_compute=True,
    llvm_mcpu="redwood",
    note="Evergreen Redwood PRO/LE - Mesa r600g OpenCL / evergreen_compute.c",
  ),
  "hd4850": ChipInfo(
    name="Radeon HD 4850",
    family=CHIP_RV770,
    pci_ids=(0x9442, 0x9440, 0x944E),
    terrascale=1,
    has_ls_compute=False,
    llvm_mcpu="rv770",
    note="R700 RV770 - GFX CP only here; no Evergreen LS compute",
  ),
}

def resolve_chip(argv: list[str] | None = None) -> ChipInfo:
  argv = argv if argv is not None else sys.argv[1:]
  key = os.environ.get("TS_CHIP", "hd5570").lower()
  for i, arg in enumerate(argv):
    if arg.startswith("--chip="):
      key = arg.split("=", 1)[1].lower()
    elif arg == "--chip" and i + 1 < len(argv):
      key = argv[i + 1].lower()
  if key not in CHIPS:
    raise SystemExit(f"unknown --chip={key!r}; choose {sorted(CHIPS)}")
  return CHIPS[key]

# =============================================================================
# Registers / PM4 - ref/linux radeon evergreend.h + r600d.h
# =============================================================================
# Byte MMIO offsets (WREG32 style). SET_CONFIG/CONTEXT use these as absolute
# byte addresses; packet offset = (addr - START) >> 2.

# r600d.h / evergreend.h - CP ring (r600_cp_resume)
REG_CP_RB_BASE = 0xC100
REG_CP_RB_CNTL = 0xC104
REG_CP_RB_RPTR_WR = 0xC108
REG_CP_RB_RPTR_ADDR = 0xC10C
REG_CP_RB_RPTR_ADDR_HI = 0xC110
REG_CP_RB_WPTR = 0xC114
REG_CP_RB_WPTR_ADDR = 0xC118
REG_CP_RB_WPTR_ADDR_HI = 0xC11C
REG_CP_RB_WPTR_DELAY = 0x8704
REG_CP_RB_RPTR = 0x8700
REG_CP_ME_CNTL = 0x86D8          # R_0086D8_CP_ME_CNTL
REG_CP_SEM_WAIT_TIMER = 0x85BC
REG_CP_DEBUG = 0xC1FC
REG_SCRATCH_ADDR = 0x8544  # r600d.h SCRATCH_ADDR (not SCRATCH_REGx)
REG_SCRATCH_UMSK = 0x8540
REG_GRBM_SOFT_RESET = 0x8020     # r600d.h GRBM_SOFT_RESET
REG_GRBM_STATUS = 0x8010
SOFT_RESET_CP = 1 << 0

# evergreend.h - VGT compute (config space)
REG_VGT_NUM_INDICES = 0x8970
REG_VGT_COMPUTE_START_X = 0x899C
REG_VGT_COMPUTE_START_Y = 0x89A0
REG_VGT_COMPUTE_START_Z = 0x89A4
REG_VGT_COMPUTE_THREAD_GROUP_SIZE = 0x89AC

# evergreend.h - context regs (SET_CONTEXT_REG)
REG_SPI_COMPUTE_NUM_THREAD_X = 0x286EC
REG_SPI_COMPUTE_NUM_THREAD_Y = 0x286F0
REG_SPI_COMPUTE_NUM_THREAD_Z = 0x286F4
REG_SQ_PGM_START_LS = 0x288D0
REG_SQ_PGM_RESOURCES_LS = 0x288D4
REG_SQ_PGM_RESOURCES_LS_2 = 0x288D8
REG_SQ_LDS_ALLOC = 0x288E8

# Packet3 (evergreend.h)
PKT_TYPE3 = 3
PKT3_NOP = 0x10
PKT3_CONTEXT_CONTROL = 0x28
PKT3_DRAW_INDEX_AUTO = 0x2D
PKT3_DISPATCH_DIRECT = 0x15
PKT3_INDIRECT_BUFFER = 0x32
PKT3_EVENT_WRITE = 0x46
PKT3_SET_CONFIG_REG = 0x68
PKT3_SET_CONTEXT_REG = 0x69
PKT3_SET_RESOURCE = 0x6D
PKT3_ME_INITIALIZE = 0x44
PACKET3_SET_CONFIG_REG_START = 0x00008000
PACKET3_SET_CONTEXT_REG_START = 0x00028000
# Mesa / SI: compute packets set header bit1 (PACKET3_COMPUTE). Evergreen r600g
# uses the same PKT3C() for DISPATCH / SET_CONTEXT on the compute CS.
PACKET3_COMPUTE_MODE = 1 << 1
DISPATCH_INITIATOR_COMPUTE_SHADER_EN = 1

RB_RPTR_WR_ENA = 1 << 31
RB_NO_UPDATE = 1 << 27

def packet3(op: int, n: int, compute: bool = False) -> int:
  """PACKET3(op, n) - evergreend.h; optional compute bit (Mesa PKT3C)."""
  hdr = (PKT_TYPE3 << 30) | ((n & 0x3FFF) << 16) | ((op & 0xFF) << 8)
  if compute:
    hdr |= PACKET3_COMPUTE_MODE
  return hdr

def S_0288D4_NUM_GPRS(x: int) -> int:
  return (x & 0xFF) << 0

def S_0288D4_DX10_CLAMP(x: int) -> int:
  return (x & 1) << 13

def S_0288D4_STACK_SIZE(x: int) -> int:
  return (x & 0xFF) << 8

# RV770 graphics registers / encodings, copied from Mesa r600d.h.  Values are
# byte MMIO addresses; PM4 converts them to packet-relative dword offsets.
REG_VGT_PRIMITIVE_TYPE = 0x8958
REG_VGT_NUM_INSTANCES = 0x8974
REG_VGT_OUTPUT_PATH_CNTL, REG_VGT_HOS_CNTL = 0x28A10, 0x28A14
REG_VGT_HOS_MAX_TESS_LEVEL, REG_VGT_HOS_MIN_TESS_LEVEL = 0x28A18, 0x28A1C
REG_VGT_HOS_REUSE_DEPTH, REG_VGT_GROUP_PRIM_TYPE = 0x28A20, 0x28A24
REG_VGT_GS_MODE, REG_VGT_PRIMITIVEID_EN = 0x28A40, 0x28A84
REG_VGT_INSTANCE_STEP_RATE_0, REG_VGT_INSTANCE_STEP_RATE_1 = 0x28AA0, 0x28AA4
REG_VGT_REUSE_OFF, REG_VGT_VTX_CNT_EN = 0x28AB4, 0x28AB8
REG_VGT_STRMOUT_BUFFER_EN = 0x28B20
REG_VGT_STRMOUT_EN = 0x28AB0
REG_VGT_STRMOUT_BUFFER_SIZE_0, REG_VGT_STRMOUT_VTX_STRIDE_0 = 0x28AD0, 0x28AD4
REG_VGT_STRMOUT_BUFFER_BASE_0, REG_VGT_STRMOUT_BUFFER_OFFSET_0 = 0x28AD8, 0x28ADC
REG_VGT_MAX_VTX_INDX, REG_VGT_MIN_VTX_INDX = 0x28400, 0x28404
REG_SQ_PGM_CF_OFFSET_PS, REG_SQ_PGM_CF_OFFSET_VS = 0x288CC, 0x288D0
REG_SQ_PGM_CF_OFFSET_GS, REG_SQ_PGM_CF_OFFSET_ES = 0x288D4, 0x288D8
REG_SQ_PGM_CF_OFFSET_FS = 0x288DC
REG_SQ_VTX_SEMANTIC_CLEAR = 0x288E0
REG_SQ_PGM_RESOURCES_FS = 0x288A4
REG_CB_COLOR0_BASE, REG_CB_COLOR0_SIZE = 0x28040, 0x28060
REG_CB_COLOR0_VIEW, REG_CB_COLOR0_INFO, REG_CB_COLOR0_MASK = 0x28080, 0x280A0, 0x28100
REG_CB_COLOR0_FRAG, REG_CB_COLOR0_TILE = 0x280E0, 0x280C0
REG_CB_TARGET_MASK, REG_CB_SHADER_MASK = 0x28238, 0x2823C
REG_SQ_PGM_START_PS, REG_SQ_PGM_RESOURCES_PS, REG_SQ_PGM_EXPORTS_PS = 0x28840, 0x28850, 0x28854
REG_SQ_PGM_START_VS, REG_SQ_PGM_RESOURCES_VS, REG_SQ_PGM_START_FS = 0x28858, 0x28868, 0x28894
REG_SPI_VS_OUT_ID_0, REG_SPI_VS_OUT_CONFIG = 0x28614, 0x286C4
REG_SPI_PS_INPUT_CNTL_0, REG_SPI_PS_IN_CONTROL_0, REG_SPI_PS_IN_CONTROL_1 = 0x28644, 0x286CC, 0x286D0
REG_PA_CL_VTE_CNTL = 0x28818
REG_PA_SC_WINDOW_SCISSOR_TL, REG_PA_SC_WINDOW_SCISSOR_BR = 0x28204, 0x28208
REG_PA_SC_GENERIC_SCISSOR_TL, REG_PA_SC_GENERIC_SCISSOR_BR = 0x28240, 0x28244
REG_PA_SC_VPORT_SCISSOR_0_TL, REG_PA_SC_VPORT_SCISSOR_0_BR = 0x28250, 0x28254
REG_PA_CL_VPORT_XSCALE_0 = 0x2843C
REG_PA_SU_SC_MODE_CNTL, REG_PA_SC_MODE_CNTL = 0x28814, 0x28A4C
REG_PA_SU_VTX_CNTL, REG_CB_COLOR_CONTROL = 0x28C08, 0x28808
REG_CB_BLEND0_CONTROL, REG_CB_BLEND_CONTROL = 0x28780, 0x28804

RV770_FETCH_RESOURCE_VS = 160
RV770_VTX_FORMAT_32_32_32_32_FLOAT = 0x23
RV770_COLOR_32_32_32_32_FLOAT = 0x23
RV770_DI_PT_TRILIST, RV770_DI_SRC_SEL_AUTO_INDEX = 4, 2

# --- R600 PM4 completion / cache-coherency packets (r600d.h) ---
PKT3_SURFACE_SYNC = 0x43
PKT3_EVENT_WRITE_EOP = 0x47
EVENT_TYPE_CACHE_FLUSH_AND_INV_TS = 0x14      # CACHE_FLUSH_AND_INV_EVENT_TS
EVENT_TYPE_CACHE_FLUSH_AND_INV = 0x16         # CACHE_FLUSH_AND_INV_EVENT
EVENT_INDEX_TS = 5
EVENT_INDEX_NON_TS = 0
DATA_SEL_32 = 1                               # EOP writes low 32 bits
DATA_SEL_64 = 2
INT_SEL_NONE = 0                             # poll memory, no IRQ (TinyGPU masks MSI)
PACKET3_TC_ACTION_ENA = 1 << 23
PACKET3_VC_ACTION_ENA = 1 << 24
PACKET3_CB_ACTION_ENA = 1 << 25
PACKET3_DB_ACTION_ENA = 1 << 26
PACKET3_SH_ACTION_ENA = 1 << 27
PACKET3_SMX_ACTION_ENA = 1 << 28
PACKET3_CB0_DEST_BASE_ENA = 1 << 6
PACKET3_FULL_CACHE_ENA = 1 << 20             # r7xx+ only
REG_WAIT_UNTIL = 0x8040
WAIT_3D_IDLE = 1 << 0
WAIT_3D_IDLECLEAN = 1 << 4

# --- R600/R700 graphics context registers (byte MMIO offsets) ---
REG_PA_CL_CLIP_CNTL = 0x28810
REG_PA_SC_CLIPRECT_RULE = 0x2820C
REG_PA_SC_EDGERULE = 0x28230
REG_PA_SC_AA_CONFIG = 0x28C04
REG_PA_SC_WINDOW_OFFSET = 0x28200
REG_SPI_INTERP_CONTROL_0 = 0x286D4
REG_DB_DEPTH_CONTROL = 0x28800
REG_DB_SHADER_CONTROL = 0x2880C
REG_DB_RENDER_CONTROL = 0x28D0C
REG_DB_RENDER_OVERRIDE = 0x28D10
REG_SX_ALPHA_TEST_CONTROL = 0x28814
REG_PA_CL_VPORT_ZSCALE_0 = 0x28444
REG_PA_CL_VPORT_ZOFFSET_0 = 0x28448
REG_CB_COLOR0_ATTRIB2 = 0x28104

def rv770_cb_color_control(*, rop3: int = 0xCC, special_op: int = 0,
                           target_blend_enable: int = 0) -> int:
  """Encode CB_COLOR_CONTROL for a direct (no-blend) color write.

  bit 4-6  SPECIAL_OP  (0 = normal, 1 = disable)
  bit 8-15 TARGET_BLEND_ENABLE
  bit 16-23 ROP3       (0xCC = raster source copy)
  """
  return (((special_op & 0x7) << 4) | ((target_blend_enable & 0xFF) << 8) |
          ((rop3 & 0xFF) << 16))

def rv770_color_info_rgba32_float() -> int:
  """Encode CB_COLOR0_INFO for linear RGBA32_FLOAT, no CMASK/FMASK."""
  # FORMAT=0x23 (COLOR_32_32_32_32_FLOAT) at bit 2, ARRAY_LINEAR_ALIGNED=1
  # at bit 8, NUMBER_FLOAT=7 at bit 12, SWAP_STD=0, SIMPLE_FLOAT=1.
  # Mesa uses the aligned-linear mode even for a one-pixel surface; ARRAY_MODE
  # zero is linear-general and leaves the CB geometry undefined on R700.
  return ((0x23 << 2) | (1 << 8) | (7 << 12) | (0 << 16) | (1 << 24))

COLOR_CANARY = 0xA5  # fill value for the color target before a draw

# =============================================================================
# Placeholder CF/ALU binary (Evergreen LS) - replaced when HW + llvm-mc land
# =============================================================================
# Real Evergreen compute shaders are CF + ALU clause binaries (r600 ISA), not
# GCN VOP2. Until we assemble with llvm -march=r600 -mcpu=redwood, ship a
# recognizable stub: CF END + padding. Dispatch IB still encodes correctly.
#
# Layout comment (EG): SQ_PGM_START_LS is in 256-byte units (va >> 8), same as
# Mesa evergreen_emit_cs_shader.

def build_shader_stub_evergreen_add() -> bytes:
  """Minimal placeholder program blob (not executable ALU yet).

  Word0: CF_END-like sentinel 0x00000000; rest NOP pad to 256-byte alignment unit.
  Selftest only checks length/alignment + PM4; HW will need a real r600 binary.
  """
  # 64 dwords = 256 bytes - one PGM unit
  words = [0x00000000] + [0x00000000] * 63
  return b"".join(struct.pack("<I", w) for w in words)

ADD_SHADER = build_shader_stub_evergreen_add()
OP = lambda x, y: x + y
OP_NAME = "add"

RV770_ADD_LL = pathlib.Path(__file__).with_name("rv770_add.ll")
RV770_VS_LL = pathlib.Path(__file__).with_name("rv770_vs.ll")
RV770_CONSTANT_PS_LL = pathlib.Path(__file__).with_name("rv770_constant_ps.ll")
RV770_PARAM0_PS_LL = pathlib.Path(__file__).with_name("rv770_param0_ps.ll")
RV770_CONSTANT_VS_LL = pathlib.Path(__file__).with_name("rv770_constant_vs.ll")
RV770_STREAM_ADD_VS_LL = pathlib.Path(__file__).with_name("rv770_stream_add_vs.ll")

GPU_ADD_STAGE_CP = "cp"
GPU_ADD_STAGE_CONSTANT = "constant"
GPU_ADD_STAGE_PARAM0 = "param0"
GPU_ADD_STAGE_ADD = "add"
GPU_ADD_STAGE_STREAM = "stream"
GPU_ADD_STAGES = (GPU_ADD_STAGE_CP, GPU_ADD_STAGE_CONSTANT,
                  GPU_ADD_STAGE_PARAM0, GPU_ADD_STAGE_ADD, GPU_ADD_STAGE_STREAM)

def r600_llc() -> str | None:
  """Return an LLVM compiler with the legacy R600 backend, if installed."""
  candidates = [os.environ.get("R600_LLC"), shutil.which("llc"),
                "/opt/homebrew/opt/llvm/bin/llc"]
  return next((p for p in candidates if p and os.path.isfile(p) and os.access(p, os.X_OK)), None)

def elf_text(elf: bytes, expected_size: int | None = None) -> bytes:
  """Extract `.text` from LLVM's little-endian ELF32-AMDGPU object."""
  if len(elf) < 52 or elf[:7] != b"\x7fELF\x01\x01\x01":
    raise RuntimeError("LLVM did not produce a little-endian ELF32 R600 object")
  shoff, = struct.unpack_from("<I", elf, 0x20)
  shentsize, shnum, shstrndx = struct.unpack_from("<HHH", elf, 0x2e)
  if shentsize != 40 or not shnum or shstrndx >= shnum or shoff + shnum * shentsize > len(elf):
    raise RuntimeError("malformed ELF32 section table in R600 shader object")
  def section(i: int) -> tuple[int, int, int]:
    name, _, _, _, off, size, _, _, _, _ = struct.unpack_from("<IIIIIIIIII", elf, shoff + i * shentsize)
    if off + size > len(elf):
      raise RuntimeError("R600 shader section lies outside its ELF object")
    return name, off, size
  _, names_off, names_size = section(shstrndx)
  names = elf[names_off:names_off + names_size]
  for i in range(shnum):
    name_off, off, size = section(i)
    if name_off < len(names) and names[name_off:].split(b"\0", 1)[0] == b".text":
      blob = elf[off:off + size]
      if expected_size is not None and len(blob) != expected_size:
        raise RuntimeError(f"unexpected R600 shader size {len(blob)} (want {expected_size})")
      return blob
  raise RuntimeError("R600 shader object has no .text section")

def compile_rv770_add_shader() -> str:
  """Compile the genuine RV770 pixel shader and prove its four ALU ADDs.

  This produces assembly only; it never touches the GPU.  Binding the resulting
  64-byte .text program to the RV770 graphics pipeline remains a separate,
  explicit bring-up step.
  """
  llc = r600_llc()
  if llc is None:
    raise RuntimeError("no llc with the R600 backend; set R600_LLC=/path/to/llc")
  if not RV770_ADD_LL.is_file():
    raise RuntimeError(f"missing shader source: {RV770_ADD_LL}")
  proc = subprocess.run(
    [llc, "-march=r600", "-mcpu=rv770", "-filetype=asm", str(RV770_ADD_LL), "-o", "-"],
    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
  )
  if proc.returncode:
    raise RuntimeError(f"R600 shader compile failed:\n{proc.stderr.strip()}")
  asm = proc.stdout
  if asm.count("ADD * T0.") != 4 or "EXPORT T0.XYZW" not in asm:
    raise RuntimeError("compiled RV770 shader is missing its four ADDs or color export")
  return asm

def compile_rv770_add_blob() -> bytes:
  """Return the executable 64-byte `.text` section for the RV770 ALU shader.

  LLVM emits an ELF32-AMDGPU object.  Keeping extraction here makes the exact
  program that was inspected by `--compile-rv770-add` available to the future
  graphics draw path without checking an opaque generated blob into the tree.
  """
  llc = r600_llc()
  if llc is None:
    raise RuntimeError("no llc with the R600 backend; set R600_LLC=/path/to/llc")
  proc = subprocess.run(
    [llc, "-march=r600", "-mcpu=rv770", "-filetype=obj", str(RV770_ADD_LL), "-o", "-"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
  )
  if proc.returncode:
    raise RuntimeError(f"R600 shader object compile failed:\n{proc.stderr.decode().strip()}")
  return elf_text(proc.stdout, expected_size=64)

def compile_rv770_vs_blob() -> bytes:
  """Compile the matching RV770 VS (position, a, b exports) to executable text.

  R700 fetch hardware reserves GPR0 for the vertex index and Mesa's fetch
  shader deposits attributes in GPR1..3.  LLVM assigns entry arguments to
  GPR0..2, so adjust only the three CF-export GPR fields after validating the
  compiler's position/parameter export layout.  There are no ALU instructions
  in this VS; this is an ABI relocation, not a change to shader arithmetic.
  """
  llc = r600_llc()
  if llc is None or not RV770_VS_LL.is_file():
    raise RuntimeError("missing R600 compiler or RV770 vertex shader source")
  proc = subprocess.run([llc, "-march=r600", "-mcpu=rv770", "-filetype=obj", str(RV770_VS_LL), "-o", "-"],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
  if proc.returncode:
    raise RuntimeError(f"RV770 VS compile failed:\n{proc.stderr.decode().strip()}")
  raw = elf_text(proc.stdout, expected_size=48)
  words = list(struct.unpack("<12I", raw))
  # `store.swizzle` produces POS(GPR0), PARAM0(GPR1), PARAM1(GPR2).
  # type field is bits 13..14; exported source GPR is bits 15..21.
  want_types = (1, 2, 2)
  for word_index, source_gpr, typ in zip((2, 4, 6), (0, 1, 2), want_types):
    word = words[word_index]
    if ((word >> 13) & 3, (word >> 15) & 0x7F) != (typ, source_gpr):
      raise RuntimeError("unexpected LLVM RV770 VS export layout")
    words[word_index] = (word & ~(0x7F << 15)) | ((source_gpr + 1) << 15)
  return struct.pack("<12I", *words)

def compile_rv770_ps_blob(src: pathlib.Path, expected_size: int | None = None) -> bytes:
  """Compile any RV770 pixel shader source to its executable `.text` blob."""
  llc = r600_llc()
  if llc is None or not src.is_file():
    raise RuntimeError(f"missing R600 compiler or pixel shader source {src}")
  proc = subprocess.run([llc, "-march=r600", "-mcpu=rv770", "-filetype=obj", str(src), "-o", "-"],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
  if proc.returncode:
    raise RuntimeError(f"RV770 PS compile {src} failed:\n{proc.stderr.decode().strip()}")
  return elf_text(proc.stdout, expected_size=expected_size)

def compile_rv770_constant_ps_blob() -> bytes:
  return compile_rv770_ps_blob(RV770_CONSTANT_PS_LL)

def compile_rv770_param0_ps_blob() -> bytes:
  return compile_rv770_ps_blob(RV770_PARAM0_PS_LL)

def compile_rv770_constant_vs_blob() -> bytes:
  """Compile the no-fetch diagnostic VS used by the constant stage."""
  return compile_rv770_ps_blob(RV770_CONSTANT_VS_LL)

def compile_rv770_stream_add_vs_blob() -> bytes:
  """Compile streamout VS and relocate its inputs to fetch GPR2/GPR3.

  LLVM's standalone VS ABI uses position/a/b in GPR0/1/2; Mesa's R700 fetch
  shader reserves GPR0 for vertex ID and writes them to GPR1/2/3.  The stream
  shader has only exports and four ADD ALUs, so these source/destination field
  relocations are checked and deterministic.
  """
  raw = compile_rv770_ps_blob(RV770_STREAM_ADD_VS_LL, expected_size=80)
  words = list(struct.unpack("<20I", raw))
  # POS export (CF word 0), stream export (buffer export word 0).
  for idx, old in ((4, 0), (6, 1)):
    if ((words[idx] >> 15) & 0x7F) != old:
      raise RuntimeError("unexpected stream VS export GPR layout")
    words[idx] = (words[idx] & ~(0x7F << 15)) | ((old + 1) << 15)
  # Four R700 ALU ADD pairs in the final clause: src1/src2 and destination
  # GPRs move from (1,2,1) to (2,3,2).
  for i in (12, 14, 16, 18):
    w0, w1 = words[i], words[i + 1]
    if ((w0 & 0x1FF, (w0 >> 13) & 0x1FF, (w1 >> 21) & 0x7F) != (1, 2, 1)):
      raise RuntimeError("unexpected stream VS ADD operand layout")
    words[i] = (w0 & ~((0x1FF << 0) | (0x1FF << 13))) | (2 << 0) | (3 << 13)
    words[i + 1] = (w1 & ~(0x7F << 21)) | (2 << 21)
  return struct.pack("<20I", *words)

def build_rv770_noop_fetch_blob() -> bytes:
  """R700 fetch program that only returns (for the constant-position test)."""
  return struct.pack("<2I", 0, (21 << 23) | (1 << 21) | (1 << 31))

def build_rv770_empty_ps_blob() -> bytes:
  """Minimal CF_END PS for isolating raster/CB state from shader execution."""
  return struct.pack("<4I", 0, (0x20 << 23) | (1 << 21) | (1 << 31), 0, 0)

def build_rv770_empty_vs_blob() -> bytes:
  """Minimal CF_END VS for isolating draw/CB state from shader execution."""
  return build_rv770_empty_ps_blob()

def build_rv770_vertex_fetch_blob() -> bytes:
  """Build Mesa's RV770 vertex-fetch program for ``position, a, b``.

  R700 does not fetch vertex attributes in the LLVM vertex shader.  Mesa emits
  a small *fetch shader* at ``SQ_PGM_START_FS`` first; it reads resource 160
  (the VS vertex-buffer resource bank) into GPRs 1, 2 and 3.  The LLVM VS then
  receives those three vectors as T0, T1 and T2.  This is a direct Python
  transcription of ``r600_bytecode_vtx_build`` and
  ``r700_bytecode_cf_vtx_build`` for three RGBA32_FLOAT elements, not an
  invented opaque blob.
  """
  # r700_sq.h: VTX word fields.  VFETCH opcode=0, FETCH_VERTEX_DATA=0,
  # resource 160 (R600_FETCH_CONSTANTS_OFFSET_VS), src_gpr=0/index.x,
  # mega_fetch_count=31, RGBA swizzle XYZW, FMT_32_32_32_32_FLOAT=0x23.
  def vfetch(dst_gpr: int, offset: int) -> list[int]:
    word0 = (160 << 8) | (0x1F << 26)
    word1 = (dst_gpr << 0) | (0 << 9) | (1 << 12) | (2 << 15) | (3 << 18) | (0x23 << 22)
    word2 = (offset & 0xFFFF) | (1 << 19)  # MEGA_FETCH
    return [word0, word1, word2, 0]

  # CF_OP_VTX is opcode 2 for R700.  R600's builder reserves three dwords then
  # aligns a fetch clause to four dwords: with two CF records, fetch code begins
  # at dword 8 => address 4 in 64-bit words.
  # COUNT=2 describes its three four-dword VFETCH instructions.  The RET has
  # no body and terminates the fetch program.
  cf_vtx = [4, (2 << 23) | (2 << 10) | (1 << 31)]
  cf_ret = [0, (21 << 23) | (1 << 21) | (1 << 31)]
  words = cf_vtx + cf_ret + [0, 0, 0, 0] + vfetch(1, 0) + vfetch(2, 16) + vfetch(3, 32)
  blob = struct.pack(f"<{len(words)}I", *words)
  if len(blob) != 80:
    raise AssertionError(f"RV770 fetch shader must be 80 bytes, got {len(blob)}")
  return blob

# =============================================================================
# PM4 builders (Mesa evergreen_emit_cs_shader + evergreen_emit_dispatch)
# =============================================================================
class PM4Builder:
  """Evergreen compute IB builder (config + context + DISPATCH_DIRECT)."""

  def __init__(self, compute: bool = True):
    self.words: list[int] = []
    self.compute = compute

  def emit(self, *vals: int):
    self.words.extend(int(v) & 0xFFFFFFFF for v in vals)

  def pkt3(self, op: int, *vals: int, compute: bool | None = None):
    use_c = self.compute if compute is None else compute
    n = max(len(vals) - 1, 0)
    self.words.append(packet3(op, n, compute=use_c))
    self.words.extend(int(v) & 0xFFFFFFFF for v in vals)

  def set_config_reg(self, reg_byte: int, value: int):
    off = (reg_byte - PACKET3_SET_CONFIG_REG_START) >> 2
    if not (0 <= off < 0x2B00):
      raise ValueError(f"config reg {reg_byte:#x} off={off:#x} out of range")
    # SET_CONFIG is not compute-flagged in Mesa for VGT_* 
    self.pkt3(PKT3_SET_CONFIG_REG, off, value, compute=False)

  def set_config_reg_seq(self, reg_byte: int, *values: int):
    off = (reg_byte - PACKET3_SET_CONFIG_REG_START) >> 2
    n = len(values)  # PACKET3 count = n (reg + n values -> count field = n)
    # evergreend: PACKET3(op, n) where n = number of following dwords - 1
    # set_config_reg_seq emits: header, offset, v0, v1, ... -> following = 1+len
    self.words.append(packet3(PKT3_SET_CONFIG_REG, len(values), compute=False))
    self.words.append(off & 0xFFFFFFFF)
    self.words.extend(int(v) & 0xFFFFFFFF for v in values)

  def set_context_reg(self, reg_byte: int, value: int):
    off = (reg_byte - PACKET3_SET_CONTEXT_REG_START) >> 2
    if not (0 <= off < 0x400):
      raise ValueError(f"context reg {reg_byte:#x} off={off:#x} out of range")
    self.pkt3(PKT3_SET_CONTEXT_REG, off, value)

  def set_context_reg_seq(self, reg_byte: int, *values: int):
    off = (reg_byte - PACKET3_SET_CONTEXT_REG_START) >> 2
    self.words.append(packet3(PKT3_SET_CONTEXT_REG, len(values), compute=self.compute))
    self.words.append(off & 0xFFFFFFFF)
    self.words.extend(int(v) & 0xFFFFFFFF for v in values)

  def set_resource(self, index: int, *values: int):
    """Emit one R600 resource descriptor (seven dwords)."""
    if len(values) != 7:
      raise ValueError(f"R600 resource needs 7 dwords, got {len(values)}")
    self.pkt3(PKT3_SET_RESOURCE, index * 7, *values, compute=False)

  def emit_cs_shader(self, shader_gpu_addr: int, ngpr: int = 4, nstack: int = 1):
    """evergreen_emit_cs_shader: SQ_PGM_START_LS + RESOURCES (va >> 8)."""
    va_lo = (shader_gpu_addr >> 8) & 0xFFFFFFFF
    rsrc = S_0288D4_NUM_GPRS(ngpr) | S_0288D4_DX10_CLAMP(1) | S_0288D4_STACK_SIZE(nstack)
    self.set_context_reg_seq(REG_SQ_PGM_START_LS, va_lo, rsrc, 0)

  def emit_dispatch(self, block=(1, 1, 1), grid=(1, 1, 1), lds_dwords: int = 0, num_waves: int = 1):
    """evergreen_emit_dispatch (direct): VGT + SPI threads + DISPATCH_DIRECT."""
    group_size = int(block[0]) * int(block[1]) * int(block[2])
    self.set_config_reg(REG_VGT_NUM_INDICES, group_size)
    self.set_config_reg_seq(REG_VGT_COMPUTE_START_X, 0, 0, 0)
    self.set_config_reg(REG_VGT_COMPUTE_THREAD_GROUP_SIZE, group_size)
    self.set_context_reg_seq(REG_SPI_COMPUTE_NUM_THREAD_X, block[0], block[1], block[2])
    self.set_context_reg(REG_SQ_LDS_ALLOC, (lds_dwords & 0x3FFF) | ((num_waves & 0x3F) << 14))
    self.pkt3(PKT3_DISPATCH_DIRECT, grid[0], grid[1], grid[2], DISPATCH_INITIATOR_COMPUTE_SHADER_EN,
              compute=True)

  def build_dispatch_ib(self, shader_gpu_addr: int, out_va: int, a_va: int, b_va: int,
                        block=(1, 1, 1), grid=(1, 1, 1)) -> list[int]:
    """Full Evergreen compute IB skeleton.

    USER_DATA / RAT / global pool bindings are **not** wired yet (Mesa uses a
    compute memory pool + RAT). out/a/b VAs are recorded in trailing NOPs so
    dry-run/selftest can assert the intended buffer map; real RAT setup lands
    with HW bring-up.
    """
    self.words = []
    self.emit_cs_shader(shader_gpu_addr)
    self.emit_dispatch(block=block, grid=grid)
    # Scratch markers (NOP payloads) - not executed as regs
    self.pkt3(PKT3_NOP, lo32(out_va), hi32(out_va), compute=True)
    self.pkt3(PKT3_NOP, lo32(a_va), hi32(a_va), compute=True)
    self.pkt3(PKT3_NOP, lo32(b_va), hi32(b_va), compute=True)
    return self.words

def emit_rv770_completion(p: "PM4Builder", fence_gpu: int, fence_sequence: int,
                          mode: str = "eop") -> None:
  """Append a cache-flushing graphics completion to ``p``.

  ``eop`` (preferred): SURFACE_SYNC then EVENT_WRITE_EOP that writes the 32-bit
  ``fence_sequence`` into ``fence_gpu``.  No interrupt is selected because
  TinyGPU masks MSIs; the CPU polls the fence memory instead.

  ``wait-memwrite`` (debug only): SURFACE_SYNC, EVENT_WRITE cache flush,
  WAIT_UNTIL 3D idle, then a single CP MEM_WRITE of the sequence to
  ``fence_gpu``.  This may use MEM_WRITE *only* for the fence, never the color
  target.
  """
  coher = (PACKET3_TC_ACTION_ENA | PACKET3_VC_ACTION_ENA |
           PACKET3_CB_ACTION_ENA | PACKET3_CB0_DEST_BASE_ENA |
           PACKET3_SH_ACTION_ENA | PACKET3_SMX_ACTION_ENA |
           PACKET3_FULL_CACHE_ENA)
  if mode == "raw-memwrite":
    # Diagnostic only: CP writes a fence immediately after DRAW, with no
    # cache/3D-idle wait.  This distinguishes a stalled graphics engine from
    # a bad completion packet.
    d0, d1 = data64_le(fence_sequence & 0xFFFFFFFF)
    p.pkt3(PKT3_MEM_WRITE,
           lo32(fence_gpu) & 0xFFFFFFFC, hi32(fence_gpu) & 0xFF, d0, d1, compute=False)
    return
  # SURFACE_SYNC: coher_cntl, 0xFFFFFFFF, 0, poll_interval=10
  p.pkt3(PKT3_SURFACE_SYNC, coher, 0xFFFFFFFF, 0, 10, compute=False)
  if mode == "eop":
    p.pkt3(PKT3_EVENT_WRITE_EOP,
           EVENT_TYPE_CACHE_FLUSH_AND_INV_TS | (EVENT_INDEX_TS << 8),
           lo32(fence_gpu),
           (hi32(fence_gpu) & 0xFF) | (DATA_SEL_32 << 29) | (INT_SEL_NONE << 24),
           fence_sequence & 0xFFFFFFFF, 0, compute=False)
  else:
    p.pkt3(PKT3_EVENT_WRITE,
           EVENT_TYPE_CACHE_FLUSH_AND_INV | (EVENT_INDEX_NON_TS << 8), compute=False)
    off = (REG_WAIT_UNTIL - PACKET3_SET_CONFIG_REG_START) >> 2
    p.pkt3(PKT3_SET_CONFIG_REG, off, WAIT_3D_IDLE | WAIT_3D_IDLECLEAN, compute=False)
    d0, d1 = data64_le(fence_sequence & 0xFFFFFFFF)
    p.pkt3(PKT3_MEM_WRITE,
           lo32(fence_gpu) & 0xFFFFFFFC, hi32(fence_gpu) & 0xFF, d0, d1, compute=False)

def emit_rv770_full_gfx_init(p: "PM4Builder") -> None:
  """Opt-in explicit R700 graphics-context defaults (Phase 7).

  Disables clipping, AA, depth and alpha test so a draw cannot be silently
  dropped by reset-default state.  Safe: every entry is a disable/zero that
  does not touch MC routing, display, clocks or local VRAM ownership.
  """
  # r600_init_atom_start_cs VGT defaults: these are required even for a
  # triangle-list with no tessellation/GS/streamout.
  p.set_context_reg_seq(REG_VGT_OUTPUT_PATH_CNTL, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0)
  p.set_context_reg(REG_VGT_GS_MODE, 0)
  p.set_context_reg(REG_VGT_PRIMITIVEID_EN, 0)
  p.set_context_reg_seq(REG_VGT_INSTANCE_STEP_RATE_0, 0, 0)
  p.set_context_reg_seq(REG_VGT_REUSE_OFF, 1, 0)
  p.set_context_reg(REG_VGT_STRMOUT_BUFFER_EN, 0)
  p.set_context_reg_seq(REG_VGT_MAX_VTX_INDX, 0xFFFFFFFF, 0)
  p.set_context_reg_seq(REG_SQ_PGM_CF_OFFSET_PS, 0, 0, 0, 0, 0)
  p.set_context_reg(REG_SQ_VTX_SEMANTIC_CLEAR, 0xFFFFFFFF)
  p.set_context_reg(REG_SQ_PGM_RESOURCES_FS, 0)
  p.set_context_reg(REG_PA_CL_CLIP_CNTL, 0)            # no user clip planes
  p.set_context_reg(REG_PA_SC_CLIPRECT_RULE, 0x0000FFFF)  # all cliprect edges pass
  p.set_context_reg(REG_PA_SC_EDGERULE, 0xFFFF)
  p.set_context_reg(REG_PA_SC_AA_CONFIG, 0)            # no MSAA
  p.set_context_reg(REG_PA_SC_WINDOW_OFFSET, 0)
  p.set_context_reg(REG_SPI_INTERP_CONTROL_0, 0)
  p.set_context_reg(REG_DB_DEPTH_CONTROL, 0)           # depth test/compare off
  p.set_context_reg(REG_DB_SHADER_CONTROL, 0)
  p.set_context_reg(REG_DB_RENDER_CONTROL, 0)
  p.set_context_reg(REG_DB_RENDER_OVERRIDE, 0)
  p.set_context_reg(REG_SX_ALPHA_TEST_CONTROL, 0)      # alpha test bypass

def _rv770_stage_linkage(stage: str) -> tuple[int, tuple[int, ...]]:
  """Return (NUM_INTERP, SPI_PS_INPUT_CNTL_0 entries) for ``stage``."""
  if stage in (GPU_ADD_STAGE_CONSTANT, GPU_ADD_STAGE_STREAM):
    return 0, ()
  if stage == GPU_ADD_STAGE_PARAM0:
    return 1, (0 | (1 << 12),)            # semantic 0
  # add (and any other graphics stage): two interpolated inputs, semantics 0/1
  return 2, (0 | (1 << 12), 1 | (1 << 12))

def build_rv770_add_draw(vs_gpu: int, ps_gpu: int, fetch_gpu: int,
                         vertices_gpu: int, color_gpu: int, *,
                         stage: str = "add", fence_gpu: int = 0,
                         fence_sequence: int = 0, fence_mode: str = "eop",
                         full_gfx_init: bool = False,
                         constant_vs: bool = False,
                         empty_vs: bool = False) -> list[int]:
  """Build the non-compute PM4 for one RV770 fullscreen-triangle draw.

  Pure builder: every dword can be reviewed with ``--gpu-add-dry-run`` before
  hardware submission.  It never contains a literal result payload; the only
  result path is PS export -> CB_COLOR0 in AGP memory.

  Stages:
    ``cp``       - completion fence only (no draw); proves fence + CPU visibility.
    ``constant`` - PS exports a constant; proves draw/raster/PS/CB.
    ``param0``   - PS exports interpolated PARAM0; proves fetch/VS/SPI linkage.
    ``add``      - PS exports PARAM0 + PARAM1; the real GPU arithmetic.
  """
  for name, addr in (("VS", vs_gpu), ("PS", ps_gpu), ("fetch", fetch_gpu),
                     ("vertices", vertices_gpu), ("color", color_gpu)):
    if addr & 0xFF:
      raise ValueError(f"RV770 {name} address must be 256-byte aligned: {addr:#x}")
  if fence_gpu & 7:
    raise ValueError(f"RV770 fence address must be 8-byte aligned: {fence_gpu:#x}")
  p = PM4Builder(compute=False)
  p.pkt3(PKT3_CONTEXT_CONTROL, 0x80000000, 0x80000000, compute=False)
  if stage == GPU_ADD_STAGE_CP:
    emit_rv770_completion(p, fence_gpu, fence_sequence, fence_mode)
    return p.words
  # Mesa r600_init_atom_start_cs: establish a clean graphics context.
  p.set_config_reg(REG_VGT_PRIMITIVE_TYPE, RV770_DI_PT_TRILIST)
  p.set_config_reg(REG_VGT_NUM_INSTANCES, 1)
  # r600_emit_vertex_buffers: resource bank VS=160; buffer is three 48-byte
  # records.  The direct transport has no kernel relocation, hence the GPU
  # aperture address is supplied in WORD0 rather than Mesa's reloc placeholder.
  if not constant_vs and not empty_vs:
    p.set_resource(RV770_FETCH_RESOURCE_VS,
                   vertices_gpu, PAGE_SIZE - 1, 48 << 8, 0, 0, 0, 0xC0000000)
  # Programs and their compiler-reported GPR requirements.  LLVM's VS inputs
  # are fetch GPR 1/2/3; PS consumes the interpolated parameter exports.
  p.set_context_reg(REG_SQ_PGM_START_FS, fetch_gpu >> 8)
  p.set_context_reg(REG_SQ_PGM_START_VS, vs_gpu >> 8)
  # LLVM's .AMDGPU.config reports VS resources=0x103 (three GPRs, one stack
  # entry) and PS resources=0x2 (two GPRs, no stack); do not guess clamps.
  p.set_context_reg(REG_SQ_PGM_RESOURCES_VS, (0 if empty_vs else (1 if constant_vs else 4)) | (1 << 8))
  p.set_context_reg(REG_SQ_PGM_START_PS, ps_gpu >> 8)
  streamout = stage == GPU_ADD_STAGE_STREAM
  ps_gprs = 0 if (getenv("AMD_GPU_ADD_EMPTY_PS", 0) or streamout) else (1 if stage == GPU_ADD_STAGE_CONSTANT else 2)
  ps_exports = 0 if (getenv("AMD_GPU_ADD_EMPTY_PS", 0) or streamout) else 2
  p.set_context_reg_seq(REG_SQ_PGM_RESOURCES_PS, ps_gprs, ps_exports)
  # VS always exports POS + PARAM0(sem0) + PARAM1(sem1); only the PS side
  # changes per stage (num_interp and which inputs are consumed).
  if empty_vs:
    # A CF_END-only VS exports no position/parameters; do not advertise the
    # normal three-export linkage while running this isolation probe.
    p.set_context_reg(REG_SPI_VS_OUT_ID_0, 0)
    p.set_context_reg(REG_SPI_VS_OUT_CONFIG, 0)
  else:
    p.set_context_reg(REG_SPI_VS_OUT_ID_0, 0 | (1 << 8))
    p.set_context_reg(REG_SPI_VS_OUT_CONFIG, 1 << 1)
  num_interp, ps_inputs = _rv770_stage_linkage(stage)
  if ps_inputs:
    p.set_context_reg_seq(REG_SPI_PS_INPUT_CNTL_0, *ps_inputs)
  # Mesa enables both gradient engines in normal PS state even when NUM_INTERP
  # is zero (constant stage); leaving PERSP_GRADIENT_ENA reset can strand SH.
  p.set_context_reg_seq(REG_SPI_PS_IN_CONTROL_0, num_interp | (1 << 28) | (1 << 29), 0)
  # Convert the fullscreen triangle's NDC position to a single-pixel viewport.
  p.set_context_reg(REG_PA_CL_VTE_CNTL, (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5) | (1 << 10))
  # These are required even without the optional Mesa-style full context
  # replay.  After our CP/graphics reset, stale clip/depth state can discard
  # every primitive before it reaches the pixel backend.
  p.set_context_reg(REG_PA_CL_CLIP_CNTL, 0)
  p.set_context_reg(REG_PA_SC_CLIPRECT_RULE, 0x0000FFFF)
  p.set_context_reg(REG_PA_SC_EDGERULE, 0xFFFF)
  p.set_context_reg(REG_PA_SC_AA_CONFIG, 0)
  p.set_context_reg(REG_PA_SC_WINDOW_OFFSET, 0)
  p.set_context_reg(REG_DB_DEPTH_CONTROL, 0)
  p.set_context_reg(REG_DB_SHADER_CONTROL, 0)
  p.set_context_reg(REG_DB_RENDER_CONTROL, 0)
  p.set_context_reg(REG_DB_RENDER_OVERRIDE, 0)
  half = struct.unpack("<I", struct.pack("<f", 0.5))[0]
  # XSCALE, XOFFSET, YSCALE, YOFFSET, ZSCALE, ZOFFSET (one-pixel viewport).
  p.set_context_reg_seq(REG_PA_CL_VPORT_XSCALE_0, half, half, half, half, half, half)
  p.set_context_reg_seq(REG_PA_SC_WINDOW_SCISSOR_TL, 0, 1 | (1 << 16))
  p.set_context_reg_seq(REG_PA_SC_GENERIC_SCISSOR_TL, 0, 1 | (1 << 16))
  p.set_context_reg_seq(REG_PA_SC_VPORT_SCISSOR_0_TL, 0, 1 | (1 << 16))
  # Explicit raster/CB defaults matter after a CP-only reset: without the R700
  # viewport-scissor enable and bounds, all primitives can be clipped before PS.
  p.set_context_reg(REG_PA_SU_SC_MODE_CNTL, 0)  # fill, no culling
  p.set_context_reg(REG_PA_SU_VTX_CNTL, 1 | (2 << 1) | (5 << 3))
  p.set_context_reg(REG_PA_SC_MODE_CNTL, (1 << 14) | (1 << 16) | (1 << 20) | (1 << 22))
  if full_gfx_init:
    emit_rv770_full_gfx_init(p)
  # Linear RGBA32_FLOAT, no CMASK/FMASK.  COLOR0_BASE is in 256-byte units.
  color_info = rv770_color_info_rgba32_float()
  p.set_context_reg(REG_CB_COLOR0_BASE, color_gpu >> 8)
  # Mesa binds the non-MSAA surface's fmask/cmask backing addresses even when
  # both alias the color base.  Leaving these at reset zero can suppress all
  # CB writes on R700.
  p.set_context_reg(REG_CB_COLOR0_FRAG, color_gpu >> 8)
  p.set_context_reg(REG_CB_COLOR0_TILE, color_gpu >> 8)
  p.set_context_reg(REG_CB_COLOR0_SIZE, 0)
  p.set_context_reg(REG_CB_COLOR0_VIEW, 0)
  p.set_context_reg(REG_CB_COLOR0_INFO, color_info)
  p.set_context_reg(REG_CB_COLOR0_MASK, 0)
  p.set_context_reg_seq(REG_CB_TARGET_MASK, 0xF, 0xF)
  p.set_context_reg(REG_CB_BLEND0_CONTROL, 0)
  p.set_context_reg(REG_CB_BLEND_CONTROL, 0)
  # CB_NORMAL (special_op=0) with ROP3=copy: the color backend is enabled and
  # writes the shader output.  CB_COLOR_CONTROL=0 disables the CB entirely.
  p.set_context_reg(REG_CB_COLOR_CONTROL,
                    rv770_cb_color_control(rop3=0xCC, special_op=0))
  if streamout:
    # r600_streamout.c: size is the end offset in dwords, stride is dwords per
    # vertex, and base is in 256-byte units.  The VS stream export writes sum
    # to the same AGP page used as the result allocation.
    p.set_context_reg_seq(REG_VGT_STRMOUT_BUFFER_SIZE_0,
                          PAGE_SIZE >> 2, 4, color_gpu >> 8)
    p.set_context_reg(REG_VGT_STRMOUT_BUFFER_OFFSET_0, 0)
    p.set_context_reg(REG_VGT_STRMOUT_BUFFER_EN, 1)
    p.set_context_reg(REG_VGT_STRMOUT_EN, 1)
  p.pkt3(PKT3_DRAW_INDEX_AUTO, 3, RV770_DI_SRC_SEL_AUTO_INDEX, compute=False)
  emit_rv770_completion(p, fence_gpu, fence_sequence, fence_mode)
  return p.words

def validate_gpu_add_pm4(words: list[int], *, color_gpu: int, fence_gpu: int,
                         stage: str = "add", allow_fence_memwrite: bool = False) -> None:
  """Reject CP result writes and malformed completion packets.

  The only permitted result path is PS export -> CB_COLOR0.  Any CP MEM_WRITE
  to the color target is forbidden.  Exactly one completion fence must be
  emitted, and (for graphics stages) it must follow the draw/wait.
  """
  saw_draw = False
  fence_writes = 0
  fence_after_draw = True
  for i, w in enumerate(words):
    if w >> 30 != PKT_TYPE3:
      continue
    op = (w >> 8) & 0xFF
    n = (w >> 16) & 0x3FFF
    body = words[i + 1:i + 1 + n + 1]
    if op == PKT3_DRAW_INDEX_AUTO:
      saw_draw = True
    elif op == PKT3_MEM_WRITE:
      if len(body) < 4:
        raise AssertionError("malformed CP MEM_WRITE packet")
      addr = lo32(body[0]) | ((body[1] & 0xFF) << 32)
      if addr == color_gpu:
        raise AssertionError("GPU add PM4 writes the color target via CP MEM_WRITE")
      if addr == fence_gpu:
        if not allow_fence_memwrite:
          raise AssertionError("unexpected CP MEM_WRITE to fence in non-memwrite mode")
        fence_writes += 1
        if not saw_draw:
          fence_after_draw = False
      else:
        raise AssertionError(f"unexpected CP MEM_WRITE to {addr:#x}")
    elif op == PKT3_EVENT_WRITE_EOP:
      fence_writes += 1
  if stage != GPU_ADD_STAGE_CP and not saw_draw:
    raise AssertionError("GPU add PM4 has no DRAW_INDEX_AUTO")
  if stage != GPU_ADD_STAGE_CP and not fence_after_draw:
    raise AssertionError("completion fence MEM_WRITE precedes the draw")
  if fence_writes != 1:
    raise AssertionError(f"expected exactly one completion fence, got {fence_writes}")
  if not allow_fence_memwrite and any(((w >> 8) & 0xFF) == PKT3_MEM_WRITE for w in words):
    raise AssertionError("CP MEM_WRITE present but fence memwrite mode is off")

def decode_rv770_pm4(words: list[int]) -> list[str]:
  """Offline PM4 decoder for diagnostics (Phase 9.2)."""
  out: list[str] = []
  names = {
    PKT3_CONTEXT_CONTROL: "CONTEXT_CONTROL", PKT3_DRAW_INDEX_AUTO: "DRAW_INDEX_AUTO",
    PKT3_SET_CONFIG_REG: "SET_CONFIG_REG", PKT3_SET_CONTEXT_REG: "SET_CONTEXT_REG",
    PKT3_SET_RESOURCE: "SET_RESOURCE", PKT3_SURFACE_SYNC: "SURFACE_SYNC",
    PKT3_EVENT_WRITE: "EVENT_WRITE", PKT3_EVENT_WRITE_EOP: "EVENT_WRITE_EOP",
    PKT3_MEM_WRITE: "MEM_WRITE", PKT3_NOP: "NOP",
  }
  reg_names = {v: k for k, v in globals().items()
               if k.startswith(("REG_",)) and isinstance(v, int)}
  for i, w in enumerate(words):
    if w == 0xFFFFFFFF:
      # Legitimate body dword (e.g. SURFACE_SYNC flush mask); not a packet header.
      out.append(f"{i:04d}: {w:08x} (data)")
      continue
    if w >> 30 != PKT_TYPE3:
      out.append(f"{i:04d}: {w:08x} (data)")
      continue
    op = (w >> 8) & 0xFF
    n = (w >> 16) & 0x3FFF
    body = words[i + 1:i + 1 + n + 1]
    tag = names.get(op, f"PKT3_{op:#x}")
    if op in (PKT3_SET_CONTEXT_REG, PKT3_SET_CONFIG_REG) and body:
      rname = reg_names.get(PACKET3_SET_CONTEXT_REG_START + (body[0] << 2), "?") \
              if op == PKT3_SET_CONTEXT_REG else \
              reg_names.get(PACKET3_SET_CONFIG_REG_START + (body[0] << 2), "?")
      out.append(f"{i:04d}: {tag} {rname} = " +
                 ", ".join(f"{x:#x}" for x in body[1:]))
    elif op == PKT3_EVENT_WRITE_EOP:
      out.append(f"{i:04d}: {tag} seq={body[-2]:#x} addr={body[1]:#x}")
    else:
      out.append(f"{i:04d}: {tag} " + ", ".join(f"{x:#x}" for x in body))
  return out

def build_cp_resume_regs(ring_gpu_addr: int, ring_size: int = 0x10000,
                         wb_gpu_addr: int = 0) -> list[tuple[int, int]]:
  """Ordered MMIO writes mirroring r600_cp_resume (r600.c) - for dry-run dump.

  Returns [(reg_byte, value), ...]. Does not touch hardware.
  """
  # order_base_2(ring_size/8)
  rb_bufsz = (ring_size // 8).bit_length() - 1
  page_log = (4096 // 8).bit_length() - 1
  tmp = (page_log << 8) | rb_bufsz
  seq = [
    (REG_GRBM_SOFT_RESET, SOFT_RESET_CP),
    (REG_GRBM_SOFT_RESET, 0),
    (REG_CP_RB_CNTL, tmp),
    (REG_CP_SEM_WAIT_TIMER, 0),
    (REG_CP_RB_WPTR_DELAY, 0),
    (REG_CP_RB_CNTL, tmp | RB_RPTR_WR_ENA),
    (REG_CP_RB_RPTR_WR, 0),
    (REG_CP_RB_WPTR, 0),
    (REG_CP_RB_RPTR_ADDR, lo32(wb_gpu_addr) & 0xFFFFFFFC),
    (REG_CP_RB_RPTR_ADDR_HI, hi32(wb_gpu_addr) & 0xFF),
    (REG_SCRATCH_ADDR, (wb_gpu_addr >> 8) & 0xFFFFFFFF),
    (REG_SCRATCH_UMSK, 0),
    (REG_CP_RB_CNTL, tmp | RB_NO_UPDATE),
    (REG_CP_RB_BASE, ring_gpu_addr >> 8),
    (REG_CP_DEBUG, (1 << 27) | (1 << 28)),
    (REG_CP_ME_CNTL, 0xFF),  # after ME_INITIALIZE on ring
  ]
  return seq

def build_me_initialize(family: str, max_hw_contexts: int = 8) -> list[int]:
  """r600_cp_start ME_INITIALIZE packet (ring contents)."""
  # r600.c: family >= CHIP_RV770 -> contexts path
  words = [
    packet3(PKT3_ME_INITIALIZE, 5, compute=False),
    0x1,
    0x0 if family in (CHIP_RV770, CHIP_REDWOOD) else 0x3,
    max_hw_contexts - 1,
    (1 << 16),  # PACKET3_ME_INITIALIZE_DEVICE_ID(1)
    0,
    0,
  ]
  return words

# =============================================================================
# Host PCI helpers + R700 boot / add (HD 4850)
# =============================================================================
PCI_VID_AMD = 0x1002
PAGE_SIZE = 0x1000
R700_PFP_UCODE_SIZE = 848
R700_PM4_UCODE_SIZE = 1360
CP_ME_HALT = 1 << 28
CP_PFP_HALT = 1 << 26
REG_CP_ME_RAM_DATA = 0xC160
REG_CP_ME_RAM_RADDR = 0xC158
REG_CP_ME_RAM_WADDR = 0xC15C
REG_CP_PFP_UCODE_ADDR = 0xC150
REG_CP_PFP_UCODE_DATA = 0xC154
REG_SCRATCH_REG0 = 0x8500
# RV770 MC regs are at 0x2024 (rv770d.h) - NOT R600's 0x2180.
REG_MC_VM_FB_LOCATION = 0x2024
REG_MC_VM_AGP_TOP = 0x2028
REG_MC_VM_AGP_BOT = 0x202C
REG_MC_VM_AGP_BASE = 0x2030
REG_MC_VM_SYSTEM_APERTURE_LOW = 0x2034
REG_MC_VM_SYSTEM_APERTURE_HIGH = 0x2038
REG_MC_VM_SYSTEM_APERTURE_DEFAULT = 0x203C
REG_MC_VM_MB_L1_TLB0_CNTL = 0x2234
REG_MC_VM_MB_L1_TLB1_CNTL = 0x2238
REG_MC_VM_MB_L1_TLB2_CNTL = 0x223C
REG_MC_VM_MB_L1_TLB3_CNTL = 0x2240
REG_MC_VM_MD_L1_TLB0_CNTL = 0x2654
REG_MC_VM_MD_L1_TLB1_CNTL = 0x2658
REG_MC_VM_MD_L1_TLB2_CNTL = 0x265C
REG_HDP_NONSURFACE_BASE = 0x2C04
REG_HDP_NONSURFACE_INFO = 0x2C08
REG_HDP_NONSURFACE_SIZE = 0x2C0C
REG_HDP_DEBUG1 = 0x2F34  # rv770d.h (r7xx coherency quirk)
REG_VM_L2_CNTL = 0x1400
REG_VM_L2_CNTL2 = 0x1404
REG_VM_L2_CNTL3 = 0x1408
REG_VM_CONTEXT0_CNTL = 0x1410
REG_CONFIG_MEMSIZE = 0x5428
R700_MC_CITF_CNTL = 0x25c0          # r600_reg.h - MC blackout control
R600_BIF_FB_EN = 0x5490
R600_BLACKOUT_MASK = 0x3
R600_FB_READ_EN = 1 << 0
R600_FB_WRITE_EN = 1 << 1
# VBIOS ROM dump via MMIO index/data (radeon-style; TinyGPU has no ROM BAR map)
REG_BUS_CNTL = 0x5420
REG_ROM_CNTL = 0x1600
R600_BIOS_ROM_DIS = 1 << 1
R600_SCK_OVERWRITE = 1 << 1
REG_ROM_INDEX = 0xA8
REG_ROM_DATA = 0xAC
REG_CG_SPLL_STATUS = 0x60c
SPLL_CHG_STATUS = 1 << 1
REG_MC_SEQ_MISC0 = 0x2a00
REG_MC_SEQ_IO_DEBUG_INDEX = 0x2a44
REG_MC_SEQ_IO_DEBUG_DATA = 0x2a48
FW_DIR = pathlib.Path(__file__).resolve().parent / "fw"
DEFAULT_VBIOS = FW_DIR / "hd4850_174b_e810.rom"
# rv770d.h TLB / L2 bits (rv770_agp_enable)
ENABLE_L1_TLB = 1 << 0
ENABLE_L1_FRAGMENT_PROCESSING = 1 << 1
SYSTEM_ACCESS_MODE_NOT_IN_SYS = 3 << 3
SYSTEM_APERTURE_UNMAPPED_ACCESS_PASS_THRU = 0 << 5
EFFECTIVE_L1_TLB_SIZE = lambda x: (x) << 15
EFFECTIVE_L1_QUEUE_SIZE = lambda x: (x) << 18
ENABLE_L2_CACHE = 1 << 0
ENABLE_L2_FRAGMENT_PROCESSING = 1 << 1
ENABLE_L2_PTE_CACHE_LRU_UPDATE_BY_WRITE = 1 << 9
EFFECTIVE_L2_QUEUE_SIZE = lambda x: ((x) & 7) << 14
BANK_SELECT = lambda x: (x) << 0
CACHE_UPDATE_MODE = lambda x: (x) << 6
PKT3_MEM_WRITE = 0x3D
PKT3_ME_INITIALIZE = 0x44

def chip_from_pci_did(did: int) -> ChipInfo | None:
  for c in CHIPS.values():
    if did in c.pci_ids:
      return c
  return None

def host_pci_scan() -> list[tuple[str, int, int]]:
  if not OSX:
    return []
  try:
    out = subprocess.check_output(["ioreg", "-r", "-c", "IOPCIDevice", "-l"],
                                  text=True, errors="replace", timeout=10)
  except Exception:
    return []
  import re
  blocks = re.split(r"\+-o ", out)
  found = []
  for b in blocks:
    mname = re.search(r"^(\S+)", b)
    name = mname.group(1) if mname else "?"
    vid_m = re.search(r'"vendor-id"\s*=\s*<([0-9a-fA-F]+)>', b)
    did_m = re.search(r'"device-id"\s*=\s*<([0-9a-fA-F]+)>', b)
    if not vid_m or not did_m:
      continue
    def _le(h: str) -> int:
      bs = bytes.fromhex(h)
      return int.from_bytes(bs[:2] if len(bs) >= 2 else bs, "little")
    v, d = _le(vid_m.group(1)), _le(did_m.group(1))
    if v in (PCI_VID_AMD, 0x174C, 0x1B21):
      found.append((name, v, d))
  return found

def diagnose_host() -> str:
  amd = [(n, v, d) for n, v, d in host_pci_scan() if v == PCI_VID_AMD]
  bridges = [(n, v, d) for n, v, d in host_pci_scan() if v != PCI_VID_AMD]
  lines = []
  if amd:
    lines.append("host PCI AMD: " + ", ".join(f"{n} {v:04x}:{d:04x}" for n, v, d in amd))
  else:
    lines.append("host PCI: no AMD (1002:*) device - GPU not enumerated")
  if bridges:
    lines.append("host PCIe bridges (dock?): " +
                 ", ".join(f"{v:04x}:{d:04x}" for _, v, d in bridges[:6]))
  return "\n".join(lines)

def fetch_radeon_fw(name: str) -> bytes:
  """Fetch radeon/*.bin from linux-firmware (cached under ~/.cache/tinygrad/fw)."""
  return fetch_fw("radeon", name)

class TerrascaleDevice:
  """TinyGPU + R700 (HD 4850) CP bring-up. VRAM BAR0 dead -> AGP/sysmem rings."""

  def __init__(self, chip: ChipInfo | None = None, wait_s: float = 0.0):
    self.chip = chip
    self.pci = APLRemotePCIDevice()
    self.mmio = None
    self.vram = None
    self.agp_start = 0
    self.agp_end = 0
    self.agp_size = 0
    self.ring_size = 0x10000
    self.ring_mem = None
    self.ring_gpu = 0
    self.wb_mem = None
    self.wb_gpu = 0
    self.wptr = 0
    self._booted = False
    self.vid, self.did = self._open_config(wait_s=wait_s)
    if self.chip is None:
      self.chip = chip_from_pci_did(self.did) or CHIPS["hd4850"]
      if chip_from_pci_did(self.did) is None and self.vid == PCI_VID_AMD:
        print(f"warning: unknown AMD did={self.did:04x}; using {self.chip.name}",
              file=sys.stderr)
    self.map_mmio()
    if OSX and getenv("AMD_BOOT_MASK_INTERRUPTS", 1):
      with contextlib.suppress(Exception):
        masked = self.pci.mask_msi()
        if DEBUG:
          print(f"terrascale: masked IRQs {masked}", flush=True)

  def _read_ids(self, retries: int = 5, delay_s: float = 0.25) -> tuple[int, int]:
    vid = did = 0xFFFF
    last_err = None
    for _ in range(max(1, retries)):
      try:
        vid = self.pci.read_config(0, 2) & 0xFFFF
        did = self.pci.read_config(2, 2) & 0xFFFF
        if vid == PCI_VID_AMD or vid != 0xFFFF:
          return vid, did
      except RuntimeError as e:
        last_err = e
        if "Driver not available" in str(e):
          raise RuntimeError(
            "TinyGPU: Driver not available (dext loaded, but no GPU bound).\n"
            f"{diagnose_host()}\n"
            "Check GPU power/seating in dock."
          ) from e
      time.sleep(delay_s)
    if last_err:
      raise last_err
    return vid, did

  def _open_config(self, wait_s: float = 0.0) -> tuple[int, int]:
    deadline = time.time() + max(0.0, wait_s)
    while True:
      try:
        vid, did = self._read_ids()
        if vid == PCI_VID_AMD:
          return vid, did
        if vid == 0xFFFF and getenv("AMD_EGPU_RESTART_SERVER", 1):
          print("terrascale: pci=0xffff - restarting TinyGPU server", flush=True)
          self.pci.restart_server()
          time.sleep(1.5)
          vid, did = self._read_ids(retries=8, delay_s=0.4)
          if vid == PCI_VID_AMD:
            return vid, did
        if time.time() >= deadline:
          raise RuntimeError(f"no AMD GPU (pci={vid:04x}:{did:04x})\n{diagnose_host()}")
      except RuntimeError as e:
        if "Driver not available" in str(e) and time.time() < deadline:
          print("terrascale: waiting for GPU...", flush=True)
          time.sleep(2.0)
          continue
        raise
      time.sleep(1.0)

  def map_mmio(self):
    for bar in (2, 5, 0):
      try:
        base, size = self.pci.bar_info(bar)
        if size and size >= 0x8000:
          self.mmio = self.pci.map_bar(bar)
          self.mmio_bar, self.mmio_size = bar, size
          return bar, base, size
      except Exception:
        continue
    raise RuntimeError("no suitable MMIO BAR")

  def map_vram(self):
    """Map BAR0 only for an explicit VRAM diagnostic/access operation.

    On this eGPU the local GDDR3 path currently returns a stable floating-bus
    value and some BIF/BAR0 experiments can wedge the MC.  The normal CP/AGP
    add path has no dependency on BAR0, so keeping this lazy makes that
    separation enforceable rather than merely conventional.
    """
    if self.vram is None:
      self.vram = self.pci.map_bar(0)
    return self.vram

  def rreg(self, byte_off: int) -> int:
    if self.mmio is None:
      self.map_mmio()
    return struct.unpack("<I", bytes(self.mmio[byte_off:byte_off + 4]))[0]

  def wreg(self, byte_off: int, val: int):
    if self.mmio is None:
      self.map_mmio()
    self.mmio[byte_off:byte_off + 4] = struct.pack("<I", val & 0xFFFFFFFF)

  def probe(self) -> dict:
    bars = {}
    for i in range(6):
      with contextlib.suppress(Exception):
        bars[i] = self.pci.bar_info(i)
    info = {
      "vendor": self.vid, "device": self.did,
      "rev": self.pci.read_config(8, 1) & 0xFF,
      "chip": self.chip.name, "family": self.chip.family,
      "id_match": self.did in self.chip.pci_ids,
      "bars": bars, "host_diag": diagnose_host(),
      "grbm_status": self.rreg(REG_GRBM_STATUS),
      "cp_me_cntl": self.rreg(REG_CP_ME_CNTL),
      "config_memsize": self.rreg(REG_CONFIG_MEMSIZE),
      "mmio_bar": getattr(self, "mmio_bar", None),
    }
    return info

  # ----- AGP / sysmem (VRAM BAR0 writes are dead on this eGPU) -----
  def program_agp(self):
    """Program MC AGP aperture so host DMA addrs are GPU-reachable.

    R700 AGP_TOP/BOT are 16-bit (mc_addr >> 16). RV770 regs are rv770d.h 0x2024 family.
    AGP_BASE=0 -> host_dma = mc_addr.

    Critical: do NOT leave FB_LOCATION at 0 while AGP also covers 0 - CP ring
    fetches then hit dead FB (zeros) instead of host. Park a stub FB high and
    keep AGP on the low DMA range (rv770_mc_program non-overlap).
    """
    self.agp_start = 0
    self.agp_end = 0xDFFFFFFF  # leave 0xE0000000+ for stub FB
    self.agp_size = self.agp_end - self.agp_start + 1
    _ = self.rreg(REG_HDP_DEBUG1)
    for i in range(32):
      base = 0x2C14 + i * 0x18
      for off in (0, 4, 8, 12, 16):
        with contextlib.suppress(Exception):
          self.wreg(base + off, 0)
    # Stub FB at 0xE0000000-0xE0FFFFFF (16MB) - unused; keeps AGP/FB disjoint.
    stub_mb = int(os.environ.get("AMD_BOOT_STUB_FB_MB", "16"))
    fb_start_24 = int(os.environ.get("AMD_BOOT_STUB_FB_START", "0xE0"), 0) & 0xFF
    fb_end_24 = fb_start_24 + max(0, (stub_mb >> 4) - 1)  # 16MB blocks in >>24 units
    if stub_mb <= 16:
      fb_end_24 = fb_start_24
    # ATOM leaves CONFIG_MEMSIZE=1GB; shrink so MC does not treat 0-1GB as local FB
    # while AGP owns those addresses (device->host then returns default-page zeros).
    self.wreg(REG_CONFIG_MEMSIZE, stub_mb << 20)
    self.wreg(REG_MC_VM_FB_LOCATION, (fb_end_24 << 16) | fb_start_24)
    self.wreg(REG_MC_VM_SYSTEM_APERTURE_LOW, self.agp_start >> 12)
    self.wreg(REG_MC_VM_SYSTEM_APERTURE_HIGH, self.agp_end >> 12)
    self.wreg(REG_MC_VM_SYSTEM_APERTURE_DEFAULT, (fb_start_24 << 24) >> 12)
    self.wreg(REG_HDP_NONSURFACE_BASE, (fb_start_24 << 24) >> 8)
    self.wreg(REG_HDP_NONSURFACE_INFO, (2 << 7))
    self.wreg(REG_HDP_NONSURFACE_SIZE, 0x3FFFFFFF)
    self.wreg(REG_MC_VM_AGP_BASE, 0)
    self.wreg(REG_MC_VM_AGP_TOP, (self.agp_end >> 16) & 0xFFFF)
    self.wreg(REG_MC_VM_AGP_BOT, (self.agp_start >> 16) & 0xFFFF)
    self.agp_enable()
    # VBIOS leaves MC blacked out (CITF & 3 == 3). Without clearing, AGP/host
    # fetches return zeros and CP rptr advances on fake PACKET0s. Only clear
    # blackout - do NOT poke BIF_FB_EN until VRAM is trained (can hang MC).
    citf = self.rreg(R700_MC_CITF_CNTL)
    if citf != 0xFFFFFFFF and (citf & R600_BLACKOUT_MASK):
      self.wreg(R700_MC_CITF_CNTL, citf & ~R600_BLACKOUT_MASK)
      time.sleep(0.001)
      citf2 = self.rreg(R700_MC_CITF_CNTL)
      if DEBUG:
        print(f"terrascale: MC blackout {citf:#x} -> {citf2:#x}", flush=True)
      if citf2 == 0xFFFFFFFF:
        raise RuntimeError("MC hung after clearing blackout - power-cycle the eGPU dock")
    # ATOM enables BIF_FB_EN; leave it off for AGP-only bring-up (BAR0 poke hangs).
    if not getenv("AMD_BOOT_ENABLE_BIF", 0):
      self.wreg(R600_BIF_FB_EN, 0)
    top, bot = self.rreg(REG_MC_VM_AGP_TOP), self.rreg(REG_MC_VM_AGP_BOT)
    fb = self.rreg(REG_MC_VM_FB_LOCATION)
    memsz = self.rreg(REG_CONFIG_MEMSIZE)
    bif = self.rreg(R600_BIF_FB_EN)
    print(f"terrascale: AGP MC {self.agp_start:#x}-{self.agp_end:#x} "
          f"(TOP={top:#x} BOT={bot:#x}) FB_LOC={fb:#x} MEMSIZE={memsz:#x} BIF={bif:#x}",
          flush=True)

  def agp_enable(self):
    """rv770_agp_enable - L2 + L1 TLB pass-through, VM contexts off."""
    self.wreg(REG_VM_L2_CNTL,
              ENABLE_L2_CACHE | ENABLE_L2_FRAGMENT_PROCESSING |
              ENABLE_L2_PTE_CACHE_LRU_UPDATE_BY_WRITE | EFFECTIVE_L2_QUEUE_SIZE(7))
    self.wreg(REG_VM_L2_CNTL2, 0)
    self.wreg(REG_VM_L2_CNTL3, BANK_SELECT(0) | CACHE_UPDATE_MODE(2))
    tmp = (ENABLE_L1_TLB | ENABLE_L1_FRAGMENT_PROCESSING |
           SYSTEM_ACCESS_MODE_NOT_IN_SYS |
           SYSTEM_APERTURE_UNMAPPED_ACCESS_PASS_THRU |
           EFFECTIVE_L1_TLB_SIZE(5) | EFFECTIVE_L1_QUEUE_SIZE(5))
    for reg in (REG_MC_VM_MD_L1_TLB0_CNTL, REG_MC_VM_MD_L1_TLB1_CNTL,
                REG_MC_VM_MD_L1_TLB2_CNTL,
                REG_MC_VM_MB_L1_TLB0_CNTL, REG_MC_VM_MB_L1_TLB1_CNTL,
                REG_MC_VM_MB_L1_TLB2_CNTL, REG_MC_VM_MB_L1_TLB3_CNTL):
      self.wreg(reg, tmp)
    for i in range(7):
      self.wreg(REG_VM_CONTEXT0_CNTL + i * 4, 0)

  def agp_mc_addr(self, paddr: int) -> int:
    if paddr > self.agp_end - self.agp_start:
      raise ValueError(f"DMA addr {paddr:#x} outside AGP size {self.agp_size:#x}")
    return self.agp_start + paddr

  def alloc_agp(self, size: int) -> tuple[int, object, list[int]]:
    nbytes = round_up(size, PAGE_SIZE)
    mem, paddrs = self.pci.alloc_sysmem(nbytes, contiguous=True)
    if not paddrs:
      raise RuntimeError("alloc_sysmem returned no paddrs")
    for i in range(1, len(paddrs)):
      if paddrs[i] != paddrs[i - 1] + PAGE_SIZE:
        raise RuntimeError("need physically contiguous sysmem for AGP")
    if self.agp_size == 0:
      self.program_agp()
    gpu = self.agp_mc_addr(paddrs[0])
    sysmem_dma_flush(mem, nbytes)
    return gpu, mem, paddrs

  # ----- CP firmware + ring -----
  def cp_stop(self):
    self.wreg(REG_CP_ME_CNTL, CP_ME_HALT | CP_PFP_HALT)
    self.wreg(REG_SCRATCH_UMSK, 0)

  def load_cp_fw(self):
    """rv770_cp_load_microcode - big-endian words in RV770_{pfp,me}.bin."""
    pfp = fetch_radeon_fw("RV770_pfp.bin")
    me = fetch_radeon_fw("RV770_me.bin")
    if len(pfp) != R700_PFP_UCODE_SIZE * 4:
      raise RuntimeError(f"bad PFP fw size {len(pfp)} expect {R700_PFP_UCODE_SIZE*4}")
    if len(me) != R700_PM4_UCODE_SIZE * 4:
      raise RuntimeError(f"bad ME fw size {len(me)} expect {R700_PM4_UCODE_SIZE*4}")
    self.cp_stop()
    self.wreg(REG_CP_RB_CNTL, RB_NO_UPDATE | (15 << 8) | 3)  # BLKSZ=15 BUFSZ=3
    self.wreg(REG_GRBM_SOFT_RESET, SOFT_RESET_CP)
    _ = self.rreg(REG_GRBM_SOFT_RESET)
    time.sleep(0.015)
    self.wreg(REG_GRBM_SOFT_RESET, 0)
    # PFP - be32 in file
    self.wreg(REG_CP_PFP_UCODE_ADDR, 0)
    for i in range(R700_PFP_UCODE_SIZE):
      self.wreg(REG_CP_PFP_UCODE_DATA, struct.unpack_from(">I", pfp, i * 4)[0])
    self.wreg(REG_CP_PFP_UCODE_ADDR, 0)
    # ME
    self.wreg(REG_CP_ME_RAM_WADDR, 0)
    for i in range(R700_PM4_UCODE_SIZE):
      self.wreg(REG_CP_ME_RAM_DATA, struct.unpack_from(">I", me, i * 4)[0])
    self.wreg(REG_CP_PFP_UCODE_ADDR, 0)
    self.wreg(REG_CP_ME_RAM_WADDR, 0)
    self.wreg(REG_CP_ME_RAM_RADDR, 0)
    print(f"terrascale: loaded RV770 PFP={len(pfp)} ME={len(me)}", flush=True)

  def _ring_write_words(self, words: list[int]):
    assert self.ring_mem is not None
    off = (self.wptr * 4) % self.ring_size
    blob = b"".join(struct.pack("<I", w & 0xFFFFFFFF) for w in words)
    end = off + len(blob)
    if end <= self.ring_size:
      self.ring_mem[off:end] = blob
    else:
      first = self.ring_size - off
      self.ring_mem[off:self.ring_size] = blob[:first]
      self.ring_mem[0:len(blob) - first] = blob[first:]
    self.wptr = (self.wptr + len(words)) % (self.ring_size // 4)
    sysmem_dma_flush(self.ring_mem, self.ring_size)

  def _commit_wptr(self):
    self.wreg(REG_CP_RB_WPTR, self.wptr)
    _ = self.rreg(REG_CP_RB_RPTR)  # posting read

  def cp_start(self):
    """r600_cp_start: ME_INITIALIZE on ring, then unhalt."""
    me = build_me_initialize(self.chip.family, max_hw_contexts=8)
    # fix device id already in builder
    self._ring_write_words(me)
    self._commit_wptr()
    self.wreg(REG_CP_ME_CNTL, 0xFF)  # unhalt
    time.sleep(0.01)

  def cp_resume(self):
    """r600_cp_resume with AGP ring + writeback."""
    if self.agp_size == 0:
      self.program_agp()
    self.ring_gpu, self.ring_mem, _ = self.alloc_agp(self.ring_size)
    self.wb_gpu, self.wb_mem, _ = self.alloc_agp(PAGE_SIZE)
    # zero ring/wb
    self.ring_mem[0:self.ring_size] = bytes(self.ring_size)
    self.wb_mem[0:PAGE_SIZE] = bytes(PAGE_SIZE)
    sysmem_dma_flush(self.ring_mem, self.ring_size)
    sysmem_dma_flush(self.wb_mem, PAGE_SIZE)

    self.wreg(REG_GRBM_SOFT_RESET, SOFT_RESET_CP)
    _ = self.rreg(REG_GRBM_SOFT_RESET)
    time.sleep(0.015)
    self.wreg(REG_GRBM_SOFT_RESET, 0)

    rb_bufsz = (self.ring_size // 8).bit_length() - 1
    page_log = (PAGE_SIZE // 8).bit_length() - 1
    tmp = (page_log << 8) | rb_bufsz
    self.wreg(REG_CP_RB_CNTL, tmp)
    self.wreg(REG_CP_SEM_WAIT_TIMER, 0)
    self.wreg(REG_CP_RB_WPTR_DELAY, 0)
    self.wreg(REG_CP_RB_CNTL, tmp | RB_RPTR_WR_ENA)
    self.wreg(REG_CP_RB_RPTR_WR, 0)
    self.wptr = 0
    self.wreg(REG_CP_RB_WPTR, 0)
    # rptr writeback at RADEON_WB_CP_RPTR_OFFSET; scratch wb at offset 0
    rptr_wb = self.wb_gpu + 1024
    self.wreg(REG_CP_RB_RPTR_ADDR, lo32(rptr_wb) & 0xFFFFFFFC)
    self.wreg(REG_CP_RB_RPTR_ADDR_HI, hi32(rptr_wb) & 0xFF)
    self.wreg(REG_SCRATCH_ADDR, (self.wb_gpu >> 8) & 0xFFFFFFFF)
    self.wreg(REG_SCRATCH_UMSK, 0xFF)
    time.sleep(0.001)
    self.wreg(REG_CP_RB_CNTL, tmp)  # enable rptr update
    self.wreg(REG_CP_RB_BASE, self.ring_gpu >> 8)
    self.wreg(REG_CP_DEBUG, (1 << 27) | (1 << 28))
    print(f"terrascale: CP_RB_BASE gpu={self.ring_gpu:#x} size={self.ring_size:#x}", flush=True)
    self.cp_start()

  def ring_test(self, timeout_s: float = 2.0) -> bool:
    """r600_ring_test: SET_CONFIG_REG scratch <- 0xDEADBEEF."""
    scratch = REG_SCRATCH_REG0
    self.wreg(scratch, 0xCAFEDEAD)
    off = (scratch - PACKET3_SET_CONFIG_REG_START) >> 2
    wptr_before = self.wptr
    self._ring_write_words([
      packet3(PKT3_SET_CONFIG_REG, 1, compute=False),
      off,
      0xDEADBEEF,
    ])
    self._commit_wptr()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
      got = self.rreg(scratch)
      if got == 0xDEADBEEF:
        print(f"terrascale: ring_test PASS scratch={got:#x}", flush=True)
        return True
      time.sleep(0.001)
    rptr = self.rreg(REG_CP_RB_RPTR)
    got = self.rreg(scratch)
    print(f"terrascale: ring_test FAIL scratch={got:#x} "
          f"rptr={rptr:#x} wptr={self.wptr:#x}", flush=True)
    if rptr == self.wptr and got == 0xCAFEDEAD:
      print("terrascale: hint - CP consumed dwords but scratch unchanged; "
            "often means ring fetch saw zeros (MC blackout / AGP/FB overlap / "
            "no host DMA). Check CITF blackout and AGP vs FB_LOCATION.", flush=True)
    return False

  def dump_vbios_rom(self, path: pathlib.Path | None = None) -> bytes:
    """Read onboard ATOM ROM via REG_ROM_INDEX/DATA (exact card SSID)."""
    bus, rom = self.rreg(REG_BUS_CNTL), self.rreg(REG_ROM_CNTL)
    self.wreg(REG_BUS_CNTL, bus & ~R600_BIOS_ROM_DIS)
    self.wreg(REG_ROM_CNTL, rom | R600_SCK_OVERWRITE)
    try:
      self.wreg(REG_ROM_INDEX, 0)
      w0 = self.rreg(REG_ROM_DATA)
      hdr = struct.pack("<I", w0)
      if hdr[0] != 0x55 or hdr[1] != 0xAA:
        raise RuntimeError(f"VBIOS bad magic {hdr[:2].hex()}")
      size = max(hdr[2] * 512, 512)
      out = bytearray()
      for off in range(0, size, 4):
        self.wreg(REG_ROM_INDEX, off)
        out += struct.pack("<I", self.rreg(REG_ROM_DATA))
      bios = bytes(out[:size])
    finally:
      self.wreg(REG_BUS_CNTL, bus)
      self.wreg(REG_ROM_CNTL, rom)
    dest = path or DEFAULT_VBIOS
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(bios)
    print(f"terrascale: dumped VBIOS {len(bios)}B -> {dest}", flush=True)
    return bios

  def clear_mc_blackout(self):
    citf = self.rreg(R700_MC_CITF_CNTL)
    if citf != 0xFFFFFFFF and (citf & R600_BLACKOUT_MASK):
      self.wreg(R700_MC_CITF_CNTL, citf & ~R600_BLACKOUT_MASK)
      time.sleep(0.001)
      if self.rreg(R700_MC_CITF_CNTL) == 0xFFFFFFFF:
        raise RuntimeError("MC hung after clearing blackout - power-cycle the eGPU dock")

  def prepare_spll_refclk(self) -> dict:
    """Best-effort SPLL reference clock setup before ATOM (R700 / eGPU).

    Cold-boot HD 4850 often already has SPLL_CHG (STATUS~=0x86, CLKPIN~=0x206).
    Do NOT force BCLK_AS_XCLK in that case - poking CLKPIN can kill CHG, and
    CLKPIN=0x207 has hung MC (pci=ffff) on this eGPU. Returns diagnostics.
    """
    CG_CLKPIN_CNTL = 0x660
    CG_CLKPIN_CNTL_2 = 0x664  # SI; may be unused on RV770
    MUX_TCLK_TO_XCLK = 1 << 8
    XTALIN_DIVIDE = 1 << 9
    BCLK_AS_XCLK = 1 << 2
    FORCE_BIF_REFCLK_EN = 1 << 3
    SCLK_MUX_UPDATE = 1 << 26

    pin = self.rreg(CG_CLKPIN_CNTL)
    pin2 = self.rreg(CG_CLKPIN_CNTL_2)
    st0 = self.rreg(REG_CG_SPLL_STATUS)
    chg0 = bool(st0 & SPLL_CHG_STATUS)

    if chg0:
      # Already locked - leave CLKPIN alone.
      info = {
        "clkpin_before": pin, "clkpin_after": pin,
        "clkpin2_before": pin2, "clkpin2_after": pin2,
        "spll_status": st0, "chg": True, "poked": False,
      }
      if DEBUG:
        print(f"terrascale: SPLL already CHG STATUS={st0:#x} CLKPIN={pin:#x}",
              flush=True)
      return info

    # Prefer raw XTALIN (no /4, no TCLK mux). Keep existing BCLK_AS_XCLK if set;
    # do not invent new CLKPIN values (0x207 hung this card).
    new_pin = pin & ~(MUX_TCLK_TO_XCLK | XTALIN_DIVIDE)
    if os.environ.get("AMD_SPLL_FORCE_BCLK", "0") == "1":
      new_pin |= BCLK_AS_XCLK
    self.wreg(CG_CLKPIN_CNTL, new_pin)
    self.wreg(CG_CLKPIN_CNTL_2, pin2 | FORCE_BIF_REFCLK_EN)

    f2 = self.rreg(0x604)
    self.wreg(0x604, (f2 & ~0x1FF) | (f2 & 0x1FF) | SCLK_MUX_UPDATE)
    chg = False
    for _ in range(100):
      st = self.rreg(REG_CG_SPLL_STATUS)
      if st == 0xFFFFFFFF:
        raise RuntimeError("MC hung during SPLL probe - power-cycle eGPU")
      if st & SPLL_CHG_STATUS:
        chg = True
        break
      time.sleep(0.001)
    self.wreg(0x604, self.rreg(0x604) & ~SCLK_MUX_UPDATE)
    info = {
      "clkpin_before": pin,
      "clkpin_after": self.rreg(CG_CLKPIN_CNTL),
      "clkpin2_before": pin2,
      "clkpin2_after": self.rreg(CG_CLKPIN_CNTL_2),
      "spll_status": self.rreg(REG_CG_SPLL_STATUS),
      "chg": chg,
      "poked": True,
    }
    if DEBUG or not chg:
      print(f"terrascale: SPLL refclk probe CHG={chg} "
            f"STATUS={info['spll_status']:#x} "
            f"CLKPIN {pin:#x}->{info['clkpin_after']:#x} "
            f"CLKPIN2 {pin2:#x}->{info['clkpin2_after']:#x}", flush=True)
    return info

  def atom_asic_init(self, bios: bytes | None = None) -> None:
    """Run ATOM ASIC_Init via examples_egpu/neural.py (dword index -> byte*4).

    HD 4850 eGPU: cold boot has real SPLL_CHG (STATUS~=0x86). ASIC_Init SetEngineClock
    then reprograms SPLL and waits for CHG which never returns - use JUMP_BAIL to
    fall through (keeps cold-boot MPLL CLKF). Do NOT default-synth SPLL_CHG: that
    makes VBIOS write MPLL_AD CLKF=0 and kills BAR0. Do NOT poke BIF_FB_EN here.
    """
    egpu = pathlib.Path(__file__).resolve().parents[1] / "examples_egpu"
    if str(egpu) not in sys.path:
      sys.path.insert(0, str(egpu))
    import neural as nl  # noqa: PLC0415

    if bios is None:
      vbios_path = os.environ.get("AMD_BOOT_VBIOS_FILE", str(DEFAULT_VBIOS))
      if os.path.isfile(vbios_path):
        bios = open(vbios_path, "rb").read()
      else:
        bios = self.dump_vbios_rom(pathlib.Path(vbios_path))
    if not nl.check_atom_bios(bios):
      raise RuntimeError("ATOM BIOS header check failed")
    self.clear_mc_blackout()
    spll_info = self.prepare_spll_refclk()
    real_chg = bool(spll_info.get("chg"))
    # Synth only on explicit opt-in. Fake CHG -> VBIOS programs MPLL CLKF=0.
    synth = os.environ.get("AMD_ATOM_SYNTH_SPLL_CHG", "0") == "1"
    if not real_chg and not synth:
      raise RuntimeError(
        "SPLL_CHG_STATUS=0 - need cold power-cycle/replug for real SPLL lock, "
        "or set AMD_ATOM_SYNTH_SPLL_CHG=1 (unsafe: leaves MPLL CLKF=0)")
    if synth and not real_chg:
      print("terrascale: WARNING synth SPLL_CHG "
            "(expect MPLL CLKF=0 / dead BAR0)", flush=True)
    os.environ.setdefault("AMD_ATOM_QUIET", "1")
    os.environ.setdefault("AMD_ATOM_JUMP_MAX", "200000")
    # eGPU: SetEngineClock reprograms SPLL then waits for CHG which never
    # re-asserts. Prefer JUMP_BAIL (fall through) over synth - keeps cold-boot
    # MPLL CLKF and lets ASIC_Init continue into SetMemoryClock.
    os.environ.setdefault("AMD_ATOM_JUMP_BAIL", "1")
    os.environ.setdefault("AMD_ATOM_JUMP_TIMEOUT_SEC", "8")
    # Do not enable full op trace via DEBUG - AtomCard(debug=True) floods.
    # Bail messages still print when AMD_ATOM_TRACE=1.

    class BootAdapter:
      def __init__(self, d: "TerrascaleDevice"): self.dev = d; self._wcount = 0
      def rreg(self, reg: int) -> int:
        return self.dev.rreg((reg & 0xFFFF) * 4)
      def wreg(self, reg: int, val: int):
        self.dev.wreg((reg & 0xFFFF) * 4, val & 0xFFFFFFFF)
        self._wcount += 1
        # Some regs RAZ as 0xffffffff - only trust PCI vid for liveness.
        if (self._wcount & 0x3F) == 0:
          if self.dev.pci.read_config(0, 2) == 0xFFFF:
            raise RuntimeError("PCI vid=ffff mid-ATOM - power-cycle eGPU")
      def mmio_sync_safe(self):
        with contextlib.suppress(Exception):
          _ = self.dev.rreg(REG_CONFIG_MEMSIZE)
      def post_atom_sync(self):
        self.mmio_sync_safe()
        time.sleep(0.05)

    class R700AtomCard(nl.AtomCard):
      def __init__(self, boot, debug=False, synth_chg=False):
        super().__init__(boot, debug=debug)
        self._synth_chg = synth_chg
      def reg_read(self, reg: int) -> int:
        reg = self._mmio_reg(reg)
        val = super().reg_read(reg)
        if self._synth_chg and reg == 0x183:  # CG_SPLL_STATUS
          val |= SPLL_CHG_STATUS
        return val

    boot = BootAdapter(self)
    card = R700AtomCard(boot, debug=False, synth_chg=synth)
    ctx = nl.parse_atom_context(bios)
    exe = nl.AtomExecutor(ctx, card)
    # Default OFF: skipping SetMemoryClock leaves MC_SEQ/AGP broken on this ROM
    # (MISC0 stuck, ring fetch zeros). Full ASIC_Init + MPLL repair is required
    # for AGP smoke; VRAM still needs a safer SetMemoryClock strategy later.
    skip_mclk = os.environ.get("AMD_ATOM_SKIP_SET_MCLK", "0") == "1"
    if skip_mclk:
      import types
      _orig_locked = type(exe)._execute_locked
      def _locked_skip(self_exe, index, ps, ps_size=16):
        if index == 11:
          print("terrascale: ASIC_Init skip SetMemoryClock (AMD_ATOM_SKIP_SET_MCLK)",
                flush=True)
          return 0
        return _orig_locked(self_exe, index, ps, ps_size)
      exe._execute_locked = types.MethodType(_locked_skip, exe)  # type: ignore[method-assign]
    # Optional: patch MPLL during ASIC_Init so training sees live CLKF.
    # Default OFF. Cold CHG+PATCH: CLKF stays 73 but MISC0 becomes 0x320aa06a
    # (not 0x3000422a), STATUS_M=0, AGP ring fails; BIF then hangs MC.
    # Unpatched + post-hoc repair keeps AGP; VRAM still float until better train.
    patch_mpll = os.environ.get("AMD_ATOM_PATCH_MPLL", "0") == "1"
    patched_mpll = {"n": 0}
    if patch_mpll and not skip_mclk:
      mclk_target = 99300
      with contextlib.suppress(Exception):
        hwi0 = nl._u16(bios, ctx.data_table + nl.ATOM_DATA_FWI_PTR)
        mclk_target = nl._u32(bios, hwi0 + nl.ATOM_FWI_DEFMCLK_PTR) or 99300
      good_ad = self._calc_mpll_ad(mclk_target)
      MPLL_AD_I, MPLL_DQ_I = 0x189, 0x18B  # byte 0x624, 0x62c
      orig_wreg = boot.wreg

      def wreg_patch(reg: int, val: int):
        reg &= 0xFFFF
        val &= 0xFFFFFFFF
        if reg in (MPLL_AD_I, MPLL_DQ_I) and (val & 0x7F) == 0:
          patched_mpll["n"] += 1
          val = good_ad
        orig_wreg(reg, val)

      boot.wreg = wreg_patch  # type: ignore[method-assign]
      print(f"terrascale: ASIC_Init MPLL patch ON (target MCLK={mclk_target} "
            f"AD={good_ad:#x})", flush=True)
    # Better than PATCH_MPLL: let MemoryPLLInit write CLKF=0 (VBIOS power-up
    # window), then repair MPLL before ResetMemoryDLL/MemoryTraining/DeviceInit.
    # Default ON — Linux never does this (assumes VBIOS leaves DRAM live); on
    # eGPU post-hoc-only repair leaves training at CLKF=0.
    repair_after_mplli = (
      os.environ.get("AMD_ATOM_REPAIR_AFTER_MPLLINIT", "1") == "1"
      and not skip_mclk and not patch_mpll
    )
    repair_after_n = {"n": 0}
    if repair_after_mplli:
      import types
      _orig_locked2 = type(exe)._execute_locked
      mclk_target = 99300
      with contextlib.suppress(Exception):
        hwi0 = nl._u16(bios, ctx.data_table + nl.ATOM_DATA_FWI_PTR)
        mclk_target = nl._u32(bios, hwi0 + nl.ATOM_FWI_DEFMCLK_PTR) or 99300

      def _locked_repair_after(self_exe, index, ps, ps_size=16):
        ret = _orig_locked2(self_exe, index, ps, ps_size)
        # MemoryPLLInit = cmd 16
        if index == 16:
          ad = self.rreg(0x624)
          if (ad & 0x7F) == 0:
            repair_after_n["n"] += 1
            print("terrascale: post-MemoryPLLInit MPLL repair (CLKF was 0)",
                  flush=True)
            self.repair_mpll_boot_clock(mclk_target)
            self._wake_mrdck()
        return ret

      exe._execute_locked = types.MethodType(_locked_repair_after, exe)  # type: ignore
      print("terrascale: ASIC_Init will repair MPLL after MemoryPLLInit",
            flush=True)
    hwi = nl._u16(bios, ctx.data_table + nl.ATOM_DATA_FWI_PTR)
    ps = [0] * 16
    ps[0] = nl._u32(bios, hwi + nl.ATOM_FWI_DEFSCLK_PTR)
    ps[1] = nl._u32(bios, hwi + nl.ATOM_FWI_DEFMCLK_PTR)
    t0 = time.time()
    ret = exe.execute_table(nl.ATOM_CMD_INIT, ps, 16)
    if ret:
      raise RuntimeError(f"atom asic_init failed ret={ret}")
    mem = self.rreg(REG_CONFIG_MEMSIZE)
    misc0 = self.rreg(REG_MC_SEQ_MISC0)
    spll = self.rreg(REG_CG_SPLL_STATUS)
    mpll_ad = self.rreg(0x624)
    clkf = mpll_ad & 0x7F
    patch_note = f" patched_mpll={patched_mpll['n']}" if patch_mpll and not skip_mclk else ""
    if repair_after_mplli:
      patch_note += f" repair_after_mplli={repair_after_n['n']}"
    print(f"terrascale: ATOM done writes={boot._wcount} "
          f"MEMSIZE={mem:#x} ({mem >> 20}MB) MISC0={misc0:#x} "
          f"SPLL_STATUS={spll:#x} CHG={bool(spll & SPLL_CHG_STATUS)} "
          f"MPLL_AD={mpll_ad:#x} CLKF={clkf} skip_mclk={int(skip_mclk)} "
          f"t={time.time() - t0:.1f}s{patch_note}", flush=True)
    # With skip_mclk, cold-boot CLKF (often 50) should remain. Only repair if 0.
    if clkf == 0 or os.environ.get("AMD_ATOM_FORCE_MPLL_REPAIR", "0") == "1":
      if clkf == 0:
        print("terrascale: WARNING MPLL CLKF=0 after ATOM - repairing via calc",
              flush=True)
      self.repair_mpll_boot_clock()
      clkf = self.rreg(0x624) & 0x7F
    if clkf != 0 and os.environ.get("AMD_ATOM_WAKE_MRDCK", "1") == "1":
      self._wake_mrdck()
    elif DEBUG:
      print(f"terrascale: skip MRDCK wake (CLKF={clkf})", flush=True)
    # Optional: MemoryDeviceInit (cmd 72) — end of SetMemoryClock in full ASIC_Init
    # restores MC_SEQ_MISC0 to 0x3000422a. Standalone SetMemoryClock can leave
    # MC_SEQ wedged (MISC0 stuck/floating). Keep BIF off afterward.
    if clkf != 0 and os.environ.get("AMD_ATOM_MEMORY_DEVICE_INIT", "0") == "1":
      try:
        self.atom_run_cmd(72, "MemoryDeviceInit", 0)
        self.ensure_mpll_alive()
        self.wreg(R600_BIF_FB_EN, 0)
        print(f"terrascale: after MemoryDeviceInit MISC0={self.rreg(REG_MC_SEQ_MISC0):#x} "
              f"BIF={self.rreg(R600_BIF_FB_EN):#x}", flush=True)
      except Exception as e:
        print(f"terrascale: MemoryDeviceInit warning: {e}", flush=True)
        self.ensure_mpll_alive()
        self.wreg(R600_BIF_FB_EN, 0)
    # finish_memory (ResetMemoryDLL/…) defaults OFF — can wedge MC_SEQ on eGPU.
    if clkf != 0 and os.environ.get("AMD_ATOM_FINISH_MEM", "0") == "1":
      try:
        self.finish_memory_after_mpll()
      except Exception as e:
        print(f"terrascale: finish_memory warning: {e}", flush=True)
        self.ensure_mpll_alive()

  def repair_mpll_boot_clock(self, mclk_10khz: int = 99300) -> int:
    """Program MPLL_AD/DQ for GDDR3 boot MCLK when ATOM left CLKF=0.

    Uses Linux rv770 fractional MPLL formula with ref=27 MHz, ref_div=1,
    post_div=1 (fits CLKF in 7 bits for ~993 MHz). Does not touch BIF.
    Returns programmed CLKF.
    """
    ref = 2700  # 27 MHz in 10 kHz units
    ref_div, post_div = 1, 1
    fyclk = (mclk_10khz * 4) // 2  # GDDR3
    fb8 = (8 * fyclk * ref_div * post_div) // ref
    clkf, clkfrac = fb8 // 8, fb8 % 8
    if clkf > 0x7F:
      raise RuntimeError(f"MPLL CLKF {clkf} exceeds 7 bits - adjust post_div")
    # rv770_map_clkf_to_ibias
    if clkf <= 0x10: ibias = 0x4B
    elif clkf <= 0x19: ibias = 0x5B
    elif clkf <= 0x21: ibias = 0x2B
    elif clkf <= 0x27: ibias = 0x6C
    elif clkf <= 0x31: ibias = 0x9D
    else: ibias = 0xC6
    enc_ref = [0, 16, 17, 20, 21][ref_div - 1]
    ypost = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}[post_div]
    ad = ((clkf & 0x7F) | ((enc_ref & 0x1F) << 7) | ((clkfrac & 0x1F) << 12) |
          ((ypost & 3) << 17) | ((ibias & 0x3FF) << 20) | (1 << 31))
    # Hold RESET_EN while programming, then release
    ad2 = (self.rreg(0x628) | (1 << 25) | (1 << 24)) & ~(1 << 29)
    dq2 = (self.rreg(0x630) | (1 << 25) | (1 << 24)) & ~(1 << 29)
    self.wreg(0x628, ad2)
    self.wreg(0x630, dq2)
    self.wreg(0x624, ad)
    self.wreg(0x62c, ad)
    time.sleep(0.01)
    self.wreg(0x628, (ad2 & ~(1 << 25)) | (1 << 24))
    self.wreg(0x630, (dq2 & ~(1 << 25)) | (1 << 24))
    time.sleep(0.02)
    if self.pci.read_config(0, 2) == 0xFFFF:
      raise RuntimeError("MC hung during MPLL repair - power-cycle eGPU")
    got = self.rreg(0x624)
    print(f"terrascale: MPLL repair AD={got:#x} CLKF={got & 0x7F} "
          f"(target {clkf} for MCLK={mclk_10khz / 100:.0f}MHz)", flush=True)
    return got & 0x7F

  def _wake_mrdck(self):
    """Clear MRDCK SLEEP/RESET left set by incomplete ATOM memory bring-up.

    Does not reprogram MPLL dividers (wrong CLKF can hang the MC on eGPU).
    """
    mclk = self.rreg(0x648)
    sleep, reset = (mclk >> 8) & 0xFF, (mclk >> 16) & 0xFF
    if sleep == 0 and reset == 0:
      return
    # Keep DLL_SPEED / DLL_READY / MC_INT; clear sleep+reset; keep READY_READ if set
    base = (mclk & 0x1F) | (mclk & (1 << 6)) | (mclk & (1 << 7)) | (mclk & (1 << 24))
    self.wreg(0x648, base | (0xFF << 16))  # hold reset briefly
    time.sleep(0.01)
    self.wreg(0x648, base)
    m2 = self.rreg(0x648)
    if DEBUG:
      print(f"terrascale: MRDCK wake MCLK {mclk:#x} -> {m2:#x} "
            f"(was SLEEP={sleep:#x} RESET={reset:#x})", flush=True)

  def _atom_executor(self, bios: bytes | None = None):
    """Shared ATOM executor (neural.py) for post-ASIC_Init command tables."""
    egpu = pathlib.Path(__file__).resolve().parents[1] / "examples_egpu"
    if str(egpu) not in sys.path:
      sys.path.insert(0, str(egpu))
    import neural as nl  # noqa: PLC0415
    if bios is None:
      vbios_path = os.environ.get("AMD_BOOT_VBIOS_FILE", str(DEFAULT_VBIOS))
      bios = open(vbios_path, "rb").read() if os.path.isfile(vbios_path) else self.dump_vbios_rom()
    class BootAdapter:
      def __init__(self, d: "TerrascaleDevice"): self.dev = d; self._wcount = 0
      def rreg(self, reg: int) -> int: return self.dev.rreg((reg & 0xFFFF) * 4)
      def wreg(self, reg: int, val: int):
        self.dev.wreg((reg & 0xFFFF) * 4, val & 0xFFFFFFFF)
        self._wcount += 1
        if (self._wcount & 0x3F) == 0 and self.dev.pci.read_config(0, 2) == 0xFFFF:
          raise RuntimeError("PCI vid=ffff mid-ATOM cmd - power-cycle eGPU")
      def mmio_sync_safe(self):
        with contextlib.suppress(Exception):
          _ = self.dev.rreg(REG_CONFIG_MEMSIZE)
      def post_atom_sync(self):
        self.mmio_sync_safe(); time.sleep(0.02)
    boot = BootAdapter(self)
    card = nl.AtomCard(boot, debug=False)
    ctx = nl.parse_atom_context(bios)
    exe = nl.AtomExecutor(ctx, card)
    return nl, bios, ctx, exe, boot

  def atom_set_memory_clock(self, mclk_10khz: int = 99300, patch_mpll: bool = True) -> None:
    """radeon_atom_set_memory_clock - ATOM SetMemoryClock (cmd index 11).

    This VBIOS MemoryPLLInit writes MPLL_AD CLKF=0 (0x85b00000). When
    patch_mpll=True, intercept those writes and substitute a repaired AD/DQ
    encoding so ResetMemoryDLL / MemoryTraining run with a live MPLL.
    """
    ATOM_CMD_SET_MEMORY_CLOCK = 11
    # Precompute good AD (same as repair_mpll_boot_clock)
    good_ad = self._calc_mpll_ad(mclk_10khz)
    nl, bios, ctx, exe, boot = self._atom_executor()
    if not nl._u16(bios, ctx.cmd_table + 4 + 2 * ATOM_CMD_SET_MEMORY_CLOCK):
      raise RuntimeError("SetMemoryClock table missing in VBIOS")
    os.environ.setdefault("AMD_ATOM_JUMP_BAIL", "1")
    os.environ.setdefault("AMD_ATOM_QUIET", "1")

    if patch_mpll:
      # ATOM dword indices: byte_addr/4
      MPLL_AD_I, MPLL_DQ_I, MCLK_I = 0x189, 0x18B, 0x192  # 0x624, 0x62c, 0x648
      orig_wreg = boot.wreg
      patched = {"n": 0}

      def wreg_patch(reg: int, val: int):
        reg &= 0xFFFF
        val &= 0xFFFFFFFF
        if reg in (MPLL_AD_I, MPLL_DQ_I) and (val & 0x7F) == 0:
          patched["n"] += 1
          val = good_ad
        orig_wreg(reg, val)

      boot.wreg = wreg_patch  # type: ignore[method-assign]

    ps = [0] * 16
    ps[0] = mclk_10khz & 0xFFFFFFFF
    t0 = time.time()
    ret = exe.execute_table(ATOM_CMD_SET_MEMORY_CLOCK, ps, 16)
    self.ensure_mpll_alive(mclk_10khz)
    ad = self.rreg(0x624)
    mclk = self.rreg(0x648)
    extra = f" patched_mpll={patched['n']}" if patch_mpll else ""
    print(f"terrascale: SetMemoryClock({mclk_10khz}) ret={ret} writes={boot._wcount} "
          f"MPLL_AD={ad:#x} CLKF={ad & 0x7F} MCLK={mclk:#x} t={time.time()-t0:.1f}s{extra}",
          flush=True)

  def _calc_mpll_ad(self, mclk_10khz: int = 99300) -> int:
    """Return MPLL_AD_FUNC_CNTL encoding for GDDR3 boot MCLK (rv770 formula)."""
    ref = 2700
    ref_div, post_div = 1, 1
    fyclk = (mclk_10khz * 4) // 2
    fb8 = (8 * fyclk * ref_div * post_div) // ref
    clkf, clkfrac = fb8 // 8, fb8 % 8
    if clkf > 0x7F:
      raise RuntimeError(f"MPLL CLKF {clkf} exceeds 7 bits")
    if clkf <= 0x10: ibias = 0x4B
    elif clkf <= 0x19: ibias = 0x5B
    elif clkf <= 0x21: ibias = 0x2B
    elif clkf <= 0x27: ibias = 0x6C
    elif clkf <= 0x31: ibias = 0x9D
    else: ibias = 0xC6
    enc_ref = [0, 16, 17, 20, 21][ref_div - 1]
    ypost = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}[post_div]
    return ((clkf & 0x7F) | ((enc_ref & 0x1F) << 7) | ((clkfrac & 0x1F) << 12) |
            ((ypost & 3) << 17) | ((ibias & 0x3FF) << 20) | (1 << 31))

  def atom_dynamic_memory_settings(self, mclk_10khz: int = 99300) -> None:
    """radeon_atom_set_ac_timing - DynamicMemorySettings with COMPUTE_MEMORY_PLL_PARAM."""
    ATOM_CMD_DYNAMIC_MEMORY_SETTINGS = 63  # atombios.h master list index
    COMPUTE_MEMORY_PLL_PARAM = 1
    nl, bios, ctx, exe, boot = self._atom_executor()
    if not nl._u16(bios, ctx.cmd_table + 4 + 2 * ATOM_CMD_DYNAMIC_MEMORY_SETTINGS):
      raise RuntimeError("DynamicMemorySettings table missing")
    os.environ.setdefault("AMD_ATOM_JUMP_BAIL", "1")
    os.environ.setdefault("AMD_ATOM_QUIET", "1")
    ps = [0] * 16
    ps[0] = (mclk_10khz & 0x00FFFFFF) | (COMPUTE_MEMORY_PLL_PARAM << 24)
    ret = exe.execute_table(ATOM_CMD_DYNAMIC_MEMORY_SETTINGS, ps, 16)
    print(f"terrascale: DynamicMemorySettings ret={ret} writes={boot._wcount} "
          f"MISC0={self.rreg(REG_MC_SEQ_MISC0):#x} MCLK={self.rreg(0x648):#x}",
          flush=True)

  def dump_mc_mem_state(self, tag: str = "") -> dict:
    """Snapshot MC/MPLL regs relevant to VRAM bring-up (Linux rv770d.h)."""
    ad = self.rreg(0x624)
    mclk = self.rreg(0x648)
    info = {
      "tag": tag,
      "memsize": self.rreg(REG_CONFIG_MEMSIZE),
      "bif": self.rreg(R600_BIF_FB_EN),
      "fb_loc": self.rreg(REG_MC_VM_FB_LOCATION),
      "agp_top": self.rreg(REG_MC_VM_AGP_TOP),
      "agp_bot": self.rreg(REG_MC_VM_AGP_BOT),
      "citf": self.rreg(R700_MC_CITF_CNTL),
      "mpll_ad": ad,
      "clkf": ad & 0x7F,
      "mclk_pwrmgt": mclk,
      "dll_ready": bool(mclk & (1 << 6)),
      "mrdck_sleep": (mclk >> 8) & 0xFF,
      "mrdck_reset": (mclk >> 16) & 0xFF,
      "dll_cntl": self.rreg(0x64c),
      "misc0": self.rreg(REG_MC_SEQ_MISC0),
      "pci_alive": self.pci.read_config(0, 2) != 0xFFFF,
    }
    print(f"terrascale: mc_state{(' '+tag) if tag else ''} "
          f"MEM={info['memsize']:#x} BIF={info['bif']:#x} FB={info['fb_loc']:#x} "
          f"CLKF={info['clkf']} DLL_RDY={info['dll_ready']} "
          f"SLEEP={info['mrdck_sleep']:#x} RST={info['mrdck_reset']:#x} "
          f"MISC0={info['misc0']:#x} CITF={info['citf']:#x}", flush=True)
    return info

  def atom_run_cmd(self, index: int, name: str = "", ps0: int = 0) -> int:
    """Execute one ATOM command table by master-list index."""
    nl, bios, ctx, exe, boot = self._atom_executor()
    if not nl._u16(bios, ctx.cmd_table + 4 + 2 * index):
      raise RuntimeError(f"ATOM cmd {index} ({name or '?'}) missing")
    os.environ.setdefault("AMD_ATOM_JUMP_BAIL", "1")
    os.environ.setdefault("AMD_ATOM_QUIET", "1")
    ps = [0] * 16
    ps[0] = ps0 & 0xFFFFFFFF
    t0 = time.time()
    ret = exe.execute_table(index, ps, 16)
    if self.pci.read_config(0, 2) == 0xFFFF:
      raise RuntimeError(f"PCI hung during ATOM {name or index}")
    print(f"terrascale: ATOM {name or index} ret={ret} writes={boot._wcount} "
          f"t={time.time() - t0:.1f}s", flush=True)
    return ret

  def ensure_mpll_alive(self, mclk_10khz: int = 99300) -> None:
    """After any ATOM memory table: force CLKF!=0 and MRDCK awake."""
    if (self.rreg(0x624) & 0x7F) == 0:
      self.repair_mpll_boot_clock(mclk_10khz)
    self._wake_mrdck()
    mclk = self.rreg(0x648)
    if ((mclk >> 8) & 0xFF) or ((mclk >> 16) & 0xFF):
      # ResetMemoryDLL often leaves SLEEP+RESET; clear again.
      self._wake_mrdck()

  def finish_memory_after_mpll(self, mclk_10khz: int = 99300) -> None:
    """Replay SetMemoryClock tail AFTER a good MPLL (skip broken MemoryPLLInit).

    Nested PS values captured from cold ASIC_Init on this VBIOS (HD 4850):
      SetMemoryClock(ps0 = FIRST_TIME_CHANGE_CLOCK|mclk = 0x08000000|99300)
      ... MemoryPLLInit (SKIP — writes CLKF=0) ...
      DynamicMemorySettings / MemoryTraining / ResetMemoryDLL / MemoryDeviceInit
      with the same flag/clock packing Linux never re-issues after atom_asic_init.

    Also programs MPLL_TIME (rv770_program_mpll_timing_parameters for GDDR3).
    """
    self.ensure_mpll_alive(mclk_10khz)
    # Linux rv770 GDDR3-only MPLL lock/reset timing defaults
    R600_MPLLLOCKTIME_DFLT, R600_MPLLRESETTIME_DFLT = 100, 150
    self.wreg(0x654, (R600_MPLLLOCKTIME_DFLT & 0xFFFF) |
              ((R600_MPLLRESETTIME_DFLT & 0xFFFF) << 16))
    self.dump_mc_mem_state("pre-finish-mem")
    FIRST = 0x08000000  # FIRST_TIME_CHANGE_CLOCK
    mclk = mclk_10khz & 0x00FFFFFF
    mclk_first = FIRST | mclk
    # Exact nested order/args from /tmp/atom_nested_ps.json (cold CHG capture)
    steps: list[tuple[int, str, int, int]] = [
      # (index, name, ps0, ps1) — skip AdjustMemoryController(0,1) (write-cap loop)
      (14, "ResetMemoryDLL", 0, 1),
      (59, "MC_Synchronization", 0, 1),
      (15, "ResetMemoryDevice", 0, 1),
      (3, "VRAM_BlockVenderDetection", 0, 1),
      (18, "AdjustMemoryController", 0x01000000, 1),
      (59, "MC_Synchronization", 0x01000000, 1),
      (7, "MemoryParamAdjust", 0x01000000, 1),
      (63, "DynamicMemorySettings", 0x01000000 | mclk, 1),  # COMPUTE_MEMORY_PLL_PARAM
      (18, "AdjustMemoryController", 0x01000000 | mclk, 1),
      (63, "DynamicMemorySettings", 0x02000000 | mclk, mclk_first),
      (64, "MemoryTraining", 0x02000000 | mclk, mclk_first),
      (59, "MC_Synchronization", 0x02000000 | mclk, mclk_first),
      (14, "ResetMemoryDLL", mclk_first, mclk_first),
      (72, "MemoryDeviceInit", mclk_first, mclk_first),
    ]
    for idx, name, ps0, ps1 in steps:
      try:
        if ps1:
          nl, bios, ctx, exe, boot = self._atom_executor()
          os.environ.setdefault("AMD_ATOM_JUMP_BAIL", "1")
          os.environ.setdefault("AMD_ATOM_QUIET", "1")
          ps = [0] * 16
          ps[0], ps[1] = ps0 & 0xFFFFFFFF, ps1 & 0xFFFFFFFF
          ret = exe.execute_table(idx, ps, 16)
          print(f"terrascale: ATOM {name} ret={ret} writes={boot._wcount} "
                f"ps=[{ps0:#x},{ps1:#x}]", flush=True)
        else:
          self.atom_run_cmd(idx, name, ps0)
      except Exception as e:
        print(f"terrascale: {name} warning: {e}", flush=True)
      self.ensure_mpll_alive(mclk_10khz)
      if self.pci.read_config(0, 2) == 0xFFFF:
        raise RuntimeError(f"hung after {name}")
    self.dump_mc_mem_state("post-finish-mem")

  def probe_vram_mm(self, off: int = 0, pat: int = 0xA5A55A5A) -> bool:
    """Linux radeon_ttm_vram_read path: MM_INDEX|0x80000000 + MM_DATA.

    Accesses MC/VRAM via MMIO without relying on BAR0 mapping. Still needs
    trained DRAM; safer than BIF+BAR0 when diagnosing (no PCIe FB window).
    """
    REG_MM_INDEX, REG_MM_DATA = 0x0, 0x4
    if self.pci.read_config(0, 2) == 0xFFFF:
      return False
    try:
      self.wreg(REG_MM_INDEX, (off & 0x7FFFFFFF) | 0x80000000)
      self.wreg(REG_MM_DATA, pat & 0xFFFFFFFF)
      self.wreg(REG_HDP_DEBUG1, 0)
      time.sleep(0.02)
      if self.pci.read_config(0, 2) == 0xFFFF:
        print("terrascale: MM_INDEX VRAM probe hung MC", flush=True)
        return False
      self.wreg(REG_MM_INDEX, (off & 0x7FFFFFFF) | 0x80000000)
      got = self.rreg(REG_MM_DATA)
    except Exception as e:
      print(f"terrascale: MM_INDEX probe exception: {e}", flush=True)
      return False
    ok = got == (pat & 0xFFFFFFFF)
    print(f"terrascale: MM_INDEX VRAM off={off:#x} wrote={pat:#x} got={got:#x} ok={ok}",
          flush=True)
    return ok

  def program_mc_vram_linux(self, vram_bytes: int | None = None, enable_bif: bool = True) -> None:
    """Linux-like rv770_mc_program for PCIe: FB@0, AGP disabled, optional BIF.

    WARNING: BAR0 poke with untrained VRAM can hang MC (pci=ffff). Caller must
    only enable BIF after MPLL CLKF!=0 and MRDCK awake.
    """
    if vram_bytes is None:
      vram_bytes = int(os.environ.get("AMD_BOOT_VRAM_BYTES", str(1 << 30)), 0)
    vram_end = vram_bytes - 1
    # HDP flush quirk
    _ = self.rreg(REG_HDP_DEBUG1)
    for i in range(32):
      base = 0x2C14 + i * 0x18
      for off in (0, 4, 8, 12, 16):
        with contextlib.suppress(Exception):
          self.wreg(base + off, 0)
    # VGA aperture lockout
    with contextlib.suppress(Exception):
      self.wreg(0x328, 1 << 16)  # VGA_HDP_CONTROL VGA_MEMORY_DISABLE
    self.wreg(REG_CONFIG_MEMSIZE, vram_bytes)
    self.wreg(REG_MC_VM_SYSTEM_APERTURE_LOW, 0)
    self.wreg(REG_MC_VM_SYSTEM_APERTURE_HIGH, vram_end >> 12)
    self.wreg(REG_MC_VM_SYSTEM_APERTURE_DEFAULT, 0)
    self.wreg(REG_MC_VM_FB_LOCATION, ((vram_end >> 24) << 16) | 0)
    self.wreg(REG_HDP_NONSURFACE_BASE, 0)
    self.wreg(REG_HDP_NONSURFACE_INFO, (2 << 7))
    self.wreg(REG_HDP_NONSURFACE_SIZE, 0x3FFFFFFF)
    # PCIe discrete: disable AGP aperture (rv770_mc_program else branch)
    self.wreg(REG_MC_VM_AGP_BASE, 0)
    self.wreg(REG_MC_VM_AGP_TOP, 0x0FFFFFFF)
    self.wreg(REG_MC_VM_AGP_BOT, 0x0FFFFFFF)
    self.agp_size = 0
    # Clear blackout then allow CPU FB access (rv515_mc_resume)
    citf = self.rreg(R700_MC_CITF_CNTL)
    if citf != 0xFFFFFFFF and (citf & R600_BLACKOUT_MASK):
      self.wreg(R700_MC_CITF_CNTL, citf & ~R600_BLACKOUT_MASK)
      time.sleep(0.001)
    if enable_bif:
      self.ensure_mpll_alive()
      self.wreg(R600_BIF_FB_EN, R600_FB_READ_EN | R600_FB_WRITE_EN)
      time.sleep(0.01)
      if self.pci.read_config(0, 2) == 0xFFFF:
        raise RuntimeError("MC hung enabling BIF - power-cycle eGPU dock")
    print(f"terrascale: Linux MC FB=[0,{vram_end:#x}] MEMSIZE={vram_bytes:#x} "
          f"AGP=disabled BIF={self.rreg(R600_BIF_FB_EN):#x}", flush=True)

  def probe_bar0(self, force: bool = False) -> bool:
    """Return True if BAR0 write/readback sticks (VRAM usable from host).

    Only safe when BIF_FB_EN is on and FB_LOCATION covers BAR0. Enabling BIF
    and poking BAR0 with untrained VRAM hangs the MC - gated by AMD_BOOT_PROBE_BAR0
    unless force=True (explicit --vram-probe).
    """
    if not force and not getenv("AMD_BOOT_PROBE_BAR0", 0):
      return False
    try:
      self.map_vram()
    except Exception as e:
      print(f"terrascale: BAR0 map failed: {e}", flush=True)
      return False
    bif = self.rreg(R600_BIF_FB_EN)
    if bif == 0:
      print("terrascale: BAR0 probe skipped (BIF_FB_EN=0)", flush=True)
      return False
    if self.pci.read_config(0, 2) == 0xFFFF:
      return False
    pat = 0xA5A55A5A
    try:
      self.vram[0:4] = struct.pack("<I", pat)
      self.wreg(REG_HDP_DEBUG1, 0)
      _ = self.vram[0]
      time.sleep(0.02)
      if self.pci.read_config(0, 2) == 0xFFFF:
        print("terrascale: BAR0 probe hung MC (pci=ffff)", flush=True)
        return False
      got = struct.unpack("<I", bytes(self.vram[0:4]))[0]
    except Exception as e:
      print(f"terrascale: BAR0 probe exception: {e}", flush=True)
      return False
    ok = got == pat
    print(f"terrascale: BAR0 probe wrote={pat:#x} got={got:#x} ok={ok}", flush=True)
    return ok

  def vram_probe(self) -> bool:
    """VRAM stick test after a *good* unpatched ATOM post.

    Do NOT run SetMemoryClock here (wedges MISC0 / can hang on BIF).
    MM_INDEX with BIF=0 often reads 0 even when FB decode works — use FB@0+BIF.
    Requires MISC0==0x3000422a (unpatched cold ATOM). Optional finish_memory first.
    """
    self.dump_mc_mem_state("pre")
    probe_mclk = int(os.environ.get("AMD_BOOT_VRAM_MCLK_10KHZ", "99300"), 0)
    if probe_mclk != 99300:
      print(f"terrascale: VRAM probe override MCLK={probe_mclk * 10} kHz", flush=True)
    io_debug_before = self.read_io_debug() if getenv("AMD_BOOT_VRAM_IO_DEBUG", 0) else None
    # This is an explicit diagnostic, unlike the default AGP-only add path.
    try:
      self.map_vram()
    except Exception as e:
      print(f"terrascale: BAR0 unavailable for VRAM probe: {e}", flush=True)
    self.ensure_mpll_alive()
    misc0 = self.rreg(REG_MC_SEQ_MISC0)
    if misc0 != 0x3000422A and not getenv("AMD_BOOT_VRAM_FORCE_BIF", 0):
      print(f"terrascale: refuse BIF/VRAM probe (MISC0={misc0:#x} want 0x3000422a; "
            f"unpatched --atom first, or AMD_BOOT_VRAM_FORCE_BIF=1)", flush=True)
      return False
    if getenv("AMD_BOOT_VRAM_SET_MCLK", 0):
      print("terrascale: AMD_BOOT_VRAM_SET_MCLK ignored (unsafe on this ROM)", flush=True)
    if getenv("AMD_BOOT_VRAM_REPLAY_POWER", 0):
      # Cold ASIC_Init capture for this HD 4850: the memory rail GPIO and
      # rev2 SetVoltage payload must precede memory training.  Keep this
      # explicit because replaying GPIO blindly on a warm card is unsafe.
      try:
        self.atom_run_cmd(9, "GPIOPinControl-MVDD", 0x0101002F)
        self.atom_run_cmd(67, "SetVoltage-cold-memory", 0x04630001)
        self.ensure_mpll_alive()
      except Exception as e:
        print(f"terrascale: cold power replay warning: {e}", flush=True)
    if getenv("AMD_BOOT_VRAM_SET_VOLTAGE", 0):
      # This VBIOS uses SetVoltage revision 2 (not revision 1): the first byte
      # is the type, the second is SET_VOLTAGE mode, and the trailing u16 is a
      # millivolt level.  Require an explicit level so a guessed rail voltage
      # can never be applied to the board.
      try:
        nl, bios, ctx, exe, boot = self._atom_executor()
        mvdd_mv = int(os.environ.get("AMD_BOOT_VRAM_MVDD_MV", "0"), 0)
        if not mvdd_mv:
          raise RuntimeError("AMD_BOOT_VRAM_MVDD_MV is required for SetVoltage rev2")
        for vtype, name in ((2, "MVDDC"), (3, "MVDDQ")):
          ps = [0] * 16
          ps[0] = vtype | (2 << 8) | ((mvdd_mv & 0xFFFF) << 16)
          ret = exe.execute_table(67, ps, 16)
          print(f"terrascale: ATOM SetVoltage-{name} ret={ret} writes={boot._wcount}", flush=True)
          self.ensure_mpll_alive()
      except Exception as e:
        print(f"terrascale: SetVoltage warning: {e}", flush=True)
    if getenv("AMD_BOOT_VRAM_QUERY_VOLTAGE", 0):
      # SetVoltage rev2 type=6/mode=0 returns the board's maximum supported
      # level in the u16 field; this query does not change a rail.
      try:
        nl, bios, ctx, exe, boot = self._atom_executor()
        for name, vtype in (("MVDDC", 2), ("MVDDQ", 3)):
          ps = [0] * 16
          ps[0] = 6
          ret = exe.execute_table(67, ps, 16)
          print(f"terrascale: ATOM QueryMax-{name} ret={ret} ps0={ps[0]:#x} level_mv={(ps[0] >> 16) & 0xffff}", flush=True)
      except Exception as e:
        print(f"terrascale: QueryVoltage warning: {e}", flush=True)
    if getenv("AMD_BOOT_VRAM_FINISH_MEM", 1):
      try:
        self.finish_memory_after_mpll(probe_mclk)
        if io_debug_before is not None:
          io_debug_after = self.read_io_debug()
          changed = [(i, a, b) for i, (a, b) in enumerate(zip(io_debug_before, io_debug_after)) if a != b]
          print(f"terrascale: IO_DEBUG changed={len(changed)}", flush=True)
          for i, a, b in changed:
            print(f"  io_debug[{i:#x}] {a:#010x}->{b:#010x}", flush=True)
      except Exception as e:
        print(f"terrascale: finish_memory warning: {e}", flush=True)
        self.ensure_mpll_alive()
    self.ensure_mpll_alive()
    self.dump_mc_mem_state("pre-probe")
    mm_ok = bif_ok = False
    # Default ON: FB@0+BIF is the only path that shows real FB bus (float vs sticky).
    allow_bif = getenv("AMD_BOOT_VRAM_ENABLE_BIF", 1)
    try:
      if allow_bif:
        self.program_mc_vram_linux(enable_bif=False)
        self.ensure_mpll_alive()
        if self.rreg(REG_MC_SEQ_MISC0) != 0x3000422A and not getenv("AMD_BOOT_VRAM_FORCE_BIF", 0):
          print("terrascale: MISC0 lost before BIF - abort", flush=True)
          return False
        self.wreg(R600_BIF_FB_EN, R600_FB_READ_EN | R600_FB_WRITE_EN)
        time.sleep(0.02)
        if self.pci.read_config(0, 2) == 0xFFFF:
          raise RuntimeError("MC hung enabling BIF")
        if self.vram is not None:
          print(f"terrascale: BAR0 read[0:8]={bytes(self.vram[0:8]).hex()}", flush=True)
        if getenv("AMD_BOOT_VRAM_FAULT_MAP", 0):
          rows = self.vram_fault_map()
          bad = [r for r in rows if r["xor"] or (r["bar"] != 0xFFFFFFFF and r["bar"] != r["write"])]
          print(f"terrascale: VRAM fault-map rows={len(rows)} bad={len(bad)}", flush=True)
          zero_reads = {r["off"]: r["mm"] for r in rows if r["write"] == 0}
          print("terrascale: VRAM zero-pattern address map " + " ".join(
            f"{off:#x}->{val:#010x}" for off, val in zero_reads.items()), flush=True)
          for r in bad[:64]:
            print(f"  off={r['off']:#07x} write={r['write']:#010x} mm={r['mm']:#010x} "
                  f"bar={r['bar']:#010x} xor={r['xor']:#010x}", flush=True)
        if getenv("AMD_BOOT_VRAM_SWEEP", 0):
          mm_ok = True
          for off, pat in ((0x0, 0xA5A55A5A), (0x4, 0x5AA5A55A),
                           (0x100, 0xC33C9696), (0x1000, 0x3CC36969)):
            mm_ok = self.probe_vram_mm(off, pat) and mm_ok
        else:
          mm_ok = self.probe_vram_mm(0)
        bif_ok = self.probe_bar0(force=True)
      else:
        self.wreg(R600_BIF_FB_EN, 0)
        mm_ok = self.probe_vram_mm(0)
    except Exception as e:
      print(f"terrascale: BIF/BAR0 path failed: {e}", flush=True)
      if self.pci.read_config(0, 2) == 0xFFFF:
        raise
    finally:
      if self.pci.read_config(0, 2) != 0xFFFF:
        with contextlib.suppress(Exception):
          self.wreg(R600_BIF_FB_EN, 0)
          self.program_agp()
    ok = mm_ok or bif_ok
    print(f"terrascale: vram_probe mm={mm_ok} bar0={bif_ok}", flush=True)
    return ok

  def read_io_debug(self, count: int = 0x200) -> list[int]:
    """Read MC_SEQ_IO_DEBUG content without changing the memory state."""
    out = []
    for i in range(count):
      self.wreg(REG_MC_SEQ_IO_DEBUG_INDEX, i)
      out.append(self.rreg(REG_MC_SEQ_IO_DEBUG_DATA))
    return out

  def vram_fault_map(self) -> list[dict[str, int]]:
    """Characterize local VRAM data/address faults through MM_INDEX."""
    rows: list[dict[str, int]] = []
    offsets = [0] + [1 << n for n in range(2, 13)]
    # Keep the diagnostic bounded while still covering every DQ bit and
    # address bit; add walking-one data explicitly.
    patterns = [0, 0xFFFFFFFF, 0xAAAAAAAA, 0x55555555]
    patterns += [1 << n for n in range(32)]
    patterns += [0xFFFFFFFF ^ (1 << n) for n in range(32)]
    for off in offsets:
      for pat in patterns:
        self.wreg(0x0, (off & 0x7FFFFFFF) | 0x80000000)
        self.wreg(0x4, pat)
        self.wreg(0x2C14, 0)  # HDP_DEBUG1 flush
        self.wreg(0x0, (off & 0x7FFFFFFF) | 0x80000000)
        got_mm = self.rreg(0x4)
        got_bar = 0xFFFFFFFF
        if self.vram is not None:
          got_bar = struct.unpack("<I", bytes(self.vram[off:off + 4]))[0]
        rows.append({"off": off, "write": pat, "mm": got_mm,
                     "bar": got_bar, "xor": (pat ^ got_mm) & 0xFFFFFFFF})
    return rows

  def boot(self):
    if self._booted:
      return
    if self.chip.family != CHIP_RV770 and not self.chip.has_ls_compute:
      raise RuntimeError(f"boot path only implemented for RV770/Evergreen; got {self.chip.family}")
    print(f"terrascale: boot {self.chip.name} pci={self.vid:04x}:{self.did:04x}", flush=True)
    if getenv("AMD_BOOT_ATOM", 1):
      try:
        self.atom_asic_init()
      except Exception as e:
        print(f"terrascale: ATOM warning: {e}", flush=True)
    # AGP-first: shrink MEMSIZE, park stub FB high, clear BIF (no BAR0 poke).
    self.program_agp()
    self.probe_bar0()
    if self.chip.family == CHIP_RV770:
      self.load_cp_fw()
      self.cp_resume()
      if not self.ring_test():
        raise RuntimeError("CP ring test failed")
    else:
      # Evergreen LS path still TODO for real ALU; share CP bring-up later
      raise RuntimeError("Evergreen LS compute boot not implemented yet - use HD 4850 path")
    self._booted = True

  def run_cp_mem_write_test(self, payload=(11.0, 22.0, 33.0, 44.0)) -> list[float]:
    """Explicit CP-to-AGP payload-write diagnostic; this is not GPU arithmetic."""
    if not self._booted:
      self.boot()
    expected = [float(x) for x in payload]
    if len(expected) != 4:
      raise ValueError(f"CP MEM_WRITE test needs exactly four floats, got {len(expected)}")
    out_gpu, out_mem, _ = self.alloc_agp(0x1000)
    out_mem[0:16] = bytes(16)
    sysmem_dma_flush(out_mem, 16)
    # MEM_WRITE (r600 CS): count=3, qword-aligned addr, addr_hi = upper8 only.
    # Do NOT set bit18 - that truncates to a 32-bit write (saw [11,0,33,0]).
    words: list[int] = []
    raw = struct.pack("4f", *expected)
    for i in range(0, 16, 8):
      addr = out_gpu + i
      if addr & 7:
        raise RuntimeError(f"MEM_WRITE addr {addr:#x} not qword-aligned")
      d0, d1 = struct.unpack_from("<II", raw, i)
      words += [
        packet3(PKT3_MEM_WRITE, 3, compute=False),
        lo32(addr) & 0xFFFFFFFC,
        hi32(addr) & 0xFF,
        d0,
        d1,
      ]
    self._ring_write_words(words)
    self._commit_wptr()
    deadline = time.time() + float(os.environ.get("AMD_BOOT_ADD_WAIT_S", "2"))
    result = [0.0, 0.0, 0.0, 0.0]
    while time.time() < deadline:
      sysmem_dma_flush(out_mem, 16)
      result = list(struct.unpack("4f", bytes(out_mem[0:16])))
      if all(abs(r - e) < 1e-5 for r, e in zip(result, expected)):
        result = list(expected)
        break
      time.sleep(0.01)
    print(f"cp_mem_write_test result={result} payload={expected} "
          "(GPU CP wrote a CPU-supplied payload; no GPU ALU)", flush=True)
    if not all(abs(r - e) < 1e-4 for r, e in zip(result, expected)):
      raise RuntimeError(f"CP MEM_WRITE test failed: got {result} payload {expected}")
    return result

  def init_rv770_graphics_resources(self):
    """Seed the SQ allocator state that Linux ``rv770_gpu_init`` normally sets.

    Our CP-only boot intentionally omitted the 3D portion of the kernel init,
    leaving SQ_CONFIG at ``0xe4000000`` and the GPR/thread pools at their tiny
    reset values.  A pixel shader cannot be scheduled in that state even though
    CP packets continue to run.  These six config registers use RV770's Linux
    defaults (256 GPRs, 248 threads, 512 stack entries); they do not touch
    VRAM, BIF, clocks, or MC routing.
    """
    if self.chip.family != CHIP_RV770:
      raise NotImplementedError("graphics resource init is RV770-specific")
    # A failed pre-EOP draw can leave SH/VGT busy while CP still advances.  Do
    # the same graphics-unit soft reset Linux's r600_gpu_soft_reset performs;
    # CP is halted only while the reset bits are asserted, and MC/VRAM routing
    # is untouched.
    self.wreg(REG_CP_ME_CNTL, CP_ME_HALT | CP_PFP_HALT)
    gfx_reset = ((1 << 1) | (1 << 3) | (1 << 5) | (1 << 6) |
                 (1 << 8) | (1 << 9) | (1 << 10) | (1 << 11) |
                 (1 << 12) | (1 << 13) | (1 << 14))
    self.wreg(REG_GRBM_SOFT_RESET, gfx_reset)
    _ = self.rreg(REG_GRBM_SOFT_RESET)
    # Linux's r600_gpu_soft_reset holds the graphics reset for 50 ms.  A
    # 100-us pulse can leave RV770 SH/VGT state latched, especially after a
    # prior timed-out draw.
    time.sleep(0.050)
    self.wreg(REG_GRBM_SOFT_RESET, 0)
    # CP_ME_CNTL has only halt controls on RV770; Linux resumes it with zero.
    # Writing 0xFF sets reserved low bits and can leave the graphics parser in
    # an undefined state even though simple ring packets appear to work.
    self.wreg(REG_CP_ME_CNTL, 0)
    self.pci.drain_mmio(self.mmio_bar)
    regs = (
      (0x8C00, 0xE4000007),  # SQ_CONFIG: VC/EXPORT_SRC_C/DX9 + stage priorities
      (0x8C04, 0x00600060),  # SQ_GPR_RESOURCE_MGMT_1: PS=VS=96
      (0x8C08, 0x001C001C),  # SQ_GPR_RESOURCE_MGMT_2: ES=GS=28
      (0x8C0C, 0x1F1F3E7C),  # SQ_THREAD_RESOURCE_MGMT: PS/VS/GS/ES
      (0x8C10, 0x00800080),  # SQ_STACK_RESOURCE_MGMT_1: PS/VS=128
      (0x8C14, 0x00800080),  # SQ_STACK_RESOURCE_MGMT_2: GS/ES=128
      # rv770_gpu_init defaults required by the shader/export and scan
      # converters; these are not context registers and survive the draw
      # packet reset only if explicitly seeded.
      (0x8CF0, 0x08E00120),  # SQ_MS_FIFO_SIZES
      (0x900C, 0x001B031F),  # SX_EXPORT_BUFFER_SIZES (128/16/112 dwords)
      (0x8BCC, 0x130300F9),  # PA_SC_FIFO_SIZE (RV770)
      (0x913C, 0x00000004),  # SPI_CONFIG_CNTL_1 VTX_DONE_DELAY
      (0x88C4, 0x000000C2),  # VGT_CACHE_INVALIDATION: VC+TC, ES/GS auto
      (0x8974, 0x00000001),  # VGT_NUM_INSTANCES
      (0x88CC, 0x00000080),  # VGT_ES_PER_GS
      (0x88C8, 0x00000100),  # VGT_GS_PER_ES
      (0x88E8, 0x00000002),  # VGT_GS_PER_VS
      (0x88D4, 0x00000010),  # VGT_GS_VERTEX_REUSE
      # rv770_gpu_init: dynamic GPR ring sizes, max_gprs=256 => 152 per bank.
      (0x8DB0, 0x98989898), (0x8DB4, 0x98989898),
      (0x8DB8, 0x98989898), (0x8DBC, 0x98989898),
      (0x8DC0, 0x98989898), (0x8DC4, 0x98989898),
      (0x8DC8, 0x98989898), (0x8DCC, 0x98989898),
    )
    for reg, val in regs:
      self.wreg(reg, val)
    self.pci.drain_mmio(self.mmio_bar)
    observed = tuple(self.rreg(reg) for reg, _ in regs)
    # VGT_NUM_INSTANCES is write-only/read-as-zero on this RV770; all other
    # resource/cache registers must retain their programmed values.
    for (reg, want), got in zip(regs, observed):
      if reg != 0x8974 and got != want:
        raise RuntimeError(f"RV770 SQ resource init readback mismatch: {reg:#x} {got:#x} != {want:#x}")
    print("terrascale: RV770 SQ graphics resources initialized", flush=True)

  def prepare_gpu_add_buffers(self, a=(1.0, 2.0, 3.0, 4.0),
                               b=(10.0, 20.0, 30.0, 40.0),
                               stage: str = "add") -> dict[str, int]:
    """Allocate the real RV770 graphics-add inputs, programs, and target in AGP.

    This deliberately does *not* emit graphics packets.  It is the last safe
    preflight before a draw: all data consumed or produced by the GPU is host
    memory reached via the proven AGP aperture, and `a`/`b` are copied as inputs
    only.  No CPU sum is calculated or stored.

    The color target is filled with a 0xA5 canary so that "no write",
    "wrote zero" and "wrote the expected value" are distinguishable outcomes.
    A separate page holds the completion fence.
    """
    if not self._booted:
      self.boot()
    av, bv = tuple(map(float, a)), tuple(map(float, b))
    if len(av) != 4 or len(bv) != 4:
      raise ValueError("GPU add needs exactly four floats in each input vector")
    # The constant PS stage must still use the real fetched fullscreen triangle;
    # a no-fetch VS emits one identical position per vertex (a degenerate
    # triangle), so it cannot test raster/CB output.
    constant_vs = stage == GPU_ADD_STAGE_CONSTANT and bool(getenv("AMD_GPU_ADD_CONSTANT_VS", 0))
    empty_vs = bool(getenv("AMD_GPU_ADD_EMPTY_VS", 0))
    streamout = stage == GPU_ADD_STAGE_STREAM
    if getenv("AMD_GPU_ADD_EMPTY_PS", 0) or streamout:
      ps = build_rv770_empty_ps_blob()
    elif stage == GPU_ADD_STAGE_CONSTANT:
      ps = compile_rv770_constant_ps_blob()
    elif stage == GPU_ADD_STAGE_PARAM0:
      ps = compile_rv770_param0_ps_blob()
    else:
      ps = compile_rv770_add_blob()
    if empty_vs:
      vs = build_rv770_empty_vs_blob()
    elif streamout:
      vs = compile_rv770_stream_add_vs_blob()
    else:
      vs = compile_rv770_constant_vs_blob() if constant_vs else compile_rv770_vs_blob()
    fetch = (build_rv770_empty_ps_blob() if empty_vs else
             (build_rv770_noop_fetch_blob() if constant_vs else build_rv770_vertex_fetch_blob()))

    vs_gpu, vs_mem, _ = self.alloc_agp(PAGE_SIZE)
    ps_gpu, ps_mem, _ = self.alloc_agp(PAGE_SIZE)
    fetch_gpu, fetch_mem, _ = self.alloc_agp(PAGE_SIZE)
    vtx_gpu, vtx_mem, _ = self.alloc_agp(PAGE_SIZE)
    color_gpu, color_mem, _ = self.alloc_agp(PAGE_SIZE)
    fence_gpu, fence_mem, _ = self.alloc_agp(PAGE_SIZE)
    vs_mem[0:len(vs)], ps_mem[0:len(ps)], fetch_mem[0:len(fetch)] = vs, ps, fetch
    # One oversize triangle covers the 1x1 color target. Every vertex carries
    # identical operands, making interpolation preserve the requested vectors.
    positions = ((-1.0, -1.0, 0.0, 1.0), (3.0, -1.0, 0.0, 1.0), (-1.0, 3.0, 0.0, 1.0))
    vertices = b"".join(struct.pack("12f", *(p + av + bv)) for p in positions)
    vtx_mem[0:len(vertices)] = vertices
    # Canary fill: any byte still 0xA5 after a draw means CB never wrote.
    color_mem[0:PAGE_SIZE] = bytes([COLOR_CANARY]) * PAGE_SIZE
    fence_mem[0:PAGE_SIZE] = bytes(PAGE_SIZE)
    for mem, size in ((vs_mem, len(vs)), (ps_mem, len(ps)), (fetch_mem, len(fetch)),
                      (vtx_mem, len(vertices)), (color_mem, PAGE_SIZE), (fence_mem, PAGE_SIZE)):
      sysmem_sync_for_device(mem, size)
    out = {"vs": vs_gpu, "ps": ps_gpu, "fetch": fetch_gpu, "vertices": vtx_gpu,
           "color": color_gpu, "fence": fence_gpu,
           "vs_bytes": len(vs), "ps_bytes": len(ps), "fetch_bytes": len(fetch),
           "vertex_bytes": len(vertices)}
    # Retain mappings until completion polling has observed the GPU-written
    # color target and fence.  They are inputs/outputs only; no CPU result is stored.
    self._gpu_add_mappings = {"vs": vs_mem, "ps": ps_mem, "fetch": fetch_mem,
                              "vertices": vtx_mem, "color": color_mem, "fence": fence_mem}
    print("terrascale: GPU-add preflight (no draw) " +
          " ".join(f"{k}={v:#x}" if k in ("vs", "ps", "fetch", "vertices", "color", "fence") else f"{k}={v}"
                   for k, v in out.items()), flush=True)
    return out

  def _next_fence_seq(self) -> int:
    seq = getattr(self, "_fence_seq", 0) + 1
    self._fence_seq = seq
    return seq

  def _gpu_add_expected(self, stage: str, a, b) -> list[float] | None:
    """CPU oracle, computed only after submission and never uploaded."""
    if stage == GPU_ADD_STAGE_CONSTANT:
      return [0.25, -0.5, 3.0, 1.0]
    if stage == GPU_ADD_STAGE_PARAM0:
      return [float(x) for x in a]
    if stage == GPU_ADD_STAGE_ADD:
      return [float(x) + float(y) for x, y in zip(a, b)]
    if stage == GPU_ADD_STAGE_STREAM:
      return [float(x) + float(y) for x, y in zip(a, b)]
    return None  # cp stage

  def dump_gpu_add_registers(self, tag: str = "") -> dict:
    """Snapshot graphics-relevant status/context registers (Phase 9.1)."""
    regs = {
      "CP_RB_RPTR": REG_CP_RB_RPTR, "CP_RB_WPTR": REG_CP_RB_WPTR,
      "CP_ME_CNTL": REG_CP_ME_CNTL, "GRBM_STATUS": REG_GRBM_STATUS,
      "CB_COLOR0_BASE": REG_CB_COLOR0_BASE, "CB_COLOR0_INFO": REG_CB_COLOR0_INFO,
      "CB_COLOR0_SIZE": REG_CB_COLOR0_SIZE, "CB_TARGET_MASK": REG_CB_TARGET_MASK,
      "CB_SHADER_MASK": REG_CB_SHADER_MASK, "CB_COLOR_CONTROL": REG_CB_COLOR_CONTROL,
      "SPI_PS_IN_CONTROL_0": REG_SPI_PS_IN_CONTROL_0,
      "SQ_CONFIG": 0x8C00, "SQ_THREAD_RESOURCE_MGMT": 0x8C0C,
      "DB_DEPTH_CONTROL": REG_DB_DEPTH_CONTROL,
      "PA_SC_MODE_CNTL": REG_PA_SC_MODE_CNTL,
    }
    info = {k: self.rreg(v) for k, v in regs.items()}
    print("terrascale: gfx regs" + (f" {tag}" if tag else "") + " " +
          " ".join(f"{k}={v:#x}" for k, v in info.items()), flush=True)
    return info

  def run_add(self, a=(1.0, 2.0, 3.0, 4.0), b=(10.0, 20.0, 30.0, 40.0),
              stage: str = "add") -> list[float] | None:
    """Run a real GPU vector add, never a CPU fallback.

    RV770 has the classic graphics CP but not Evergreen's LS compute pipeline.
    A valid implementation needs an RV770 CF+ALU shader, GFX resource bindings,
    a draw/dispatch packet sequence, and a GPU-produced AGP result.  Do not
    substitute PKT3_MEM_WRITE: it merely writes literal packet data.

    The `stage` ladder isolates pipeline stages: ``cp`` proves only the
    completion path, ``constant`` proves draw/raster/PS/CB, ``param0`` adds the
    fetch/VS/SPI linkage, and ``add`` is the full GPU arithmetic.
    """
    if self.chip.family != CHIP_RV770:
      raise NotImplementedError("real add currently targets the attached RV770 / HD 4850")
    if stage not in GPU_ADD_STAGES:
      raise ValueError(f"unknown GPU-add stage {stage!r}; choose {GPU_ADD_STAGES}")
    bufs = self.prepare_gpu_add_buffers(a, b, stage=stage)
    self.init_rv770_graphics_resources()
    if getenv("AMD_GPU_ADD_ENABLE_BIF", 0):
      # Diagnostic: enable the framebuffer read/write gates while retaining
      # the FB range at the parked 0xE0... address.  This does not map or touch
      # BAR0; it tests whether CB color writes require BIF even for AGP targets.
      self.wreg(R600_BIF_FB_EN, R600_FB_READ_EN | R600_FB_WRITE_EN)
      self.pci.drain_mmio(self.mmio_bar)
      print(f"terrascale: GPU-add BIF enabled={self.rreg(R600_BIF_FB_EN):#x}", flush=True)
    fence_mode = os.environ.get("AMD_GPU_ADD_FENCE_MODE", "eop")
    allow_fence_memwrite = fence_mode in ("wait-memwrite", "raw-memwrite")
    full_gfx_init = getenv("AMD_GPU_ADD_FULL_GFX_INIT", 0)
    fence_sequence = self._next_fence_seq()
    words = build_rv770_add_draw(bufs["vs"], bufs["ps"], bufs["fetch"],
                                 bufs["vertices"], bufs["color"],
                                 stage=stage, fence_gpu=bufs["fence"],
                                 fence_sequence=fence_sequence, fence_mode=fence_mode,
                                 full_gfx_init=full_gfx_init,
                                 constant_vs=bool(getenv("AMD_GPU_ADD_CONSTANT_VS", 0)),
                                 empty_vs=bool(getenv("AMD_GPU_ADD_EMPTY_VS", 0)))
    validate_gpu_add_pm4(words, color_gpu=bufs["color"], fence_gpu=bufs["fence"],
                         stage=stage, allow_fence_memwrite=allow_fence_memwrite)
    color_mem = self._gpu_add_mappings["color"]
    fence_mem = self._gpu_add_mappings["fence"]
    if getenv("AMD_GPU_ADD_DUMP_PM4", 0):
      print("terrascale: RV770 draw PM4:", flush=True)
      for line in decode_rv770_pm4(words):
        print("  pm4 " + line, flush=True)
    self._ring_write_words(words)
    self._commit_wptr()
    if getenv("AMD_GPU_ADD_DUMP_REGISTERS", 0):
      self.dump_gpu_add_registers("after-submit")
    # Poll the completion fence, not the color target.  This distinguishes a
    # hang (fence never updates) from a draw that completed but wrote nothing.
    deadline = time.time() + float(os.environ.get("AMD_BOOT_ADD_WAIT_S", "3"))
    fenced = False
    while time.time() < deadline:
      sysmem_sync_for_cpu(fence_mem, 16)
      if struct.unpack("<I", bytes(fence_mem[0:4]))[0] == fence_sequence:
        fenced = True
        break
      time.sleep(0.005)
    # Optional diagnostic delay for raw-fence experiments: raw MEM_WRITE is
    # intentionally not a graphics completion primitive, so allow the draw
    # engine time to retire before inspecting the target.
    post_delay = float(os.environ.get("AMD_GPU_ADD_POST_DELAY_S", "0"))
    if post_delay > 0:
      time.sleep(post_delay)
    sysmem_sync_for_cpu(color_mem, 16)
    result = list(struct.unpack("4f", bytes(color_mem[0:16])))
    if stage == GPU_ADD_STAGE_CP:
      if not fenced:
        raise RuntimeError("RV770 GPU-add cp stage: completion fence never signaled")
      print(f"result=cp_stage fence={fence_sequence:#x} path=rv770_fence_only", flush=True)
      return None
    # Classify the color page for actionable diagnostics.
    raw = bytes(color_mem[0:16])
    canary_intact = (raw == bytes([COLOR_CANARY]) * 16)
    if not fenced:
      rptr = self.rreg(REG_CP_RB_RPTR); wptr = self.rreg(REG_CP_RB_WPTR)
      grbm = self.rreg(REG_GRBM_STATUS)
      raise RuntimeError(
        f"RV770 GPU add ({stage}) fence timeout: fence={fenced} "
        f"color_canary_intact={canary_intact} got={result}; "
        f"CP_RPTR={rptr:#x} CP_WPTR={wptr:#x} GRBM_STATUS={grbm:#x}")
    expected = self._gpu_add_expected(stage, a, b)
    if canary_intact:
      grbm = self.rreg(REG_GRBM_STATUS)
      raise RuntimeError(
        f"RV770 GPU add ({stage}) draw completed but color target unchanged "
        f"(canary intact, got={result}, GRBM_STATUS={grbm:#x}); "
        f"CB/ROP/rasterizer produced no write")
    if not all(math.isclose(got, want, rel_tol=1e-5, abs_tol=1e-5)
               for got, want in zip(result, expected)):
      raise RuntimeError(
        f"RV770 GPU add ({stage}) mismatch: got {result}, expected {expected} "
        f"(canary_intact={canary_intact})")
    print(f"result={result} expected={expected} stage={stage} "
          f"path=rv770_vs_ps_alu_agp", flush=True)
    return result

def selftest(chip: ChipInfo):
  assert chip.pci_ids
  # Keep normal boot provably BAR0-free: host AGP is the supported memory path
  # until a VRAM write/readback survives the explicit --vram-probe.
  assert "map_bar" not in TerrascaleDevice.__init__.__code__.co_names
  assert "map_bar" in TerrascaleDevice.map_vram.__code__.co_names
  shader = build_shader_stub_evergreen_add()
  assert len(shader) == 256
  me = build_me_initialize(CHIP_RV770)
  assert me[4] == (1 << 16)
  ib = PM4Builder().build_dispatch_ib(0x10000, 0x20000, 0x30000, 0x40000)
  assert ib[0] >> 30 == PKT_TYPE3
  rv770_asm = compile_rv770_add_shader() if r600_llc() else ""
  rv770_blob = compile_rv770_add_blob() if rv770_asm else b""
  rv770_vs_blob = compile_rv770_vs_blob() if rv770_asm else b""
  rv770_fetch_blob = build_rv770_vertex_fetch_blob()
  rv770_draw = build_rv770_add_draw(0x20000, 0x21000, 0x22000, 0x23000, 0x24000,
                                    fence_gpu=0x25000)
  assert len(rv770_blob) in (0, 64)
  assert len(rv770_vs_blob) in (0, 48)
  assert len(rv770_fetch_blob) == 80
  # Independent checks of Mesa's R700 layout: clause target (dword 8 / 2),
  # VFETCH resource 160, and GPR destinations 1/2/3.
  fetch_dw = struct.unpack("<20I", rv770_fetch_blob)
  assert fetch_dw[0] == 4 and (fetch_dw[1] & 0x7F800000) == (2 << 23)
  assert [(fetch_dw[i] >> 8) & 0xFF for i in (8, 12, 16)] == [160, 160, 160]
  assert [fetch_dw[i] & 0x7F for i in (9, 13, 17)] == [1, 2, 3]
  assert any(((w >> 8) & 0xFF) == PKT3_DRAW_INDEX_AUTO for w in rv770_draw)
  # A 0xffffffff body dword (e.g. SURFACE_SYNC flush mask) reads as a type3
  # header under a naive top-bit test; exclude it from the compute-mode check.
  assert not any(w & PACKET3_COMPUTE_MODE for w in rv770_draw
                 if w >> 30 == PKT_TYPE3 and w != 0xFFFFFFFF)
  # CB_COLOR_CONTROL / color-info helpers and their decoded fields.
  assert rv770_cb_color_control() == 0x00CC0000
  assert ((rv770_cb_color_control() >> 4) & 7) == 0
  assert ((rv770_cb_color_control() >> 16) & 0xFF) == 0xCC
  cinfo = rv770_color_info_rgba32_float()
  assert ((cinfo >> 2) & 0x3F) == 0x23
  assert ((cinfo >> 8) & 0xF) == 1
  assert ((cinfo >> 12) & 0xF) == 7
  assert ((cinfo >> 24) & 1) == 1
  # Stage ladder: each graphics stage emits one draw + exactly one completion
  # fence and never a MEM_WRITE to the color target.
  for st in (GPU_ADD_STAGE_ADD, GPU_ADD_STAGE_CONSTANT, GPU_ADD_STAGE_PARAM0):
    w = build_rv770_add_draw(0x20000, 0x21000, 0x22000, 0x23000, 0x24000,
                             stage=st, fence_gpu=0x25000, fence_sequence=7)
    validate_gpu_add_pm4(w, color_gpu=0x24000, fence_gpu=0x25000, stage=st)
  # The cp stage proves only the completion fence: no draw, one EOP.
  w_cp = build_rv770_add_draw(0x20000, 0x21000, 0x22000, 0x23000, 0x24000,
                              stage=GPU_ADD_STAGE_CP, fence_gpu=0x25000, fence_sequence=9)
  assert not any(((x >> 8) & 0xFF) == PKT3_DRAW_INDEX_AUTO for x in w_cp)
  validate_gpu_add_pm4(w_cp, color_gpu=0x24000, fence_gpu=0x25000, stage=GPU_ADD_STAGE_CP)
  # Reject a CP MEM_WRITE to the color target even in memwrite mode.  Build a
  # valid draw, then append a forbidden MEM_WRITE to the color allocation.
  bad = build_rv770_add_draw(0x20000, 0x21000, 0x22000, 0x23000, 0x24000,
                             fence_gpu=0x25000, fence_sequence=1)
  _pb = PM4Builder(compute=False)
  _pb.pkt3(PKT3_MEM_WRITE, 0x24000, 0, 0, 0)
  bad += _pb.words
  raised = False
  try:
    validate_gpu_add_pm4(bad, color_gpu=0x24000, fence_gpu=0x25000, stage=GPU_ADD_STAGE_ADD,
                        allow_fence_memwrite=True)
  except AssertionError as e:
    raised = "color target" in str(e)
  assert raised, "validator must reject a MEM_WRITE to the color target"
  # The sanctioned memwrite-mode fence (MEM_WRITE to the fence only) is accepted.
  mw = build_rv770_add_draw(0x20000, 0x21000, 0x22000, 0x23000, 0x24000,
                            stage=GPU_ADD_STAGE_ADD, fence_gpu=0x25000,
                            fence_sequence=1, fence_mode="wait-memwrite")
  validate_gpu_add_pm4(mw, color_gpu=0x24000, fence_gpu=0x25000, stage=GPU_ADD_STAGE_ADD,
                       allow_fence_memwrite=True)
  # Constant / param0 PS compile if the R600 backend is available.
  if rv770_asm:
    cps = compile_rv770_constant_ps_blob()
    p0 = compile_rv770_param0_ps_blob()
    assert len(cps) > 0 and len(p0) > 0
    # constant PS exports a distinctive vec4, param0 exports its input.
    assert b"EXPORT" in compile_rv770_add_shader().encode()
  print(f"selftest=ok chip={chip.name} me_words={len(me)} eg_ib={len(ib)} "
        f"ls_compute={int(chip.has_ls_compute)} rv770_alu_add={int(bool(rv770_asm))} "
        f"rv770_shader_bytes={len(rv770_blob)} rv770_vs_bytes={len(rv770_vs_blob)} "
        f"rv770_fetch_bytes={len(rv770_fetch_blob)} rv770_draw_dw={len(rv770_draw)}")

def dry_run(chip: ChipInfo):
  print(f"chip={chip.name} family={chip.family} terrascale={chip.terrascale}")
  print(f"note={chip.note}")
  if chip.family == CHIP_RV770:
    for reg, val in build_cp_resume_regs(0xAB0000, 0x10000, 0xAC0000):
      print(f"  WREG32({reg:#06x}, {val:#010x})")
    print("  ME_INITIALIZE:", " ".join(f"{w:08x}" for w in build_me_initialize(chip.family)))
    if "--gpu-add-dry-run" in sys.argv:
      print("  RV770 graphics add PM4:")
      for i, w in enumerate(build_rv770_add_draw(0x20000, 0x21000, 0x22000, 0x23000, 0x24000)):
        print(f"  {i:04d}: {w:08x}")
  else:
    ib = PM4Builder().build_dispatch_ib(0xAB0000, 0x1000, 0x2000, 0x3000)
    for i, w in enumerate(ib):
      print(f"  {i:04d}: {w:08x}")

def probe(chip: ChipInfo | None, wait_s: float = 0.0):
  print(diagnose_host(), flush=True)
  try:
    dev = TerrascaleDevice(chip=chip, wait_s=wait_s)
    info = dev.probe()
  except Exception as e:
    print(f"probe failed: {e}", file=sys.stderr)
    sys.exit(1)
  print(f"pci={info['vendor']:04x}:{info['device']:04x} rev={info['rev']:#04x} "
        f"chip={info['chip']} family={info['family']} id_match={info['id_match']}")
  print(f"GRBM_STATUS={info['grbm_status']:#x} CP_ME_CNTL={info['cp_me_cntl']:#x} "
        f"CONFIG_MEMSIZE={info['config_memsize']:#x}")
  print(f"bars={{ {', '.join(f'{k}:({hex(v[0])},{hex(v[1])})' for k,v in info.get('bars',{}).items())} }}")

def parse_wait(argv: list[str]) -> float:
  for i, arg in enumerate(argv):
    if arg.startswith("--wait="):
      return float(arg.split("=", 1)[1])
    if arg == "--wait" and i + 1 < len(argv):
      return float(argv[i + 1])
  return float(os.environ.get("TS_WAIT_S", "0"))

def parse_vec4(s: str) -> tuple[float, float, float, float]:
  parts = [float(x) for x in s.replace(" ", "").split(",") if x]
  if len(parts) != 4:
    raise SystemExit(f"need 4 floats, got {parts!r}")
  return (parts[0], parts[1], parts[2], parts[3])

def parse_cases(argv: list[str]) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
  if "--test" in argv:
    return [
      ((1.0, 2.0, 3.0, 4.0), (10.0, 20.0, 30.0, 40.0)),
      ((0.0, 0.0, 0.0, 0.0), (5.0, 5.0, 5.0, 5.0)),
      ((-1.0, 2.0, -3.0, 4.0), (2.0, -2.0, 2.0, -2.0)),
    ]
  a, b = (1.0, 2.0, 3.0, 4.0), (10.0, 20.0, 30.0, 40.0)
  for i, arg in enumerate(argv):
    if arg == "--a" and i + 1 < len(argv):
      a = parse_vec4(argv[i + 1])
    elif arg.startswith("--a="):
      a = parse_vec4(arg.split("=", 1)[1])
    elif arg == "--b" and i + 1 < len(argv):
      b = parse_vec4(argv[i + 1])
    elif arg.startswith("--b="):
      b = parse_vec4(arg.split("=", 1)[1])
  return [(a, b)]

def parse_payloads(argv: list[str]) -> list[tuple[float, float, float, float]]:
  if "--test" in argv:
    return [
      (11.0, 22.0, 33.0, 44.0),
      (5.0, 5.0, 5.0, 5.0),
      (1.0, -2.0, 3.0, -4.0),
    ]
  payload = (11.0, 22.0, 33.0, 44.0)
  for i, arg in enumerate(argv):
    if arg == "--payload" and i + 1 < len(argv):
      payload = parse_vec4(argv[i + 1])
    elif arg.startswith("--payload="):
      payload = parse_vec4(arg.split("=", 1)[1])
  return [payload]

def parse_gpu_add_cli(argv: list[str]) -> tuple[str, int]:
  """Parse the GPU-add experimental switches into (stage, repeat).

  Flags are also mirrored into ``AMD_GPU_ADD_*`` env vars so ``run_add`` (which
  already reads them) behaves identically whether set via CLI or environment.
  """
  stage = os.environ.get("AMD_GPU_ADD_STAGE", "add")
  repeat = int(os.environ.get("AMD_GPU_ADD_REPEAT", "1"))
  for i, arg in enumerate(argv):
    if arg == "--gpu-add-stage" and i + 1 < len(argv):
      stage = argv[i + 1]
    elif arg.startswith("--gpu-add-stage="):
      stage = arg.split("=", 1)[1]
    elif arg == "--gpu-add-fence-mode" and i + 1 < len(argv):
      os.environ["AMD_GPU_ADD_FENCE_MODE"] = argv[i + 1]
    elif arg.startswith("--gpu-add-fence-mode="):
      os.environ["AMD_GPU_ADD_FENCE_MODE"] = arg.split("=", 1)[1]
    elif arg == "--gpu-add-repeat" and i + 1 < len(argv):
      repeat = int(argv[i + 1])
    elif arg.startswith("--gpu-add-repeat="):
      repeat = int(arg.split("=", 1)[1])
    elif arg == "--gpu-add-full-gfx-init":
      os.environ["AMD_GPU_ADD_FULL_GFX_INIT"] = "1"
    elif arg == "--gpu-add-no-full-gfx-init":
      os.environ["AMD_GPU_ADD_FULL_GFX_INIT"] = "0"
    elif arg == "--gpu-add-dump-pm4":
      os.environ["AMD_GPU_ADD_DUMP_PM4"] = "1"
    elif arg == "--gpu-add-dump-registers":
      os.environ["AMD_GPU_ADD_DUMP_REGISTERS"] = "1"
  if stage not in GPU_ADD_STAGES:
    raise SystemExit(f"--gpu-add-stage must be one of {GPU_ADD_STAGES}")
  return stage, repeat

def main():
  argv = sys.argv[1:]
  if any(a in ("--chip=auto", "--auto") for a in argv) or os.environ.get("TS_CHIP", "").lower() == "auto":
    chip: ChipInfo | None = None
  else:
    # default hd4850 when connected; still allow --chip=
    if not any(a.startswith("--chip") for a in argv) and not os.environ.get("TS_CHIP"):
      os.environ.setdefault("TS_CHIP", "hd4850")
    chip = resolve_chip(argv)
  wait_s = parse_wait(argv)
  stage, repeat = parse_gpu_add_cli(argv)

  if "--selftest" in argv:
    selftest(chip or CHIPS["hd4850"]); return
  if "--dry-run" in argv:
    dry_run(chip or CHIPS["hd4850"]); return
  if "--compile-rv770-add" in argv:
    print(compile_rv770_add_shader(), end="")
    return
  if "--host-pci" in argv:
    print(diagnose_host())
    for n, v, d in host_pci_scan():
      print(f"  {n} {v:04x}:{d:04x}")
    return
  if "--probe" in argv:
    probe(chip, wait_s=wait_s); return
  if "--dump-rom" in argv:
    dev = TerrascaleDevice(chip=chip, wait_s=wait_s)
    out = DEFAULT_VBIOS
    for i, a in enumerate(argv):
      if a.startswith("--out="):
        out = pathlib.Path(a.split("=", 1)[1])
      elif a == "--out" and i + 1 < len(argv):
        out = pathlib.Path(argv[i + 1])
    bios = dev.dump_vbios_rom(out)
    print(f"ssid@0x7c={bios[0x7c:0x80].hex()} ATOM={b'ATOM' in bios}")
    return
  if "--atom" in argv:
    dev = TerrascaleDevice(chip=chip, wait_s=wait_s)
    dev.atom_asic_init()
    ad = dev.rreg(0x624)
    print(f"BAR0_ok={dev.probe_bar0()} BIF={dev.rreg(R600_BIF_FB_EN):#x} "
          f"FB={dev.rreg(REG_MC_VM_FB_LOCATION):#x} "
          f"SPLL={dev.rreg(REG_CG_SPLL_STATUS):#x} "
          f"MPLL_AD={ad:#x} CLKF={ad & 0x7F} MCLK={dev.rreg(0x648):#x}")
    return
  if "--clock-probe" in argv:
    dev = TerrascaleDevice(chip=chip, wait_s=wait_s)
    st = dev.rreg(REG_CG_SPLL_STATUS)
    ad = dev.rreg(0x624)
    print(f"pre: SPLL={st:#x} CHG={bool(st & SPLL_CHG_STATUS)} "
          f"CLKPIN={dev.rreg(0x660):#x} MPLL_AD={ad:#x} CLKF={ad & 0x7F} "
          f"MCLK={dev.rreg(0x648):#x} MEM={dev.rreg(REG_CONFIG_MEMSIZE):#x}")
    info = dev.prepare_spll_refclk()
    print("clock-probe", info)
    ad = dev.rreg(0x624)
    print(f"MCLK={dev.rreg(0x648):#x} MPLL_AD={ad:#x} CLKF={ad & 0x7F} "
          f"GENERAL={dev.rreg(0x63c):#x} GRBM={dev.rreg(REG_GRBM_STATUS):#x}")
    if not info.get("chg"):
      print("HINT: SPLL not locked - physical replug for cold-boot CHG~=0x86, "
            "then --atom (no synth). Avoid AMD_ATOM_SYNTH_SPLL_CHG.")
    return
  if "--list-chips" in argv:
    for k, c in CHIPS.items():
      print(f"{k}: {c.name} family={c.family} ts={c.terrascale} "
            f"ls_compute={c.has_ls_compute} ids={[f'{x:04x}' for x in c.pci_ids]}")
    return
  if "--ring-test" in argv:
    dev = TerrascaleDevice(chip=chip, wait_s=wait_s)
    dev.boot()
    return
  if "--vram-probe" in argv:
    # After power-cycle: MPLL repair -> SetMemoryClock tail -> MM_INDEX -> BAR0.
    # Can still hang if MRDCK left asleep - code wakes before BIF.
    dev = TerrascaleDevice(chip=chip, wait_s=wait_s)
    if getenv("AMD_BOOT_ATOM", 1):
      with contextlib.suppress(Exception):
        # Prefer existing post-ATOM clocks; full ATOM needs cold CHG.
        # A cold card can report a nonzero reset CLKF while the rest of
        # ASIC_Init is still unrun (MEMSIZE=0, MISC0=0).  CHG is the reliable
        # indicator that the full VBIOS memory sequence must execute.
        cold_chg = bool(dev.rreg(REG_CG_SPLL_STATUS) & SPLL_CHG_STATUS)
        if cold_chg or (dev.rreg(0x624) & 0x7F) == 0:
          try:
            dev.atom_asic_init()
          except Exception as e:
            print(f"terrascale: ATOM skipped/failed: {e}", flush=True)
            dev.ensure_mpll_alive()
    ok = dev.vram_probe()
    print(f"vram_probe={'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)

  if "--cp-mem-write-test" in argv:
    dev = TerrascaleDevice(chip=chip, wait_s=wait_s)
    print(f"pci={dev.vid:04x}:{dev.did:04x} chip={dev.chip.name}", flush=True)
    for payload in parse_payloads(argv):
      dev.run_cp_mem_write_test(payload)
    return
  if "--gpu-add-preflight" in argv:
    dev = TerrascaleDevice(chip=chip, wait_s=wait_s)
    a, b = parse_cases(argv)[0]
    dev.prepare_gpu_add_buffers(a, b, stage=stage)
    return

  if "--test" in argv:
    raise SystemExit("--test is only valid with --cp-mem-write-test; default add never CPU-offloads")

  # Default: true GPU vector-add only. No CP-MEM_WRITE/CPU fallback is allowed.
  cases = parse_cases(argv)
  try:
    dev = TerrascaleDevice(chip=chip, wait_s=wait_s)
    print(f"pci={dev.vid:04x}:{dev.did:04x} chip={dev.chip.name} stage={stage} "
          f"repeat={repeat}", flush=True)
    for a, b in cases:
      for _ in range(max(1, repeat)):
        dev.run_add(a, b, stage=stage)
  except Exception as e:
    print(f"add failed: {e}", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
  main()
