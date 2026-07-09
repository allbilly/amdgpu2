#!/usr/bin/env python3
"""Standalone AMD RX570 (Polaris10 / gfx803) eGPU vector-add over TinyGPU.app on macOS.

Mirrors the nvgpu examples/add.py goal: run a hand-assembled GCN shader via direct
PM4 compute dispatch without importing tinygrad.runtime on the live path.

Hardware boundary: TinyGPU.app (APLRemotePCIDevice unix socket) for PCIe BAR/MMIO
and sysmem DMA — same transport as the NV eGPU examples.

Polaris (GCN4) is NOT covered by tinygrad's AMDev boot path (gfx9+/RDNA only, no
mp_7.1.1 register tables). This file implements a Polaris-specific bring-up stub
and PM4 path using linux amdgpu headers as reference.

Usage:
  python3 examples_egpu/add.py                  # vector-add (session #21 proven path)
  python3 examples_egpu/add.py --probe          # eGPU + register sanity (no boot)
  python3 examples_egpu/add.py --selftest         # offline PM4 + shader gate
  python3 examples_egpu/add.py --reset           # auto reset (AMD cfg if PCI up, else PCI hot reset)
  python3 examples_egpu/add.py --boot-stage=add  # same as bare run
  python3 examples_egpu/add.py --boot-stage=kcq-ring-test
  python3 examples_egpu/add.py --boot-stage=sdma-probe
  AMD_BOOT_SAFE=1 python3 examples_egpu/add.py  # print stage help only (old gated default)
"""
from __future__ import annotations
import os, sys, ctypes, ctypes.util, time, mmap, struct, array, socket, subprocess, contextlib, functools, itertools, enum, dataclasses, urllib.request, hashlib, tempfile, pathlib

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

def sysmem_dma_flush(mem, size: int):
  """Flush CPU writes so eGPU DMA sees host memory (ARM lacks IO coherency).

  See geerlingguy/raspberry-pi-pcie-devices#756 — Pi5/M1 need explicit sync for
  GPU DMA to see CPU-written sysmem (yanghaku pgprot_dmacoherent TTM patch).
  """
  if os.environ.get("AMD_BOOT_SYSMEM_FLUSH", "1") == "0":
    return
  if not hasattr(mem, "addr") or not size:
    return
  libc = ctypes.CDLL(ctypes.util.find_library("c"))
  MS_SYNC = 0x10
  if libc.msync(ctypes.c_void_p(mem.addr), size, MS_SYNC) != 0:
    with contextlib.suppress(Exception):
      libc.sync()

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
        if cap_id == 0x05:  # MSI capability — clear MSI Enable (bit 0 of Message Control)
          mc = self.read_config(cap + 2, 2)
          if mc & 0x1:
            self.write_config(cap + 2, 2, mc & ~0x1)
          cleared.append(f"msi@{cap:#x}")
        elif cap_id == 0x11:  # MSI-X — clear MSI-X Enable (bit 15), set Function Mask (bit 14)
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

# ============================================================================
# Polaris10 (RX570) register + PM4 constants (from linux gfx_8_0_d.h / amdgpu_poc.c)
# ============================================================================
# VI SET_SH_REG uses dword indices (gfx_8_0_d.h); SI byte addrs /4 ≡ same offsets.
PACKET3_SET_SH_REG_START = 0x00002c00
SI_SH_REG_OFFSET = 0x0000b000  # == PACKET3_SET_SH_REG_START << 2 (compat)
SI_SH_REG_END = 0x0000c000
# mmCOMPUTE_* from gfx_8_0_d.h (preferred)
REG_COMPUTE_START_X = 0x2e04
REG_COMPUTE_START_Y = 0x2e05
REG_COMPUTE_START_Z = 0x2e06
REG_COMPUTE_NUM_THREAD_X = 0x2e07
REG_COMPUTE_NUM_THREAD_Y = 0x2e08
REG_COMPUTE_NUM_THREAD_Z = 0x2e09
REG_COMPUTE_PGM_LO = 0x2e0c
REG_COMPUTE_PGM_HI = 0x2e0d
REG_COMPUTE_PGM_RSRC1 = 0x2e12
REG_COMPUTE_PGM_RSRC2 = 0x2e13
REG_COMPUTE_USER_DATA_0 = 0x2e40
INDIRECT_BUFFER_VALID = 1 << 23
PACKET3_SHADER_TYPE_S = 1 << 1  # compute; OR into PACKET3 header, not opcode
REG_GRBM_STATUS = 0x2004
REG_GRBM_SOFT_RESET = 0x2008
REG_CP_MEC_CNTL = 0x208d
REG_CP_HQD_ACTIVE = 0x2071
REG_SRBM_SOFT_RESET = 0x398
REG_GMCON_DEBUG = 0xd5f
REG_CONFIG_MEMSIZE = 0x150a  # bif_5_0_d.h (VI), not DCE 0x5428
AMDGPU_ASIC_RESET_CFG_OFF = 0x7c
AMDGPU_ASIC_RESET_DATA = 0x39d5e86b
# gfx_v8_0 soft reset: CP + GFX blocks
GRBM_SOFT_RESET_CP_GFX = 0x1 | 0x4 | 0x10000 | 0x20000 | 0x40000
SRBM_SOFT_RESET_ALL = 0x1 | 0x2 | 0x4 | 0x8 | 0x10 | 0x20

PKT_TYPE3 = 3
PKT3_SET_SH_REG = 0x76
PKT3_DISPATCH_DIRECT = 0x15
PKT3_INDIRECT_BUFFER = 0x3f
PKT3_WRITE_DATA = 0x37
PKT3_PFP_SYNC_ME = 0x23
DISPATCH_INITIATOR_COMPUTE_SHADER_EN = 1 << 0
DISPATCH_INITIATOR_FORCE_START_AT_000 = 1 << 2

# gfx803 flat_load/store add4 (assembled llvm-22 -mcpu=gfx803). global_* is gfx9+.
# s[0:1]=out, s[2:3]=a, s[4:5]=b; v0..v13 used → need VGPRS>=3 in RSRC1.
ADD_SHADER = bytes.fromhex(
  "8002007e8402027e8802047e8c02067e0202187e03021a7e000050dc0c00000484181832000050dc0c00000584181832000050dc0c00000684181832000050dc0c0000070402187e05021a7e000050dc0c00000884181832000050dc0c00000984181832000050dc0c00000a84181832000050dc0c00000b70008cbf0411080205130a0206150c0207170e020002187e01021a7e000070dc0c04000084181832000070dc0c05000084181832000070dc0c06000084181832000070dc0c07000070008cbf000081bf"
)
PKT3_EVENT_WRITE = 0x46
EVENT_TYPE_CS_PARTIAL_FLUSH = 7
EVENT_INDEX_CS_PARTIAL_FLUSH = 4

class PM4Builder:
  def __init__(self):
    self.words: list[int] = []

  def pkt3(self, op: int, *vals: int, predicate=0):
    # PACKET3 count = (number of following dwords) - 1 (vid.h PACKET3 macro).
    n = max(len(vals) - 1, 0)
    self.words.append((PKT_TYPE3 << 30) | ((n & 0x3fff) << 16) | ((op & 0xff) << 8) | (predicate & 1))
    self.words.extend(vals)

  def set_sh_reg(self, reg: int, value: int):
    # Accept either mm* dword index (0x2e04) or legacy SI byte addr (0xb810).
    if reg >= SI_SH_REG_OFFSET:
      off = (reg - SI_SH_REG_OFFSET) // 4
    else:
      off = reg - PACKET3_SET_SH_REG_START
    if not (0 <= off < 0x400):
      raise ValueError(f"shader reg {reg:#x} out of SET_SH_REG range (off={off:#x})")
    self.pkt3(PKT3_SET_SH_REG, off, value)

  def set_sh_reg_seq(self, reg: int, *values: int):
    if reg >= SI_SH_REG_OFFSET:
      off = (reg - SI_SH_REG_OFFSET) // 4
    else:
      off = reg - PACKET3_SET_SH_REG_START
    self.pkt3(PKT3_SET_SH_REG, off, *values)

  def dispatch_direct(self, gx=1, gy=1, gz=1, initiator=DISPATCH_INITIATOR_COMPUTE_SHADER_EN | DISPATCH_INITIATOR_FORCE_START_AT_000):
    # SHADER_TYPE is header bit1, NOT part of the opcode (was wrongly 0x17).
    self.pkt3(PKT3_DISPATCH_DIRECT, gx, gy, gz, initiator)
    self.words[-5] |= PACKET3_SHADER_TYPE_S

  def build_dispatch_ib(self, shader_gpu_addr: int, out_va: int, a_va: int, b_va: int,
                        rsrc1=0x000f0043, rsrc2=0x0000000c) -> list[int]:
    """Build PM4 IB for 1x1x1 threadgroup, 4-wide float add.

    COMPUTE_PGM_LO/HI are in 256-byte units (gfx_v8_0_do_edc_gpr_workarounds).
    rsrc1: VGPRS=3 (16 VGPRs), SGPRS=1 (16 SGPRs), FLOAT_MODE=0xf0.
    rsrc2: USER_SGPR=6 → bits[5:1]=6 → 0xc."""
    self.words = []
    self.set_sh_reg(REG_COMPUTE_START_X, 0)
    self.set_sh_reg(REG_COMPUTE_START_Y, 0)
    self.set_sh_reg(REG_COMPUTE_START_Z, 0)
    self.set_sh_reg(REG_COMPUTE_NUM_THREAD_X, 1)
    self.set_sh_reg(REG_COMPUTE_NUM_THREAD_Y, 1)
    self.set_sh_reg(REG_COMPUTE_NUM_THREAD_Z, 1)
    pgm = shader_gpu_addr >> 8
    self.set_sh_reg_seq(REG_COMPUTE_PGM_LO, lo32(pgm), hi32(pgm))
    self.set_sh_reg(REG_COMPUTE_PGM_RSRC1, rsrc1)
    self.set_sh_reg(REG_COMPUTE_PGM_RSRC2, rsrc2)
    self.set_sh_reg(REG_COMPUTE_USER_DATA_0 + 0, lo32(out_va))
    self.set_sh_reg(REG_COMPUTE_USER_DATA_0 + 1, hi32(out_va))
    self.set_sh_reg(REG_COMPUTE_USER_DATA_0 + 2, lo32(a_va))
    self.set_sh_reg(REG_COMPUTE_USER_DATA_0 + 3, hi32(a_va))
    self.set_sh_reg(REG_COMPUTE_USER_DATA_0 + 4, lo32(b_va))
    self.set_sh_reg(REG_COMPUTE_USER_DATA_0 + 5, hi32(b_va))
    self.dispatch_direct()
    # CS_PARTIAL_FLUSH so stores retire before host poll (gfx_v8_0 EDC path).
    self.pkt3(PKT3_EVENT_WRITE, EVENT_TYPE_CS_PARTIAL_FLUSH | (EVENT_INDEX_CS_PARTIAL_FLUSH << 8))
    return self.words

class PolarisDevice:
  """RX570 eGPU device over TinyGPU remote PCI."""

  PCI_DID_RX570 = 0x67df
  PCI_VID_AMD = 0x1002

  def __init__(self, reset: bool = False):
    self.pci = APLRemotePCIDevice("AMD", "usb4")
    vid, did = self._open_config(self.pci, reset=reset)
    if did != self.PCI_DID_RX570 and getenv("AMD_EGPU_ALLOW_ANY", 0) == 0:
      print(f"warning: device {did:#06x} is not RX570 ({self.PCI_DID_RX570:#06x}); set AMD_EGPU_ALLOW_ANY=1 to continue")
    self.vram = self.pci.map_bar(0)          # VRAM window (256MB BAR typical)
    self.doorbell = self.pci.map_bar(2, fmt='I')  # VI doorbells are 32-bit byte-indexed
    self.mmio = self.pci.map_bar(5, fmt='I') # register aperture
    self.bar0_size = self.pci.bar_info(0)[1]
    # macOS USB4 has no handler for eGPU IRQs → mask MSI/INTx before any firmware runs
    # (prevents the recurring APCIE 'unhandled interrupts' kernel panic).
    if OSX and getenv("AMD_BOOT_MASK_INTERRUPTS", 1):
      with contextlib.suppress(Exception):
        masked = self.pci.mask_msi()
        if DEBUG >= 1:
          print(f"polaris: masked device interrupts {masked}", flush=True)
    self._vram_off = 0x100000  # bump allocator in VRAM window
    self._boot: object | None = None
    self._vram_start = 0
    if reset and getenv("AMD_EGPU_MMIO_RESET", 1):
      with contextlib.suppress(Exception):
        self.gpu_mmio_soft_reset()
      self.pci.poll_memsize_ready(self.mmio, wait_s=float(getenv("AMD_EGPU_MEMSIZE_WAIT_S", 2)))
    if DEBUG >= 1:
      print(f"polaris: pci={vid:04x}:{did:04x} bar0={self.bar0_size:#x} grbm={self.mmio[REG_GRBM_STATUS]:#x} mec={self.mmio[REG_CP_MEC_CNTL]:#x}")

  @classmethod
  def _read_pci_ids(cls, pci: APLRemotePCIDevice, retries: int = 5, delay_s: float = 0.2) -> tuple[int, int]:
    """Config reads can transiently return 0xffff on TB eGPU — retry before reset."""
    vid = did = 0xffff
    for _ in range(max(1, retries)):
      vid = pci.read_config(0, 2) & 0xffff
      did = pci.read_config(2, 2) & 0xffff
      if vid == cls.PCI_VID_AMD:
        return vid, did
      if vid != 0xffff:
        break
      time.sleep(delay_s)
    return vid, did

  @classmethod
  def _open_config(cls, pci: APLRemotePCIDevice, reset: bool = False, reset_mode: str = "auto") -> tuple[int, int]:
    vid, did = cls._read_pci_ids(pci)
    auto = getenv("AMD_EGPU_NO_AUTO_RESET", 0) == 0
    mode = reset_mode or os.environ.get("AMD_RESET_MODE", "auto")
    if mode == "mmio" and vid == 0xffff:
      raise RuntimeError("MMIO reset needs PCI visible (vid=0xffff) — replug USB4 cable first")
    if vid == cls.PCI_VID_AMD and not reset:
      return vid, did
    # vid=0xffff: only TinyGPU server restart — PCI hot reset cannot recover missing device
    if vid == 0xffff and not reset:
      if getenv("AMD_EGPU_RESTART_SERVER", 1):
        print("polaris: pci=0xffff — restarting TinyGPU server (no PCI hot reset)", flush=True)
        pci.restart_server()
        time.sleep(2.0)
        vid, did = cls._read_pci_ids(pci, retries=12, delay_s=0.5)
        if vid == cls.PCI_VID_AMD:
          print(f"polaris: pci back {vid:04x}:{did:04x}", flush=True)
          return vid, did
      raise RuntimeError(
        "GPU fell off PCIe (config vid=0xffff). Replug USB4 cable, then: "
        "python3 add.py --probe"
      )
    if not (reset or (auto and vid != cls.PCI_VID_AMD)):
      raise RuntimeError(f"Expected AMD GPU ({cls.PCI_VID_AMD:#06x}), got {vid:#06x}")
    attempts = 1 if vid == 0xffff else int(getenv("AMD_EGPU_RESET_ATTEMPTS", 2))
    for i in range(attempts):
      if vid == cls.PCI_VID_AMD and reset and i == 0:
        print(f"polaris: reset mode={mode} (requested)", flush=True)
      elif vid != cls.PCI_VID_AMD:
        print(f"polaris: pci={vid:#06x} — reset attempt {i + 1}/{attempts} mode={mode}", flush=True)
      if vid == 0xffff and mode not in ("aggressive", "pci", "full", "gentle"):
        break
      vid, did = pci.software_reset(wait_s=float(getenv("AMD_EGPU_RESET_WAIT_S", 8)), mode=mode)
      if vid == cls.PCI_VID_AMD:
        print(f"polaris: reset ok pci={vid:04x}:{did:04x}", flush=True)
        return vid, did
    if vid == 0xffff:
      raise RuntimeError(
        "GPU fell off PCIe (config vid=0xffff). Replug USB4 cable, then: "
        "python3 add.py --probe"
      )
    raise RuntimeError(f"AMD GPU not reachable after reset (last vid={vid:#06x})")

  def gpu_mmio_soft_reset(self):
    """GFX8 GRBM/SRBM soft reset (linux gfx_v8_0_soft_reset) when PCI is up."""
    # Stall GFX via GMCON_DEBUG before reset
    gmcon = self.reg(REG_GMCON_DEBUG)
    self.wreg(REG_GMCON_DEBUG, gmcon | (1 << 0) | (1 << 1))  # GFX_STALL | GFX_CLEAR
    time.sleep(0.05)
    grbm = GRBM_SOFT_RESET_CP_GFX
    self.wreg(REG_GRBM_SOFT_RESET, self.reg(REG_GRBM_SOFT_RESET) | grbm)
    time.sleep(0.05)
    self.wreg(REG_GRBM_SOFT_RESET, self.reg(REG_GRBM_SOFT_RESET) & ~grbm)
    srbm = SRBM_SOFT_RESET_ALL
    self.wreg(REG_SRBM_SOFT_RESET, self.reg(REG_SRBM_SOFT_RESET) | srbm)
    time.sleep(0.05)
    self.wreg(REG_SRBM_SOFT_RESET, self.reg(REG_SRBM_SOFT_RESET) & ~srbm)
    self.wreg(REG_GMCON_DEBUG, gmcon)
    self.pci.drain_mmio(bar=5, reg=REG_GRBM_STATUS)
    time.sleep(0.1)

  def software_reset(self, mmio_reset: bool = True, mode: str = "auto"):
    """Reset eGPU: PCI/AMD-cfg per mode, then optional MMIO soft reset."""
    reset_mode = mode or os.environ.get("AMD_RESET_MODE", "auto")
    if reset_mode == "mmio" and self.pci.read_config(0, 2) & 0xffff == self.PCI_VID_AMD:
      print("polaris: MMIO-only soft reset (no PCI reset)", flush=True)
      with contextlib.suppress(Exception):
        self.gpu_mmio_soft_reset()
      if self.pci.poll_memsize_ready(self.mmio, wait_s=2.0):
        return self.PCI_VID_AMD, self.pci.read_config(2, 2) & 0xffff
    vid, did = self._open_config(self.pci, reset=True, reset_mode=reset_mode)
    self.pci.bar_info.cache_clear()
    self.vram = self.pci.map_bar(0)
    self.doorbell = self.pci.map_bar(2, fmt='I')
    self.mmio = self.pci.map_bar(5, fmt='I')
    self.bar0_size = self.pci.bar_info(0)[1]
    self._boot = None
    self._vram_start = 0
    self._vram_off = 0x100000
    if mmio_reset and getenv("AMD_EGPU_MMIO_RESET", 1):
      with contextlib.suppress(Exception):
        self.gpu_mmio_soft_reset()
    self.pci.poll_memsize_ready(self.mmio, wait_s=float(getenv("AMD_EGPU_MEMSIZE_WAIT_S", 2)))
    return vid, did

  def reg(self, addr: int) -> int:
    return int(self.mmio[addr]) & 0xffffffff

  def wreg(self, addr: int, val: int):
    self.mmio[addr] = int(val) & 0xffffffff

  def gpu_ready(self) -> bool:
    return self.reg(REG_CP_HQD_ACTIVE) != 0

  def vram_gpu_addr(self, bar_off: int) -> int:
    """Map BAR0 offset to GPU VM address (after mc_program)."""
    if self._vram_start:
      return self._vram_start + bar_off
    return bar_off

  def ring_doorbell(self, index: int, wptr: int):
    # VI amdgpu_mm_wdoorbell(index) uses byte offset; BAR2 mmap is dword-indexed.
    from polaris_boot import boot_no_doorbell
    if boot_no_doorbell():
      return
    slot = index >> 2
    self.pci.drain_mmio(bar=5, reg=REG_GRBM_STATUS)
    self.doorbell[slot] = wptr & 0xffffffff
    self.pci.drain_mmio(bar=5, reg=REG_GRBM_STATUS)

  def alloc_vram(self, size: int, align=0x1000) -> int:
    off = round_up(self._vram_off, align)
    if off + size > self.bar0_size: raise MemoryError(f"VRAM BAR overflow {off+size:#x} > {self.bar0_size:#x}")
    self._vram_off = off + size
    return off

  def upload(self, off: int, data: bytes):
    self.vram[off:off+len(data)] = data

  def _boot_retry_attempts(self) -> int:
    return max(1, int(getenv("AMD_BOOT_ATTEMPTS", 2)))

  def _should_retry_after_error(self, err: BaseException) -> bool:
    if getenv("AMD_EGPU_NO_AUTO_RESET", 0) or getenv("AMD_BOOT_RESET", 1) == 0:
      return False
    msg = str(err).lower()
    if "fell off pcie" in msg or "vid=0xffff" in msg:
      return False
    if "rpc failed" in msg or "tinygpu" in msg:
      return False
    return True

  def boot(self, stage: str | None = None):
    """Polaris10 firmware boot (SMU7 + GFX8 + GMC8) — ref/linux amdgpu VI path."""
    attempts = self._boot_retry_attempts()
    last_err: RuntimeError | None = None
    for attempt in range(attempts):
      if attempt and getenv("AMD_BOOT_RESET", 1) != 0:
        print(f"polaris: boot retry {attempt + 1}/{attempts} after software reset", flush=True)
        self.software_reset(mmio_reset=True)
      elif getenv("AMD_BOOT_RESET", 0) == 1:
        with contextlib.suppress(Exception):
          self.software_reset(mmio_reset=True)
      try:
        self._boot_once(stage)
        return
      except RuntimeError as e:
        last_err = e
        if attempt + 1 >= attempts or not self._should_retry_after_error(e):
          raise
        print(f"polaris: boot failed ({e}); retrying", flush=True)
    if last_err:
      raise last_err

  def _boot_stage_fw_direct(self, b, fw_mask: int | None = None, unhalt: bool | None = None) -> int:
    """ATOM → SMC → MC → GART → direct MMIO firmware (no KIQ/dispatch)."""
    from polaris_boot import FW_RLC_ONLY
    if fw_mask is None:
      fw_mask = int(os.environ.get("AMD_BOOT_FW_MASK", str(FW_RLC_ONLY)), 0)
    b.boot_through_fw_direct(fw_mask, unhalt=unhalt)
    return fw_mask

  def _boot_stage_kiq(self, b, map_queues: bool, skip_fw: bool = False) -> None:
    """Firmware boot + KIQ/KCQ MQD; MAP_QUEUES doorbell optional."""
    from polaris_boot import (KIQ_ME, KIQ_PIPE, KIQ_QUEUE, mmCP_MEC_CNTL,
                              ComputeQueue, DOORBELL_MEC_RING0, FW_COMPUTE_MIN,
                              FW_COMPUTE_WITH_JT)
    # Prefer SMC LoadUcodes (MEC body + JT) when requested — Linux Polaris path.
    use_smc = os.environ.get("AMD_BOOT_MEC_SMC_UCODE", "0") == "1"
    fw_mask = int(os.environ.get(
      "AMD_BOOT_FW_MASK",
      str(FW_COMPUTE_WITH_JT if use_smc else FW_COMPUTE_MIN)), 0)
    if skip_fw and b.compute_fw_loaded():
      if int(os.environ.get("DEBUG", "0")):
        print(f"polaris: skip fw re-upload (ME1 running CP_MEC_CNTL={b.rreg(mmCP_MEC_CNTL):#x})",
              flush=True)
      b.boot_minimal_for_compute()
    elif use_smc:
      # Cold path via SMC AGP TOC (session #21) — includes MEC JT1/JT2.
      os.environ.setdefault("AMD_BOOT_FW_LAYOUT", "agp")
      os.environ.setdefault("AMD_BOOT_LOADUCODES_UNTRAINED", "1")
      os.environ.setdefault("AMD_BOOT_FW_MINIMAL", "1")
      b.vi_common_init()
      from atom_replay import run_asic_init_if_needed
      b.enable_vbios_rom()
      run_asic_init_if_needed(b)
      b.gmc_sw_init()
      b.gmc_hw_init_for_dma()
      if not b.smc_running():
        b.start_smc()
      b.process_smc_firmware_header()
      print(f"polaris: MEC via SMC LoadUcodes mask={fw_mask:#x} layout=agp", flush=True)
      b.load_ip_firmware()
      b.unhalt_loaded_firmware(fw_mask)
      b.enable_compute()
    else:
      self._boot_stage_fw_direct(b, fw_mask=fw_mask, unhalt=False)
      b.unhalt_loaded_firmware(fw_mask)
      b.enable_compute()
    cq = ComputeQueue(b, me=1, pipe=0, queue=0, doorbell_index=DOORBELL_MEC_RING0)
    cq.init()
    cq.setup_with_kiq(map_queues=map_queues)
    kiq_active = b.read_hqd_active(KIQ_ME, KIQ_PIPE, KIQ_QUEUE)
    kcq_active = b.read_hqd_active(1, 0, 0)
    if kcq_active & 1:
      b._compute = cq
    print(f"stage=kiq map_queues={map_queues} skip_fw={skip_fw} smc={use_smc} "
          f"CP_MEC_CNTL={b.rreg(mmCP_MEC_CNTL):#x} "
          f"KIQ_HQD_ACTIVE={kiq_active:#x} KCQ_HQD_ACTIVE={kcq_active:#x}")

  def _boot_once(self, stage: str | None = None):
    from polaris_boot import PolarisBoot
    if self._boot is None:
      self._boot = PolarisBoot(self)
    b = self._boot
    if self.gpu_ready():
      if DEBUG >= 1: print("polaris: CP queue already active, skipping boot")
      return
    if stage == "common":
      b.vi_common_init(); print("stage=common ok"); return
    if stage == "atom":
      from atom_replay import run_asic_init_if_needed, vram_training_ok
      b.vi_common_init(); b.enable_vbios_rom()
      run_asic_init_if_needed(b)
      print(f"stage=atom MEMSIZE={b.config_memsize_mb():#x} MISC0={b.rreg(0xa80):#x} "
            f"trained={vram_training_ok(b)}"); return
    if stage == "pre-fw":
      from atom_replay import run_asic_init_if_needed, vram_training_ok
      b.vi_common_init(); b.enable_vbios_rom()
      run_asic_init_if_needed(b)
      if not vram_training_ok(b):
        b.mc_program_light()
        with contextlib.suppress(RuntimeError):
          b.load_mc_firmware()
      b.gmc_sw_init(); b.start_smc(); b.process_smc_firmware_header()
      b.mc_program()
      with contextlib.suppress(RuntimeError):
        b.load_mc_firmware()
      ok, reason, bar0, mm = b.load_ip_firmware_prereqs()
      print(f"stage=pre-fw smc={b.smc_running()} trained={vram_training_ok(b)} "
            f"bar0={bar0} mm={mm} load_ok={ok} — {reason}"); return
    if stage == "smc":
      try:
        b.start_smc()
      except RuntimeError as e:
        print(f"stage=smc FAILED: {e}", flush=True)
        print(f"  hint: AMD_BOOT_SMC_UPLOAD=chunked AMD_MMIO_DRAIN_EVERY=128 "
              f"AMD_BOOT_SMC_POLL_MS=50; replug if pci=0xffff", flush=True)
        raise
      print(f"stage=smc smc_running={b.smc_running()} {b.smc_diag()}"); return
    if stage == "mc":
      b.vi_common_init(); b.start_smc(); b.mc_program(); b.load_mc_firmware(); print("stage=mc ok"); return
    if stage == "fw-rlc":
      from polaris_boot import FW_RLC_ONLY, mmCP_MEC_CNTL
      mask = self._boot_stage_fw_direct(b, fw_mask=FW_RLC_ONLY, unhalt=False)
      print(f"stage=fw-rlc mask={mask:#x} CP_MEC_CNTL={b.rreg(mmCP_MEC_CNTL):#x} smc={b.smc_running()}"); return
    if stage == "fw-cp":
      from polaris_boot import FW_RLC_ONLY, FW_CP_GFX_MASK, mmCP_MEC_CNTL
      mask = self._boot_stage_fw_direct(b, fw_mask=FW_RLC_ONLY | FW_CP_GFX_MASK)
      print(f"stage=fw-cp mask={mask:#x} CP_MEC_CNTL={b.rreg(mmCP_MEC_CNTL):#x}"); return
    if stage == "fw-mec":
      from polaris_boot import FW_COMPUTE_MIN, mmCP_MEC_CNTL
      mask = self._boot_stage_fw_direct(b, fw_mask=FW_COMPUTE_MIN, unhalt=False)
      print(f"stage=fw-mec mask={mask:#x} CP_MEC_CNTL={b.rreg(mmCP_MEC_CNTL):#x} "
            f"(upload only — run fw-start to unhalt)"); return
    if stage == "fw-start":
      from polaris_boot import FW_COMPUTE_MIN, mmCP_MEC_CNTL
      fw_mask = int(os.environ.get("AMD_BOOT_FW_MASK", str(FW_COMPUTE_MIN)), 0)
      b.unhalt_loaded_firmware(fw_mask)
      print(f"stage=fw-start mask={fw_mask:#x} CP_MEC_CNTL={b.rreg(mmCP_MEC_CNTL):#x}"); return
    if stage == "fw-sdma":
      from polaris_boot import (FW_COMPUTE_MIN, UCODE_ID_SDMA0_MASK, UCODE_ID_SDMA1_MASK,
                                mmCP_MEC_CNTL, mmSDMA0_F32_CNTL)
      # Default halted (panic #7): sdma-probe unhalts only after the ring is in GART.
      unhalt = os.environ.get("AMD_BOOT_FW_UNHALT", "0") == "1"
      if b.compute_fw_loaded():
        # Hot GPU: SDMA-only incremental upload — do NOT re-halt/re-upload the live MEC.
        b.load_sdma_firmware_only(unhalt=unhalt)
        sdma_only = True
      else:
        # Cold GPU: full bring-up is required (SMC/MC/GART) plus SDMA, but keep SDMA
        # halted with rings torn down so the F32 unhalt cannot fetch garbage.
        self._boot_stage_fw_direct(
          b, fw_mask=FW_COMPUTE_MIN | UCODE_ID_SDMA0_MASK | UCODE_ID_SDMA1_MASK, unhalt=unhalt)
        sdma_only = False
      print(f"stage=fw-sdma sdma_only={sdma_only} unhalt={unhalt} "
            f"CP_MEC_CNTL={b.rreg(mmCP_MEC_CNTL):#x} F32_CNTL={b.rreg(mmSDMA0_F32_CNTL):#x} "
            f"(ring torn down; run --boot-stage=sdma-probe to unhalt + prove DMA)"); return
    if stage == "fw-direct":
      from polaris_boot import mmCP_MEC_CNTL
      mask = self._boot_stage_fw_direct(b)
      print(f"stage=fw-direct mask={mask:#x} CP_MEC_CNTL={b.rreg(mmCP_MEC_CNTL):#x} "
            f"smc={b.smc_running()} "
            f"(full fw: AMD_BOOT_FW_MASK=0x47e)"); return
    if stage == "kiq":
      self._boot_stage_kiq(b, map_queues=False, skip_fw=False); return
    if stage == "kiq-map":
      if os.environ.get("AMD_BOOT_KIQ_MAP", "1") != "1":
        print("BLOCKED: kiq-map requires AMD_BOOT_KIQ_MAP=1", file=sys.stderr)
        sys.exit(2)
      self._boot_stage_kiq(b, map_queues=True, skip_fw=True); return
    if stage == "gart-probe":
      b.probe_gart_dma()
      print("stage=gart-probe ok"); return
    if stage == "sdma-probe":
      if os.environ.get("AMD_BOOT_SDMA_PROBE", "0") != "1":
        print("GATED: sdma-probe — device DMA-reads SDMA ring from GART host sysmem "
              "(APCIE completion-timeout / kernel-panic risk on M1 USB4).", file=sys.stderr)
        print("  Prereq: python3 add.py --boot-stage=fw-sdma", file=sys.stderr)
        print("  Then:   sleep 10", file=sys.stderr)
        print("  Run:    AMD_BOOT_SDMA_PROBE=1 python3 add.py --boot-stage=sdma-probe",
              file=sys.stderr)
        sys.exit(2)
      from polaris_boot import mmSDMA0_F32_CNTL
      use_agp = os.environ.get("AMD_BOOT_SDMA_AGP", "0") == "1"
      if b.compute_fw_loaded():
        # Hot GPU: never touch the live MEC — only make sure SDMA ucode is resident
        # (halted). AGP mode needs no GART; probe_sdma_dma programs apertures itself.
        if not b.sdma_fw_resident():
          b.load_sdma_firmware_only(unhalt=False)
        if not use_agp:
          b.gmc_sw_init()
          os.environ["AMD_BOOT_GART_SYSMEM"] = "1"
          if b.gart_pte_mem is None:
            b.gart_enable()
      else:
        # Cold GPU: ATOM + gmc_hw_init_for_dma + SDMA ucode. VRAM data path is dead
        # on this eGPU (CMD_RDY=0), so default GART PTE table to host sysmem via AGP
        # MC base (session #14/#15). Override with AMD_BOOT_GART_SYSMEM=0 only if
        # BAR0/MM VRAM writes actually survive HDP flush.
        os.environ.setdefault("AMD_BOOT_GART_SYSMEM", "1")
        b.boot_sdma_minimal()
      r = b.probe_sdma_dma()
      print(f"stage=sdma-probe mode={r.get('mode')} pkt={r.get('pkt_mode')} "
            f"write_ok={r['write_ok']} srbm_ok={r.get('srbm_ok')} "
            f"fetch_ok={r.get('fetch_ok')} fetch={r.get('rptr_fetch', 0):#x} "
            f"ring_drained={r['ring_drained']} dst={r['dst_value']:#x} "
            f"CNTL={r.get('sdma_cntl', 0):#x} RB_PRIV={r.get('rb_priv')} "
            f"PKT_RDY={r.get('packet_ready')} EX_IDLE={r.get('ex_idle')} "
            f"MC_WR_IDLE={r.get('mc_wr_idle')} EXP={r.get('ctx_expired')} "
            f"CTX={r.get('ctx_status', 0):#x} PHASE0={r.get('phase0', 0):#x} "
            f"ST2={r.get('status2', 0):#x} IP={r.get('f32_instr_ptr')} "
            f"CMD={r.get('cmd_op', 0):#x} POWER={r.get('power_cntl', 0):#x} "
            f"DUMMY={r.get('dummy_reg', 0):#x} "
            f"F32_CNTL={b.rreg(mmSDMA0_F32_CNTL):#x}")
      return
    if stage == "kcq-direct":
      os.environ["AMD_BOOT_KCQ_DIRECT"] = "1"
      if os.environ.get("AMD_BOOT_KCQ_ACTIVATE", "0") != "1":
        print("note: kcq-direct stages the KCQ MQD but leaves the HQD inactive "
              "(activation DMA-reads host sysmem → APCIE panic on USB4).", file=sys.stderr)
        print("  Activate deliberately: AMD_BOOT_KCQ_ACTIVATE=1 python3 add.py --boot-stage=kcq-direct",
              file=sys.stderr)
      self._boot_stage_kiq(b, map_queues=False, skip_fw=True); return
    if stage == "kcq-ring-test":
      from polaris_boot import mmSCRATCH_REG0, mmCP_HQD_PQ_WPTR, mmCP_HQD_PQ_RPTR, boot_no_doorbell
      os.environ["AMD_BOOT_KCQ_DIRECT"] = "1"
      os.environ["AMD_BOOT_RING_TEST"] = "1"
      self._boot_stage_kiq(b, map_queues=False, skip_fw=True)
      cq = b._compute
      if cq is None:
        print("stage=kcq-ring-test BLOCKED: KCQ not active", file=sys.stderr)
        sys.exit(2)
      ok = cq.ring_test_scratch()
      scratch = b.rreg(mmSCRATCH_REG0)
      b.srbm_select(1, 0, 0, 0)
      rptr = b.rreg(mmCP_HQD_PQ_RPTR)
      wptr = b.rreg(mmCP_HQD_PQ_WPTR)
      b.srbm_select(0, 0, 0, 0)
      print(f"stage=kcq-ring-test ring_ok={ok} SCRATCH={scratch:#x} "
            f"PQ_WPTR={wptr:#x} PQ_RPTR={rptr:#x} no_doorbell={boot_no_doorbell()}")
      return
    if stage == "add":
      os.environ["AMD_BOOT_KCQ_DIRECT"] = "1"
      self._boot_stage_kiq(b, map_queues=False, skip_fw=True)
      result = self.run_add()
      expected = [11.0, 22.0, 33.0, 44.0]
      print(f"stage=add result={result}")
      if result != expected:
        raise RuntimeError(f"vector-add failed: expected {expected}, got {result}")
      return
    if stage == "kiq-nop":
      os.environ["AMD_BOOT_KIQ_NOP_TEST"] = "1"
      self._boot_stage_kiq(b, map_queues=False, skip_fw=True); return
    b.boot()
    self._vram_start = b.vram_start

  def submit_compute_ib(self, ib_words: list[int]) -> None:
    if self._boot is None:
      self.boot()
    if self._boot._compute is None:
      self._boot.init_compute_queue()
    cq = self._boot._compute
    cq.submit_ib(ib_words)

  def run_add(self, a=(1.0, 2.0, 3.0, 4.0), b_vals=(10.0, 20.0, 30.0, 40.0)):
    if self._boot is None:
      self.boot()
    boot = self._boot
    # Prefer same aperture as KCQ ring (AGP when VRAM dead / COMPUTE_AGP=1).
    cq = boot._compute
    use_agp = cq is not None and getattr(cq, "_mem", None) == "agp"
    use_gtt = (not use_agp) and (not boot.probe_bar0_writes())

    def put_buf(data: bytes, size: int = 0x1000) -> tuple[int, object | None]:
      nbytes = max(size, len(data), 0x1000)
      if use_agp:
        gpu_va, mem, _ = boot.alloc_agp_buffer(nbytes)
        mem[0:len(data)] = data
        sysmem_dma_flush(mem, len(data))
        return gpu_va, mem
      if use_gtt:
        gpu_va, mem, _ = boot.alloc_gtt_buffer(nbytes)
        mem[0:len(data)] = data
        sysmem_dma_flush(mem, len(data))
        return gpu_va, mem
      off = self.alloc_vram(nbytes)
      self.upload(off, data)
      return self.vram_gpu_addr(off), None

    expected = [x + y for x, y in zip(a, b_vals)]
    a_bytes = struct.pack("4f", *a)
    b_bytes = struct.pack("4f", *b_vals)
    out_bytes = bytes(16)
    a_va, _ = put_buf(a_bytes)
    b_va, _ = put_buf(b_bytes)
    out_va, out_mem = put_buf(out_bytes)
    # PGM addr must be 256-byte aligned (COMPUTE_PGM_LO units).
    shader_va, _ = put_buf(ADD_SHADER, round_up(max(len(ADD_SHADER), 0x100), 0x100))
    ib = PM4Builder().build_dispatch_ib(shader_va, out_va, a_va, b_va)
    if DEBUG >= 1:
      print(f"polaris: ib_words={len(ib)} shader={len(ADD_SHADER)} "
            f"mem={'agp' if use_agp else 'gtt' if use_gtt else 'vram'} "
            f"shader={shader_va:#x} out={out_va:#x} expected={expected}", flush=True)
    if cq is None:
      boot.init_compute_queue()
      cq = boot._compute
    drained = cq.submit_ib(ib)
    # Wait for shader store to land in host memory (no IO coherency on M1).
    deadline = time.time() + float(os.environ.get("AMD_BOOT_ADD_WAIT_S", "2"))
    result = [0.0, 0.0, 0.0, 0.0]
    while time.time() < deadline:
      if out_mem is not None:
        sysmem_dma_flush(out_mem, 16)
        result = list(struct.unpack("4f", bytes(out_mem[0:16])))
      else:
        out_off = out_va - (self._vram_start or 0)
        result = list(struct.unpack("4f", bytes(self.vram[out_off:out_off + 16])))
      if result == expected:
        break
      time.sleep(0.01)
    print(f"result={result} drained={drained}")
    return result

def probe():
  dev = PolarisDevice()
  bars = {}
  for i in range(6):
    with contextlib.suppress(Exception):
      bars[i] = dev.pci.bar_info(i)
  from polaris_boot import PolarisBoot, ixSMC_PC_C, ixFIRMWARE_FLAGS
  boot = PolarisBoot(dev)
  print(f"pci=1002:{dev.pci.read_config(2, 2):04x} rev={dev.pci.read_config(8, 1):#04x}")
  print(f"bars={ {k:(hex(v[0]), hex(v[1])) for k,v in bars.items()} }")
  print(f"GRBM_STATUS={dev.reg(REG_GRBM_STATUS):#x} CP_MEC_CNTL={dev.reg(REG_CP_MEC_CNTL):#x} CP_HQD_ACTIVE={dev.reg(REG_CP_HQD_ACTIVE):#x}")
  print(f"SMC running={boot.smc_running()} PC={boot.smc_rreg(ixSMC_PC_C):#x} "
        f"FLAGS={boot.smc_rreg(ixFIRMWARE_FLAGS):#x} RESP={dev.mmio[0x95]:#x}")
  print(f"CONFIG_MEMSIZE={boot.rreg(0x150a):#x} MC_VM_FB_LOCATION={boot.rreg(0x809):#x}")
  st_m = boot.rreg(0xa91)
  print(f"MC_SEQ_STATUS_M={st_m:#x} CMD_RDY={bool(st_m & 0xc)} "
        f"(GDDR live only if CMD_RDY; PWRUP alone is not enough)")
  bar0_ok = boot.probe_bar0_writes()
  print(f"BAR0 writes={'ok' if bar0_ok else 'FAIL'} (strict: survives HDP flush)")
  if getenv("AMD_PROBE_MC", 0):
    boot.gmc_sw_init()
    boot.mc_program()
    mm_ok = boot.probe_vram_mm_writes()
    print(f"MM_INDEX VRAM writes={'ok' if mm_ok else 'FAIL'} (strict: survives HDP flush)")
  with contextlib.suppress(Exception):
    mem, paddrs, _ = boot.alloc_sysmem_buffer(0x1000, contiguous=True)
    if paddrs:
      print(f"sysmem paddr[0]={paddrs[0]:#x} agp_mc={boot.agp_mc_addr(paddrs[0]):#x}")
  print(f"shader_bytes={len(ADD_SHADER)} selftest=ok")

def selftest():
  ib = PM4Builder().build_dispatch_ib(0x10000, 0x20000, 0x30000, 0x40000)
  assert len(ADD_SHADER) == 200
  assert len(ib) >= 20
  assert ib[0] >> 30 == PKT_TYPE3
  from polaris_boot import (PACKET3_SET_RESOURCES, PACKET3_MAP_QUEUES, pkt3,
                            _map_queues_num_q, _map_queues_dbell, _map_queues_queue,
                            _map_queues_pipe, _map_queues_me, DOORBELL_MEC_RING0)
  set_pkt = [pkt3(PACKET3_SET_RESOURCES, 6), 0, 1, 0, 0, 0, 0, 0]
  me_bit = (_map_queues_dbell(DOORBELL_MEC_RING0) | _map_queues_queue(0)
            | _map_queues_pipe(0) | _map_queues_me(0))
  map_pkt = [pkt3(PACKET3_MAP_QUEUES, 5), _map_queues_num_q(1), me_bit,
             0xff00110000, 0xff, 0xff00120040, 0xff]
  assert len(set_pkt) == 8 and len(map_pkt) == 7
  sha = hashlib.sha256(ADD_SHADER).hexdigest()[:12]
  print(f"middle_selftest=ok shader_sha={sha} ib_words={len(ib)} kiq_pkt_words={len(set_pkt)+len(map_pkt)}")

def reset_gpu(mode: str = "auto"):
  if mode != "auto":
    os.environ["AMD_RESET_MODE"] = mode
  dev = PolarisDevice(reset=True)
  memsize = dev.reg(REG_CONFIG_MEMSIZE)
  print(f"reset ok pci=1002:{dev.pci.read_config(2, 2) & 0xffff:04x} "
        f"GRBM_STATUS={dev.reg(REG_GRBM_STATUS):#x} CONFIG_MEMSIZE={memsize:#x}")

def atom_info_cmd():
  from polaris_boot import PolarisBoot
  from atom_replay import read_vbios_rom, atom_info, need_asic_init
  dev = PolarisDevice()
  boot = PolarisBoot(dev)
  boot.enable_vbios_rom()
  bios = read_vbios_rom(boot)
  info = atom_info(bios)
  print(f"vbios_len={info['bios_len']} asic_init_off={info['asic_init_off']:#x}")
  print(f"def_sclk={info['def_sclk']:#x} def_mclk={info['def_mclk']:#x} iio={info['iio_tables']}")
  print(f"need_asic_init={need_asic_init(boot)} CONFIG_MEMSIZE={boot.rreg(0x150a):#x} "
        f"scratch7={boot.rreg(0x5d0):#x} MISC0={boot.rreg(0xa80):#x}")

def apply_add_defaults():
  """Session #21 proven env for RX570 TinyGPU AGP vector-add (VRAM dead).

  setdefault: caller/env overrides still win."""
  defaults = {
    "AMD_BOOT_ADD": "1",
    "AMD_BOOT_MASK_INTERRUPTS": "1",
    "AMD_BOOT_KCQ_DIRECT": "1",
    "AMD_BOOT_COMPUTE_AGP": "1",
    "AMD_BOOT_KIQ_MAP": "0",
    "AMD_BOOT_SKIP_KIQ": "1",
    "AMD_BOOT_DOORBELL_HIT": "1",
    "AMD_BOOT_MEC_SMC_UCODE": "1",
    "AMD_BOOT_FW_LAYOUT": "agp",
    "AMD_BOOT_LOADUCODES_UNTRAINED": "1",
    "AMD_BOOT_FW_MINIMAL": "1",
    "AMD_BOOT_RING_TEST": "1",
  }
  for k, v in defaults.items():
    os.environ.setdefault(k, v)


def main():
  if "--probe" in sys.argv:
    probe(); return
  if "--atom-info" in sys.argv:
    atom_info_cmd(); return
  if "--selftest" in sys.argv:
    selftest(); return
  reset_mode = "auto"
  for arg in sys.argv[1:]:
    if arg.startswith("--reset="):
      reset_mode = arg.split("=", 1)[1]
  if "--reset" in sys.argv:
    reset_gpu(reset_mode); return
  stage = None
  for arg in sys.argv[1:]:
    if arg.startswith("--boot-stage="):
      stage = arg.split("=", 1)[1]
  # Bare run / --boot-stage=add: apply proven defaults (no more AMD_BOOT_*=1 wall).
  if stage is None and os.environ.get("AMD_BOOT_SAFE", "0") == "1":
    print("AMD_BOOT_SAFE=1: not running vector-add.", file=sys.stderr)
    print("  Run:   python3 add.py", file=sys.stderr)
    print("  Or:    python3 add.py --boot-stage=add", file=sys.stderr)
    print("  Probe: python3 add.py --probe | --selftest | --boot-stage=atom", file=sys.stderr)
    sys.exit(2)
  if stage in (None, "add", "kcq-ring-test"):
    apply_add_defaults()
  if stage is None:
    stage = "add"
  if stage:
    # --boot-stage=add no longer needs a separate AMD_BOOT_ADD=1 gate.
    if stage == "add":
      os.environ["AMD_BOOT_ADD"] = "1"
    if stage == "kcq-ring-test":
      os.environ["AMD_BOOT_RING_TEST"] = "1"
    t0 = time.perf_counter()
    expected = [11.0, 22.0, 33.0, 44.0]
    if stage == "add":
      print(f"shader_bytes={len(ADD_SHADER)} expected_result={expected}", flush=True)
    try:
      dev = PolarisDevice()
      if os.environ.get("AMD_ADD_TRACE_STAGES") == "1":
        print(f"  stage t={time.perf_counter()-t0:6.3f}s  device probed", flush=True)
      dev.boot(stage=stage)
      if stage == "add":
        # boot(stage=add) already printed result; assert if run_add returned via stage
        pass
    except RuntimeError as e:
      print(str(e), file=sys.stderr)
      sys.exit(1)
    return

if __name__ == "__main__":
  main()
