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
  python3 examples_egpu/add.py --probe          # eGPU + register sanity (no boot)
  python3 examples_egpu/add.py --selftest         # offline PM4 + shader gate
  python3 examples_egpu/add.py --reset           # TinyGPU PCI software reset
  python3 examples_egpu/add.py --boot-stage=smc   # incremental boot debug
  python3 examples_egpu/add.py                    # boot + add kernel
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
      "mmio": [],  # MMIO only handled by PolarisDevice
      "gentle": [try_amd_cfg, do_pci],
      "full": [try_amd_cfg, do_pci, try_amd_cfg],
      "aggressive": [do_pci, try_amd_cfg, do_pci],
    }
    steps = strategies.get(mode, strategies["gentle"] if vid == 0x1002 else strategies["aggressive"])

    if mode == "auto":
      if vid == 0xffff:
        steps = strategies["aggressive"]
      elif vid == 0x1002:
        steps = [try_amd_cfg]  # AMD cfg reset when PCI still visible
      else:
        steps = [do_pci]

    attempts = max(1, int(os.environ.get("AMD_EGPU_RESET_ATTEMPTS", 3)))
    for i in range(attempts):
      if i:
        print(f"polaris: reset retry {i + 1}/{attempts} mode={mode}", flush=True)
        time.sleep(min(0.5 * (2 ** i), 3.0))
      for step in steps:
        step()
      vid, did = self.poll_config(wait_s=wait_s / attempts)
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
SI_SH_REG_OFFSET = 0x0000b000
SI_SH_REG_END = 0x0000c000
REG_COMPUTE_NUM_THREAD_X = 0x0000b81c
REG_COMPUTE_NUM_THREAD_Y = 0x0000b820
REG_COMPUTE_NUM_THREAD_Z = 0x0000b824
REG_COMPUTE_START_X = 0x0000b810
REG_COMPUTE_START_Y = 0x0000b814
REG_COMPUTE_START_Z = 0x0000b818
REG_COMPUTE_PGM_LO = 0x0000b830
REG_COMPUTE_PGM_HI = 0x0000b834
REG_COMPUTE_PGM_RSRC1 = 0x0000b848
REG_COMPUTE_PGM_RSRC2 = 0x0000b84c
REG_COMPUTE_USER_DATA_0 = 0x0000b900
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

# gfx900 ISA add4 kernel (clang cannot assemble global_* for gfx803 on LLVM 22; Polaris runs gfx900 ISA for this smoke test)
ADD_SHADER = bytes.fromhex(
  "8002007e8402027e8802047e8c02067e008050dc00000204008050dc01000205008050dc02000206008050dc03000207008050dc00000408008050dc01000409008050dc0200040a008050dc0300040b700f8cbf0411080205130a0206150c0207170e02008070dc00040000008070dc01050000008070dc02060000008070dc03070000700f8cbf000081bf"
)

class PM4Builder:
  def __init__(self):
    self.words: list[int] = []

  def pkt3(self, op: int, *vals: int, predicate=0):
    self.words.append((PKT_TYPE3 << 30) | ((len(vals) & 0x3fff) << 16) | ((op & 0xff) << 8) | (predicate & 1))
    self.words.extend(vals)

  def set_sh_reg(self, reg: int, value: int):
    if not (SI_SH_REG_OFFSET <= reg < SI_SH_REG_END):
      raise ValueError(f"shader reg {reg:#x} out of range")
    self.pkt3(PKT3_SET_SH_REG, (reg - SI_SH_REG_OFFSET) // 4, value)

  def dispatch_direct(self, gx=1, gy=1, gz=1, initiator=DISPATCH_INITIATOR_COMPUTE_SHADER_EN | DISPATCH_INITIATOR_FORCE_START_AT_000):
    self.pkt3(PKT3_DISPATCH_DIRECT | (1 << 1), gx, gy, gz, initiator)  # PKT3_SHADER_TYPE_S(1) for compute

  def build_dispatch_ib(self, shader_gpu_addr: int, out_va: int, a_va: int, b_va: int, rsrc1=0x00000240, rsrc2=0x00000008) -> list[int]:
    """Build PM4 IB for 1x1x1 threadgroup, 4-wide float add."""
    self.words = []
    self.set_sh_reg(REG_COMPUTE_START_X, 0)
    self.set_sh_reg(REG_COMPUTE_START_Y, 0)
    self.set_sh_reg(REG_COMPUTE_START_Z, 0)
    self.set_sh_reg(REG_COMPUTE_NUM_THREAD_X, 1)
    self.set_sh_reg(REG_COMPUTE_NUM_THREAD_Y, 1)
    self.set_sh_reg(REG_COMPUTE_NUM_THREAD_Z, 1)
    self.set_sh_reg(REG_COMPUTE_PGM_LO, lo32(shader_gpu_addr))
    self.set_sh_reg(REG_COMPUTE_PGM_HI, hi32(shader_gpu_addr))
    self.set_sh_reg(REG_COMPUTE_PGM_RSRC1, rsrc1)
    self.set_sh_reg(REG_COMPUTE_PGM_RSRC2, rsrc2)
    self.set_sh_reg(REG_COMPUTE_USER_DATA_0 + 0, lo32(out_va))
    self.set_sh_reg(REG_COMPUTE_USER_DATA_0 + 1, hi32(out_va))
    self.set_sh_reg(REG_COMPUTE_USER_DATA_0 + 2, lo32(a_va))
    self.set_sh_reg(REG_COMPUTE_USER_DATA_0 + 3, hi32(a_va))
    self.set_sh_reg(REG_COMPUTE_USER_DATA_0 + 4, lo32(b_va))
    self.set_sh_reg(REG_COMPUTE_USER_DATA_0 + 5, hi32(b_va))
    self.dispatch_direct()
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
  def _open_config(cls, pci: APLRemotePCIDevice, reset: bool = False, reset_mode: str = "auto") -> tuple[int, int]:
    vid = pci.read_config(0, 2) & 0xffff
    did = pci.read_config(2, 2) & 0xffff
    auto = getenv("AMD_EGPU_NO_AUTO_RESET", 0) == 0
    mode = reset_mode or os.environ.get("AMD_RESET_MODE", "auto")
    attempts = int(getenv("AMD_EGPU_RESET_ATTEMPTS", 4 if vid == 0xffff else 2))
    if vid == cls.PCI_VID_AMD and not reset:
      return vid, did
    if not (reset or (auto and vid != cls.PCI_VID_AMD)):
      raise RuntimeError(f"Expected AMD GPU ({cls.PCI_VID_AMD:#06x}), got {vid:#06x}")
    for i in range(attempts):
      if vid == cls.PCI_VID_AMD and reset and i == 0:
        print(f"polaris: reset mode={mode} (requested)", flush=True)
      elif vid != cls.PCI_VID_AMD:
        print(f"polaris: pci={vid:#06x} — reset attempt {i + 1}/{attempts} mode={mode}", flush=True)
        if vid == 0xffff and i == 0 and getenv("AMD_EGPU_RESTART_SERVER", 1):
          print("polaris: restarting TinyGPU server", flush=True)
          pci.restart_server()
      vid, did = pci.software_reset(wait_s=float(getenv("AMD_EGPU_RESET_WAIT_S", 8)), mode=mode)
      if vid == cls.PCI_VID_AMD:
        print(f"polaris: reset ok pci={vid:04x}:{did:04x}", flush=True)
        return vid, did
    if vid == 0xffff:
      raise RuntimeError(
        "GPU fell off PCIe (config vid=0xffff). Tried PCI/AMD-cfg reset — replug USB4 cable, "
        "then: AMD_RESET_MODE=aggressive python3 add.py --reset"
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
    self.doorbell[index] = wptr & 0xffffffff

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
    if getenv("AMD_EGPU_NO_AUTO_RESET", 0):
      return False
    msg = str(err).lower()
    if "fell off pcie" in msg or "vid=0xffff" in msg:
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
        print(f"polaris: boot failed ({e}); retrying after software reset", flush=True)
    if last_err:
      raise last_err

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
    b.boot()
    self._vram_start = b.vram_start

  def submit_compute_ib(self, ib_words: list[int]) -> None:
    if self._boot is None:
      self.boot()
    cq = self._boot.init_compute_queue()
    cq.submit_ib(ib_words)

  def run_add(self, a=(1.0, 2.0, 3.0, 4.0), b=(10.0, 20.0, 30.0, 40.0)):
    self.boot()
    expected = [x + y for x, y in zip(a, b)]
    a_bytes = struct.pack("4f", *a)
    b_bytes = struct.pack("4f", *b)
    out_bytes = bytes(16)
    a_off = self.alloc_vram(0x1000)
    b_off = self.alloc_vram(0x1000)
    out_off = self.alloc_vram(0x1000)
    shader_off = self.alloc_vram(round_up(len(ADD_SHADER), 0x100))
    self.upload(a_off, a_bytes)
    self.upload(b_off, b_bytes)
    self.upload(out_off, out_bytes)
    self.upload(shader_off, ADD_SHADER)
    # GPU virtual addresses in VRAM aperture (set during mc_program)
    a_va = self.vram_gpu_addr(a_off)
    b_va = self.vram_gpu_addr(b_off)
    out_va = self.vram_gpu_addr(out_off)
    shader_va = self.vram_gpu_addr(shader_off)
    ib = PM4Builder().build_dispatch_ib(shader_va, out_va, a_va, b_va)
    if DEBUG >= 1: print(f"polaris: ib_words={len(ib)} shader={len(ADD_SHADER)} expected={expected}")
    self.submit_compute_ib(ib)
    result = list(struct.unpack("4f", bytes(self.vram[out_off:out_off+16])))
    print(f"result={result}")
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
  bar0_ok = boot.probe_bar0_writes()
  print(f"BAR0 writes={'ok' if bar0_ok else 'FAIL'}")
  boot.gmc_sw_init()
  boot.mc_program()
  mm_ok = boot.probe_vram_mm_writes()
  print(f"MM_INDEX VRAM writes={'ok' if mm_ok else 'FAIL'}")
  with contextlib.suppress(Exception):
    mem, paddrs, _ = boot.alloc_sysmem_buffer(0x1000, contiguous=True)
    if paddrs:
      print(f"sysmem paddr[0]={paddrs[0]:#x} agp_mc={boot.agp_mc_addr(paddrs[0]):#x}")
  print(f"shader_bytes={len(ADD_SHADER)} selftest=ok")

def selftest():
  ib = PM4Builder().build_dispatch_ib(0x10000, 0x20000, 0x30000, 0x40000)
  assert len(ADD_SHADER) == 140
  assert len(ib) >= 20
  assert ib[0] >> 30 == PKT_TYPE3
  sha = hashlib.sha256(ADD_SHADER).hexdigest()[:12]
  print(f"middle_selftest=ok shader_sha={sha} ib_words={len(ib)}")

def reset_gpu(mode: str = "auto"):
  if mode != "auto":
    os.environ["AMD_RESET_MODE"] = mode
  dev = PolarisDevice(reset=True)
  memsize = dev.reg(REG_CONFIG_MEMSIZE)
  print(f"reset ok pci=1002:{dev.pci.read_config(2, 2) & 0xffff:04x} "
        f"GRBM_STATUS={dev.reg(REG_GRBM_STATUS):#x} CONFIG_MEMSIZE={memsize:#x}")

def main():
  if "--probe" in sys.argv:
    probe(); return
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
  if stage:
    dev = PolarisDevice()
    dev.boot(stage=stage)
    return
  t0 = time.perf_counter()
  dev = PolarisDevice()
  if os.environ.get("AMD_ADD_TRACE_STAGES") == "1":
    print(f"  stage t={time.perf_counter()-t0:6.3f}s  device probed", flush=True)
  expected = [x + y for x, y in zip((1.0, 2.0, 3.0, 4.0), (10.0, 20.0, 30.0, 40.0))]
  print(f"shader_bytes={len(ADD_SHADER)} expected_result={expected}")
  try:
    result = dev.run_add()
    assert result == expected, f"expected {expected}, got {result}"
  except RuntimeError as e:
    print(str(e), file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
  main()
