#!/usr/bin/env python3
"""Standalone AMD RX570 (Polaris10 / gfx803) eGPU vector-add over TinyGPU.app on macOS.

Vendored single-file (nvgpu examples/add.py style): TinyGPU transport + ATOM BIOS
interpreter + Polaris boot/ComputeQueue + PM4 vector-add. No runtime import of
polaris_boot / atom_replay.

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
import os, sys, ctypes, ctypes.util, time, mmap, struct, array, socket, subprocess
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

# gfx803 ISA (GCN3) — named encoders like nvgpu CubinHelper / build_cubin.
# User SGPRs: s[0:1]=out, s[2:3]=a, s[4:5]=b. VGPRs v0..v13 → RSRC1 VGPRS>=3.
# VOP2: OP[30:25] | VSRC1[24:17] | SRC0[16:9] | VDST[8:0]
class Gcn3:
  class Reg:
    V0, V1, V2, V3, V4, V5, V6, V7 = range(8)
    V8, V9, V10, V11, V12, V13 = range(8, 14)
    S0, S1, S2, S3, S4, S5 = range(6)

  class Op:
    V_ADD_F32 = 1
    V_MUL_F32 = 5
    V_ADD_U32 = 0x19
    S_WAITCNT_VM0_LGKM0 = 0xBF8C0070
    S_ENDPGM = 0xBF810000

  @staticmethod
  def words_blob(words):
    return b"".join(struct.pack("<I", w) for w in words)

  @staticmethod
  def v_mov_b32(vd: int, src: int) -> int:
    """VOP1 v_mov_b32: src = SGPR index, or 0x80|imm for inline constant."""
    return 0x7E000200 | ((vd & 0xFF) << 17) | (src & 0x1FF)

  @classmethod
  def v_mov_b32_imm(cls, vd: int, imm: int) -> int:
    return cls.v_mov_b32(vd, 0x80 | (imm & 0x7F))

  @classmethod
  def v_mov_b32_sgpr(cls, vd: int, s: int) -> int:
    return cls.v_mov_b32(vd, s & 0xFF)

  @staticmethod
  def flat_load_dword(vd: int, vaddr: int) -> tuple[int, int]:
    return (0xDC500000, ((vd & 0xFF) << 24) | (vaddr & 0xFF))

  @staticmethod
  def flat_store_dword(vaddr: int, vdata: int) -> tuple[int, int]:
    return (0xDC700000, ((vdata & 0xFF) << 8) | (vaddr & 0xFF))

  @staticmethod
  def v_add_u32(vd: int, imm: int, vs1: int) -> int:
    """v_add_u32 vd, vcc, imm, vs1 (inline imm lands in VDST field; matches llvm-mc)."""
    return ((Gcn3.Op.V_ADD_U32 & 0x3F) << 25) | ((vs1 & 0xFF) << 17) | ((vd & 0x1FF) << 9) | (0x80 | (imm & 0x7F))

  @staticmethod
  def v_binop_f32(op6: int, vd: int, src0: int, vsrc1: int) -> int:
    """v_{add,mul}_f32_e32 vd, src0, vsrc1 (VDST bit8 set; HW-proven blob encoding)."""
    return ((op6 & 0x3F) << 25) | ((vsrc1 & 0xFF) << 17) | ((src0 & 0x1FF) << 9) | (0x100 | (vd & 0xFF))

def build_shader(alu_op: int) -> bytes:
  """4-wide float kernel: out[i] = a[i] OP b[i] via flat load/store (gfx803)."""
  R, Op = Gcn3.Reg, Gcn3.Op
  w: list[int] = []
  # COMMON_PREFIX: byte offsets + load a[0..3]
  w += [Gcn3.v_mov_b32_imm(i, 4 * i) for i in (R.V0, R.V1, R.V2, R.V3)]
  w += [Gcn3.v_mov_b32_sgpr(R.V12, R.S2), Gcn3.v_mov_b32_sgpr(R.V13, R.S3)]
  for vd in (R.V4, R.V5, R.V6, R.V7):
    w += list(Gcn3.flat_load_dword(vd, R.V12))
    if vd != R.V7:
      w.append(Gcn3.v_add_u32(R.V12, 4, R.V12))
  # load b[0..3]
  w += [Gcn3.v_mov_b32_sgpr(R.V12, R.S4), Gcn3.v_mov_b32_sgpr(R.V13, R.S5)]
  for vd in (R.V8, R.V9, R.V10, R.V11):
    w += list(Gcn3.flat_load_dword(vd, R.V12))
    if vd != R.V11:
      w.append(Gcn3.v_add_u32(R.V12, 4, R.V12))
  w.append(Op.S_WAITCNT_VM0_LGKM0)
  # ARITHMETIC: v_op_f32 v{4+i}, v{8+i}, v{4+i}
  for i in range(4):
    w.append(Gcn3.v_binop_f32(alu_op, R.V4 + i, R.V8 + i, R.V4 + i))
  # COMMON_SUFFIX: store out[0..3]
  w += [Gcn3.v_mov_b32_sgpr(R.V12, R.S0), Gcn3.v_mov_b32_sgpr(R.V13, R.S1)]
  for vd in (R.V4, R.V5, R.V6, R.V7):
    w += list(Gcn3.flat_store_dword(R.V12, vd))
    if vd != R.V7:
      w.append(Gcn3.v_add_u32(R.V12, 4, R.V12))
  w += [Op.S_WAITCNT_VM0_LGKM0, Op.S_ENDPGM]
  return Gcn3.words_blob(w)

ALU_OP = Gcn3.Op.V_ADD_F32
OP = lambda x, y: x + y
OP_NAME = "add"
ADD_SHADER = build_shader(ALU_OP)
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



# =============================================================================
# === atom_replay ===  (ATOM BIOS interpreter)
# =============================================================================
# atom.h / atom.c constants
ATOM_BIOS_MAGIC = 0xAA55
ATOM_ATI_MAGIC_PTR = 0x30
ATOM_ATI_MAGIC = b" 761295520"  # 10 bytes at ROM offset 0x30
ATOM_ROM_TABLE_PTR = 0x48
ATOM_ROM_MAGIC = b"ATOM"
ATOM_ROM_ALT_MAGIC = b"MOTA"  # byte-swapped ATOM (NootedRed checkAtomBios)
ATOM_ROM_MAGIC_PTR = 4
ATOM_ROM_CMD_PTR = 0x1E   # offset in ROM header → command table (not MDT index 0x1E)
ATOM_ROM_DATA_PTR = 0x20
ATOMBIOS_IMAGE_SIZE = 0x10000  # ChefKiss/NootedRed ATOMBIOS.hpp
ATOM_CMD_INIT = 0
# Offsets within master *data table header* (linux atom.c, not MDT uint16[] indices)
ATOM_DATA_FWI_PTR = 0xC
ATOM_DATA_IIO_PTR = 0x32
ATOM_FWI_DEFSCLK_PTR = 8
ATOM_FWI_DEFMCLK_PTR = 0xC
ATOM_FWI_REVISION_PTR = 4
ATOM_FWI_MAIN_PARSER_PTR = 0x14
ATOM_FWI_SCRATCH_REG_PTR = 0x18
# Master data table uint16[] indices (atom_master_list_of_data_tables_v2_1 field order).
# NootedRed getVBIOSDataTable<T>(index) uses these, not ATOM_ROM_CMD_PTR.
MDT_IDX_FIRMWAREINFO = 0x04
MDT_IDX_POWERPLAYINFO = 0x0F
MDT_IDX_DISPLAYOBJECTINFO = 0x16
MDT_IDX_INDIRECT_IO = 0x17
MDT_IDX_UMC_INFO = 0x18
MDT_IDX_DCE_INFO = 0x1B
MDT_IDX_VRAM_INFO = 0x1C
MDT_IDX_INTEGRATED_SYS = 0x1E
MDT_IDX_ASIC_PROFILING = 0x1F
# Polaris dGPU VRAM types (atom_dgpu_vram_type, subset)
ATOM_MEM_TYPE_GDDR5 = 0x0B
_DGPU_MEM_TYPE = {
  0x01: "DDR2", 0x02: "DDR3", 0x03: "DDR4", 0x0B: "GDDR5", 0x10: "GDDR6", 0x11: "HBM",
}
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
ATOM_WS_SHIFT, ATOM_WS_OR_MASK, ATOM_WS_AND_MASK = 0x43, 0x44, 0x45
ATOM_WS_FB_WINDOW, ATOM_WS_ATTRIBUTES = 0x46, 0x47
ATOM_WS_REGPTR = 0x48
ATOM_EXECUTE_MAX_DEPTH = 32
ATOM_CMD_TIMEOUT_SEC = float(os.environ.get("AMD_ATOM_JUMP_TIMEOUT_SEC", "30"))
# Real asic_init timed poll loops (e.g. DELAY_MS settle at 0xd03f) legitimately
# iterate a few thousand times; 512 falsely aborted them. Wall-clock timeout is
# the true stuck-loop guard.
ATOM_JUMP_MAX_ITERS = int(os.environ.get("AMD_ATOM_JUMP_MAX", "50000"))
# MC / idle registers polled during asic_init memory training
ATOM_MC_POLL_REGS = frozenset({0xa80, 0x150a, 0xa29, 0xa2a, 0xe50, 0x2004, 0x1})
ATOM_MAX_WRITES = int(os.environ.get("AMD_ATOM_MAX_WRITES", "65536"))
ATOM_JUMP_BAIL_MAX = int(os.environ.get("AMD_ATOM_JUMP_BAIL_MAX", "0"))
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
    self._jump_bail_count = 0
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
      return 0  # mmMM_INDEX is write-only in ATOM indirect sequences
    retries = int(os.environ.get("AMD_ATOM_POLL_RETRIES", "0"))
    poll_sleep = float(os.environ.get("AMD_ATOM_POLL_SLEEP_MS", "2")) / 1000.0
    if reg in ATOM_MC_POLL_REGS:
      retries = max(retries, int(os.environ.get("AMD_ATOM_MC_POLL_RETRIES", "64")))
    for attempt in range(max(1, retries)):
      val = self.boot.rreg(reg)
      if reg in ATOM_MC_POLL_REGS and attempt + 1 < retries:
        done = False
        if reg == 0xa80 and (val & 0x80):
          done = True
        elif reg == 0x150a and (val & 0xffff) >= 128:
          done = True
        elif reg == 0x1 and val != 0 and val != 0xffffffff:
          done = True
        elif reg not in (0xa80, 0x150a, 0x1) and val not in (0, 0xffffffff):
          done = True
        if done:
          break
        with contextlib.suppress(Exception):
          self.boot.dev.pci.drain_mmio(bar=5, reg=0x2004)
        if poll_sleep > 0:
          time.sleep(poll_sleep)
      else:
        break
    if val == 0 and os.environ.get("AMD_ATOM_POLL_HACK", "0") != "0":
      n = self._poll_reads.get(reg, 0) + 1
      self._poll_reads[reg] = n
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


def check_atom_bios(bios: bytes) -> bool:
  """NootedRed checkAtomBios + linux amdgpu_atom_parse header checks."""
  if len(bios) < 0x49:
    return False
  if _u16(bios, 0) != ATOM_BIOS_MAGIC:
    return False
  base = _u16(bios, ATOM_ROM_TABLE_PTR)
  if not base or base + 8 > len(bios):
    return False
  magic = bios[base + ATOM_ROM_MAGIC_PTR:base + ATOM_ROM_MAGIC_PTR + 4]
  return magic in (ATOM_ROM_MAGIC, ATOM_ROM_ALT_MAGIC)


def mdt_offset(bios: bytes, data_table: int, index: int) -> int:
  """NootedRed getVBIOSDataTable — offset of master data table entry, or 0."""
  if data_table + 4 + index * 2 + 2 > len(bios):
    return 0
  return _u16(bios, data_table + 4 + index * 2)


def parse_firmware_info(bios: bytes, data_table: int) -> dict:
  """AtomFirmwareInfo via ATOM_DATA_FWI_PTR (asic_init ps[0]/ps[1] source)."""
  off = _u16(bios, data_table + ATOM_DATA_FWI_PTR)
  if not off or off + 0x1C > len(bios):
    return {}
  return {
    "off": off,
    "revision": _u32(bios, off + ATOM_FWI_REVISION_PTR),
    "def_sclk_10khz": _u32(bios, off + ATOM_FWI_DEFSCLK_PTR),
    "def_mclk_10khz": _u32(bios, off + ATOM_FWI_DEFMCLK_PTR),
    "main_call_parser": _u32(bios, off + ATOM_FWI_MAIN_PARSER_PTR),
    "scratch_reg_start": _u32(bios, off + ATOM_FWI_SCRATCH_REG_PTR),
  }


def parse_vram_info(bios: bytes, data_table: int) -> dict | None:
  """atom_vram_info_header_v2_3 + first atom_vram_module_v9 (Polaris GDDR5)."""
  off = mdt_offset(bios, data_table, MDT_IDX_VRAM_INFO)
  if not off or off + 0x18 > len(bios):
    return None
  hdr_size = _u16(bios, off)
  if hdr_size < 0x18 or off + hdr_size > len(bios):
    return None
  mod = off + hdr_size
  if mod + 0x34 > len(bios):
    return None
  mem_type = _u8(bios, mod + 23)
  return {
    "off": off,
    "format_rev": _u8(bios, off + 2),
    "content_rev": _u8(bios, off + 3),
    "mc_phyinit_off": _u16(bios, off + 0xA),
    "post_ucode_init_off": _u16(bios, off + 0x10),
    "module_num": _u8(bios, off + 0x14),
    "memory_size_mb": _u32(bios, mod),
    "channel_enable": _u32(bios, mod + 4),
    "max_mem_clk_10khz": _u32(bios, mod + 8),
    "memory_type": mem_type,
    "memory_type_name": _DGPU_MEM_TYPE.get(mem_type, f"0x{mem_type:02x}"),
    "channel_num": _u8(bios, mod + 24),
    "channel_width": _u8(bios, mod + 25),
    "tuning_set_id": _u8(bios, mod + 27),
    "part_number": _cstr(bios, mod + 32, 20).decode("ascii", "replace"),
  }


def list_mdt_entries(bios: bytes, data_table: int) -> dict[int, int]:
  """Non-zero master data table slots → ROM offsets (debug)."""
  out: dict[int, int] = {}
  end = min(len(bios), data_table + 4 + 0x23 * 2)
  for idx in range((end - data_table - 4) // 2):
    off = _u16(bios, data_table + 4 + idx * 2)
    if off:
      out[idx] = off
  return out


def read_vbios_rom(boot: "PolarisBoot", length: int | None = None) -> bytes:
  """vi_read_bios_from_rom via SMC ind-port."""
  vbios_file = os.environ.get("AMD_BOOT_VBIOS_FILE")
  if vbios_file and os.path.isfile(vbios_file):
    return open(vbios_file, "rb").read()
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
    length = min(_u8(hdr, 2) << 9, ATOMBIOS_IMAGE_SIZE)
  length = min((length + 3) & ~3, ATOMBIOS_IMAGE_SIZE)
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
  if not check_atom_bios(bios):
    raise ValueError("invalid ATOM BIOS (magic/header)")
  if bios[ATOM_ATI_MAGIC_PTR:ATOM_ATI_MAGIC_PTR + len(ATOM_ATI_MAGIC)] != ATOM_ATI_MAGIC:
    raise ValueError("invalid ATI magic")
  base = _u16(bios, ATOM_ROM_TABLE_PTR)
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
  fw = parse_firmware_info(bios, ctx.data_table)
  vram = parse_vram_info(bios, ctx.data_table)
  mdt = list_mdt_entries(bios, ctx.data_table)
  info = {
    "bios_len": len(bios),
    "cmd_table": ctx.cmd_table,
    "data_table": ctx.data_table,
    "asic_init_off": init_off,
    "def_sclk": fw.get("def_sclk_10khz", 0),
    "def_mclk": fw.get("def_mclk_10khz", 0),
    "firmware_revision": fw.get("revision"),
    "main_call_parser": fw.get("main_call_parser"),
    "bios_scratch_reg_start": fw.get("scratch_reg_start"),
    "iio_tables": len(ctx.iio),
    "mdt_count": len(mdt),
    "mdt_vram_off": mdt.get(MDT_IDX_VRAM_INFO),
    "mdt_umc_off": mdt.get(MDT_IDX_UMC_INFO),
    "mdt_pp_off": mdt.get(MDT_IDX_POWERPLAYINFO),
  }
  if vram:
    info["vram_mb"] = vram["memory_size_mb"]
    info["vram_type"] = vram["memory_type_name"]
    info["vram_channels"] = vram["channel_num"]
    info["vram_pn"] = vram["part_number"]
  return info


def vram_training_ok(boot: "PolarisBoot") -> bool:
  mem_mb = boot.rreg(0x150a) & 0xffff
  misc0 = boot.rreg(0xa80)
  fb_loc = boot.rreg(0x809)
  # MEMSIZE+MISC0 alone can survive a soft reset while FB_LOCATION is cleared —
  # require a real FB aperture before treating the ASIC as posted.
  return (mem_mb >= 128 and bool(misc0 & 0x80)
          and fb_loc not in (0, 0xffffffff) and (fb_loc & 0xffff) != 0)


def need_asic_init(boot: "PolarisBoot") -> bool:
  if os.environ.get("AMD_BOOT_ATOM_FORCE", "0") == "1":
    return True
  if vram_training_ok(boot):
    return False
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
      # atom.c: special WS registers 0x40-0x48 take priority over the ws[] array.
      ws_map = {
        ATOM_WS_QUOTIENT: g.divmul[0], ATOM_WS_REMAINDER: g.divmul[1],
        ATOM_WS_DATAPTR: g.data_block, ATOM_WS_SHIFT: g.shift,
        ATOM_WS_OR_MASK: (1 << g.shift) & 0xFFFFFFFF,
        ATOM_WS_AND_MASK: (~(1 << g.shift)) & 0xFFFFFFFF,
        ATOM_WS_FB_WINDOW: g.fb_base, ATOM_WS_ATTRIBUTES: g.io_attr,
        ATOM_WS_REGPTR: g.reg_block,
      }
      if idx in ws_map:
        val = ws_map[idx]
      elif idx < len(self.ws):
        val = self.ws[idx]
      else:
        val = 0
    elif arg == ATOM_ARG_ID:
      idx = _u16(bios, p); ptr[0] = p + 2
      # atom.c: ID reads the dword at ROM offset (idx + data_block), not the address.
      off = (idx + g.data_block) & 0xffff
      val = _u32(bios, off) if off + 4 <= len(bios) else 0
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
      # atom.c: special WS registers 0x40-0x48 take priority over ws[]; OR/AND mask read-only.
      if idx == ATOM_WS_QUOTIENT: g.divmul[0] = val
      elif idx == ATOM_WS_REMAINDER: g.divmul[1] = val
      elif idx == ATOM_WS_DATAPTR: g.data_block = val
      elif idx == ATOM_WS_SHIFT: g.shift = val
      elif idx in (ATOM_WS_OR_MASK, ATOM_WS_AND_MASK): pass
      elif idx == ATOM_WS_FB_WINDOW: g.fb_base = val
      elif idx == ATOM_WS_ATTRIBUTES: g.io_attr = val
      elif idx == ATOM_WS_REGPTR: g.reg_block = val & 0xFFFF
      elif idx < len(self.ws): self.ws[idx] = val
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
              if stuck_t or stuck_n:
                if os.environ.get("AMD_ATOM_JUMP_BAIL", "0") == "1":
                  self.card._jump_bail_count += 1
                  if ATOM_JUMP_BAIL_MAX > 0 and self.card._jump_bail_count > ATOM_JUMP_BAIL_MAX:
                    misc0 = self.card.boot.rreg(0xa80)
                    mem = self.card.boot.rreg(0x150a) & 0xffff
                    raise RuntimeError(
                      f"ATOM jump bail limit ({ATOM_JUMP_BAIL_MAX}) — training incomplete "
                      f"MISC0={misc0:#x} MEMSIZE={mem}")
                  if self.debug or os.environ.get("AMD_ATOM_TRACE"):
                    why = "iters" if stuck_n else "time"
                    print(f"  atom: bail stuck jump op={op} ({why}) fall through", flush=True)
                  self.card._jump_counts[abs_t] = 0
                  last_jump = 0
                else:
                  why = "iters" if stuck_n else f"time>{ATOM_CMD_TIMEOUT_SEC}s"
                  misc0 = self.card.boot.rreg(0xa80)
                  mem = self.card.boot.rreg(0x150a) & 0xffff
                  raise RuntimeError(
                    f"ATOM jump loop stuck op={op} target={abs_t:#x} ({why}) "
                    f"MISC0={misc0:#x} MEMSIZE={mem} — set AMD_ATOM_JUMP_BAIL=1 to fall through (unsafe)")
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
  boot.post_atom_sync()
  mem = boot.rreg(0x150a) & 0xffff
  misc0 = boot.rreg(0xa80)
  trained = vram_training_ok(boot)
  if debug or os.environ.get("AMD_ATOM_TRACE"):
    print(f"atom: asic_init done writes={ctx.reg_write_count} MEMSIZE={mem:#x} "
          f"MISC0={misc0:#x} trained={trained} pci={boot.dev.pci.read_config(0,2)&0xffff:#06x}", flush=True)
  if os.environ.get("AMD_BOOT_STRICT_ATOM", "0") == "1" and not trained:
    raise RuntimeError(
      f"ATOM training incomplete MEMSIZE={mem} (want >=128) MISC0={misc0:#x} (want bit 0x80)")
  if not trained and debug:
    print("atom: warning — training incomplete; LoadUcodes likely fails until MEMSIZE/MISC0 ok", flush=True)
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


# =============================================================================
# === polaris_boot ===  (Polaris bring-up / ComputeQueue)
# =============================================================================
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


def boot_allow_sdma_probe() -> bool:
  """Opt-in for --boot-stage=sdma-probe (device DMA via SDMA ring fetch).

  Enabling the SDMA GFX ring makes the GPU DMA-read the ring from GART host sysmem —
  same APCIE completion-timeout risk as KCQ HQD preload. Verification uses SDMA
  WRITE_LINEAR (posted MemWr to host) + CPU readback of the dst buffer (no device read
  for the pass/fail check). Gated behind AMD_BOOT_SDMA_PROBE=1."""
  return os.environ.get("AMD_BOOT_SDMA_PROBE", "0") == "1"


# smu_7_1_3_d.h (polaris10_smumgr.c) / gmc_8_1_d.h / gfx_8_0_d.h

mmSMC_IND_ACCESS_CNTL = 0x92
mmSMC_MESSAGE_0 = 0x94
mmSMC_RESP_0 = 0x95
mmSMC_MSG_ARG_0 = 0xa4
mmSMC_IND_INDEX_11 = 0x1ac
mmSMC_IND_DATA_11 = 0x1ad
mmMC_SEQ_MISC0 = 0xa80
mmMC_SEQ_STATUS_M = 0xa91  # PWRUP_COMPL[1:0], CMD_RDY[3:2]
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
# oss_3_0_d.h — SDMA0 GFX ring (sdma_v2_4_gfx_resume)
mmSDMA0_POWER_CNTL = 0x3402
mmSDMA0_CLK_CTRL = 0x3403
mmSDMA0_CNTL = 0x3404
mmSDMA0_CHICKEN_BITS = 0x3405
mmSDMA0_TILING_CONFIG = 0x3406
mmGB_ADDR_CONFIG = 0x263e  # gfx_8_0_d.h; polaris10 golden 0x22011003
mmSDMA0_SEM_WAIT_FAIL_TIMER_CNTL = 0x3409
mmSDMA0_FREEZE = 0x3413
# sdma_v3_0.c fiji/polaris golden: POWER_CNTL mask 0x800 → 0x3c800
SDMA0_POWER_CNTL_GOLDEN = 0x0003c800
# golden_settings_polaris10_a11: CHICKEN mask 0xfc910007 → 0x00810007
SDMA0_CHICKEN_BITS_GOLDEN = 0x00810007
SDMA0_CHICKEN_BITS_MASK = 0xfc910007
SDMA0_GFX_RB_WPTR_POLL_CNTL__F32_POLL_ENABLE_MASK = 0x4
mmSDMA0_GFX_RB_CNTL = 0x3480
mmSDMA0_GFX_RB_BASE = 0x3481
mmSDMA0_GFX_RB_BASE_HI = 0x3482
mmSDMA0_GFX_RB_RPTR = 0x3483
mmSDMA0_GFX_RB_WPTR = 0x3484
mmSDMA0_GFX_IB_CNTL = 0x348a
mmSDMA0_GFX_CONTEXT_CNTL = 0x3493
mmSDMA0_GFX_VIRTUAL_ADDR = 0x34a7
mmSDMA0_GFX_APE1_CNTL = 0x34a8
SDMA0_CNTL__TRAP_ENABLE_MASK = 0x1
SDMA0_CNTL__AUTO_CTXSW_ENABLE_MASK = 0x40000
SDMA0_CNTL__ATC_L1_ENABLE_MASK = 0x2
mmSDMA0_GFX_RB_WPTR_POLL_CNTL = 0x3485
mmSDMA0_GFX_RB_RPTR_ADDR_HI = 0x3488
mmSDMA0_GFX_RB_RPTR_ADDR_LO = 0x3489
mmSDMA0_GFX_DOORBELL = 0x3492
mmSDMA0_STATUS_REG = 0x340d
mmSDMA0_STATUS1_REG = 0x340e
mmSDMA0_STATUS2_REG = 0x3423
mmSDMA0_GFX_DUMMY_REG = 0x34b1
SDMA0_STATUS_REG__IDLE_MASK = 0x1
SDMA0_STATUS_REG__MC_RD_IDLE_MASK = 0x80000
SDMA0_STATUS2_REG__F32_INSTR_PTR_MASK = 0xffc
SDMA0_STATUS2_REG__F32_INSTR_PTR__SHIFT = 2
SDMA0_STATUS2_REG__CMD_OP_MASK = 0xffff0000
SDMA0_STATUS2_REG__CMD_OP__SHIFT = 16
SDMA0_POWER_CNTL__MEM_POWER_OVERRIDE_MASK = 0x100
mmSRBM_SOFT_RESET = 0x0398
SRBM_SOFT_RESET__SOFT_RESET_SDMA_MASK = 0x100000
SRBM_SOFT_RESET__SOFT_RESET_SDMA1_MASK = 0x40
# gmc_8_1_sh_mask.h MC_VM_MX_L1_TLB_CNTL
MC_VM_MX_L1_TLB_CNTL__ENABLE_L1_TLB_MASK = 0x1
MC_VM_MX_L1_TLB_CNTL__ENABLE_L1_FRAGMENT_PROCESSING_MASK = 0x2
MC_VM_MX_L1_TLB_CNTL__SYSTEM_ACCESS_MODE_MASK = 0x18
MC_VM_MX_L1_TLB_CNTL__SYSTEM_ACCESS_MODE__SHIFT = 3
MC_VM_MX_L1_TLB_CNTL__SYSTEM_APERTURE_UNMAPPED_ACCESS_MASK = 0x20
MC_VM_MX_L1_TLB_CNTL__ENABLE_ADVANCED_DRIVER_MODEL_MASK = 0x40
# dce_11_0 — VGA + CRTC (gmc_v8_0_mc_program / dce_v8_0_disable_dce)
mmVGA_RENDER_CONTROL = 0xc0
mmVGA_HDP_CONTROL = 0xca
VGA_HDP_CONTROL__VGA_MEMORY_DISABLE_MASK = 0x10
VGA_RENDER_CONTROL__VGA_VSTATUS_CNTL_MASK = 0x30000
mmCRTC_CONTROL = 0x1b9c
mmCRTC_UPDATE_LOCK = 0x1bb5
CRTC_CONTROL__CRTC_MASTER_EN_MASK = 0x1
# Polaris10 CRTC register offsets relative to CRTC0 (dce_11_0_d.h)
CRTC_REG_OFFSETS = (0x0, 0x200, 0x400, 0x2600, 0x2800, 0x2a00)
SDMA0_GFX_RB_CNTL__RPTR_WRITEBACK_ENABLE_MASK = 0x1000
SDMA0_GFX_RB_CNTL__RB_SWAP_ENABLE_MASK = 0x200
SDMA0_GFX_RB_CNTL__RPTR_WRITEBACK_TIMER_MASK = 0x1f0000
SDMA0_GFX_RB_CNTL__RPTR_WRITEBACK_TIMER__SHIFT = 16
SDMA0_GFX_RB_WPTR_POLL_CNTL__ENABLE_MASK = 0x1
SDMA0_GFX_DOORBELL__ENABLE_MASK = 0x10000000
SDMA0_GFX_RB_CNTL__RB_ENABLE_MASK = 0x1
SDMA0_GFX_RB_CNTL__RB_ENABLE__SHIFT = 0
SDMA0_GFX_RB_CNTL__RB_SIZE_MASK = 0x3e
SDMA0_GFX_RB_CNTL__RB_SIZE__SHIFT = 1
# TrustOS bare-metal: RB_PRIV (bit 23) required without IOMMU for packet execute.
SDMA0_GFX_RB_CNTL__RB_PRIV_MASK = 0x800000
SDMA0_GFX_RB_CNTL__RB_PRIV__SHIFT = 23
SDMA0_GFX_IB_CNTL__IB_ENABLE_MASK = 0x1
SDMA0_GFX_IB_CNTL__IB_ENABLE__SHIFT = 0
SDMA_RING_SIZE = 4096  # bytes — amdgpu default; rb_size field = order_base_2(size/4)
# tonga_sdma_pkt_open.h (VI / Polaris)
SDMA_OP_NOP = 0
SDMA_OP_WRITE = 2
SDMA_OP_SRBM_WRITE = 14
SDMA_SUBOP_WRITE_LINEAR = 0


def _sdma_pkt_hdr(op: int, sub_op: int = 0, byte_en: int = 0) -> int:
  return (op & 0xff) | ((sub_op & 0xff) << 8) | ((byte_en & 0xf) << 28)


def _reg_set_field(val: int, mask: int, shift: int, field: int) -> int:
  return (val & ~mask) | ((field << shift) & mask)


def _order_base_2(n: int) -> int:
  o = 0
  while (1 << o) < n:
    o += 1
  return o

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
mmMC_VM_AGP_TOP = 0x80a
mmMC_VM_AGP_BOT = 0x80b
mmMC_VM_AGP_BASE = 0x80c
mmMC_VM_SYSTEM_APERTURE_LOW_ADDR = 0x80d
mmMC_VM_SYSTEM_APERTURE_HIGH_ADDR = 0x80e
mmMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR = 0x80f
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
# gmc_8_1_d.h — MC_VM_MX_L1 is 0x819 (NOT 0x518); VM_L2_CNTL4 is 0x578 (NOT 0x503)
mmMC_VM_MX_L1_TLB_CNTL = 0x819
mmVM_L2_CNTL = 0x500
mmVM_L2_CNTL2 = 0x501
mmVM_L2_CNTL3 = 0x502
mmVM_L2_CNTL4 = 0x578
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
# Correct VI offset is 0x3083 (gfx_8_0_d.h). Do NOT use 0x2148.
mmCP_PQ_WPTR_POLL_CNTL = 0x3083
mmCP_PQ_WPTR_POLL_CNTL__EN_MASK = 0x80000000
mmCP_PQ_WPTR_POLL_CNTL1 = 0x3084

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
# Linux smu7 also loads MEC JT1/JT2 as separate TOC entries.
FW_COMPUTE_WITH_JT = (FW_COMPUTE_MIN | UCODE_ID_CP_MEC_JT1_MASK | UCODE_ID_CP_MEC_JT2_MASK)
FW_TO_LOAD = (FW_COMPUTE_WITH_JT | UCODE_ID_SDMA0_MASK | UCODE_ID_SDMA1_MASK)

SMU_FW_BUF_SIZE = 200 * 4096
SMU_HDR_BUF_SIZE = 4096
PAGE_SIZE = 4096

# gmc_v8_0 GTT PTE: VALID|SYSTEM|SNOOPED|EXECUTABLE|READABLE|WRITEABLE (amdgpu_vm.h)
# GCN VI page-table entries are 64-bit: pte = (dma_addr & mask) | flags.
GART_PTE_FLAGS = 0x77
GART_PTE_SIZE = 8            # gfx8 PTE is 8 bytes (amdgpu_gmc_set_pte_pde writes 64-bit)
GART_PTE_ADDR_MASK = 0x0000FFFFFFFFF000  # [47:12] physical page address
# VM_L2_CNTL4 — read PDE/PTE from host system RAM (gmc_8_1_sh_mask.h)
VM_L2_CNTL4__CTX0_PDE_REQUEST_PHYSICAL = 0x40    # shift 6
VM_L2_CNTL4__CTX0_PTE_REQUEST_PHYSICAL = 0x200   # shift 9
VM_L2_CNTL4__CTX1_PDE_REQUEST_PHYSICAL = 0x1000  # shift 12
VM_L2_CNTL4__CTX1_PTE_REQUEST_PHYSICAL = 0x8000  # shift 15

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
    """USB4/TinyGPU: MMIO writes are queued; wait for backlog before unhalt.

    Defaults are short: drain_mmio already flushes the TinyGPU queue; long
    sleeps were the main reason bare `add.py` took ~10s vs nvgpu's instant path."""
    if heavy:
      rounds = int(os.environ.get("AMD_MMIO_SETTLE_ROUNDS", "5"))
      pause_ms = int(os.environ.get("AMD_MMIO_SETTLE_MS", "15"))
    else:
      rounds = int(os.environ.get("AMD_MMIO_SETTLE_ROUNDS_LIGHT", "2"))
      pause_ms = int(os.environ.get("AMD_MMIO_SETTLE_MS_LIGHT", "5"))
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

  def boot_sdma_minimal(self):
    """Cold-GPU bring-up for the SDMA DMA proof with the smallest crash surface.

    [optional ATOM asic_init] → [optional SMC for SCLK/MCLK] → apertures → SDMA ucode.
    Default: NO SMC (session #10 panic surface). TrustOS needed SMU running
    (SCLK/MCLK) before SDMA RPTR advanced — set AMD_BOOT_SDMA_SMC=1 to enable.
    AMD_BOOT_SDMA_ATOM=1 (default) runs ATOM asic_init with the full jump budget."""
    self.vi_common_init()
    os.environ.setdefault("AMD_BOOT_SDMA_ATOM", "1")
    if os.environ.get("AMD_BOOT_SDMA_ATOM", "1") == "1":
      self.enable_vbios_rom()
      run_asic_init_if_needed(self)
    self.gmc_sw_init()
    self.gmc_hw_init_for_dma()
    if os.environ.get("AMD_BOOT_SDMA_SMC", "0") == "1" or \
       os.environ.get("AMD_BOOT_SDMA_SMC_UCODE", "0") == "1":
      # TrustOS: SMU bring-up → SCLK/MCLK before SDMA ring fetch works.
      # Linux Polaris: SDMA ucode is loaded by SMC LoadUcodes (smu7_request_smu_load_fw),
      # not CIK-style MMIO UCODE_DATA — set AMD_BOOT_SDMA_SMC_UCODE=1 for that path.
      print("polaris: SDMA path starting SMC (AMD_BOOT_SDMA_SMC / SMC_UCODE)",
            flush=True)
      if not self.smc_running():
        self.start_smc()
      print(f"polaris: SMC running={self.smc_running()} {self.smc_diag()}", flush=True)
    # Cheap liveness gate: if the SDMA block isn't clocked without asic_init, fail
    # cleanly here instead of uploading into a dead block / unhalting garbage.
    self.wreg(mmSDMA0_GFX_RB_WPTR, 0xA5A58)
    got = self.rreg(mmSDMA0_GFX_RB_WPTR)
    self.wreg(mmSDMA0_GFX_RB_WPTR, 0)
    if got != 0xA5A58:
      raise RuntimeError(
        f"SDMA regs not responding (wrote 0xA5A58, read {got:#x}) — "
        "block needs asic_init; retry with AMD_BOOT_SDMA_ATOM=1")
    # Prefer Linux Polaris path: SMC LoadUcodes with AGP-hosted TOC (VRAM TOC
    # is unsafe — BAR0/MM data path dead). Falls back to direct MMIO upload.
    if os.environ.get("AMD_BOOT_SDMA_SMC_UCODE", "0") == "1" and self.smc_running():
      os.environ.setdefault("AMD_BOOT_FW_LAYOUT", "agp")
      os.environ.setdefault("AMD_BOOT_FW_MASK", "0x6")  # SDMA0|SDMA1 only
      os.environ.setdefault("AMD_BOOT_FW_MINIMAL", "1")
      os.environ.setdefault("AMD_BOOT_SMC_SKIP_DRAM", "1")
      os.environ.setdefault("AMD_BOOT_LOADUCODES_UNTRAINED", "1")
      print("polaris: SDMA ucode via SMC LoadUcodes (AGP TOC, mask=SDMA0|1)",
            flush=True)
      self.sdma_enable(False)
      self.load_ip_firmware()
      self._sdma_fw_resident = True
    else:
      self.load_sdma_firmware_only(unhalt=False)

  def disable_vga_dce(self):
    """gmc_v8_0_mc_program VGA lockout + dce_v8_0_disable_dce CRTC master off.

    TrustOS: VBIOS leaves DCE/DMIF scanout running; it faults VMID0 and can wedge
    the MC while SDMA tries its first host read. Quiesce before GART/SDMA."""
    with contextlib.suppress(Exception):
      tmp = self.rreg(mmVGA_HDP_CONTROL)
      self.wreg(mmVGA_HDP_CONTROL, tmp | VGA_HDP_CONTROL__VGA_MEMORY_DISABLE_MASK)
      tmp = self.rreg(mmVGA_RENDER_CONTROL)
      self.wreg(mmVGA_RENDER_CONTROL, tmp & ~VGA_RENDER_CONTROL__VGA_VSTATUS_CNTL_MASK)
    for off in CRTC_REG_OFFSETS:
      with contextlib.suppress(Exception):
        ctl = self.rreg(mmCRTC_CONTROL + off)
        if ctl & CRTC_CONTROL__CRTC_MASTER_EN_MASK:
          self.wreg(mmCRTC_UPDATE_LOCK + off, 1)
          self.wreg(mmCRTC_CONTROL + off, ctl & ~CRTC_CONTROL__CRTC_MASTER_EN_MASK)
          self.wreg(mmCRTC_UPDATE_LOCK + off, 0)
    self.mmio_sync_safe()


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
    fb_loc = self.rreg(mmMC_VM_FB_LOCATION)
    if (mem_mb not in (0, 0xffff) and (misc0 & 0x80)
        and fb_loc not in (0, 0xffffffff) and (fb_loc & 0xffff) != 0):
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
    """True only if BAR0 writes survive HDP flush (posted writeback is a false positive).

    TinyGPU BAR0 often latches the last write in the CPU mapping; after
    hdp_flush/invalidate the same offset returns open-bus garbage (session #15:
    constant 0xbde1aebe) while MC_SEQ_STATUS_M lacks CMD_RDY — GDDR path dead."""
    pat = 0xA5A5A5A5
    off = 0x2000
    try:
      self.dev.vram[off:off + 4] = struct.pack('<I', pat)
      if struct.unpack('<I', bytes(self.dev.vram[off:off + 4]))[0] != pat:
        return False
      self.hdp_flush()
      self.hdp_invalidate()
      got = struct.unpack('<I', bytes(self.dev.vram[off:off + 4]))[0]
      return got == pat
    except Exception:
      return False

  def probe_vram_mm_writes(self) -> bool:
    """True only if MM_INDEX VRAM writes survive HDP flush (same latch trap as BAR0)."""
    pat = 0xA5A5A5A5
    offs = [0x3000, 0x10000]
    if self.vram_start and self.vram_visible_mc:
      offs.append((self.vram_visible_mc - self.vram_start + 0x3000) & 0xffffffff)
    for off in offs:
      off &= 0xffffffff
      mc = self.vram_mc_addr(off)
      try:
        self.vram_mm_write(mc, struct.pack('<I', pat))
        # Immediate read can return the MM_DATA write latch — require flush.
        self.hdp_flush()
        self.hdp_invalidate()
        got = struct.unpack('<I', self.vram_mm_read(mc, 4))[0]
        ok = got == pat
        if int(os.environ.get("DEBUG", "0")):
          print(f"polaris: probe_vram_mm off={off:#x} mc={mc:#x} wrote={pat:#x} "
                f"read={got:#x} ok={ok}", flush=True)
        if ok:
          return True
      except Exception as e:
        if int(os.environ.get("DEBUG", "0")):
          print(f"polaris: probe_vram_mm off={off:#x} failed: {e}", flush=True)
    return False

  def vram_data_path_live(self) -> bool:
    """CPU can persist a dword into GDDR (BAR0 or MM_INDEX)."""
    return self.probe_bar0_writes() or self.probe_vram_mm_writes()

  def mc_seq_cmd_ready(self) -> bool:
    """MC_SEQ_STATUS_M CMD_RDY_D0|D1 — GDDR command interface accepting traffic."""
    return bool(self.rreg(mmMC_SEQ_STATUS_M) & 0xc)

  def mc_init_locations(self):
    """Place VRAM / GART / AGP from CONFIG_MEMSIZE + MC_VM_FB_LOCATION (gmc_v8_0_mc_init).

    FB_LOCATION is 16 MB granules: bits[15:0]=FB_BASE, bits[31:16]=FB_TOP.
    Example ATOM value 0xf4fff400 → [0xf4000000, 0xf4ffffff] (256 MB BAR window),
    while CONFIG_MEMSIZE may still report full 4096 MB GDDR5."""
    mem_mb = self.rreg(mmCONFIG_MEMSIZE) & 0xffff
    self.mmio_sync_safe()
    fb_loc = self.rreg(mmMC_VM_FB_LOCATION)
    fb_base = (fb_loc & 0xffff) << 24
    fb_top = (((fb_loc >> 16) & 0xffff) << 24) | 0xffffff
    if fb_loc not in (0, 0xffffffff) and fb_base and fb_top >= fb_base:
      self.vram_start = fb_base
      fb_span = fb_top - fb_base + 1
    else:
      self.vram_start = 0
      fb_span = 0
    if mem_mb in (0, 0xffff) or mem_mb < 128:
      mem_mb = int(os.environ.get("AMD_VRAM_MB", "4096"))
    self.vram_size = mem_mb * 1024 * 1024
    # Prefer the hardware FB window for aperture placement (AGP sits above it).
    # Full GDDR5 size stays in vram_size for MEMSIZE reporting.
    if fb_span:
      self.vram_end = fb_top
    else:
      self.vram_end = (self.vram_start + self.vram_size - 1) & 0xffffffffffffffff
    bar_bytes = self.dev.bar0_size
    vis = self.vram_end - self.vram_start + 1
    if vis > bar_bytes:
      self.vram_visible_mc = (self.vram_end - bar_bytes + 1) & 0xffffffffffffffff
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
    # AGP above the real FB window (amdgpu_gmc_agp_location)
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

  def mc_program_apertures(self):
    """System + AGP apertures (gmc_v8_0_mc_program + TrustOS host-DMA stretch).

    Linux programs SYSTEM_APERTURE to the FB window only and leaves AGP unused.
    TrustOS (working Polaris SDMA) stretches SYS_APR through the AGP window so
    host-bound MC addresses stay inside the system aperture decode. AGP_BASE=0
    → host_phys = mc_addr - agp_start (amdgpu_gmc_agp_addr)."""
    scratch = self.vram_visible_mc or self.vram_start
    # Prefer a page inside the FB window for the default/fault page.
    if self.vram_start:
      scratch = self.vram_start + 0x400000  # TrustOS SYS_APR_DEFAULT = FB+0x400000
    sys_lo = min(self.vram_start, self.agp_start) if self.agp_start else self.vram_start
    sys_hi = max(self.vram_end, self.agp_end) if self.agp_end else self.vram_end
    self.wreg(mmMC_VM_SYSTEM_APERTURE_LOW_ADDR, sys_lo >> 12)
    self.wreg(mmMC_VM_SYSTEM_APERTURE_HIGH_ADDR, sys_hi >> 12)
    self.wreg(mmMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR, scratch >> 12)
    self.wreg(mmMC_VM_AGP_BASE, 0)
    self.wreg(mmMC_VM_AGP_TOP, self.agp_end >> 22)
    self.wreg(mmMC_VM_AGP_BOT, self.agp_start >> 22)

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
    self.mc_program_apertures()
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
    self.mc_program_apertures()
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

  def _gart_program_vm(self, pte_base_addr: int, pte_physical: bool = False):
    # gmc_v8_0_gart_enable TLB control, field-exact (was 0x98000b: SYSTEM_ACCESS_MODE=1,
    # no ADVANCED_DRIVER_MODEL — aperture/VM decode not fully active on gfx8).
    self.mc_setup_tlb_apertures()
    self.wreg(mmVM_L2_CNTL, 0x30103)
    self.wreg(mmVM_L2_CNTL2, 0x30003)
    self.wreg(mmVM_L2_CNTL3, 0x24100003)
    # Linux gmc_v8_0_gart_enable clears all VMC_TAP_*_REQUEST_PHYSICAL bits (table
    # lives in VRAM / is an MC address). Host-sysmem tables must be addressed via
    # the AGP aperture (agp_start+dma) so the walker MC-read routes to PCIe — NOT
    # via PTE_REQUEST_PHYSICAL with a raw host phys (small IOVAs like 0x4000 fall
    # outside SYSTEM_APERTURE → default-page / VRAM garbage → apciec 0x200000).
    self.wreg(mmVM_L2_CNTL4, 0)
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
    gart_entries = self.gart_size // PAGE_SIZE           # one 64-bit PTE per 4K page
    gart_bytes = gart_entries * GART_PTE_SIZE
    use_sysmem = os.environ.get("AMD_BOOT_GART_SYSMEM", "auto")
    if use_sysmem == "auto":
      # Prefer VRAM-backed table (Linux/TrustOS): walker never touches host.
      # BAR0 may be dead for CPU readback but MM_INDEX writes can still stick.
      use_sysmem = "0"
    self.gart_pte_mem = bytearray(gart_bytes)
    invalid_pte = struct.pack('<Q', 0)
    for i in range(gart_entries):
      self.gart_pte_mem[i * GART_PTE_SIZE:i * GART_PTE_SIZE + GART_PTE_SIZE] = invalid_pte
    if use_sysmem == "1":
      # Host-RAM table via AGP MC base (experimental). Prefer VRAM table.
      mem, paddrs, _ = self.alloc_sysmem_buffer(gart_bytes, contiguous=True)
      if not self._paddrs_contiguous(paddrs):
        raise RuntimeError(
          f"GART PTE table not physically contiguous ({len(paddrs)} pages) — "
          f"host page-table walk needs a contiguous DMA buffer")
      if not self.agp_start:
        self.gmc_sw_init()
        self.mc_program_apertures()
      self.gart_pte_sysmem = mem
      self.gart_pte_phys = paddrs[0] & ~0xfff
      self.gart_base = self.agp_mc_addr(self.gart_pte_phys)
      for i in range(gart_entries):
        off = i * GART_PTE_SIZE
        self.gart_pte_mem[off:off + GART_PTE_SIZE] = invalid_pte
        mem[off:off + GART_PTE_SIZE] = invalid_pte
      self._gart_pte_flush()
      self.hdp_flush()
      self._gart_program_vm(self.gart_base, pte_physical=False)
      if int(os.environ.get("DEBUG", "0")):
        print(f"polaris: GART PTE table in host RAM via AGP "
              f"base_phys={self.gart_pte_phys:#x} base_mc={self.gart_base:#x} "
              f"entries={gart_entries} bytes={gart_bytes:#x}", flush=True)
    else:
      # TrustOS: table at FB+0x380000. We use bump-alloc + MM_INDEX when BAR0 dead.
      self.gart_pte_off = self.dev.alloc_vram(gart_bytes, align=PAGE_SIZE)
      self.gart_base = self.vram_start + self.gart_pte_off
      self.gart_pte_sysmem = None
      if self.probe_bar0_writes():
        self.dev.upload(self.gart_pte_off, bytes(self.gart_pte_mem))
      else:
        # BAR0 dead — fire-and-forget MM_INDEX writes (readback may still fail).
        self.vram_mm_write(self.gart_base, bytes(self.gart_pte_mem))
      self.hdp_flush()
      self._gart_program_vm(self.gart_base, pte_physical=False)
      if int(os.environ.get("DEBUG", "0")):
        print(f"polaris: GART PTE table in VRAM mc={self.gart_base:#x} "
              f"off={self.gart_pte_off:#x} bar0={'ok' if self.probe_bar0_writes() else 'mm'}",
              flush=True)

  def gart_flush_tlb(self):
    self.wreg(mmVM_INVALIDATE_REQUEST, 1)
    self.mmio_sync_safe()

  def alloc_sysmem_buffer(self, size: int, contiguous: bool = False) -> tuple[object, list[int], int]:
    mem, paddrs = self.dev.pci.alloc_sysmem(size, contiguous=contiguous)
    return mem, paddrs, size

  def _gart_write_pte(self, pte_off: int, pte_val: int):
    if self.gart_pte_mem is None:
      return
    chunk = struct.pack('<Q', pte_val & 0xffffffffffffffff)
    self.gart_pte_mem[pte_off:pte_off + GART_PTE_SIZE] = chunk
    if self.gart_pte_sysmem is not None:
      self.gart_pte_sysmem[pte_off:pte_off + GART_PTE_SIZE] = chunk
    elif self.gart_pte_off is not None:
      if self.probe_bar0_writes():
        self.dev.upload(self.gart_pte_off + pte_off, chunk)
      else:
        self.vram_mm_write(self.gart_base + pte_off, chunk)

  @staticmethod
  def _encode_pte(paddr: int) -> int:
    """gfx8 64-bit GTT PTE: physical page addr [47:12] | flags."""
    return (paddr & GART_PTE_ADDR_MASK) | GART_PTE_FLAGS

  @staticmethod
  def _paddrs_contiguous(paddrs: list[int]) -> bool:
    return all(paddrs[i] == paddrs[0] + i * PAGE_SIZE for i in range(len(paddrs)))

  def _gart_pte_flush(self):
    # Flush host PTE table so eGPU DMA sees GART updates (M1 lacks IO coherency).
    if self.gart_pte_sysmem is None:
      return
    sysmem_dma_flush(self.gart_pte_sysmem, len(self.gart_pte_mem))

  def map_sysmem_gpu(self, paddrs: list[int], size: int, gpu_va: int | None = None) -> int:
    if gpu_va is None:
      gpu_va = self._next_gart_va(size)
    if self.gart_pte_mem is not None:
      npages = (size + 0xfff) // 0x1000
      base_pfn = (gpu_va - self.gart_start) >> 12
      for i, paddr in enumerate(paddrs[:npages]):
        off = (base_pfn + i) * GART_PTE_SIZE
        if off + GART_PTE_SIZE <= len(self.gart_pte_mem):
          self._gart_write_pte(off, self._encode_pte(paddr))
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

  def agp_mc_addr(self, paddr: int, size: int = PAGE_SIZE) -> int:
    """amdgpu_gmc_agp_addr: agp_start + dma_address (full phys, VI rarely uses AGP)."""
    if paddr + size > self.agp_size:
      raise ValueError(f"paddr {paddr:#x} outside AGP aperture size {self.agp_size:#x}")
    return self.agp_start + paddr

  def alloc_agp_buffer(self, size: int) -> tuple[int, object, list[int]]:
    """Host sysmem reachable through the AGP aperture — NO GART page table involved.

    The MC routes agp_start+dma_addr straight out to PCIe (mc_program_apertures), so
    the only device→host transaction for an SDMA ring here is the ring fetch itself:
    no page-table-walk read that could hit the FB aperture / a bogus address
    (progress.md session #10 hypothesis for the apciec 0x200000 panics)."""
    nbytes = round_up(size, PAGE_SIZE)
    mem, paddrs, _ = self.alloc_sysmem_buffer(nbytes, contiguous=True)
    if not paddrs or not self._paddrs_contiguous(paddrs):
      raise RuntimeError("alloc_agp_buffer: need physically contiguous DMA pages")
    mc_addr = self.agp_mc_addr(paddrs[0], nbytes)
    sysmem_dma_flush(mem, nbytes)
    return mc_addr, mem, paddrs

  def vm_context0_disable(self):
    """Force VMID0 back to pure physical addressing (VBIOS-default, no translation).

    Needed for the AGP probe: if a previous GART bring-up left VM context0 enabled,
    an AGP MC address (>= 4 GB) would be pushed through the GART page table, miss its
    range and fault instead of routing via the AGP aperture."""
    self.wreg(mmVM_CONTEXT0_CNTL, 0)
    self.wreg(mmVM_INVALIDATE_REQUEST, 1)
    self.mmio_sync_safe()

  def gmc_program_vm_l2(self):
    """gmc_v8_0_gart_enable L2 cache setup — needed even for AGP physical addressing."""
    self.wreg(mmVM_L2_CNTL, 0x30103)
    self.wreg(mmVM_L2_CNTL2, 0x30003)   # invalidate L1 TLB + L2
    self.wreg(mmVM_L2_CNTL3, 0x24100003)
    self.wreg(mmVM_INVALIDATE_REQUEST, 1)
    self.mmio_sync_safe()

  def _sdma_disable_auto_ctxsw(self):
    """sdma_v3_0_ctx_switch_enable(false) — TrustOS session #16 critical fix.

    Do NOT clobber SDMA0_CNTL to 0x1. TrustOS found that wiping preamble bits
    (MC_RDREQ_CREDIT / MC_WRREQ_CREDIT / …) left RPTR_FETCH stuck at 0 forever;
    preserving them and only clearing AUTO_CTXSW yielded CNTL≈0x08050402 and
    RPTR_FETCH advanced. Linux likewise RMW: AUTO_CTXSW=0, ATC_L1=1.
    AMD_BOOT_SDMA_CNTL=trap forces the old TRAP-only 0x1 (debug only)."""
    mode = os.environ.get("AMD_BOOT_SDMA_CNTL", "preserve")
    for off in (0, SDMA1_REG_OFFSET):
      if mode == "trap":
        cntl = SDMA0_CNTL__TRAP_ENABLE_MASK
        if os.environ.get("AMD_BOOT_SDMA_ATC", "0") == "1":
          cntl |= SDMA0_CNTL__ATC_L1_ENABLE_MASK
      else:
        # Linux sdma_v3_0_ctx_switch_enable: preserve credits/preamble.
        # Default AUTO_CTXSW=0 (TrustOS fetch-ok baseline). Linux gfx_resume
        # enables it; AMD_BOOT_SDMA_AUTO_CTXSW=1 matches that.
        cntl = self.rreg(mmSDMA0_CNTL + off)
        if os.environ.get("AMD_BOOT_SDMA_AUTO_CTXSW", "0") == "1":
          cntl |= SDMA0_CNTL__AUTO_CTXSW_ENABLE_MASK
        else:
          cntl &= ~SDMA0_CNTL__AUTO_CTXSW_ENABLE_MASK
        # ATC_L1: Linux enables it; TrustOS sometimes cleared UTC_L1 for AGP
        # physical addressing when WRITE_LINEAR stalled with MC_WR_IDLE=1.
        # AMD_BOOT_SDMA_ATC=0 forces ATC_L1 off (default: on, matching Linux).
        if os.environ.get("AMD_BOOT_SDMA_ATC", "1") == "1":
          cntl |= SDMA0_CNTL__ATC_L1_ENABLE_MASK
        else:
          cntl &= ~SDMA0_CNTL__ATC_L1_ENABLE_MASK
        if os.environ.get("AMD_BOOT_SDMA_TRAP", "0") == "1":
          cntl |= SDMA0_CNTL__TRAP_ENABLE_MASK
        else:
          # TrustOS stable baseline kept TRAP off (TRAP=1 reintroduced MC0 fault).
          cntl &= ~SDMA0_CNTL__TRAP_ENABLE_MASK
      self.wreg(mmSDMA0_CNTL + off, cntl)
      if int(os.environ.get("DEBUG", "0")):
        print(f"polaris: SDMA{off and 1 or 0}_CNTL={self.rreg(mmSDMA0_CNTL + off):#x} "
              f"(mode={mode})", flush=True)

  def gmc_hw_init_for_dma(self):
    """Linux gmc_v8_0_hw_init minus VRAM-backed GART table: mc_program → MC ucode → VM.

    Minimal SDMA probe was stalling on ring fetch (MC_RD_IDLE=0) because we only ran
    mc_program_light without polaris10_mc.bin — Linux always loads MC ucode in
    gmc_v8_0_hw_init before gmc_v8_0_gart_enable."""
    self.disable_vga_dce()
    self.mc_program()
    mc_ok = False
    try:
      self.load_mc_firmware()
      mc_ok = self.vram_trained()
    except RuntimeError as e:
      print(f"polaris: MC ucode load: {e}", flush=True)
    if mc_ok:
      self.mc_program()
    elif int(os.environ.get("DEBUG", "0")):
      print(f"polaris: MC not trained MISC0={self.rreg(mmMC_SEQ_MISC0):#x} "
            f"MEMSIZE={self.config_memsize_mb()} (ATOM asic_init may still be required)",
            flush=True)
    self.mc_program_apertures()
    self.mc_setup_tlb_apertures()
    self.gmc_program_vm_l2()

  def mc_setup_tlb_apertures(self):
    """gmc_v8_0_gart_enable TLB control — REQUIRED for FB/AGP aperture routing.

    First AGP probe attempt stalled forever (SDMA0_STATUS_REG IDLE=0, MC_RD_IDLE=0):
    the ring-fetch MC read never left the chip because the VBIOS-default
    MC_VM_MX_L1_TLB_CNTL (0x503) has SYSTEM_ACCESS_MODE=0 and
    ENABLE_ADVANCED_DRIVER_MODEL=0, i.e. system/AGP aperture decoding is OFF.
    Program it exactly like Linux gmc_v8_0_gart_enable: L1 TLB on, fragment
    processing on, SYSTEM_ACCESS_MODE=3 (not-in-sys), advanced driver model on,
    unmapped-access=0 (default page instead of dropped request)."""
    tmp = self.rreg(mmMC_VM_MX_L1_TLB_CNTL)
    tmp |= (MC_VM_MX_L1_TLB_CNTL__ENABLE_L1_TLB_MASK |
            MC_VM_MX_L1_TLB_CNTL__ENABLE_L1_FRAGMENT_PROCESSING_MASK |
            MC_VM_MX_L1_TLB_CNTL__ENABLE_ADVANCED_DRIVER_MODEL_MASK)
    tmp = (tmp & ~MC_VM_MX_L1_TLB_CNTL__SYSTEM_ACCESS_MODE_MASK) | \
          (3 << MC_VM_MX_L1_TLB_CNTL__SYSTEM_ACCESS_MODE__SHIFT)
    tmp &= ~MC_VM_MX_L1_TLB_CNTL__SYSTEM_APERTURE_UNMAPPED_ACCESS_MASK
    self.wreg(mmMC_VM_MX_L1_TLB_CNTL, tmp)
    self.wreg(mmVM_INVALIDATE_REQUEST, 1)
    self.mmio_sync_safe()

  def sdma_soft_reset(self):
    """sdma_v3_0_soft_reset: SRBM soft reset both SDMA instances.

    Recovers a wedged engine (e.g. stuck MC read with MC_RD_IDLE=0 after an AGP
    fetch that could not route). A stuck in-flight read is also a suspect for the
    delayed 'spontaneous' apciec panics, so always clear it before re-probing."""
    self.sdma_enable(False)
    tmp = self.rreg(mmSRBM_SOFT_RESET)
    tmp |= SRBM_SOFT_RESET__SOFT_RESET_SDMA_MASK | SRBM_SOFT_RESET__SOFT_RESET_SDMA1_MASK
    self.wreg(mmSRBM_SOFT_RESET, tmp)
    self.mmio_sync_safe()
    time.sleep(0.05)
    tmp &= ~(SRBM_SOFT_RESET__SOFT_RESET_SDMA_MASK | SRBM_SOFT_RESET__SOFT_RESET_SDMA1_MASK)
    self.wreg(mmSRBM_SOFT_RESET, tmp)
    self.mmio_sync_safe()
    time.sleep(0.05)
    # soft reset re-halts F32 and wipes the uploaded ucode state
    self._sdma_fw_resident = False

  def sdma_engine_idle(self) -> bool:
    return bool(self.rreg(mmSDMA0_STATUS_REG) & SDMA0_STATUS_REG__IDLE_MASK)

  def ensure_gart_ready(self):
    """Linux amdgpu_ttm_alloc_gart binds PTEs at sw_init; enable GART before LoadUcodes on eGPU."""
    if self.gart_pte_mem is None:
      self.gart_enable()

  def _flush_fw_sysmem(self, layout: str, fw_mem, extra=None):
    """ARM/M1: CPU cache may hide sysmem writes from eGPU DMA (rpi-pcie #756)."""
    if layout not in ("hybrid", "agp", "gtt"):
      return
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
    sysmem_dma_flush(mem, nbytes)
    return gpu_va, mem, paddrs

  def probe_gart_dma(self) -> dict:
    """Validate GART PTE self-map + sysmem page mapping (CPU-side; run before kiq-map)."""
    self.gmc_sw_init()
    if self.gart_pte_mem is None:
      # Default: VRAM-backed table (walker stays on-chip). Override with
      # AMD_BOOT_GART_SYSMEM=1 for host-table experiments.
      os.environ.setdefault("AMD_BOOT_GART_SYSMEM", "0")
      self.gart_enable()
    pat = 0xCAFEBABE
    src_va, src_mem, paddrs = self.alloc_gtt_buffer(PAGE_SIZE)
    src_mem[0:4] = struct.pack('<I', pat)
    sysmem_dma_flush(src_mem, PAGE_SIZE)
    base_pfn = (src_va - self.gart_start) >> 12
    pte_off = base_pfn * GART_PTE_SIZE
    pte_cpu = struct.unpack_from('<Q', self.gart_pte_mem, pte_off)[0]
    expected_pte = self._encode_pte(paddrs[0])
    pte_host = 0
    if self.gart_pte_sysmem is not None:
      pte_host = struct.unpack_from('<Q', bytes(self.gart_pte_sysmem[pte_off:pte_off + GART_PTE_SIZE]), 0)[0]
    self_map = struct.unpack_from('<Q', self.gart_pte_mem, 0)[0]
    ok = (pte_cpu == expected_pte) and (not self.gart_pte_sysmem or pte_host == expected_pte)
    result = {
      "gart_start": self.gart_start,
      "gart_base": self.gart_base,
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
          f"pte={pte_cpu:#x} expected={expected_pte:#x} gart_base={self.gart_base:#x} "
          f"self_map={self_map:#x}")
    if not ok:
      raise RuntimeError(f"GART PTE mismatch: got {pte_cpu:#x} expected {expected_pte:#x}")
    return result

  def sdma_fw_ready(self) -> bool:
    """SDMA F32 unhalted (firmware resident and running)."""
    return (self.rreg(mmSDMA0_F32_CNTL) & SDMA_F32_CNTL_HALT) == 0

  def _sdma_gfx_ring_disable(self, off: int = 0):
    """sdma_v3_0_gfx_stop for one instance: clear RB_ENABLE + IB_ENABLE, zero base/ptrs.

    Panic #7 root cause: unhalting SDMA F32 while a stale RB_ENABLE=1 / garbage
    RB_BASE survived from a prior session made the engine immediately DMA-read a
    bogus ring address over USB4 (APCIE completion timeout). Always fully quiesce
    the ring — and zero RB_BASE — before any F32 halt/unhalt."""
    rb_cntl = self.rreg(mmSDMA0_GFX_RB_CNTL + off)
    self.wreg(mmSDMA0_GFX_RB_CNTL + off,
              rb_cntl & ~SDMA0_GFX_RB_CNTL__RB_ENABLE_MASK)
    ib_cntl = self.rreg(mmSDMA0_GFX_IB_CNTL + off)
    self.wreg(mmSDMA0_GFX_IB_CNTL + off,
              ib_cntl & ~SDMA0_GFX_IB_CNTL__IB_ENABLE_MASK)
    self.wreg(mmSDMA0_GFX_RB_WPTR + off, 0)
    self.wreg(mmSDMA0_GFX_RB_RPTR + off, 0)
    self.wreg(mmSDMA0_GFX_RB_BASE + off, 0)
    self.wreg(mmSDMA0_GFX_RB_BASE_HI + off, 0)

  def _sdma_gfx_ring_setup(self, ring_gpu_va: int, ring_bytes: int = SDMA_RING_SIZE,
                           unhalt: bool = True):
    """sdma_v3_0_gfx_resume ring programming — ring buffer must live in GART/AGP sysmem.

    Leaves F32 halted when unhalt=False so the caller can preload WPTR first
    (avoids an empty-ring unhalt race on USB4)."""
    self.disable_gpu_interrupts("pre-sdma-ring")
    self._sdma_gfx_ring_disable()
    self._sdma_disable_auto_ctxsw()
    # Session #19: after a successful retire, FREEZE can stick at 1 and the next
    # probe never fetches (FETCH=0). Always clear before reprogramming.
    for off in (0, SDMA1_REG_OFFSET):
      frz = self.rreg(mmSDMA0_FREEZE + off)
      if frz:
        self.wreg(mmSDMA0_FREEZE + off, 0)
        if int(os.environ.get("DEBUG", "0")):
          print(f"polaris: cleared SDMA{off and 1 or 0} FREEZE was={frz:#x}", flush=True)
    # Linux sdma_v3_0_init_golden_registers (polaris10_a11): CHICKEN + CLK + POWER.
    # Missing CHICKEN_BITS=0x00810007 left execute stalled (PKT_RDY + MC_WR_IDLE)
    # after fetch_ok on this eGPU — same STATUS TrustOS saw mid-debug.
    for off in (0, SDMA1_REG_OFFSET):
      chicken = self.rreg(mmSDMA0_CHICKEN_BITS + off)
      chicken = (chicken & ~SDMA0_CHICKEN_BITS_MASK) | (
          SDMA0_CHICKEN_BITS_GOLDEN & SDMA0_CHICKEN_BITS_MASK)
      self.wreg(mmSDMA0_CHICKEN_BITS + off, chicken)
      clk = self.rreg(mmSDMA0_CLK_CTRL + off)
      self.wreg(mmSDMA0_CLK_CTRL + off, (clk & ~0xff000fff) | 0x0)
      pwr = self.rreg(mmSDMA0_POWER_CNTL + off)
      # Golden LS/DS/SD + delay (0x3c800). Also force MEM_POWER_OVERRIDE (bit8)
      # like sdma_v3_0_update_sdma_medium_grain_light_sleep — keeps SDMA mem on.
      pwr = (pwr & ~0x800) | SDMA0_POWER_CNTL_GOLDEN
      if os.environ.get("AMD_BOOT_SDMA_MEM_PWR_OVR", "1") == "1":
        pwr |= SDMA0_POWER_CNTL__MEM_POWER_OVERRIDE_MASK
      self.wreg(mmSDMA0_POWER_CNTL + off, pwr)
      if int(os.environ.get("DEBUG", "0")):
        print(f"polaris: SDMA{off and 1 or 0} CHICKEN={self.rreg(mmSDMA0_CHICKEN_BITS + off):#x} "
              f"CLK={self.rreg(mmSDMA0_CLK_CTRL + off):#x} "
              f"POWER={self.rreg(mmSDMA0_POWER_CNTL + off):#x} "
              f"FREEZE={self.rreg(mmSDMA0_FREEZE + off):#x}", flush=True)
    # Linux sdma_v3_0_gfx_resume: TILING_CONFIG = gb_addr_config & 0x70
    gb = self.rreg(mmGB_ADDR_CONFIG)
    self.wreg(mmSDMA0_TILING_CONFIG, gb & 0x70)
    self.wreg(mmSDMA0_TILING_CONFIG + SDMA1_REG_OFFSET, gb & 0x70)
    # Linux sdma_v3_0_gfx_resume: clear VIRTUAL_ADDR + APE1 for every VMID via SRBM.
    for vmid in range(AMDGPU_NUM_VMID):
      self.srbm_select(0, 0, 0, vmid)
      self.wreg(mmSDMA0_GFX_VIRTUAL_ADDR, 0)
      self.wreg(mmSDMA0_GFX_APE1_CNTL, 0)
    self.srbm_select(0, 0, 0, 0)
    # TrustOS: CONTEXT_CNTL!=0 can make F32 touch VRAM outside GART → VM fault.
    self.wreg(mmSDMA0_GFX_CONTEXT_CNTL, 0)
    self.wreg(mmSDMA0_SEM_WAIT_FAIL_TIMER_CNTL, 0)
    rb_bufsz = _order_base_2(ring_bytes // 4)
    # TrustOS stable baseline 0x31015 = RB_ENABLE|RB_SIZE(10)|RPTR_WB_TIMER(3).
    # Do NOT leave VBIOS RB_SWAP_ENABLE (bit 9) set — that is the 0x1017 stall
    # (endian-swap corrupts LE ring words). No RPTR writeback / wptr poll / doorbell:
    # only the ring fetch itself is a device→host read for this probe.
    # RB_PRIV (bit 23): TrustOS marks REQUIRED for bare-metal without IOMMU —
    # without it, PACKET_READY can stick with MC_WR_IDLE=1 and no host write.
    # AMD_BOOT_SDMA_RB_PRIV=0 disables (Linux gfx_resume does not set it).
    rb_cntl = ((rb_bufsz << SDMA0_GFX_RB_CNTL__RB_SIZE__SHIFT) &
               SDMA0_GFX_RB_CNTL__RB_SIZE_MASK)
    rb_cntl = _reg_set_field(rb_cntl, SDMA0_GFX_RB_CNTL__RPTR_WRITEBACK_TIMER_MASK,
                             SDMA0_GFX_RB_CNTL__RPTR_WRITEBACK_TIMER__SHIFT, 3)
    if os.environ.get("AMD_BOOT_SDMA_RB_PRIV", "1") == "1":
      rb_cntl |= SDMA0_GFX_RB_CNTL__RB_PRIV_MASK
    self.wreg(mmSDMA0_GFX_RB_CNTL, rb_cntl)  # RB_ENABLE=0 while programming
    self.wreg(mmSDMA0_GFX_RB_RPTR, 0)
    self.wreg(mmSDMA0_GFX_RB_WPTR, 0)
    # Linux gfx_resume always programs RPTR_ADDR + RPTR_WRITEBACK_ENABLE=1.
    # TrustOS also needs WB for RB_RPTR publish. Default ON with AGP/GART page;
    # AMD_BOOT_SDMA_RPTR_WB=0 disables (old probe baseline).
    self._sdma_rptr_wb_va = None
    self._sdma_rptr_wb_mem = None
    rptr_wb = os.environ.get("AMD_BOOT_SDMA_RPTR_WB", "1") == "1"
    if rptr_wb:
      try:
        if getattr(self, "_probe_use_agp", False):
          wb_va, wb_mem, _ = self.alloc_agp_buffer(PAGE_SIZE)
        else:
          wb_va, wb_mem, _ = self.alloc_gtt_buffer(PAGE_SIZE)
        for i in range(PAGE_SIZE // 4):
          wb_mem[i * 4:(i + 1) * 4] = struct.pack('<I', 0xDEADBEEF)
        sysmem_dma_flush(wb_mem, PAGE_SIZE)
        self.wreg(mmSDMA0_GFX_RB_RPTR_ADDR_HI, (wb_va >> 32) & 0xffffffff)
        self.wreg(mmSDMA0_GFX_RB_RPTR_ADDR_LO, wb_va & 0xfffffffc)
        rb_cntl |= SDMA0_GFX_RB_CNTL__RPTR_WRITEBACK_ENABLE_MASK
        self._sdma_rptr_wb_va = wb_va
        self._sdma_rptr_wb_mem = wb_mem
      except Exception as e:
        if int(os.environ.get("DEBUG", "0")):
          print(f"polaris: RPTR_WB alloc failed: {e}", flush=True)
        self.wreg(mmSDMA0_GFX_RB_RPTR_ADDR_HI, 0)
        self.wreg(mmSDMA0_GFX_RB_RPTR_ADDR_LO, 0)
    else:
      self.wreg(mmSDMA0_GFX_RB_RPTR_ADDR_HI, 0)
      self.wreg(mmSDMA0_GFX_RB_RPTR_ADDR_LO, 0)
    self.wreg(mmSDMA0_GFX_RB_WPTR_POLL_CNTL,
              self.rreg(mmSDMA0_GFX_RB_WPTR_POLL_CNTL) & ~SDMA0_GFX_RB_WPTR_POLL_CNTL__ENABLE_MASK)
    # TrustOS: Linux doorbell OFFSET|ENABLE (0x100001E0) stopped init MC0 faults
    # even when doorbell mailbox is dead — F32 may require the OFFSET field.
    # Default: ENABLE=0 but write Linux-like OFFSET. AMD_BOOT_SDMA_DOORBELL=linux|0|1
    door_mode = os.environ.get("AMD_BOOT_SDMA_DOORBELL", "linux")
    door = self.rreg(mmSDMA0_GFX_DOORBELL)
    if door_mode == "1":
      door = (door & ~0x3ff) | 0x1e0 | SDMA0_GFX_DOORBELL__ENABLE_MASK
    elif door_mode == "linux":
      # OFFSET=0x1e0, ENABLE=0 — matches TrustOS "doorbell Linux values" without
      # waiting for a doorbell ring that never comes on TinyGPU.
      door = (door & ~0x3ff) | 0x1e0
      door &= ~SDMA0_GFX_DOORBELL__ENABLE_MASK
    else:
      door &= ~SDMA0_GFX_DOORBELL__ENABLE_MASK
    self.wreg(mmSDMA0_GFX_DOORBELL, door)
    self.wreg(mmSDMA0_GFX_RB_BASE, ring_gpu_va >> 8)
    self.wreg(mmSDMA0_GFX_RB_BASE_HI, (ring_gpu_va >> 40) & 0xffffffff)
    # TrustOS: IB_ENABLE=1 (0x101) reintroduced SDM0 MC0 fault; stable baseline IB=0.
    # Linux golden_settings_polaris10_a11: IB_CNTL mask→0x100 (SWITCH_INSIDE_IB only,
    # IB_ENABLE=0). Full gfx_resume later sets IB_ENABLE=1; ring_test only needs RB.
    # AMD_BOOT_SDMA_IB=1 enables IB; =golden (default) applies 0x100; =0 clears all.
    ib_mode = os.environ.get("AMD_BOOT_SDMA_IB", "golden")
    ib_cntl = self.rreg(mmSDMA0_GFX_IB_CNTL)
    if ib_mode == "1":
      ib_cntl = _reg_set_field(ib_cntl, SDMA0_GFX_IB_CNTL__IB_ENABLE_MASK,
                               SDMA0_GFX_IB_CNTL__IB_ENABLE__SHIFT, 1)
    elif ib_mode == "golden":
      ib_cntl = (ib_cntl & ~0x800f0111) | 0x00000100
    else:
      ib_cntl &= ~SDMA0_GFX_IB_CNTL__IB_ENABLE_MASK
      ib_cntl &= ~0x100  # SWITCH_INSIDE_IB
    self.wreg(mmSDMA0_GFX_IB_CNTL, ib_cntl)
    # PHASE*_QUANTUM: Linux leaves 0 unless amdgpu_sdma_phase_quantum is set.
    # TrustOS: non-zero (0x2000) reintroduced MC0 faults. Live eGPU (session #19):
    # PHASE=0 → EXPIRED=1; PHASE=0xff0f clears EXPIRED and (with RPTR_WB +
    # MEM_POWER_OVERRIDE + doorbell OFFSET) allows F32 execute/retire.
    # AMD_BOOT_SDMA_PHASE=0|leave|<hex> (default 0xff0f after write_ok).
    phase_mode = os.environ.get("AMD_BOOT_SDMA_PHASE", "0xff0f")
    if phase_mode != "leave":
      phase = int(phase_mode, 0)
      self.wreg(0x3414, phase)
      self.wreg(0x3415, phase)
      self.wreg(0x3614, phase)
      self.wreg(0x3615, phase)
    rb_cntl = _reg_set_field(rb_cntl, SDMA0_GFX_RB_CNTL__RB_ENABLE_MASK,
                             SDMA0_GFX_RB_CNTL__RB_ENABLE__SHIFT, 1)
    self.wreg(mmSDMA0_GFX_RB_CNTL, rb_cntl)
    if unhalt and not self.sdma_fw_ready():
      self.sdma_enable(True)
      self.wreg(mmSDMA0_GFX_CONTEXT_CNTL, 0)
    self.hdp_flush()
    self.gart_flush_tlb()
    self.mmio_sync_safe()

  def _sdma_gfx_ring_commit(self, ring_mem, pkt_dwords: list[int], ring_va: int | None = None):
    """Write pkt_dwords into ring[0..] and bump WPTR (byte offset in register)."""
    nbytes = len(pkt_dwords) * 4
    data = b''.join(struct.pack('<I', dw) for dw in pkt_dwords)
    if ring_mem is not None:
      for i, dw in enumerate(pkt_dwords):
        ring_mem[i * 4:(i + 1) * 4] = struct.pack('<I', dw)
      sysmem_dma_flush(ring_mem, nbytes)
    elif ring_va is not None:
      self.vram_mm_write(ring_va, data)
    else:
      raise RuntimeError("ring commit needs ring_mem or ring_va")
    self.hdp_flush()
    self.gart_flush_tlb()
    self.wreg(mmSDMA0_GFX_RB_WPTR, len(pkt_dwords) << 2)

  def probe_sdma_dma(self) -> dict:
    """Device DMA proof: SDMA WRITE_LINEAR → host buffer (sdma_v2_4_ring_test_ring).

    Linux amdgpu uses the same pattern: one WRITE_LINEAR dword to a wb buffer, CPU
    polls the dst for 0xDEADBEEF. Ring fetch is a device→host read (APCIE risk on
    M1/USB4); dst verification is CPU-only (posted write, no completion).

    Two addressing modes for the ring/dst:
      AMD_BOOT_SDMA_AGP=0 (default): GART page table in host RAM via AGP MC base
        (agp_start+dma) — walker routes to PCIe; PTEs carry host DMA + SYSTEM.
      AMD_BOOT_SDMA_AGP=1: AGP aperture — linear MC window, no page-table walk.

    Requires SDMA ucode resident (upload happens halted; this probe unhalts only
    after RB_BASE points at a valid ring). Gated by AMD_BOOT_SDMA_PROBE=1."""
    if not boot_allow_sdma_probe():
      raise RuntimeError(
        "SDMA device-DMA probe gated — set AMD_BOOT_SDMA_PROBE=1 (APCIE panic risk)")
    use_agp = os.environ.get("AMD_BOOT_SDMA_AGP", "0") == "1"
    self._probe_use_agp = use_agp
    if not self.sdma_fw_resident():
      raise RuntimeError(
        "SDMA ucode not resident — run: python3 add.py --boot-stage=fw-sdma first "
        "(this probe unhalts SDMA itself once the ring is programmed).")
    use_vram = os.environ.get("AMD_BOOT_SDMA_VRAM", "0") == "1"
    pkt_mode = os.environ.get("AMD_BOOT_SDMA_PKT", "write")  # write|srbm|nop
    if use_vram:
      # Isolation test: ring+dst in VRAM (no host DMA). HARD GATE: if the CPU
      # cannot persist a dword into GDDR, RB_BASE at 0xf4… is not real VRAM —
      # MC may route it out PCIe as a ≥32-bit TLP → apciec 0x200000 (panic #15).
      if not self.vram_data_path_live():
        st = self.rreg(mmMC_SEQ_STATUS_M)
        raise RuntimeError(
          f"AMD_BOOT_SDMA_VRAM=1 refused: VRAM data path dead "
          f"(BAR0/MM_INDEX writes vanish after HDP flush; "
          f"MC_SEQ_STATUS_M={st:#x} CMD_RDY={bool(st & 0xc)}). "
          f"Use AMD_BOOT_SDMA_AGP=1 (host ring via AGP) instead.")
      self.gmc_sw_init()
      self.mc_program_apertures()
      self.mc_setup_tlb_apertures()
      self.gmc_program_vm_l2()
      self.vm_context0_disable()
      pte = {"mode": "vram", "vram_start": self.vram_start}
      ring_off = self.dev.alloc_vram(SDMA_RING_SIZE)
      dst_off = self.dev.alloc_vram(PAGE_SIZE)
      ring_va = self.vram_start + ring_off
      dst_va = self.vram_start + dst_off
      ring_mem = None
      dst_mem = None
      dst_paddrs = [dst_va]
    elif use_agp:
      self.gmc_sw_init()
      self.mc_program_apertures()
      self.mc_setup_tlb_apertures()
      self.gmc_program_vm_l2()
      self.vm_context0_disable()
      pte = {"mode": "agp", "agp_start": self.agp_start}
      ring_va, ring_mem, _ = self.alloc_agp_buffer(SDMA_RING_SIZE)
      dst_va, dst_mem, dst_paddrs = self.alloc_agp_buffer(PAGE_SIZE)
    else:
      pte = self.probe_gart_dma()
      pte["mode"] = "gart"
      ring_va, ring_mem, _ = self.alloc_gtt_buffer(SDMA_RING_SIZE)
      dst_va, dst_mem, dst_paddrs = self.alloc_gtt_buffer(PAGE_SIZE)
    sentinel = 0xCAFEDEAD
    expect = 0xDEADBEEF
    if use_vram:
      zeros = bytes(SDMA_RING_SIZE)
      self.vram_mm_write(ring_va, zeros)
      self.vram_mm_write(dst_va, struct.pack('<I', sentinel) + bytes(PAGE_SIZE - 4))
    else:
      for i in range(SDMA_RING_SIZE // 4):
        ring_mem[i * 4:(i + 1) * 4] = struct.pack('<I', 0)
      sysmem_dma_flush(ring_mem, SDMA_RING_SIZE)
      dst_mem[0:4] = struct.pack('<I', sentinel)
      sysmem_dma_flush(dst_mem, PAGE_SIZE)
    # Program ring while F32 halted, preload WPTR, then unhalt — so the first
    # fetch already has work (Linux commits after enable; USB4 prefers preload).
    self._sdma_gfx_ring_setup(ring_va, SDMA_RING_SIZE, unhalt=False)
    # Optional: write into the ring page itself (same AGP mapping as fetch) to
    # isolate dst-aperture issues. AMD_BOOT_SDMA_DST=ring → ring_va+0x100.
    dst_mode = os.environ.get("AMD_BOOT_SDMA_DST", "buf")
    dst_off = 0
    if dst_mode == "ring" and not use_vram and ring_mem is not None:
      dst_va = ring_va + 0x100
      dst_mem = ring_mem
      dst_off = 0x100
      ring_mem[dst_off:dst_off + 4] = struct.pack('<I', sentinel)
      sysmem_dma_flush(ring_mem, SDMA_RING_SIZE)
      if use_agp:
        dst_paddrs = [dst_va - self.agp_start]
      else:
        dst_paddrs = [dst_va]
    # Packet modes:
    #   write (default): WRITE_LINEAR → host (needs host DMA write path)
    #   srbm: SRBM_WRITE → GFX_DUMMY_REG (on-chip; proves F32 execute w/o host DMA)
    #   nop:  NOP only (proves retire / RPTR advance)
    count_field = int(os.environ.get("AMD_BOOT_SDMA_COUNT", "1"))
    dummy_expect = 0xA5A5A5A5
    if pkt_mode == "srbm":
      self.wreg(mmSDMA0_GFX_DUMMY_REG, 0)
      pkt = [
        _sdma_pkt_hdr(SDMA_OP_SRBM_WRITE, 0, byte_en=0xf),
        mmSDMA0_GFX_DUMMY_REG & 0xffff,
        dummy_expect,
      ]
    elif pkt_mode == "nop":
      pkt = [0]  # single NOP
    else:
      pkt = [
        _sdma_pkt_hdr(SDMA_OP_WRITE, SDMA_SUBOP_WRITE_LINEAR),
        dst_va & 0xffffffff,
        (dst_va >> 32) & 0xffffffff,
        count_field,
        expect,
      ]
    # TrustOS: NOP first then WRITE — F32 must see a trivial packet before WRITE_LINEAR.
    # Session #18: also pad trailing NOPs so speculative ring prefetch past the
    # packet does not leave RB_MC_RREQ outstanding on garbage (FETCH still advances
    # through NOPs; execute/retire remains the blocker).
    nop_first = os.environ.get("AMD_BOOT_SDMA_NOP_FIRST", "1") == "1"
    nop_pad = int(os.environ.get("AMD_BOOT_SDMA_NOP_PAD", "32"))
    if pkt_mode == "nop":
      full = pkt + ([0] * max(0, nop_pad))
    else:
      full = ([0] if nop_first else []) + pkt + ([0] * max(0, nop_pad))
    self._sdma_gfx_ring_commit(ring_mem, full,
                               ring_va=ring_va if (use_vram and ring_mem is None) else None)
    wptr_dwords = len(full)
    if not self.sdma_fw_ready():
      self.sdma_enable(True)
      self.wreg(mmSDMA0_GFX_CONTEXT_CNTL, 0)
      # F32 may rewrite CNTL/CONTEXT on unhalt — re-apply TrustOS baseline.
      self._sdma_disable_auto_ctxsw()
      self.wreg(mmSDMA0_GFX_CONTEXT_CNTL, 0)
      self.mmio_sync_safe()
    timeout_s = float(os.environ.get("AMD_BOOT_SDMA_PROBE_TIMEOUT_S", "5"))
    deadline = time.time() + timeout_s
    write_ok = False
    srbm_ok = False
    last_val = sentinel
    while time.time() < deadline:
      if pkt_mode == "srbm":
        dummy = self.rreg(mmSDMA0_GFX_DUMMY_REG)
        if dummy == dummy_expect:
          srbm_ok = True
          write_ok = True
          last_val = dummy
          break
        last_val = dummy
      elif pkt_mode == "nop":
        rptr_now = self.rreg(mmSDMA0_GFX_RB_RPTR) >> 2
        if rptr_now >= 1:
          write_ok = True
          last_val = rptr_now
          break
      else:
        if dst_mem is not None:
          last_val = struct.unpack('<I', bytes(dst_mem[dst_off:dst_off + 4]))[0]
        else:
          with contextlib.suppress(Exception):
            last_val = struct.unpack('<I', self.vram_mm_read(dst_va, 4))[0]
        if last_val == expect:
          write_ok = True
          break
      # Session #16: RPTR_FETCH advance proves host ring READ works (partial win).
      fetch = self.rreg(0x340a)
      if pkt_mode == "write" and fetch >= (wptr_dwords << 2) and last_val == expect:
        write_ok = True
        break
      if use_vram and (self.rreg(mmSDMA0_GFX_RB_RPTR) >> 2) >= wptr_dwords:
        write_ok = True
        break
      time.sleep(0.001)
    rptr = self.rreg(mmSDMA0_GFX_RB_RPTR) >> 2
    wptr = wptr_dwords
    status = self.rreg(mmSDMA0_STATUS_REG)
    status2 = self.rreg(mmSDMA0_STATUS2_REG)
    # TrustOS diag: internal fetch ptr can move while RB_RPTR stays 0.
    rptr_fetch = self.rreg(0x340a)  # mmSDMA0_RB_RPTR_FETCH
    fetch_ok = rptr_fetch >= (wptr << 2)
    f32_ip = (status2 & SDMA0_STATUS2_REG__F32_INSTR_PTR_MASK) >> SDMA0_STATUS2_REG__F32_INSTR_PTR__SHIFT
    cmd_op = (status2 & SDMA0_STATUS2_REG__CMD_OP_MASK) >> SDMA0_STATUS2_REG__CMD_OP__SHIFT
    result = {
      **pte,
      "ring_va": ring_va,
      "dst_va": dst_va,
      "dst_paddr": dst_paddrs[0],
      "write_ok": write_ok,
      "srbm_ok": srbm_ok,
      "pkt_mode": pkt_mode,
      "fetch_ok": fetch_ok,
      "dst_value": last_val,
      "expect": expect if pkt_mode == "write" else (dummy_expect if pkt_mode == "srbm" else 1),
      "rptr_dw": rptr,
      "wptr_dw": wptr,
      "rptr_fetch": rptr_fetch,
      "ring_drained": rptr >= wptr,
      "f32_cntl": self.rreg(mmSDMA0_F32_CNTL),
      "rb_cntl": self.rreg(mmSDMA0_GFX_RB_CNTL),
      "sdma_cntl": self.rreg(mmSDMA0_CNTL),
      "ib_cntl": self.rreg(mmSDMA0_GFX_IB_CNTL),
      "context_cntl": self.rreg(mmSDMA0_GFX_CONTEXT_CNTL),
      "status_reg": status,
      "status2": status2,
      "f32_instr_ptr": f32_ip,
      "cmd_op": cmd_op,
      "power_cntl": self.rreg(mmSDMA0_POWER_CNTL),
      "freeze": self.rreg(mmSDMA0_FREEZE),
      "dummy_reg": self.rreg(mmSDMA0_GFX_DUMMY_REG),
      "doorbell": self.rreg(mmSDMA0_GFX_DOORBELL),
      "agp_start": self.agp_start,
      "vram_start": self.vram_start,
      "sys_apr_lo": self.rreg(mmMC_VM_SYSTEM_APERTURE_LOW_ADDR),
      "sys_apr_hi": self.rreg(mmMC_VM_SYSTEM_APERTURE_HIGH_ADDR),
      "sys_apr_def": self.rreg(mmMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR),
    }
    if self._sdma_rptr_wb_mem is not None:
      result["rptr_wb"] = struct.unpack('<I', bytes(self._sdma_rptr_wb_mem[0:4]))[0]
    else:
      result["rptr_wb"] = None
    result["engine_idle"] = bool(status & SDMA0_STATUS_REG__IDLE_MASK)
    result["mc_rd_idle"] = bool(status & SDMA0_STATUS_REG__MC_RD_IDLE_MASK)
    result["rb_mc_rreq_idle"] = bool(status & 0x20000)  # RB_MC_RREQ_IDLE
    result["mc_rd_ret_stall"] = bool(status & 0x200000)  # MC_RD_RET_STALL
    result["packet_ready"] = bool(status & (1 << 12))
    result["ex_idle"] = bool(status & (1 << 10))
    result["mc_wr_idle"] = bool(status & (1 << 13))
    # oss_3_0: bit22 = MC_RD_NO_POLL_IDLE (idle flag), NOT a write-return stall.
    result["mc_rd_no_poll_idle"] = bool(status & (1 << 22))
    result["ctx_status"] = self.rreg(0x3491)  # mmSDMA0_GFX_CONTEXT_STATUS
    result["ctx_expired"] = bool(result["ctx_status"] & 0x8)
    result["ctx_selected"] = bool(result["ctx_status"] & 0x1)
    result["rb_priv"] = bool(result["rb_cntl"] & SDMA0_GFX_RB_CNTL__RB_PRIV_MASK)
    result["chicken"] = self.rreg(mmSDMA0_CHICKEN_BITS)
    result["phase0"] = self.rreg(0x3414)
    print(f"sdma_probe pkt={pkt_mode} write_ok={write_ok} srbm_ok={srbm_ok} "
          f"dst={last_val:#x} expect={result['expect']:#x} "
          f"rptr_dw={rptr} wptr_dw={wptr} fetch={rptr_fetch:#x} "
          f"ring_drained={result['ring_drained']} "
          f"ring_va={ring_va:#x} dst_paddr={dst_paddrs[0]:#x} "
          f"vram={self.vram_start:#x} agp={self.agp_start:#x} "
          f"CNTL={result['sdma_cntl']:#x} RB={result['rb_cntl']:#x} "
          f"RB_PRIV={result['rb_priv']} CTX={result['context_cntl']:#x} "
          f"IB={result['ib_cntl']:#x} DOOR={result['doorbell']:#x} "
          f"status={status:#x} ST2={status2:#x} IP={f32_ip} CMD={cmd_op:#x} "
          f"POWER={result['power_cntl']:#x} FRZ={result['freeze']:#x} "
          f"DUMMY={result['dummy_reg']:#x} "
          f"idle={result['engine_idle']} "
          f"PKT_RDY={result['packet_ready']} EX_IDLE={result['ex_idle']} "
          f"MC_WR_IDLE={result['mc_wr_idle']} "
          f"mc_rd_idle={result['mc_rd_idle']} rb_rreq_idle={result['rb_mc_rreq_idle']} "
          f"rd_ret_stall={result['mc_rd_ret_stall']} "
          f"CTX={result['ctx_status']:#x} EXP={result['ctx_expired']} "
          f"CHICKEN={result['chicken']:#x} PHASE0={result['phase0']:#x} "
          f"RPTR_WB={result['rptr_wb']} "
          f"SYS_APR={result['sys_apr_lo']:#x}/{result['sys_apr_hi']:#x}/{result['sys_apr_def']:#x}",
          flush=True)
    if not write_ok:
      if pkt_mode == "srbm":
        print("sdma_probe: SRBM_WRITE did not update DUMMY_REG — F32 execute stuck",
              flush=True)
      elif pkt_mode == "nop":
        print("sdma_probe: NOP did not advance RB_RPTR — F32 retire stuck",
              flush=True)
      else:
        print("sdma_probe: WRITE_LINEAR did not update dst — execute or host DMA failed",
              flush=True)
      # Outstanding ring-fetch (RB_MC_RREQ_IDLE=0) or MC_RD_RET_STALL — soft-reset
      # so a wedged completion cannot trip delayed apciec panics.
      if not result["rb_mc_rreq_idle"] or result["mc_rd_ret_stall"] or not result["mc_rd_idle"]:
        print("sdma_probe: ring MC request outstanding/stalled — soft-resetting SDMA",
              flush=True)
        self.sdma_soft_reset()
        result["status_after_reset"] = self.rreg(mmSDMA0_STATUS_REG)
        print(f"sdma_probe: post-reset status={result['status_after_reset']:#x}", flush=True)
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
    # sdma_v3_0_enable: when halting, stop the gfx rings FIRST so the next unhalt
    # never fetches a stale RB_BASE (panic #7). RB_ENABLE / RB_BASE are left zeroed,
    # so unhalting here is safe until _sdma_gfx_ring_setup programs a real ring.
    if not enable:
      for off in (0, SDMA1_REG_OFFSET):
        self._sdma_gfx_ring_disable(off)
    for off in (0, SDMA1_REG_OFFSET):
      reg = mmSDMA0_F32_CNTL + off
      tmp = self.rreg(reg)
      if enable:
        tmp &= ~SDMA_F32_CNTL_HALT
      else:
        tmp |= SDMA_F32_CNTL_HALT
      self.wreg(reg, tmp)
    time.sleep(0.05)

  def load_sdma_firmware_only(self, unhalt: bool = False):
    """Upload ONLY SDMA0/SDMA1 ucode via MMIO, leaving a live CP/MEC untouched.

    fw-sdma on a hot GPU must not re-halt/re-upload the running MEC (panic #7 was a
    full ~200s compute-fw re-bootstrap on USB4). SDMA ucode is small and independent:
    halt SDMA (with ring teardown), stream the two ucode blobs, keep F32 halted by
    default. The GART ring is only programmed + unhalted later in _sdma_gfx_ring_setup
    (option 4 in the fix plan) so the engine never fetches before a valid RB_BASE."""
    want = (
      (UCODE_ID_SDMA0, "polaris10_sdma.bin", mmSDMA0_UCODE_ADDR, mmSDMA0_UCODE_DATA),
      (UCODE_ID_SDMA1, "polaris10_sdma1.bin", mmSDMA1_UCODE_ADDR, mmSDMA1_UCODE_DATA),
    )
    self.sdma_enable(False)  # gfx_stop (RB/IB off, base zeroed) + F32 HALT
    self._sdma_disable_auto_ctxsw()
    loaded = []
    for _ucode_id, name, addr_reg, data_reg in want:
      blob = self.fw(name)
      words, version = self._fw_ucode_words(blob)
      if int(os.environ.get("DEBUG", "0")):
        print(f"polaris: direct sdma ucode {name} words={len(words)} ver={version:#x}", flush=True)
      self._mmio_load_ucode(addr_reg, data_reg, words, version, label=name)
      loaded.append(name)
    self._sdma_fw_resident = True
    if unhalt:
      self.disable_gpu_interrupts("pre-unhalt-sdma")
      self.sdma_enable(True)
    print(f"polaris: SDMA-only firmware loaded ({', '.join(loaded)}) "
          f"unhalt={unhalt} F32_CNTL={self.rreg(mmSDMA0_F32_CNTL):#x}", flush=True)

  def sdma_fw_resident(self) -> bool:
    """Best-effort: SDMA ucode present. F32 unhalted proves it; else trust our flag
    set by an upload earlier in this process (GPU state persists across CLI runs but
    the object does not, so cold callers should just upload)."""
    return self.sdma_fw_ready() or getattr(self, "_sdma_fw_resident", False)

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
                        UCODE_ID_CP_MEC_MASK, 0),  # body only — see below
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
      # Polaris/VI: SMC LoadUcodes splits mec.bin into MEC body + JT1/JT2.
      # Direct MMIO of the FULL blob (body||JT) into ME1_UCODE puts JT bytes
      # into instruction RAM and leaves JT unloaded — MEC runs but never
      # fetches rings. Strip JT for direct path (session #21).
      if ucode_id == UCODE_ID_CP_MEC and os.environ.get("AMD_BOOT_MEC_STRIP_JT", "1") == "1":
        _uo, _usz, _ver, jt_off, jt_sz = parse_gfx_fw(blob)
        if 0 < jt_off < len(words):
          if int(os.environ.get("DEBUG", "0")):
            print(f"polaris: MEC strip JT jt_off={jt_off} jt_sz={jt_sz} "
                  f"body_words={jt_off} (was {len(words)})", flush=True)
          words = words[:jt_off]
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
    if fw_mask & (UCODE_ID_SDMA0_MASK | UCODE_ID_SDMA1_MASK):
      self._sdma_fw_resident = True
    if fw_mask & (UCODE_ID_SDMA0_MASK | UCODE_ID_SDMA1_MASK):
      self._sdma_fw_resident = True
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
    """Whether LoadUcodes is safe.

    Linux needs a CPU-writable VRAM path for the SMC TOC — dead on this eGPU.
    Session #18/#21: AGP-hosted TOC works for SDMA (SMC DMA via AGP→PCIe). Allow
    that layout without BAR0. GART-sysmem also OK once walker is proven."""
    bar0_ok = self.probe_bar0_writes()
    mm_ok = self.probe_vram_mm_writes() if not bar0_ok else False
    trained = self.vram_trained()
    layout = os.environ.get("AMD_BOOT_FW_LAYOUT", "auto")
    if bar0_ok or mm_ok:
      return True, f"trained={trained} bar0={bar0_ok} mm_index={mm_ok}", bar0_ok, mm_ok
    if layout == "agp" or os.environ.get("AMD_BOOT_LOADUCODES_UNTRAINED", "0") == "1":
      return True, f"agp/forced TOC (trained={trained} bar0=0) — SMC DMA via AGP", False, False
    return False, (
      f"VRAM trained={trained} but no CPU-visible VRAM data path (BAR0+MM_INDEX both "
      f"dead on this TinyGPU/USB4 transport) — SMC cannot DMA the firmware TOC/header; "
      f"LoadUcodes will hang and drop the USB4 link. Use AMD_BOOT_FW_LAYOUT=agp "
      f"AMD_BOOT_LOADUCODES_UNTRAINED=1 (proven for SDMA) or fix BAR0."
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
        # Session #21: AGP TOC proven for SDMA LoadUcodes; prefer over GTT.
        layout = "agp"
        if int(os.environ.get("DEBUG", "0")):
          print(f"polaris: auto layout agp ({reason})", flush=True)
    if layout == "gtt" and not allowed and os.environ.get("AMD_BOOT_LOADUCODES_UNTRAINED", "0") != "1":
      raise RuntimeError(
        "GTT-only firmware layout unusable without VRAM path — use AMD_BOOT_FW_LAYOUT=agp")
    if layout == "agp":
      # Ensure AGP aperture programmed before SMC DMA.
      if not self.agp_start:
        self.gmc_sw_init()
      self.mc_program_apertures()
      self.mc_setup_tlb_apertures()
      self.gmc_program_vm_l2()
      self.vm_context0_disable()
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

  def init_sh_mem_vmid0(self):
    """gfx_v8_0_constants_init VMID0 SH_MEM — required for flat_load/store.

    Without this, DISPATCH can retire while shader memory ops go nowhere.
    Linux: DEFAULT_MTYPE=UC, APE1_MTYPE=UC, ALIGNMENT=UNALIGNED, BASES=0,
    APE1 disabled (base=1, limit=0)."""
    mmSH_MEM_BASES = 0x230a
    mmSH_MEM_APE1_BASE = 0x230b
    mmSH_MEM_APE1_LIMIT = 0x230c
    mmSH_MEM_CONFIG = 0x230d
    MTYPE_UC = 3
    ALIGN_UNALIGNED = 3  # SH_MEM_ALIGNMENT_MODE_UNALIGNED
    # bits: ALIGNMENT_MODE[4:3], DEFAULT_MTYPE[7:5], APE1_MTYPE[10:8]
    cfg = ((ALIGN_UNALIGNED & 3) << 3) | ((MTYPE_UC & 7) << 5) | ((MTYPE_UC & 7) << 8)
    # SRBM VMID select is the 4th arg of srbm_select(me, pipe, queue, vmid)
    self.srbm_select(0, 0, 0, 0)
    self.wreg(mmSH_MEM_CONFIG, cfg)
    self.wreg(mmSH_MEM_BASES, 0)
    self.wreg(mmSH_MEM_APE1_BASE, 1)
    self.wreg(mmSH_MEM_APE1_LIMIT, 0)
    self.mmio_sync_safe()
    if int(os.environ.get("DEBUG", "0")):
      print(f"polaris: SH_MEM_CONFIG={self.rreg(mmSH_MEM_CONFIG):#x} "
            f"BASES={self.rreg(mmSH_MEM_BASES):#x} (VMID0 flat/UC)", flush=True)

  def enable_compute(self):
    self.disable_gpu_interrupts("pre-enable-compute")
    self.init_sh_mem_vmid0()
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
    poll = max(0.01, float(os.environ.get("AMD_BOOT_UCODE_POLL_MS", "20")) / 1000.0)
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
CP_HQD_PERSISTENT_STATE_PRELOAD_SIZE_MASK = 0x3ff00
CP_HQD_PERSISTENT_STATE_PRELOAD_SIZE__SHIFT = 8
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
  # KFD kgd_hqd_load always sets DOORBELL_EN=1 before ACTIVE — arms MEC to
  # watch WPTR even when we never ring BAR2 (TinyGPU MSI panic). Override:
  # AMD_BOOT_HQD_DOORBELL_EN=0|1 (default 1).
  door_en = os.environ.get("AMD_BOOT_HQD_DOORBELL_EN", "1") == "1"
  if not door_en:
    door_en = not boot_no_doorbell()
  dbell = _reg_field(dbell, CP_HQD_PQ_DOORBELL_EN_MASK, 30, 1 if door_en else 0)
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
  # gfx_8_0: PRELOAD_SIZE is bits[17:8] (mask 0x3ff00), PRELOAD_REQ is bit0.
  # Bug (session #20): old mask 0xff/shift0 wrote 0x53 into the low byte and
  # permanently set PRELOAD_REQ → MEC hung on MQD DMA (RPTR stuck forever).
  ps = _reg_field(ps, CP_HQD_PERSISTENT_STATE_PRELOAD_SIZE_MASK,
                  CP_HQD_PERSISTENT_STATE_PRELOAD_SIZE__SHIFT, 0x53)
  ps &= ~CP_HQD_PERSISTENT_STATE_PRELOAD_REQ_MASK
  if activate and os.environ.get("AMD_BOOT_HQD_PRELOAD", "0") == "1":
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
  """Port of gfx_v8_0_mqd_commit (ref/linux gfx_v8_0.c).

  Linux clears CP_PQ_WPTR_POLL EN (doorbell notifies MEC). TinyGPU has no BAR2
  doorbell — without poll EN, MEC never notices MMIO PQ_WPTR / shadow updates
  (KIQ/KCQ RPTR stuck at 0). Re-enable poll when AMD_BOOT_NO_DOORBELL."""
  if deactivate:
    boot.deactivate_hqd(cq.me, cq.pipe, cq.queue)
  boot.srbm_select(cq.me, cq.pipe, cq.queue, 0)
  boot.wreg(mmCP_PQ_WPTR_POLL_CNTL, boot.rreg(mmCP_PQ_WPTR_POLL_CNTL) & ~mmCP_PQ_WPTR_POLL_CNTL__EN_MASK)
  for reg in range(mmCP_HQD_VMID, mmCP_HQD_EOP_CONTROL + 1):
    boot.wreg(reg, mqd.hqd(reg))
  boot.wreg(mmCP_HQD_EOP_RPTR, mqd.hqd(mmCP_HQD_EOP_RPTR))
  boot.wreg(mmCP_HQD_EOP_WPTR, mqd.hqd(mmCP_HQD_EOP_WPTR))
  boot.wreg(mmCP_HQD_EOP_WPTR_MEM, mqd.hqd(mmCP_HQD_EOP_WPTR_MEM))
  for reg in range(mmCP_HQD_EOP_EVENTS, mmCP_HQD_ERROR + 1):
    boot.wreg(reg, mqd.hqd(reg))
  for reg in range(mmCP_MQD_BASE_ADDR, mmCP_HQD_ACTIVE + 1):
    boot.wreg(reg, mqd.hqd(reg))
  if boot_no_doorbell() and os.environ.get("AMD_BOOT_PQ_WPTR_POLL", "0") == "1":
    # Default OFF: live eGPU saw POLL EN corrupt PQ_WPTR→0x3fff (bad shadow
    # read) while RPTR stayed 0. Prefer DOORBELL_HIT + MMIO WPTR instead.
    # Enable global poll + queue mask (gfx9+ KFD writes POLL_CNTL1; VI same reg).
    poll = boot.rreg(mmCP_PQ_WPTR_POLL_CNTL)
    boot.wreg(mmCP_PQ_WPTR_POLL_CNTL, poll | mmCP_PQ_WPTR_POLL_CNTL__EN_MASK)
    boot.wreg(mmCP_PQ_WPTR_POLL_CNTL1, 0xffffffff)
    if int(os.environ.get("DEBUG", "0")):
      print(f"polaris: CP_PQ_WPTR_POLL_CNTL={boot.rreg(mmCP_PQ_WPTR_POLL_CNTL):#x} "
            f"POLL1={boot.rreg(mmCP_PQ_WPTR_POLL_CNTL1):#x} "
            f"wptr_poll={cq.wptr_gpu:#x} door={mqd.hqd(mmCP_HQD_PQ_DOORBELL_CONTROL):#x}",
            flush=True)
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
    # Session #19: SDMA WRITE_LINEAR proved AGP host DMA. Prefer AGP for MEC
    # ring/MQD/EOP (no GART walk) — same path as sdma-probe. Override:
    # AMD_BOOT_COMPUTE_AGP=0 → GART; =1 force AGP; default auto (AGP if no VRAM).
    agp_mode = os.environ.get("AMD_BOOT_COMPUTE_AGP", "auto")
    no_vram = not boot.probe_bar0_writes()
    if agp_mode == "1" or (agp_mode == "auto" and no_vram):
      self._mem = "agp"
    elif boot.gart_pte_sysmem is not None or no_vram:
      self._mem = "gtt"
    else:
      self._mem = "vram"
    self._gtt = self._mem != "vram"  # host-backed (agp or gtt)
    self.ring_off = self.mqd_off = self.eop_off = self.wptr_off = 0
    self.ring_gpu = self.mqd_gpu = self.eop_gpu = self.wptr_gpu = self.rptr_gpu = 0
    self.ring_mem = self.mqd_mem = self.eop_mem = self.wptr_mem = None
    self.wptr = 0

  def _alloc_buf(self, size: int, align=0x1000) -> tuple[int, object | None, int]:
    if self._mem == "agp":
      # Ensure AGP aperture + VMID0 physical (same as sdma AGP probe).
      if not getattr(self.boot, "agp_start", 0):
        self.boot.gmc_sw_init()
      self.boot.mc_program_apertures()
      self.boot.mc_setup_tlb_apertures()
      self.boot.gmc_program_vm_l2()
      self.boot.vm_context0_disable()
      gpu_va, mem, _ = self.boot.alloc_agp_buffer(size)
      return gpu_va, mem, 0
    if self._mem == "gtt":
      gpu_va, mem, _ = self.boot.alloc_gtt_buffer(size, align)
      return gpu_va, mem, 0
    off = self.dev.alloc_vram(size, align)
    return self.dev.vram_gpu_addr(off), None, off

  def _write_bytes(self, off_or_mem, data: bytes, mem=None):
    if mem is not None:
      mem[0:len(data)] = data
      sysmem_dma_flush(mem, len(data))
    else:
      self.dev.upload(off_or_mem, data)

  def _write_ring(self, words: list[int], offset_dwords: int = 0):
    data = struct.pack('<' + 'I' * len(words), *words)
    byte_off = offset_dwords * 4
    if self.ring_mem is not None:
      self.ring_mem[byte_off:byte_off + len(data)] = data
      sysmem_dma_flush(self.ring_mem, byte_off + len(data))
    else:
      self.dev.upload(self.ring_off + byte_off, data)

  def _publish_wptr(self, wptr: int):
    """CPU shadow of wptr for CP poll (gfx_v8_0_ring_set_wptr_compute)."""
    if self.wptr_mem is None:
      return
    self.wptr_mem[64:68] = struct.pack('<I', wptr & 0xffffffff)
    sysmem_dma_flush(self.wptr_mem, 128)

  def _signal_wptr(self, wptr: int, doorbell_index: int):
    """Linux: doorbell + wptr shadow. macOS/TinyGPU: MMIO PQ_WPTR + optional HIT.

    Without BAR2 doorbell, also pulse CP_HQD_PQ_DOORBELL_CONTROL.DOORBELL_HIT
    (bit31) so MEC treats the MMIO WPTR update like a doorbell edge."""
    self._publish_wptr(wptr)
    boot = self.boot
    if not boot_no_doorbell():
      self.dev.ring_doorbell(doorbell_index, wptr)
    if boot_use_mmio_wptr():
      boot.srbm_select(self.me, self.pipe, self.queue, 0)
      boot.wreg(mmCP_HQD_PQ_WPTR, wptr & 0xffffffff)
      if os.environ.get("AMD_BOOT_DOORBELL_HIT", "1") == "1":
        door = boot.rreg(mmCP_HQD_PQ_DOORBELL_CONTROL)
        door |= CP_HQD_PQ_DOORBELL_EN_MASK
        boot.wreg(mmCP_HQD_PQ_DOORBELL_CONTROL, door | 0x80000000)  # HIT
        boot.wreg(mmCP_HQD_PQ_DOORBELL_CONTROL, door & ~0x80000000)
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
      sysmem_dma_flush(self.wptr_mem, 128)
    if int(os.environ.get("DEBUG", "0")):
      print(f"polaris: ComputeQueue mem={self._mem} ring={self.ring_gpu:#x} "
            f"mqd={self.mqd_gpu:#x} eop={self.eop_gpu:#x} wb={wb_gpu:#x}", flush=True)

  def ring_test_scratch(self, timeout_s: float = 2.0) -> bool:
    """gfx_v8_0_ring_test_ring: SET_UCONFIG_REG on SCRATCH_REG0 (ref/linux gfx_v8_0.c).

    Linux writes (mmSCRATCH_REG0 - PACKET3_SET_UCONFIG_REG_START) as the offset
    dword — NOT shifted. Session #21: >>2 made offset 0x10 instead of 0x40 so
    MEC drained the ring (RPTR==WPTR) but never touched SCRATCH_REG0."""
    boot = self.boot
    boot.wreg(mmSCRATCH_REG0, 0xCAFEDEAD)
    # gfx_v8_0_ring_test_ring: offset is register-index delta, not byte/4.
    scratch_idx = mmSCRATCH_REG0 - PACKET3_SET_UCONFIG_REG_START
    pkt = [pkt3(PACKET3_SET_UCONFIG_REG, 1), scratch_idx, 0xDEADBEEF]
    self._ring_commit(pkt, self.doorbell_index)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
      if boot.rreg(mmSCRATCH_REG0) == 0xDEADBEEF:
        return True
      time.sleep(0.001)
    if int(os.environ.get("DEBUG", "0")):
      boot.srbm_select(self.me, self.pipe, self.queue, 0)
      rptr = boot.rreg(mmCP_HQD_PQ_RPTR)
      wptr = boot.rreg(mmCP_HQD_PQ_WPTR)
      err = boot.rreg(mmCP_HQD_ERROR)
      active = boot.rreg(mmCP_HQD_ACTIVE)
      boot.srbm_select(0, 0, 0, 0)
      print(f"polaris: ring_test FAIL SCRATCH={boot.rreg(mmSCRATCH_REG0):#x} "
            f"ACTIVE={active:#x} WPTR={wptr:#x} RPTR={rptr:#x} ERR={err:#x} "
            f"idx={scratch_idx:#x} mem={self._mem} ring={self.ring_gpu:#x}", flush=True)
    return False

  def submit_ib(self, ib_words: list[int], timeout_s: float = 5.0) -> bool:
    """gfx_v8_0_ring_emit_ib_compute: INDIRECT_BUFFER → AGP/GTT IB body.

    Session #21 bug: used opcode 0x10 (NOP) and inlined IB words on the ring.
    Correct packet is PACKET3_INDIRECT_BUFFER (0x3F) with IB GPU addr + VALID."""
    ib_bytes = struct.pack('<' + 'I' * len(ib_words), *ib_words)
    ib_gpu, ib_mem, _ = self._alloc_buf(max(len(ib_bytes), 0x1000))
    self._write_bytes(0, ib_bytes, ib_mem)
    control = INDIRECT_BUFFER_VALID | (len(ib_words) & 0xfffff)
    pkt = [
      pkt3(PKT3_INDIRECT_BUFFER, 2),
      ib_gpu & 0xfffffffc,
      (ib_gpu >> 32) & 0xffff,
      control,
    ]
    self._ring_commit(pkt, self.doorbell_index)
    boot = self.boot
    deadline = time.time() + timeout_s
    drained = False
    while time.time() < deadline:
      boot.srbm_select(self.me, self.pipe, self.queue, 0)
      rptr = boot.rreg(mmCP_HQD_PQ_RPTR)
      wptr = boot.rreg(mmCP_HQD_PQ_WPTR)
      boot.srbm_select(0, 0, 0, 0)
      if rptr == wptr:
        drained = True
        break
      time.sleep(0.001)
    if int(os.environ.get("DEBUG", "0")):
      boot.srbm_select(self.me, self.pipe, self.queue, 0)
      err = boot.rreg(mmCP_HQD_ERROR)
      boot.srbm_select(0, 0, 0, 0)
      print(f"polaris: KCQ submit_ib drained={drained} PQ_WPTR={wptr:#x} "
            f"PQ_RPTR={rptr:#x} ERR={err:#x} ib={ib_gpu:#x} ndw={len(ib_words)}",
            flush=True)
    return drained

  def setup_with_kiq(self, map_queues: bool | None = None):
    """gfx_v8_0_kiq_resume + kcq_resume (ref/linux gfx_v8_0.c)."""
    if map_queues is None:
      map_queues = os.environ.get("AMD_BOOT_KIQ_MAP", "1") == "1"
    boot = self.boot
    boot.deactivate_hqd(self.me, self.pipe, self.queue)
    kcq_mode = os.environ.get("AMD_BOOT_KCQ_DIRECT", "auto")
    direct_kcq = kcq_mode == "1" or (kcq_mode == "auto" and not map_queues)
    # Session #21: for direct KCQ, skip KIQ HQD — two ACTIVE queues with no
    # doorbell can leave MEC1_BUSY without draining either ring. Linux uses
    # KIQ only to MAP_QUEUES; direct path programs KCQ HQD alone.
    skip_kiq = direct_kcq and os.environ.get("AMD_BOOT_SKIP_KIQ", "1") == "1"
    if not skip_kiq:
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
    else:
      kiq = None
      if int(os.environ.get("DEBUG", "0")):
        print("polaris: skip KIQ (AMD_BOOT_SKIP_KIQ=1, direct KCQ)", flush=True)
      # Still tell RLC about the compute queue we will activate (KFD HIQ path).
      boot.kiq_setting(self.me, self.pipe, self.queue)

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

    if not map_queues or kiq is None:
      if int(os.environ.get("DEBUG", "0")):
        print("polaris: KIQ MAP_QUEUES skipped", flush=True)
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



# =============================================================================
# === device + main ===
# =============================================================================
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
    if fw_mask is None:
      fw_mask = int(os.environ.get("AMD_BOOT_FW_MASK", str(FW_RLC_ONLY)), 0)
    b.boot_through_fw_direct(fw_mask, unhalt=unhalt)
    return fw_mask

  def _boot_stage_kiq(self, b, map_queues: bool, skip_fw: bool = False) -> None:
    """Firmware boot + KIQ/KCQ MQD; MAP_QUEUES doorbell optional."""
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
    if self._boot is None:
      self._boot = PolarisBoot(self)
    b = self._boot
    if self.gpu_ready():
      if DEBUG >= 1: print("polaris: CP queue already active, skipping boot")
      return
    if stage == "common":
      b.vi_common_init(); print("stage=common ok"); return
    if stage == "atom":
      b.vi_common_init(); b.enable_vbios_rom()
      run_asic_init_if_needed(b)
      print(f"stage=atom MEMSIZE={b.config_memsize_mb():#x} MISC0={b.rreg(0xa80):#x} "
            f"trained={vram_training_ok(b)}"); return
    if stage == "pre-fw":
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
      mask = self._boot_stage_fw_direct(b, fw_mask=FW_RLC_ONLY, unhalt=False)
      print(f"stage=fw-rlc mask={mask:#x} CP_MEC_CNTL={b.rreg(mmCP_MEC_CNTL):#x} smc={b.smc_running()}"); return
    if stage == "fw-cp":
      mask = self._boot_stage_fw_direct(b, fw_mask=FW_RLC_ONLY | FW_CP_GFX_MASK)
      print(f"stage=fw-cp mask={mask:#x} CP_MEC_CNTL={b.rreg(mmCP_MEC_CNTL):#x}"); return
    if stage == "fw-mec":
      mask = self._boot_stage_fw_direct(b, fw_mask=FW_COMPUTE_MIN, unhalt=False)
      print(f"stage=fw-mec mask={mask:#x} CP_MEC_CNTL={b.rreg(mmCP_MEC_CNTL):#x} "
            f"(upload only — run fw-start to unhalt)"); return
    if stage == "fw-start":
      fw_mask = int(os.environ.get("AMD_BOOT_FW_MASK", str(FW_COMPUTE_MIN)), 0)
      b.unhalt_loaded_firmware(fw_mask)
      print(f"stage=fw-start mask={fw_mask:#x} CP_MEC_CNTL={b.rreg(mmCP_MEC_CNTL):#x}"); return
    if stage == "fw-sdma":
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
      # Cross-process: prior add.py left ME1 running with a stale KCQ MQD → zeros.
      # Warm skip_fw alone fails; must soft-reset + cold LoadUcodes. Use MMIO-only
      # reset (not PCI) — clears HQD/CP without the slow PCI settle path.
      # Same-process re-dispatch is already fast (run_add ~20ms); nvgpu stays
      # instant because GSP/queues persist in one long-lived process.
      if b.compute_fw_loaded():
        print("polaris: hot MEC — MMIO soft-reset then cold LoadUcodes", flush=True)
        self.software_reset(mmio_reset=True, mode="mmio")
        self._boot = PolarisBoot(self)
        b = self._boot
      self._boot_stage_kiq(b, map_queues=False, skip_fw=False)
      cases = getattr(self, "_op_cases", None) or [((1.0, 2.0, 3.0, 4.0), (10.0, 20.0, 30.0, 40.0))]
      for a, b_vals in cases:
        result = self.run_add(a, b_vals)
        expected = [OP(x, y) for x, y in zip(a, b_vals)]
        print(f"stage=add a={list(a)} b={list(b_vals)} result={result}")
        if not all(abs(r - e) < 1e-4 for r, e in zip(result, expected)):
          raise RuntimeError(f"vector-{OP_NAME} failed: expected {expected}, got {result}")
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

    expected = [OP(x, y) for x, y in zip(a, b_vals)]
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
      # float compare: exact for these smoke values; allow tiny noise
      if all(abs(r - e) < 1e-5 for r, e in zip(result, expected)):
        result = list(expected)
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
  assert ADD_SHADER == build_shader(ALU_OP)
  assert build_shader(Gcn3.Op.V_MUL_F32) != ADD_SHADER
  assert len(ib) >= 20
  assert ib[0] >> 30 == PKT_TYPE3
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

  setdefault: caller/env overrides still win. Speed knobs cut cold boot from
  ~10s (30×100ms mmio_settle + 100ms ucode poll) to ~3s without changing the
  LoadUcodes path that nvgpu avoids by keeping GSP alive across launches."""
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
    # Boot latency (USB4 drain_mmio is the real barrier; long sleeps were padding)
    "AMD_MMIO_SETTLE_ROUNDS": "5",
    "AMD_MMIO_SETTLE_MS": "15",
    "AMD_MMIO_SETTLE_ROUNDS_LIGHT": "2",
    "AMD_MMIO_SETTLE_MS_LIGHT": "5",
    "AMD_BOOT_UCODE_POLL_MS": "20",
    "AMD_BOOT_SMC_SETTLE_MS": "30",
    "AMD_BOOT_LOADUCODES_SETTLE_MS": "30",
    "AMD_BOOT_SMC_POLL_MS": "8",
    "AMD_BOOT_DOORBELL_SETTLE_MS": "2",
    "AMD_BOOT_ADD_WAIT_S": "1",
  }
  for k, v in defaults.items():
    os.environ.setdefault(k, v)


def parse_vec4(s: str) -> tuple[float, float, float, float]:
  parts = [float(x) for x in s.replace(" ", "").split(",") if x]
  if len(parts) != 4:
    raise SystemExit(f"need 4 floats, got {parts!r} from {s!r}")
  return (parts[0], parts[1], parts[2], parts[3])


def parse_op_cases(argv: list[str]) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
  """CLI: default one case; --a/--b one case; --test several; --cases a:b;a:b"""
  if "--test" in argv:
    return [
      ((1.0, 2.0, 3.0, 4.0), (10.0, 20.0, 30.0, 40.0)),
      ((0.0, 0.0, 0.0, 0.0), (5.0, 5.0, 5.0, 5.0)),
      ((-1.0, 2.0, -3.0, 4.0), (2.0, -2.0, 2.0, -2.0)),
      ((0.5, 1.5, 2.5, 3.5), (2.0, 2.0, 2.0, 2.0)),
      ((1e3, -1e3, 0.125, -0.5), (2.0, 0.5, 8.0, -4.0)),
    ]
  a = (1.0, 2.0, 3.0, 4.0)
  b = (10.0, 20.0, 30.0, 40.0)
  for i, arg in enumerate(argv):
    if arg == "--a" and i + 1 < len(argv):
      a = parse_vec4(argv[i + 1])
    elif arg.startswith("--a="):
      a = parse_vec4(arg.split("=", 1)[1])
    elif arg == "--b" and i + 1 < len(argv):
      b = parse_vec4(argv[i + 1])
    elif arg.startswith("--b="):
      b = parse_vec4(arg.split("=", 1)[1])
    elif arg == "--cases" and i + 1 < len(argv):
      out = []
      for pair in argv[i + 1].split(";"):
        left, right = pair.split(":")
        out.append((parse_vec4(left), parse_vec4(right)))
      return out
    elif arg.startswith("--cases="):
      out = []
      for pair in arg.split("=", 1)[1].split(";"):
        left, right = pair.split(":")
        out.append((parse_vec4(left), parse_vec4(right)))
      return out
  return [(a, b)]


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
    if stage == "add":
      os.environ["AMD_BOOT_ADD"] = "1"
    if stage == "kcq-ring-test":
      os.environ["AMD_BOOT_RING_TEST"] = "1"
    cases = parse_op_cases(sys.argv[1:])
    if stage == "add":
      print(f"shader_bytes={len(ADD_SHADER)} op={OP_NAME} cases={len(cases)}", flush=True)
    try:
      dev = PolarisDevice()
      if stage == "add":
        dev._op_cases = cases
      dev.boot(stage=stage)
    except RuntimeError as e:
      print(str(e), file=sys.stderr)
      sys.exit(1)
    return

if __name__ == "__main__":
  main()
