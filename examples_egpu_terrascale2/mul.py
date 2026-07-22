#!/usr/bin/env python3
"""GPU vector multiply on TeraScale 2 / Redwood. Thin wrapper over add.py.

Ponytail: add.py already has the full Evergreen LS compute dispatch.
mul.py only swaps the shader (fmul instead of fadd) and the expected-result
lambda.  Everything else — firmware, golden registers, RAT setup, cache
flush, fence — is identical and reused.
"""
import sys, pathlib, subprocess
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import add as _add

MUL_LL = pathlib.Path(__file__).with_name("redwood_mul.ll")

def _compile_mul_blob() -> bytes:
  """Compile the mul LS kernel and verify its VTX/ALU/RAT instruction path."""
  llc = _add.r600_llc()
  if llc is None or not MUL_LL.is_file():
    raise RuntimeError("missing R600 compiler or redwood_mul.ll")
  asm = subprocess.run(
    [llc, "-march=r600", "-mcpu=redwood", "-filetype=asm", str(MUL_LL), "-o", "-"],
    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
  )
  if asm.returncode:
    raise RuntimeError(f"Redwood mul assembly failed:\n{asm.stderr.strip()}")
  if asm.stdout.count("MUL") != 4 or "VTX_READ_128" not in asm.stdout or "MEM_RAT_CACHELESS STORE_RAW" not in asm.stdout:
    raise RuntimeError("Redwood mul kernel lacks expected VTX reads, four MULs, or RAT store")
  obj = subprocess.run(
    [llc, "-march=r600", "-mcpu=redwood", "-filetype=obj", str(MUL_LL), "-o", "-"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
  )
  if obj.returncode:
    raise RuntimeError(f"Redwood mul object failed:\n{obj.stderr.decode().strip()}")
  return _add.elf_text(obj.stdout, expected_size=144)

# Monkey-patch the two things that differ from add:
#   1. Which shader to compile and load
#   2. How to compute the expected result
_add.compile_redwood_add_blob = _compile_mul_blob
_add.OP = lambda x, y: x * y
_add.OP_NAME = "mul"

if __name__ == "__main__":
  _add.main()
