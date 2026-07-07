#!/bin/bash
# Capture amdgpu MMIO writes during RX570 hotplug asic_init on Linux.
# Run on Pi 5 / any machine with amdgpu + eGPU. Requires root.
#
# Usage:
#   sudo ./linux_trace_asic_init.sh [out.json]
#
# Alternative: patch amdgpu WREG32 macro temporarily with trace_printk(reg, val).

set -euo pipefail
OUT="${1:-asic_init_trace.json}"

if [[ ! -d /sys/kernel/debug/tracing ]]; then
  echo "debugfs tracing not mounted"; exit 1
fi

TRACE=/sys/kernel/debug/tracing
echo "Unbind GPU, then rebind to trigger amdgpu probe + asic_init..."
echo "  echo 0000:bus:00.0 > /sys/bus/pci/drivers/amdgpu/unbind"
echo "  echo 0000:bus:00.0 > /sys/bus/pci/drivers/amdgpu/bind"
echo

# Generic writeback tracer on mmio writes needs a custom kprobe on amdgpu MMIO.
# Practical approach: use amdgpu debug=1 + dmesg grep, or bpftrace:
if command -v bpftrace >/dev/null 2>&1; then
  echo "Starting bpftrace (Ctrl-C after GPU init completes)..."
  sudo bpftrace -e '
    kprobe:amdgpu_device_asic_init { printf("asic_init start\n"); }
    kprobe:amdgpu_atom_execute_table { printf("atom_execute_table idx=%d\n", arg1); }
  ' &
  BPFPID=$!
  trap "kill $BPFPID 2>/dev/null || true" EXIT
fi

echo "Reload amdgpu and watch dmesg:"
echo "  dmesg -w | grep -E 'GPU posting|asic_init|MC_SEQ|CONFIG_MEMSIZE|amdgpu.*VRAM'"
echo
echo "For full register trace, add to drivers/gpu/drm/amd/amdgpu/amdgpu_device.c WREG32:"
echo '  trace_printk("WREG32 %x %x\n", reg, val);'
echo "Then: cat $TRACE/trace_pipe | tee mmio.trace"
echo
echo "Convert trace to JSON for eGPU replay:"
echo '  python3 -c "
import re, json, sys
regs = []
for line in open(\"mmio.trace\"):
  m = re.search(r\"WREG32 ([0-9a-f]+) ([0-9a-f]+)\", line, re.I)
  if m: regs.append({\"reg\": int(m.group(1),16), \"val\": int(m.group(2),16)})
json.dump({\"regs\": regs}, open(\"'$OUT'\", \"w\"), indent=2)
"'
