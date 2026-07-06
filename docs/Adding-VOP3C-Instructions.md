# Adding VOP3C Instructions to MGPUSim

## Instruction Format Overview

VOP3C (VOP3 with a Constant) instructions are encoded as 64-bit (8-byte) instructions:
```
63                                           0
┌──────────────────────────────────────────────────────────────┐
│  VOP3C Encoding (MIMG/DS/MUBUF/MEM cycles)                 │
│  opcode[8] | vdst[8] | src0[7] | offen | idxen | const      │
│  abs[1] | neg[1] | opsel[4] | 0x3D | 1 | 1 | vcc/en         │
└──────────────────────────────────────────────────────────────┘
```

## Step 1: Add Instruction to Decode Table

File: `amd/insts/decodetable.go`

```go
// Add VOP3C instruction type
d.addInstType(&InstType{
    Name:       "v_add_co_u32",
    Opcode:     0x68,  // V_ADD_CO_U32_e32 encoding
    Format:     FormatTable[VOP3b],  // VOP3b format for VOP3C
    ExeUnit:    ExeUnitVALU,
    WaveSize:   64,
    ...
})
```

## Step 2: Add Instruction to Format Table

File: `amd/insts/format.go`

```go
// VOP3C uses VOP3b format (64-bit with optional vcc)
const VOP3b InstFormat = 15  // existing VOP3b format
```

## Step 3: Implement Instruction Handler

File: `amd/emu/aluvop3b.go` (or create new file)

```go
package emu

// runVaddCoU32VOP3b handles V_ADD_CO_U32_e32 instruction
// Encoding: VOP3b with VCC destination
// Syntax: v_add_co_u32_e32 vdst, vcc, src0, src1, vcc_in
func (u *ALUImpl) runVaddCoU32VOP3b(state InstEmuState) {
    inst := state.Inst()
    
    // Get operands
    src0 := asUint32(state.Src0())
    src1 := asUint32(state.Src1())
    
    // Execute addition
    result := src0 + src1
    
    // Write result to VGPR
    state.Vdst().WriteUint32(result)
    
    // Handle VCC (carry flag for 64-bit operations)
    // V_ADD_CO_U32 sets VCC if overflow occurs
    overflow := (result < src0) || (result < src1)
    vcc := uint64(0)
    if overflow {
        vcc = 1
    }
    state.Vcc().WriteUint64(vcc)
}
```

## Step 4: Register in Instruction Switch

File: `amd/emu/aluvop3b.go` (in switch statement)

```go
func (u *ALUImpl) runVOP3B(state InstEmuState) {
    inst := state.Inst()
    
    u.vop3Preprocess(state)
    
    switch inst.Opcode {
    case 0x68: // V_ADD_CO_U32_e32
        u.runVaddCoU32VOP3b(state)
    case 0x69: // V_ADDC_CO_U32_e32
        u.runVaddcCoU32VOP3b(state)
    // ... more cases
    }
}
```

## Step 5: Add Test Case

File: `amd/emu/aluvop3b_test.go`

```go
package emu

import "testing"

func TestVaddCoU32VOP3b(t *testing.T) {
    // Setup wavefront
    wf := newTestWavefront()
    
    // Set inputs
    wf.VGPR[0] = uint64(0xFFFFFFFF)
    wf.VGPR[1] = uint64(2)
    
    // Create instruction
    inst := &insts.Inst{
        Opcode:   0x68,
        Vdst:     2,
        Src0:     0,
        Src1:     1,
    }
    
    // Execute
    alu := newALU(wf)
    alu.runVaddCoU32VOP3b(&testState{inst: inst, wf: wf})
    
    // Verify result
    result := wf.VGPR[2]
    expected := uint64(1) // 0xFFFFFFFF + 2 = 1 (overflow)
    
    if result != expected {
        t.Errorf("V_ADD_CO_U32: expected 0x%x, got 0x%x", expected, result)
    }
    
    // Verify VCC (should be set due to overflow)
    if wf.SGPR[inst.Vcc()].Value() != 1 {
        t.Errorf("VCC should be set for overflow")
    }
}
```

## Common VOP3C Instructions to Add

| Opcode | Name | Description |
|--------|------|-------------|
| 0x68 | V_ADD_CO_U32_e32 | Add with carry out |
| 0x69 | V_ADDC_CO_U32_e32 | Add with carry in/out |
| 0x6A | V_SUB_CO_U32_e32 | Subtract with carry out |
| 0x6B | V_SUBB_CO_U32_e32 | Subtract with borrow |

## File Locations

```
mgpusim/
└── amd/
    ├── insts/           # Instruction decoding
    │   ├── decodetable.go    # Add to decode table
    │   └── format.go         # Define format constants
    └── emu/             # Instruction emulation
        ├── aluvop3b.go       # Add handlers
        └── aluvop3b_test.go  # Add tests
```

## Build and Test

```bash
cd /home/fedora/amdgpu/mgpusim

# Build
go build ./...

# Run tests
go test ./amd/emu/... -v -run TestVaddCoU32

# Run integration test
./custom_test -kernel your_kernel.hsaco -name kernel_name
```
