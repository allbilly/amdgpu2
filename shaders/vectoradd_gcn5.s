; Simple vector add for GCN5 (Vega)
; Compile: clang -c -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx900 -o vectoradd_gcn5.o vectoradd_gcn5.s
; Disassemble: llvm-objdump -d -m=amdgpu --print-imm-hex vectoradd_gcn5.o

.text
.p2align 8

.globl vectoradd_gcn5
.type vectoradd_gcn5,@function

vectoradd_gcn5:
    ; Get workitem ID
    v_mov_b32 v0, v0
    
    ; Load two floats from global memory
    flat_load_dword v1, v[0:1] offset:0
    flat_load_dword v2, v[0:1] offset:4
    
    ; Add: v3 = v1 + v2
    v_add_f32 v3, v1, v2
    
    ; Store result
    flat_store_dword v[0:1], v3 offset:8
    
    s_endpgm
