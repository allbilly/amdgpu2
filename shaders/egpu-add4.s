  .text
  .globl start
  .p2align 8
start:
  v_mov_b32 v0, 0
  v_mov_b32 v1, 4
  v_mov_b32 v2, 8
  v_mov_b32 v3, 12
  global_load_dword v4, v0, s[2:3]
  global_load_dword v5, v1, s[2:3]
  global_load_dword v6, v2, s[2:3]
  global_load_dword v7, v3, s[2:3]
  global_load_dword v8, v0, s[4:5]
  global_load_dword v9, v1, s[4:5]
  global_load_dword v10, v2, s[4:5]
  global_load_dword v11, v3, s[4:5]
  s_waitcnt vmcnt(0)
  v_add_f32_e32 v4, v4, v8
  v_add_f32_e32 v5, v5, v9
  v_add_f32_e32 v6, v6, v10
  v_add_f32_e32 v7, v7, v11
  global_store_dword v0, v4, s[0:1]
  global_store_dword v1, v5, s[0:1]
  global_store_dword v2, v6, s[0:1]
  global_store_dword v3, v7, s[0:1]
  s_waitcnt vmcnt(0)
  s_endpgm
