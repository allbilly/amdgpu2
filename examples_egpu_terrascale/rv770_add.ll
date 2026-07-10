; RV770/R600 pixel-shader candidate: color = a + b.
; Values arrive as interpolated graphics inputs; the shader exports its sum to
; color target 0.  The host must bind a/b vertex attributes and an AGP render
; target before this can be submitted.  This is intentionally separate from
; add.py's CP payload-write diagnostic.
target triple = "r600--"

declare void @llvm.r600.store.swizzle(<4 x float>, i32 immarg, i32 immarg)

define amdgpu_ps void @rv770_add(<4 x float> inreg %a, <4 x float> inreg %b) {
entry:
  %sum = fadd <4 x float> %a, %b
  ; R700 SQ_EXPORT_PIXEL is type 0.  Passing 15 aliases to type 3 in the
  ; two-bit hardware field (SQ_EXPORT_SX), which does not write CB_COLOR0.
  call void @llvm.r600.store.swizzle(<4 x float> %sum, i32 0, i32 0)
  ret void
}
