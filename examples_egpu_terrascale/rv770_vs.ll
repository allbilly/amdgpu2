; RV770 vertex shader for the graphics-add experiment.
; T0 is clip-space position; T1 and T2 carry the two vec4 add inputs to the
; pixel stage as semantics 1 and 2.  The host still has to configure matching
; vertex fetch and PS interpolation state.
target triple = "r600--"

declare void @llvm.r600.store.swizzle(<4 x float>, i32 immarg, i32 immarg)

define amdgpu_vs void @rv770_vs(<4 x float> inreg %position,
                                <4 x float> inreg %a,
                                <4 x float> inreg %b) {
entry:
  ; type 1 = SQ_EXPORT_POS, type 2 = SQ_EXPORT_PARAM (r700_sq.h).
  ; Array bases 0 and 1 identify the two pixel-shader interpolants.
  call void @llvm.r600.store.swizzle(<4 x float> %position, i32 0, i32 1)
  call void @llvm.r600.store.swizzle(<4 x float> %a, i32 0, i32 2)
  call void @llvm.r600.store.swizzle(<4 x float> %b, i32 1, i32 2)
  ret void
}
