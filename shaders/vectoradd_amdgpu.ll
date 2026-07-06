; Simple vectoradd using AMDGPU intrinsics
; This should compile cleanly without external dependencies

define amdgpu_kernel void @vectoradd(
  float* nocapture readonly %a,
  float* nocapture readonly %b,
  float* nocapture %c,
  i32 %n
) #0 {
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %cmp = icmp slt i32 %id, %n
  br i1 %cmp, label %body, label %exit

body:
  %gep_a = getelementptr float, float* %a, i32 %id
  %gep_b = getelementptr float, float* %b, i32 %id
  %gep_c = getelementptr float, float* %c, i32 %id
  %val_a = load float, float* %gep_a, align 4
  %val_b = load float, float* %gep_b, align 4
  %sum = fadd float %val_a, %val_b
  store float %sum, float* %gep_c, align 4
  br label %exit

exit:
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()

attributes #0 = { noinline }
!amdgpu.targets = !{!0}
!0 = !{!"amdgcn-amd-amdhsa--gfx900:xnack+"}
