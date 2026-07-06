define amdgpu_kernel void @vectoradd(float* nocapture readonly %a, float* nocapture readonly %b, float* nocapture %c) #0 {
entry:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %idxprom = sext i32 %id to i64
  %a_ptr = getelementptr float, float* %a, i64 %idxprom
  %b_ptr = getelementptr float, float* %b, i64 %idxprom
  %c_ptr = getelementptr float, float* %c, i64 %idxprom
  %a_val = load float, float* %a_ptr, align 4
  %b_val = load float, float* %b_ptr, align 4
  %sum = fadd float %a_val, %b_val
  store float %sum, float* %c_ptr, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()

attributes #0 = { noinline }

!amdgpu.targets = !{!0}
!0 = !{!"amdgcn-amd-amdhsa--gfx900:xnack+"}
