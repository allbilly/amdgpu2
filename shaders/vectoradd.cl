__kernel void vectoradd(
  __global float* a,
  __global float* b,
  __global float* c,
  int n)
{
  int id = get_global_id(0);
  if (id < n) {
    c[id] = a[id] + b[id];
  }
}
