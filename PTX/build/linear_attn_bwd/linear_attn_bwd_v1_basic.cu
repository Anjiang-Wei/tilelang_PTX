#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>

extern "C" __global__ void chunk_linear_attn_bwd_kernel(half_t* __restrict__ K, half_t* __restrict__ Q, half_t* __restrict__ V, half_t* __restrict__ dK, half_t* __restrict__ dO, half_t* __restrict__ dQ, half_t* __restrict__ dV);
extern "C" __global__ void __launch_bounds__(128, 1) chunk_linear_attn_bwd_kernel(half_t* __restrict__ K, half_t* __restrict__ Q, half_t* __restrict__ V, half_t* __restrict__ dK, half_t* __restrict__ dO, half_t* __restrict__ dQ, half_t* __restrict__ dV) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float h[32];
  float dh[32];
  float ds[32];
  float dq[32];
  float dk[32];
  float dv[32];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    *(float2*)(h + (i * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 16; ++i_1) {
    *(float2*)(dh + (i_1 * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((((i_2 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 40960), dO+((((((((int)blockIdx.z) >> 2) * 32768) + (i_2 * 4096)) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((((i_3 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), K+((((((((int)blockIdx.z) >> 2) * 32768) + (i_3 * 4096)) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    tl::cp_async_gs<16>(buf_dyn_shmem+(((((i_4 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)), V+((((((((int)blockIdx.z) >> 2) * 32768) + (i_4 * 4096)) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
  }
  tl::cp_async_commit();
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 0, 1, 1>((&(((half_t*)buf_dyn_shmem)[20480])), (&(((half_t*)buf_dyn_shmem)[0])), (&(ds[0])));
  __syncthreads();
  #pragma unroll
  for (int i_5 = 0; i_5 < 16; ++i_5) {
    for (int vec_s = 0; vec_s < 2; ++vec_s) {
      float condval;
      if (((((((i_5 >> 2) * 16) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s) <= ((((((i_5 & 3) >> 1) * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_5 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
        condval = ds[((i_5 * 2) + vec_s)];
      } else {
        condval = 0.000000e+00f;
      }
      ((half_t*)buf_dyn_shmem)[((((((((((((i_5 & 3) >> 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((i_5 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((((i_5 >> 2) * 16) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_5 & 7) >> 2)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s) + 4096)] = ((half_t)condval);
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 0, 0, 1>((&(((half_t*)buf_dyn_shmem)[4096])), (&(((half_t*)buf_dyn_shmem)[8192])), (&(dq[0])));
  __syncthreads();
  #pragma unroll
  for (int i_6 = 0; i_6 < 16; ++i_6) {
    uint1 __1;
    float2 v_ = *(float2*)(h + (i_6 * 2));
    ((half2*)(&(__1.x)))->x = (half_t)(v_.x);
    ((half2*)(&(__1.x)))->y = (half_t)(v_.y);
    *(uint1*)(((half_t*)buf_dyn_shmem) + (((((((((((i_6 & 3) >> 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((i_6 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_6 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_6 & 7) >> 2)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 16384)) = __1;
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 0, 0, 0>((&(((half_t*)buf_dyn_shmem)[20480])), (&(((half_t*)buf_dyn_shmem)[16384])), (&(dq[0])));
  __syncthreads();
  #pragma unroll
  for (int i_7 = 0; i_7 < 4; ++i_7) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((((i_7 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 40960), dO+(((((((((int)blockIdx.z) >> 2) * 32768) + (i_7 * 4096)) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 16384));
  }
  tl::cp_async_commit();
  tl::cp_async_wait<1>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 1, 0, 0>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[8192])), (&(h[0])));
  __syncthreads();
  #pragma unroll
  for (int i_8 = 0; i_8 < 4; ++i_8) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((((i_8 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), K+(((((((((int)blockIdx.z) >> 2) * 32768) + (i_8 * 4096)) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 16384));
  }
  #pragma unroll
  for (int i_9 = 0; i_9 < 4; ++i_9) {
    tl::cp_async_gs<16>(buf_dyn_shmem+(((((i_9 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)), V+(((((((((int)blockIdx.z) >> 2) * 32768) + (i_9 * 4096)) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 16384));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_10 = 0; i_10 < 32; ++i_10) {
    dq[i_10] = (dq[i_10] * 1.250000e-01f);
  }
  #pragma unroll
  for (int i_11 = 0; i_11 < 16; ++i_11) {
    uint1 __2;
    float2 v__1 = *(float2*)(dq + (i_11 * 2));
    ((half2*)(&(__2.x)))->x = (half_t)(v__1.x);
    ((half2*)(&(__2.x)))->y = (half_t)(v__1.y);
    *(uint1*)(dQ + ((((((((((((int)blockIdx.z) >> 2) * 32768) + (((i_11 & 3) >> 1) * 8192)) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + ((i_11 & 1) * 2048)) + (((((int)threadIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((i_11 >> 2) * 16)) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __2;
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 0, 1, 1>((&(((half_t*)buf_dyn_shmem)[20480])), (&(((half_t*)buf_dyn_shmem)[0])), (&(ds[0])));
  __syncthreads();
  #pragma unroll
  for (int i_12 = 0; i_12 < 16; ++i_12) {
    for (int vec_s_1 = 0; vec_s_1 < 2; ++vec_s_1) {
      float condval_1;
      if (((((((i_12 >> 2) * 16) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s_1) <= ((((((i_12 & 3) >> 1) * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_12 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
        condval_1 = ds[((i_12 * 2) + vec_s_1)];
      } else {
        condval_1 = 0.000000e+00f;
      }
      ((half_t*)buf_dyn_shmem)[((((((((((((i_12 & 3) >> 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((i_12 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((((i_12 >> 2) * 16) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s_1) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_12 & 7) >> 2)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s_1) + 4096)] = ((half_t)condval_1);
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 0, 0, 1>((&(((half_t*)buf_dyn_shmem)[4096])), (&(((half_t*)buf_dyn_shmem)[8192])), (&(dq[0])));
  __syncthreads();
  #pragma unroll
  for (int i_13 = 0; i_13 < 16; ++i_13) {
    uint1 __3;
    float2 v__2 = *(float2*)(h + (i_13 * 2));
    ((half2*)(&(__3.x)))->x = (half_t)(v__2.x);
    ((half2*)(&(__3.x)))->y = (half_t)(v__2.y);
    *(uint1*)(((half_t*)buf_dyn_shmem) + (((((((((((i_13 & 3) >> 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((i_13 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_13 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_13 & 7) >> 2)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 16384)) = __3;
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 0, 0, 0>((&(((half_t*)buf_dyn_shmem)[20480])), (&(((half_t*)buf_dyn_shmem)[16384])), (&(dq[0])));
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 1, 0, 0>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[8192])), (&(h[0])));
  #pragma unroll
  for (int i_14 = 0; i_14 < 32; ++i_14) {
    dq[i_14] = (dq[i_14] * 1.250000e-01f);
  }
  #pragma unroll
  for (int i_15 = 0; i_15 < 16; ++i_15) {
    uint1 __4;
    float2 v__3 = *(float2*)(dq + (i_15 * 2));
    ((half2*)(&(__4.x)))->x = (half_t)(v__3.x);
    ((half2*)(&(__4.x)))->y = (half_t)(v__3.y);
    *(uint1*)(dQ + (((((((((((((int)blockIdx.z) >> 2) * 32768) + (((i_15 & 3) >> 1) * 8192)) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + ((i_15 & 1) * 2048)) + (((((int)threadIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((i_15 >> 2) * 16)) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 16384)) = __4;
  }
  __syncthreads();
  #pragma unroll
  for (int i_16 = 0; i_16 < 4; ++i_16) {
    tl::cp_async_gs<16>(buf_dyn_shmem+(((((i_16 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)), V+(((((((((int)blockIdx.z) >> 2) * 32768) + (i_16 * 4096)) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 16384));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_17 = 0; i_17 < 4; ++i_17) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((((i_17 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), K+(((((((((int)blockIdx.z) >> 2) * 32768) + (i_17 * 4096)) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 16384));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_18 = 0; i_18 < 4; ++i_18) {
    uint4 __5;
    ulonglong4 __6;
      ulonglong4 __7;
      uint4 v__4 = *(uint4*)(Q + (((((((((int)blockIdx.z) >> 2) * 32768) + (i_18 * 4096)) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 16384));
      ((float2*)(&(__7.x)))->x = (float)(((half2*)(&(v__4.x)))->x);
      ((float2*)(&(__7.x)))->y = (float)(((half2*)(&(v__4.x)))->y);
      ((float2*)(&(__7.y)))->x = (float)(((half2*)(&(v__4.y)))->x);
      ((float2*)(&(__7.y)))->y = (float)(((half2*)(&(v__4.y)))->y);
      ((float2*)(&(__7.z)))->x = (float)(((half2*)(&(v__4.z)))->x);
      ((float2*)(&(__7.z)))->y = (float)(((half2*)(&(v__4.z)))->y);
      ((float2*)(&(__7.w)))->x = (float)(((half2*)(&(v__4.w)))->x);
      ((float2*)(&(__7.w)))->y = (float)(((half2*)(&(v__4.w)))->y);
      ulonglong4 v__5 = make_ulonglong4(*(unsigned long long*)&make_float2(1.250000e-01f, 1.250000e-01f), *(unsigned long long*)&make_float2(1.250000e-01f, 1.250000e-01f), *(unsigned long long*)&make_float2(1.250000e-01f, 1.250000e-01f), *(unsigned long long*)&make_float2(1.250000e-01f, 1.250000e-01f));
      ((float2*)(&(__6.x)))->x = (((float2*)(&(__7.x)))->x*((float2*)(&(v__5.x)))->x);
      ((float2*)(&(__6.x)))->y = (((float2*)(&(__7.x)))->y*((float2*)(&(v__5.x)))->y);
      ((float2*)(&(__6.y)))->x = (((float2*)(&(__7.y)))->x*((float2*)(&(v__5.y)))->x);
      ((float2*)(&(__6.y)))->y = (((float2*)(&(__7.y)))->y*((float2*)(&(v__5.y)))->y);
      ((float2*)(&(__6.z)))->x = (((float2*)(&(__7.z)))->x*((float2*)(&(v__5.z)))->x);
      ((float2*)(&(__6.z)))->y = (((float2*)(&(__7.z)))->y*((float2*)(&(v__5.z)))->y);
      ((float2*)(&(__6.w)))->x = (((float2*)(&(__7.w)))->x*((float2*)(&(v__5.w)))->x);
      ((float2*)(&(__6.w)))->y = (((float2*)(&(__7.w)))->y*((float2*)(&(v__5.w)))->y);
    ((half2*)(&(__5.x)))->x = (half_t)(((float2*)(&(__6.x)))->x);
    ((half2*)(&(__5.x)))->y = (half_t)(((float2*)(&(__6.x)))->y);
    ((half2*)(&(__5.y)))->x = (half_t)(((float2*)(&(__6.y)))->x);
    ((half2*)(&(__5.y)))->y = (half_t)(((float2*)(&(__6.y)))->y);
    ((half2*)(&(__5.z)))->x = (half_t)(((float2*)(&(__6.z)))->x);
    ((half2*)(&(__5.z)))->y = (half_t)(((float2*)(&(__6.z)))->y);
    ((half2*)(&(__5.w)))->x = (half_t)(((float2*)(&(__6.w)))->x);
    ((half2*)(&(__5.w)))->y = (half_t)(((float2*)(&(__6.w)))->y);
    *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((i_18 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)) = __5;
  }
  #pragma unroll
  for (int i_19 = 0; i_19 < 4; ++i_19) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((((i_19 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 40960), dO+(((((((((int)blockIdx.z) >> 2) * 32768) + (i_19 * 4096)) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 16384));
  }
  tl::cp_async_commit();
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 0, 1, 1>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[20480])), (&(ds[0])));
  __syncthreads();
  #pragma unroll
  for (int i_20 = 0; i_20 < 16; ++i_20) {
    for (int vec_s_2 = 0; vec_s_2 < 2; ++vec_s_2) {
      float condval_2;
      if ((((((((i_20 & 3) >> 1) * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_20 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) <= (((((i_20 >> 2) * 16) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s_2))) {
        condval_2 = ds[((i_20 * 2) + vec_s_2)];
      } else {
        condval_2 = 0.000000e+00f;
      }
      ((half_t*)buf_dyn_shmem)[((((((((((((i_20 & 3) >> 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((i_20 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((((i_20 >> 2) * 16) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s_2) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_20 & 7) >> 2)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s_2) + 4096)] = ((half_t)condval_2);
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 0, 0, 1>((&(((half_t*)buf_dyn_shmem)[4096])), (&(((half_t*)buf_dyn_shmem)[16384])), (&(dk[0])));
  __syncthreads();
  #pragma unroll
  for (int i_21 = 0; i_21 < 16; ++i_21) {
    uint1 __8;
    float2 v__6 = *(float2*)(dh + (i_21 * 2));
    ((half2*)(&(__8.x)))->x = (half_t)(v__6.x);
    ((half2*)(&(__8.x)))->y = (half_t)(v__6.y);
    *(uint1*)(((half_t*)buf_dyn_shmem) + (((((((((((i_21 & 3) >> 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((i_21 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_21 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_21 & 7) >> 2)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 12288)) = __8;
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 0, 1, 0>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[12288])), (&(dk[0])));
  __syncthreads();
  #pragma unroll
  for (int i_22 = 0; i_22 < 4; ++i_22) {
    tl::cp_async_gs<16>(buf_dyn_shmem+(((((i_22 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)), V+((((((((int)blockIdx.z) >> 2) * 32768) + (i_22 * 4096)) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
  }
  tl::cp_async_commit();
  tl::cp_async_wait<1>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 0, 1, 1>((&(((half_t*)buf_dyn_shmem)[8192])), (&(((half_t*)buf_dyn_shmem)[16384])), (&(ds[0])));
  __syncthreads();
  #pragma unroll
  for (int i_23 = 0; i_23 < 16; ++i_23) {
    for (int vec_s_3 = 0; vec_s_3 < 2; ++vec_s_3) {
      float condval_3;
      if ((((((((i_23 & 3) >> 1) * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_23 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) <= (((((i_23 >> 2) * 16) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s_3))) {
        condval_3 = ds[((i_23 * 2) + vec_s_3)];
      } else {
        condval_3 = 0.000000e+00f;
      }
      ((half_t*)buf_dyn_shmem)[((((((((((((i_23 & 3) >> 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((i_23 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((((i_23 >> 2) * 16) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s_3) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_23 & 7) >> 2)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s_3) + 4096)] = ((half_t)condval_3);
    }
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 0, 0, 1>((&(((half_t*)buf_dyn_shmem)[4096])), (&(((half_t*)buf_dyn_shmem)[20480])), (&(dv[0])));
  tl::cp_async_wait<2>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 0, 0, 0>((&(((half_t*)buf_dyn_shmem)[8192])), (&(((half_t*)buf_dyn_shmem)[12288])), (&(dv[0])));
  __syncthreads();
  #pragma unroll
  for (int i_24 = 0; i_24 < 4; ++i_24) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((((i_24 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), K+((((((((int)blockIdx.z) >> 2) * 32768) + (i_24 * 4096)) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
  }
  tl::cp_async_commit();
  tl::cp_async_wait<2>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 1, 0, 0>((&(((half_t*)buf_dyn_shmem)[16384])), (&(((half_t*)buf_dyn_shmem)[20480])), (&(dh[0])));
  __syncthreads();
  #pragma unroll
  for (int i_25 = 0; i_25 < 4; ++i_25) {
    uint4 __9;
    ulonglong4 __10;
      ulonglong4 __11;
      uint4 v__7 = *(uint4*)(Q + ((((((((int)blockIdx.z) >> 2) * 32768) + (i_25 * 4096)) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
      ((float2*)(&(__11.x)))->x = (float)(((half2*)(&(v__7.x)))->x);
      ((float2*)(&(__11.x)))->y = (float)(((half2*)(&(v__7.x)))->y);
      ((float2*)(&(__11.y)))->x = (float)(((half2*)(&(v__7.y)))->x);
      ((float2*)(&(__11.y)))->y = (float)(((half2*)(&(v__7.y)))->y);
      ((float2*)(&(__11.z)))->x = (float)(((half2*)(&(v__7.z)))->x);
      ((float2*)(&(__11.z)))->y = (float)(((half2*)(&(v__7.z)))->y);
      ((float2*)(&(__11.w)))->x = (float)(((half2*)(&(v__7.w)))->x);
      ((float2*)(&(__11.w)))->y = (float)(((half2*)(&(v__7.w)))->y);
      ulonglong4 v__8 = make_ulonglong4(*(unsigned long long*)&make_float2(1.250000e-01f, 1.250000e-01f), *(unsigned long long*)&make_float2(1.250000e-01f, 1.250000e-01f), *(unsigned long long*)&make_float2(1.250000e-01f, 1.250000e-01f), *(unsigned long long*)&make_float2(1.250000e-01f, 1.250000e-01f));
      ((float2*)(&(__10.x)))->x = (((float2*)(&(__11.x)))->x*((float2*)(&(v__8.x)))->x);
      ((float2*)(&(__10.x)))->y = (((float2*)(&(__11.x)))->y*((float2*)(&(v__8.x)))->y);
      ((float2*)(&(__10.y)))->x = (((float2*)(&(__11.y)))->x*((float2*)(&(v__8.y)))->x);
      ((float2*)(&(__10.y)))->y = (((float2*)(&(__11.y)))->y*((float2*)(&(v__8.y)))->y);
      ((float2*)(&(__10.z)))->x = (((float2*)(&(__11.z)))->x*((float2*)(&(v__8.z)))->x);
      ((float2*)(&(__10.z)))->y = (((float2*)(&(__11.z)))->y*((float2*)(&(v__8.z)))->y);
      ((float2*)(&(__10.w)))->x = (((float2*)(&(__11.w)))->x*((float2*)(&(v__8.w)))->x);
      ((float2*)(&(__10.w)))->y = (((float2*)(&(__11.w)))->y*((float2*)(&(v__8.w)))->y);
    ((half2*)(&(__9.x)))->x = (half_t)(((float2*)(&(__10.x)))->x);
    ((half2*)(&(__9.x)))->y = (half_t)(((float2*)(&(__10.x)))->y);
    ((half2*)(&(__9.y)))->x = (half_t)(((float2*)(&(__10.y)))->x);
    ((half2*)(&(__9.y)))->y = (half_t)(((float2*)(&(__10.y)))->y);
    ((half2*)(&(__9.z)))->x = (half_t)(((float2*)(&(__10.z)))->x);
    ((half2*)(&(__9.z)))->y = (half_t)(((float2*)(&(__10.z)))->y);
    ((half2*)(&(__9.w)))->x = (half_t)(((float2*)(&(__10.w)))->x);
    ((half2*)(&(__9.w)))->y = (half_t)(((float2*)(&(__10.w)))->y);
    *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((i_25 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)) = __9;
  }
  #pragma unroll
  for (int i_26 = 0; i_26 < 4; ++i_26) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((((i_26 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 40960), dO+((((((((int)blockIdx.z) >> 2) * 32768) + (i_26 * 4096)) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_27 = 0; i_27 < 16; ++i_27) {
    uint1 __12;
    float2 v__9 = *(float2*)(dk + (i_27 * 2));
    ((half2*)(&(__12.x)))->x = (half_t)(v__9.x);
    ((half2*)(&(__12.x)))->y = (half_t)(v__9.y);
    *(uint1*)(dK + (((((((((((((int)blockIdx.z) >> 2) * 32768) + (((i_27 & 3) >> 1) * 8192)) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + ((i_27 & 1) * 2048)) + (((((int)threadIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((i_27 >> 2) * 16)) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 16384)) = __12;
  }
  #pragma unroll
  for (int i_28 = 0; i_28 < 16; ++i_28) {
    uint1 __13;
    float2 v__10 = *(float2*)(dv + (i_28 * 2));
    ((half2*)(&(__13.x)))->x = (half_t)(v__10.x);
    ((half2*)(&(__13.x)))->y = (half_t)(v__10.y);
    *(uint1*)(dV + (((((((((((((int)blockIdx.z) >> 2) * 32768) + (((i_28 & 3) >> 1) * 8192)) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + ((i_28 & 1) * 2048)) + (((((int)threadIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((i_28 >> 2) * 16)) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 16384)) = __13;
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 0, 1, 1>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[20480])), (&(ds[0])));
  __syncthreads();
  #pragma unroll
  for (int i_29 = 0; i_29 < 16; ++i_29) {
    for (int vec_s_4 = 0; vec_s_4 < 2; ++vec_s_4) {
      float condval_4;
      if ((((((((i_29 & 3) >> 1) * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_29 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) <= (((((i_29 >> 2) * 16) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s_4))) {
        condval_4 = ds[((i_29 * 2) + vec_s_4)];
      } else {
        condval_4 = 0.000000e+00f;
      }
      ((half_t*)buf_dyn_shmem)[((((((((((((i_29 & 3) >> 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((i_29 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((((i_29 >> 2) * 16) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s_4) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_29 & 7) >> 2)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s_4) + 4096)] = ((half_t)condval_4);
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 0, 0, 1>((&(((half_t*)buf_dyn_shmem)[4096])), (&(((half_t*)buf_dyn_shmem)[16384])), (&(dk[0])));
  __syncthreads();
  #pragma unroll
  for (int i_30 = 0; i_30 < 16; ++i_30) {
    uint1 __14;
    float2 v__11 = *(float2*)(dh + (i_30 * 2));
    ((half2*)(&(__14.x)))->x = (half_t)(v__11.x);
    ((half2*)(&(__14.x)))->y = (half_t)(v__11.y);
    *(uint1*)(((half_t*)buf_dyn_shmem) + (((((((((((i_30 & 3) >> 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((i_30 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_30 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_30 & 7) >> 2)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 12288)) = __14;
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 0, 1, 0>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[12288])), (&(dk[0])));
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 0, 1, 1>((&(((half_t*)buf_dyn_shmem)[8192])), (&(((half_t*)buf_dyn_shmem)[16384])), (&(ds[0])));
  __syncthreads();
  #pragma unroll
  for (int i_31 = 0; i_31 < 16; ++i_31) {
    for (int vec_s_5 = 0; vec_s_5 < 2; ++vec_s_5) {
      float condval_5;
      if ((((((((i_31 & 3) >> 1) * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_31 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) <= (((((i_31 >> 2) * 16) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s_5))) {
        condval_5 = ds[((i_31 * 2) + vec_s_5)];
      } else {
        condval_5 = 0.000000e+00f;
      }
      ((half_t*)buf_dyn_shmem)[((((((((((((i_31 & 3) >> 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((i_31 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((((i_31 >> 2) * 16) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s_5) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_31 & 7) >> 2)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s_5) + 4096)] = ((half_t)condval_5);
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 0, 0, 1>((&(((half_t*)buf_dyn_shmem)[4096])), (&(((half_t*)buf_dyn_shmem)[20480])), (&(dv[0])));
  tl::cp_async_wait<1>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 0, 0, 0>((&(((half_t*)buf_dyn_shmem)[8192])), (&(((half_t*)buf_dyn_shmem)[12288])), (&(dv[0])));
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 1, 0, 0>((&(((half_t*)buf_dyn_shmem)[16384])), (&(((half_t*)buf_dyn_shmem)[20480])), (&(dh[0])));
  #pragma unroll
  for (int i_32 = 0; i_32 < 16; ++i_32) {
    uint1 __15;
    float2 v__12 = *(float2*)(dk + (i_32 * 2));
    ((half2*)(&(__15.x)))->x = (half_t)(v__12.x);
    ((half2*)(&(__15.x)))->y = (half_t)(v__12.y);
    *(uint1*)(dK + ((((((((((((int)blockIdx.z) >> 2) * 32768) + (((i_32 & 3) >> 1) * 8192)) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + ((i_32 & 1) * 2048)) + (((((int)threadIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((i_32 >> 2) * 16)) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __15;
  }
  #pragma unroll
  for (int i_33 = 0; i_33 < 16; ++i_33) {
    uint1 __16;
    float2 v__13 = *(float2*)(dv + (i_33 * 2));
    ((half2*)(&(__16.x)))->x = (half_t)(v__13.x);
    ((half2*)(&(__16.x)))->y = (half_t)(v__13.y);
    *(uint1*)(dV + ((((((((((((int)blockIdx.z) >> 2) * 32768) + (((i_33 & 3) >> 1) * 8192)) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + ((i_33 & 1) * 2048)) + (((((int)threadIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((i_33 >> 2) * 16)) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __16;
  }
}


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int init() {
    error_buf[0] = '\0';
    
    cudaError_t result_chunk_linear_attn_bwd_kernel = cudaFuncSetAttribute(chunk_linear_attn_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152);
    if (result_chunk_linear_attn_bwd_kernel != CUDA_SUCCESS) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 49152, cudaGetErrorString(result_chunk_linear_attn_bwd_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(half_t* __restrict__ Q, half_t* __restrict__ K, half_t* __restrict__ V, half_t* __restrict__ dO, half_t* __restrict__ dQ, half_t* __restrict__ dK, half_t* __restrict__ dV, cudaStream_t stream=cudaStreamDefault) {
	chunk_linear_attn_bwd_kernel<<<dim3(1, 1, 8), dim3(128, 1, 1), 49152, stream>>>(K, Q, V, dK, dO, dQ, dV);
	TILELANG_CHECK_LAST_ERROR("chunk_linear_attn_bwd_kernel");

	return 0;
}
