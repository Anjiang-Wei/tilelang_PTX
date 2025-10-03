#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>

extern "C" __global__ void chunk_linear_attn_fwd_kernel(half_t* __restrict__ K, half_t* __restrict__ O, half_t* __restrict__ Q, half_t* __restrict__ V, float* __restrict__ final_state);
extern "C" __global__ void __launch_bounds__(128, 1) chunk_linear_attn_fwd_kernel(half_t* __restrict__ K, half_t* __restrict__ O, half_t* __restrict__ Q, half_t* __restrict__ V, float* __restrict__ final_state) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float h[32];
  float s[32];
  float o[32];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    *(float2*)(h + (i * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
  }
  const dim3 blockIdx = tl::rasterization2DRow<10>();
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((((i_1 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 8192), K+((((((((int)blockIdx.z) >> 2) * 32768) + (i_1 * 4096)) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((((i_2 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), V+((((((((int)blockIdx.z) >> 2) * 32768) + (i_2 * 4096)) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    uint4 __1;
    ulonglong4 __2;
      ulonglong4 __3;
      uint4 v_ = *(uint4*)(Q + ((((((((int)blockIdx.z) >> 2) * 32768) + (i_3 * 4096)) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
      ((float2*)(&(__3.x)))->x = (float)(((half2*)(&(v_.x)))->x);
      ((float2*)(&(__3.x)))->y = (float)(((half2*)(&(v_.x)))->y);
      ((float2*)(&(__3.y)))->x = (float)(((half2*)(&(v_.y)))->x);
      ((float2*)(&(__3.y)))->y = (float)(((half2*)(&(v_.y)))->y);
      ((float2*)(&(__3.z)))->x = (float)(((half2*)(&(v_.z)))->x);
      ((float2*)(&(__3.z)))->y = (float)(((half2*)(&(v_.z)))->y);
      ((float2*)(&(__3.w)))->x = (float)(((half2*)(&(v_.w)))->x);
      ((float2*)(&(__3.w)))->y = (float)(((half2*)(&(v_.w)))->y);
      ulonglong4 v__1 = make_ulonglong4(*(unsigned long long*)&make_float2(1.250000e-01f, 1.250000e-01f), *(unsigned long long*)&make_float2(1.250000e-01f, 1.250000e-01f), *(unsigned long long*)&make_float2(1.250000e-01f, 1.250000e-01f), *(unsigned long long*)&make_float2(1.250000e-01f, 1.250000e-01f));
      ((float2*)(&(__2.x)))->x = (((float2*)(&(__3.x)))->x*((float2*)(&(v__1.x)))->x);
      ((float2*)(&(__2.x)))->y = (((float2*)(&(__3.x)))->y*((float2*)(&(v__1.x)))->y);
      ((float2*)(&(__2.y)))->x = (((float2*)(&(__3.y)))->x*((float2*)(&(v__1.y)))->x);
      ((float2*)(&(__2.y)))->y = (((float2*)(&(__3.y)))->y*((float2*)(&(v__1.y)))->y);
      ((float2*)(&(__2.z)))->x = (((float2*)(&(__3.z)))->x*((float2*)(&(v__1.z)))->x);
      ((float2*)(&(__2.z)))->y = (((float2*)(&(__3.z)))->y*((float2*)(&(v__1.z)))->y);
      ((float2*)(&(__2.w)))->x = (((float2*)(&(__3.w)))->x*((float2*)(&(v__1.w)))->x);
      ((float2*)(&(__2.w)))->y = (((float2*)(&(__3.w)))->y*((float2*)(&(v__1.w)))->y);
    ((half2*)(&(__1.x)))->x = (half_t)(((float2*)(&(__2.x)))->x);
    ((half2*)(&(__1.x)))->y = (half_t)(((float2*)(&(__2.x)))->y);
    ((half2*)(&(__1.y)))->x = (half_t)(((float2*)(&(__2.y)))->x);
    ((half2*)(&(__1.y)))->y = (half_t)(((float2*)(&(__2.y)))->y);
    ((half2*)(&(__1.z)))->x = (half_t)(((float2*)(&(__2.z)))->x);
    ((half2*)(&(__1.z)))->y = (half_t)(((float2*)(&(__2.z)))->y);
    ((half2*)(&(__1.w)))->x = (half_t)(((float2*)(&(__2.w)))->x);
    ((half2*)(&(__1.w)))->y = (half_t)(((float2*)(&(__2.w)))->y);
    *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((i_3 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 12288)) = __1;
  }
  tl::cp_async_commit();
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 0, 1, 1>((&(((half_t*)buf_dyn_shmem)[12288])), (&(((half_t*)buf_dyn_shmem)[4096])), (&(s[0])));
  __syncthreads();
  #pragma unroll
  for (int i_4 = 0; i_4 < 16; ++i_4) {
    for (int vec_s = 0; vec_s < 2; ++vec_s) {
      float condval;
      if (((((((i_4 >> 2) * 16) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s) <= ((((((i_4 & 3) >> 1) * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_4 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
        condval = s[((i_4 * 2) + vec_s)];
      } else {
        condval = 0.000000e+00f;
      }
      ((half_t*)buf_dyn_shmem)[(((((((((((i_4 & 3) >> 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((i_4 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((((i_4 >> 2) * 16) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_4 & 7) >> 2)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s)] = ((half_t)condval);
    }
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 0, 0, 1>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[8192])), (&(o[0])));
  __syncthreads();
  #pragma unroll
  for (int i_5 = 0; i_5 < 16; ++i_5) {
    uint1 __4;
    float2 v__2 = *(float2*)(h + (i_5 * 2));
    ((half2*)(&(__4.x)))->x = (half_t)(v__2.x);
    ((half2*)(&(__4.x)))->y = (half_t)(v__2.y);
    *(uint1*)(((half_t*)buf_dyn_shmem) + (((((((((((i_5 & 3) >> 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((i_5 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_5 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_5 & 7) >> 2)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 16384)) = __4;
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 1, 0, 0>((&(((half_t*)buf_dyn_shmem)[4096])), (&(((half_t*)buf_dyn_shmem)[8192])), (&(h[0])));
  __syncthreads();
  #pragma unroll
  for (int i_6 = 0; i_6 < 4; ++i_6) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((((i_6 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 8192), K+(((((((((int)blockIdx.z) >> 2) * 32768) + (i_6 * 4096)) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 16384));
  }
  #pragma unroll
  for (int i_7 = 0; i_7 < 4; ++i_7) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((((i_7 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), V+(((((((((int)blockIdx.z) >> 2) * 32768) + (i_7 * 4096)) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 16384));
  }
  tl::cp_async_commit();
  tl::cp_async_wait<1>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 0, 0, 0>((&(((half_t*)buf_dyn_shmem)[12288])), (&(((half_t*)buf_dyn_shmem)[16384])), (&(o[0])));
  __syncthreads();
  #pragma unroll
  for (int i_8 = 0; i_8 < 4; ++i_8) {
    uint4 __5;
    ulonglong4 __6;
      ulonglong4 __7;
      uint4 v__3 = *(uint4*)(Q + (((((((((int)blockIdx.z) >> 2) * 32768) + (i_8 * 4096)) + ((((int)threadIdx.x) >> 3) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 16384));
      ((float2*)(&(__7.x)))->x = (float)(((half2*)(&(v__3.x)))->x);
      ((float2*)(&(__7.x)))->y = (float)(((half2*)(&(v__3.x)))->y);
      ((float2*)(&(__7.y)))->x = (float)(((half2*)(&(v__3.y)))->x);
      ((float2*)(&(__7.y)))->y = (float)(((half2*)(&(v__3.y)))->y);
      ((float2*)(&(__7.z)))->x = (float)(((half2*)(&(v__3.z)))->x);
      ((float2*)(&(__7.z)))->y = (float)(((half2*)(&(v__3.z)))->y);
      ((float2*)(&(__7.w)))->x = (float)(((half2*)(&(v__3.w)))->x);
      ((float2*)(&(__7.w)))->y = (float)(((half2*)(&(v__3.w)))->y);
      ulonglong4 v__4 = make_ulonglong4(*(unsigned long long*)&make_float2(1.250000e-01f, 1.250000e-01f), *(unsigned long long*)&make_float2(1.250000e-01f, 1.250000e-01f), *(unsigned long long*)&make_float2(1.250000e-01f, 1.250000e-01f), *(unsigned long long*)&make_float2(1.250000e-01f, 1.250000e-01f));
      ((float2*)(&(__6.x)))->x = (((float2*)(&(__7.x)))->x*((float2*)(&(v__4.x)))->x);
      ((float2*)(&(__6.x)))->y = (((float2*)(&(__7.x)))->y*((float2*)(&(v__4.x)))->y);
      ((float2*)(&(__6.y)))->x = (((float2*)(&(__7.y)))->x*((float2*)(&(v__4.y)))->x);
      ((float2*)(&(__6.y)))->y = (((float2*)(&(__7.y)))->y*((float2*)(&(v__4.y)))->y);
      ((float2*)(&(__6.z)))->x = (((float2*)(&(__7.z)))->x*((float2*)(&(v__4.z)))->x);
      ((float2*)(&(__6.z)))->y = (((float2*)(&(__7.z)))->y*((float2*)(&(v__4.z)))->y);
      ((float2*)(&(__6.w)))->x = (((float2*)(&(__7.w)))->x*((float2*)(&(v__4.w)))->x);
      ((float2*)(&(__6.w)))->y = (((float2*)(&(__7.w)))->y*((float2*)(&(v__4.w)))->y);
    ((half2*)(&(__5.x)))->x = (half_t)(((float2*)(&(__6.x)))->x);
    ((half2*)(&(__5.x)))->y = (half_t)(((float2*)(&(__6.x)))->y);
    ((half2*)(&(__5.y)))->x = (half_t)(((float2*)(&(__6.y)))->x);
    ((half2*)(&(__5.y)))->y = (half_t)(((float2*)(&(__6.y)))->y);
    ((half2*)(&(__5.z)))->x = (half_t)(((float2*)(&(__6.z)))->x);
    ((half2*)(&(__5.z)))->y = (half_t)(((float2*)(&(__6.z)))->y);
    ((half2*)(&(__5.w)))->x = (half_t)(((float2*)(&(__6.w)))->x);
    ((half2*)(&(__5.w)))->y = (half_t)(((float2*)(&(__6.w)))->y);
    *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((i_8 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 12288)) = __5;
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_9 = 0; i_9 < 16; ++i_9) {
    uint1 __8;
    float2 v__5 = *(float2*)(o + (i_9 * 2));
    ((half2*)(&(__8.x)))->x = (half_t)(v__5.x);
    ((half2*)(&(__8.x)))->y = (half_t)(v__5.y);
    *(uint1*)(O + ((((((((((((int)blockIdx.z) >> 2) * 32768) + (((i_9 & 3) >> 1) * 8192)) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + ((i_9 & 1) * 2048)) + (((((int)threadIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((i_9 >> 2) * 16)) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __8;
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 0, 1, 1>((&(((half_t*)buf_dyn_shmem)[12288])), (&(((half_t*)buf_dyn_shmem)[4096])), (&(s[0])));
  __syncthreads();
  #pragma unroll
  for (int i_10 = 0; i_10 < 16; ++i_10) {
    for (int vec_s_1 = 0; vec_s_1 < 2; ++vec_s_1) {
      float condval_1;
      if (((((((i_10 >> 2) * 16) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s_1) <= ((((((i_10 & 3) >> 1) * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_10 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
        condval_1 = s[((i_10 * 2) + vec_s_1)];
      } else {
        condval_1 = 0.000000e+00f;
      }
      ((half_t*)buf_dyn_shmem)[(((((((((((i_10 & 3) >> 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((i_10 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((((i_10 >> 2) * 16) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s_1) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_10 & 7) >> 2)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s_1)] = ((half_t)condval_1);
    }
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 0, 0, 1>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[8192])), (&(o[0])));
  __syncthreads();
  #pragma unroll
  for (int i_11 = 0; i_11 < 16; ++i_11) {
    uint1 __9;
    float2 v__6 = *(float2*)(h + (i_11 * 2));
    ((half2*)(&(__9.x)))->x = (half_t)(v__6.x);
    ((half2*)(&(__9.x)))->y = (half_t)(v__6.y);
    *(uint1*)(((half_t*)buf_dyn_shmem) + (((((((((((i_11 & 3) >> 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((i_11 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_11 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_11 & 7) >> 2)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 16384)) = __9;
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 1, 0, 0>((&(((half_t*)buf_dyn_shmem)[4096])), (&(((half_t*)buf_dyn_shmem)[8192])), (&(h[0])));
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 2, 2, 0, 0, 0>((&(((half_t*)buf_dyn_shmem)[12288])), (&(((half_t*)buf_dyn_shmem)[16384])), (&(o[0])));
  #pragma unroll
  for (int i_12 = 0; i_12 < 16; ++i_12) {
    uint1 __10;
    float2 v__7 = *(float2*)(o + (i_12 * 2));
    ((half2*)(&(__10.x)))->x = (half_t)(v__7.x);
    ((half2*)(&(__10.x)))->y = (half_t)(v__7.y);
    *(uint1*)(O + (((((((((((((int)blockIdx.z) >> 2) * 32768) + (((i_12 & 3) >> 1) * 8192)) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + ((i_12 & 1) * 2048)) + (((((int)threadIdx.x) & 31) >> 2) * 256)) + ((((int)blockIdx.z) & 3) * 64)) + ((i_12 >> 2) * 16)) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 16384)) = __10;
  }
  #pragma unroll
  for (int i_13 = 0; i_13 < 16; ++i_13) {
    *(float2*)(final_state + ((((((((((int)blockIdx.z) * 4096) + (((i_13 & 3) >> 1) * 2048)) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((i_13 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_13 >> 2) * 16)) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(h + (i_13 * 2));
  }
}


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int init() {
    error_buf[0] = '\0';
    
    cudaError_t result_chunk_linear_attn_fwd_kernel = cudaFuncSetAttribute(chunk_linear_attn_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 40960);
    if (result_chunk_linear_attn_fwd_kernel != CUDA_SUCCESS) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 40960, cudaGetErrorString(result_chunk_linear_attn_fwd_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(half_t* __restrict__ Q, half_t* __restrict__ K, half_t* __restrict__ V, half_t* __restrict__ O, float* __restrict__ final_state, cudaStream_t stream=cudaStreamDefault) {
	chunk_linear_attn_fwd_kernel<<<dim3(1, 1, 8), dim3(128, 1, 1), 40960, stream>>>(K, O, Q, V, final_state);
	TILELANG_CHECK_LAST_ERROR("chunk_linear_attn_fwd_kernel");

	return 0;
}
