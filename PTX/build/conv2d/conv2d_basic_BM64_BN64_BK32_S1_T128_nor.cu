#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(half_t* __restrict__ data, half_t* __restrict__ out_flat, half_t* __restrict__ w_flat);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(half_t* __restrict__ data, half_t* __restrict__ out_flat, half_t* __restrict__ w_flat) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float out_local[32];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    *(float4*)(out_local + (i * 4)) = make_float4(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  for (int kt = 0; kt < 12; ++kt) {
    #pragma unroll
    for (int i_1 = 0; i_1 < 16; ++i_1) {
      bool inb = (((((1 <= ((((((((((int)blockIdx.x) >> 1) * 64) + (i_1 * 4)) + (((int)threadIdx.x) >> 5)) % 3025) / 55) * 2) + (((kt * 32) + (((int)threadIdx.x) & 31)) / 66))) && (((((((((((int)blockIdx.x) >> 1) * 64) + (i_1 * 4)) + (((int)threadIdx.x) >> 5)) % 3025) / 55) * 2) + (((kt * 32) + (((int)threadIdx.x) & 31)) / 66)) < 113)) && (1 <= (((((((((int)blockIdx.x) >> 1) * 64) + (i_1 * 4)) + (((int)threadIdx.x) >> 5)) % 55) * 2) + ((((kt * 32) + (((int)threadIdx.x) & 31)) % 33) / 6)))) && ((((((((((int)blockIdx.x) >> 1) * 64) + (i_1 * 4)) + (((int)threadIdx.x) >> 5)) % 55) * 2) + ((((kt * 32) + (((int)threadIdx.x) & 31)) % 33) / 6)) < 113)) && (((kt * 32) + (((int)threadIdx.x) & 31)) < 363));
      half_t condval;
      if (((((((1 <= (((((((((int)blockIdx.x) >> 1) * 64) + (i_1 * 4)) + (((int)threadIdx.x) >> 5)) % 55) * 2) + ((((kt * 32) + (((int)threadIdx.x) & 31)) % 33) / 6))) && ((((((((((int)blockIdx.x) >> 1) * 64) + (i_1 * 4)) + (((int)threadIdx.x) >> 5)) % 55) * 2) + ((((kt * 32) + (((int)threadIdx.x) & 31)) % 33) / 6)) < 113)) && (1 <= ((((((((((int)blockIdx.x) >> 1) * 64) + (i_1 * 4)) + (((int)threadIdx.x) >> 5)) % 3025) / 55) * 2) + (((kt * 32) + (((int)threadIdx.x) & 31)) / 66)))) && (((((((((((int)blockIdx.x) >> 1) * 64) + (i_1 * 4)) + (((int)threadIdx.x) >> 5)) % 3025) / 55) * 2) + (((kt * 32) + (((int)threadIdx.x) & 31)) / 66)) < 113)) && ((((((int)blockIdx.x) >> 1) * 16) + i_1) < 75625)) && inb)) {
        condval = data[(((((((((((((int)blockIdx.x) >> 1) * 64) + (i_1 * 4)) + (((int)threadIdx.x) >> 5)) / 3025) * 150528) + (((((((((int)blockIdx.x) >> 1) * 64) + (i_1 * 4)) + (((int)threadIdx.x) >> 5)) % 3025) / 55) * 2688)) + ((((kt * 32) + (((int)threadIdx.x) & 31)) / 33) * 672)) + ((((((((int)blockIdx.x) >> 1) * 64) + (i_1 * 4)) + (((int)threadIdx.x) >> 5)) % 55) * 12)) + (((kt * 32) + (((int)threadIdx.x) & 31)) % 33)) - 1350)];
      } else {
        condval = half_t(0x0p+0f/*0.000000e+00*/);
      }
      ((half_t*)buf_dyn_shmem)[(((i_1 * 128) + ((int)threadIdx.x)) + 2048)] = condval;
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      bool inb_w = (((((kt * 32) + (i_2 * 16)) + (((int)threadIdx.x) >> 3)) < 363) && ((((((int)blockIdx.x) & 1) * 2) + ((((int)threadIdx.x) & 7) >> 2)) < 3));
      uint4 condval_1;
      if (((((((((int)blockIdx.x) & 1) * 2) + ((((int)threadIdx.x) & 7) >> 2)) < 3) && ((((kt * 32) + (i_2 * 16)) + (((int)threadIdx.x) >> 3)) < 363)) && inb_w)) {
        condval_1 = *(uint4*)(w_flat + (((((kt * 3072) + (i_2 * 1536)) + ((((int)threadIdx.x) >> 3) * 96)) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
      } else {
        condval_1 = make_uint4(__pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)));
      }
      *(uint4*)(((half_t*)buf_dyn_shmem) + ((i_2 * 1024) + (((int)threadIdx.x) * 8))) = condval_1;
    }
    __syncthreads();
    for (int kk = 0; kk < 32; ++kk) {
      #pragma unroll
      for (int i_3 = 0; i_3 < 32; ++i_3) {
        out_local[i_3] = (out_local[i_3] + (((float)((half_t*)buf_dyn_shmem)[(((((i_3 >> 2) * 256) + ((((int)threadIdx.x) >> 4) * 32)) + kk) + 2048)]) * ((float)((half_t*)buf_dyn_shmem)[(((kk * 64) + ((((int)threadIdx.x) & 15) * 4)) + (i_3 & 3))])));
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 8; ++i_4) {
    uint2 __1;
    float4 v_ = *(float4*)(out_local + (i_4 * 4));
    ((half2*)(&(__1.x)))->x = (half_t)(v_.x);
    ((half2*)(&(__1.x)))->y = (half_t)(v_.y);
    ((half2*)(&(__1.y)))->x = (half_t)(v_.z);
    ((half2*)(&(__1.y)))->y = (half_t)(v_.w);
    *(uint2*)(((half_t*)buf_dyn_shmem) + ((i_4 * 512) + (((int)threadIdx.x) * 4))) = __1;
  }
  __syncthreads();
  #pragma unroll
  for (int i_5 = 0; i_5 < 4; ++i_5) {
    if ((((((((int)blockIdx.x) >> 1) * 16) + (i_5 * 4)) + (((int)threadIdx.x) >> 5)) < 75625) && ((((((int)blockIdx.x) & 1) * 2) + ((((int)threadIdx.x) & 7) >> 2)) < 3)) {
      *(uint4*)(out_flat + ((((((((int)blockIdx.x) >> 1) * 6144) + (i_5 * 1536)) + ((((int)threadIdx.x) >> 3) * 96)) + ((((int)blockIdx.x) & 1) * 64)) + ((((int)threadIdx.x) & 7) * 8))) = *(uint4*)(((half_t*)buf_dyn_shmem) + ((i_5 * 1024) + (((int)threadIdx.x) * 8)));
    }
  }
}


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int init() {
    error_buf[0] = '\0';
    
    cudaError_t result_main_kernel = cudaFuncSetAttribute(main_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192);
    if (result_main_kernel != CUDA_SUCCESS) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 8192, cudaGetErrorString(result_main_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(half_t* __restrict__ data, half_t* __restrict__ w_flat, half_t* __restrict__ out_flat, cudaStream_t stream=cudaStreamDefault) {
	main_kernel<<<dim3(9454, 1, 1), dim3(128, 1, 1), 8192, stream>>>(data, out_flat, w_flat);
	TILELANG_CHECK_LAST_ERROR("main_kernel");

	return 0;
}
