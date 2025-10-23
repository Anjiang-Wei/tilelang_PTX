#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_loc[32];
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    C_loc[i] = 0x0p+0f/*0.000000e+00*/;
  }
  for (int kt = 0; kt < 128; ++kt) {
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      *(uint4*)(((half_t*)buf_dyn_shmem) + ((i_1 * 1024) + (((int)threadIdx.x) * 8))) = *(uint4*)(A + (((((((int)blockIdx.y) * 262144) + (i_1 * 131072)) + ((((int)threadIdx.x) >> 2) * 4096)) + (kt * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      *(uint4*)(((half_t*)buf_dyn_shmem) + (((i_2 * 1024) + (((int)threadIdx.x) * 8)) + 2048)) = *(uint4*)(B + (((((((int)blockIdx.x) * 262144) + (i_2 * 131072)) + ((((int)threadIdx.x) >> 2) * 4096)) + (kt * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    }
    __syncthreads();
    for (int kk = 0; kk < 32; ++kk) {
      #pragma unroll
      for (int i_3 = 0; i_3 < 32; ++i_3) {
        C_loc[i_3] = (C_loc[i_3] + (((float)((half_t*)buf_dyn_shmem)[(((i_3 * 64) + ((((int)threadIdx.x) >> 6) * 32)) + kk)]) * ((float)((half_t*)buf_dyn_shmem)[((((((int)threadIdx.x) & 63) * 32) + kk) + 2048)])));
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 32; ++i_4) {
    C[(((((((int)blockIdx.y) * 262144) + (i_4 * 8192)) + ((((int)threadIdx.x) >> 6) * 4096)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 63))] = ((half_t)C_loc[i_4]);
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

extern "C" int call(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C, cudaStream_t stream=cudaStreamDefault) {
	main_kernel<<<dim3(64, 64, 1), dim3(128, 1, 1), 8192, stream>>>(A, B, C);
	TILELANG_CHECK_LAST_ERROR("main_kernel");

	return 0;
}
