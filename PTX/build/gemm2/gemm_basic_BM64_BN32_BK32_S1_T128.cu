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
  float C_loc[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    C_loc[i] = 0x0p+0f/*0.000000e+00*/;
  }
  for (int kt = 0; kt < 64; ++kt) {
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      *(uint4*)(((half_t*)buf_dyn_shmem) + ((i_1 * 1024) + (((int)threadIdx.x) * 8))) = *(uint4*)(A + ((((((((int)blockIdx.x) >> 6) * 131072) + (i_1 * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + (kt * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    }
    *(uint4*)(((half_t*)buf_dyn_shmem) + ((((int)threadIdx.x) * 8) + 2048)) = *(uint4*)(B + (((((((int)blockIdx.x) & 63) * 65536) + ((((int)threadIdx.x) >> 2) * 2048)) + (kt * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    __syncthreads();
    for (int kk = 0; kk < 32; ++kk) {
      #pragma unroll
      for (int i_2 = 0; i_2 < 16; ++i_2) {
        C_loc[i_2] = (C_loc[i_2] + (((float)((half_t*)buf_dyn_shmem)[(((i_2 * 128) + ((((int)threadIdx.x) >> 5) * 32)) + kk)]) * ((float)((half_t*)buf_dyn_shmem)[((((((int)threadIdx.x) & 31) * 32) + kk) + 2048)])));
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 16; ++i_3) {
    C[((((((((int)blockIdx.x) >> 6) * 131072) + (i_3 * 8192)) + ((((int)threadIdx.x) >> 5) * 2048)) + ((((int)blockIdx.x) & 63) * 32)) + (((int)threadIdx.x) & 31))] = ((half_t)C_loc[i_3]);
  }
}


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int init() {
    error_buf[0] = '\0';
    
    cudaError_t result_main_kernel = cudaFuncSetAttribute(main_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 6144);
    if (result_main_kernel != CUDA_SUCCESS) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 6144, cudaGetErrorString(result_main_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C, cudaStream_t stream=cudaStreamDefault) {
	main_kernel<<<dim3(2048, 1, 1), dim3(128, 1, 1), 6144, stream>>>(A, B, C);
	TILELANG_CHECK_LAST_ERROR("main_kernel");

	return 0;
}
