#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>

extern "C" __global__ void main_kernel(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[64];
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    *(float2*)(C_local + (i * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
  }
  for (int k = 0; k < 16; ++k) {
    __syncthreads();
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      *(uint4*)(((half_t*)buf_dyn_shmem) + (((((i_1 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(A + (((((((int)blockIdx.y) * 131072) + (i_1 * 32768)) + ((((int)threadIdx.x) >> 3) * 1024)) + (k * 64)) + ((((int)threadIdx.x) & 7) * 8)));
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((i_2 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)) = *(uint4*)(B + (((((((int)blockIdx.x) * 131072) + (i_2 * 32768)) + ((((int)threadIdx.x) >> 3) * 1024)) + (k * 64)) + ((((int)threadIdx.x) & 7) * 8)));
    }
    __syncthreads();
    tl::gemm_ss<128, 128, 64, 2, 4, 0, 1, 0>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[8192])), (&(C_local[0])));
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 32; ++i_3) {
    uint1 __1;
    float2 v_ = *(float2*)(C_local + (i_3 * 2));
    ((half2*)(&(__1.x)))->x = (half_t)(v_.x);
    ((half2*)(&(__1.x)))->y = (half_t)(v_.y);
    *(uint1*)(C + (((((((((((int)blockIdx.y) * 131072) + (((i_3 & 7) >> 1) * 32768)) + (((((int)threadIdx.x) & 63) >> 5) * 16384)) + ((i_3 & 1) * 8192)) + (((((int)threadIdx.x) & 31) >> 2) * 1024)) + (((int)blockIdx.x) * 128)) + ((i_3 >> 3) * 32)) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __1;
  }
}


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int init() {
    error_buf[0] = '\0';
    
    cudaError_t result_main_kernel = cudaFuncSetAttribute(main_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768);
    if (result_main_kernel != CUDA_SUCCESS) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 32768, cudaGetErrorString(result_main_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C, cudaStream_t stream=cudaStreamDefault) {
	main_kernel<<<dim3(8, 8, 1), dim3(256, 1, 1), 32768, stream>>>(A, B, C);
	TILELANG_CHECK_LAST_ERROR("main_kernel");

	return 0;
}
