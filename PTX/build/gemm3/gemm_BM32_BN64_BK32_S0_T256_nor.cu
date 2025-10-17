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
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[8];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    *(float2*)(C_local + (i * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  for (int k = 0; k < 32; ++k) {
    __syncthreads();
    *(uint2*)(((half_t*)buf_dyn_shmem) + ((((((((int)threadIdx.x) >> 3) * 32) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 2048)) = *(uint2*)(A + ((((((int)blockIdx.y) * 32768) + ((((int)threadIdx.x) >> 3) * 1024)) + (k * 32)) + ((((int)threadIdx.x) & 7) * 4)));
    *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 2) * 32) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(B + ((((((int)blockIdx.x) * 65536) + ((((int)threadIdx.x) >> 2) * 1024)) + (k * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    __syncthreads();
    tl::gemm_ss<32, 64, 32, 2, 4, 0, 1, 0, 32, 32, 0, 0>((&(((half_t*)buf_dyn_shmem)[2048])), (&(((half_t*)buf_dyn_shmem)[0])), (&(C_local[0])));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    uint1 __1;
    float2 v_ = *(float2*)(C_local + (i_1 * 2));
    ((half2*)(&(__1.x)))->x = (half_t)(v_.x);
    ((half2*)(&(__1.x)))->y = (half_t)(v_.y);
    *(uint1*)(C + ((((((((((int)blockIdx.y) * 32768) + (((((int)threadIdx.x) & 63) >> 5) * 16384)) + ((i_1 & 1) * 8192)) + (((((int)threadIdx.x) & 31) >> 2) * 1024)) + (((int)blockIdx.x) * 64)) + ((i_1 >> 1) * 32)) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __1;
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
	main_kernel<<<dim3(16, 32, 1), dim3(256, 1, 1), 6144, stream>>>(A, B, C);
	TILELANG_CHECK_LAST_ERROR("main_kernel");

	return 0;
}
