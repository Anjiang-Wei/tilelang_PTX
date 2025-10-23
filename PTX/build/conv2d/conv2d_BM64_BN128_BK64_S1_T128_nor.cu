#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(half_t* __restrict__ data, half_t* __restrict__ kernel_flat, half_t* __restrict__ out_flat);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(half_t* __restrict__ data, half_t* __restrict__ kernel_flat, half_t* __restrict__ out_flat) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float out_local[64];
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    *(float2*)(out_local + (i * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    bool in_bound = ((1 <= (((int)blockIdx.y) & 63)) && (1 <= ((i_1 * 16) + (((int)threadIdx.x) >> 3))));
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((((((i_1 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), data+(((((((int)blockIdx.y) * 8192) + (i_1 * 2048)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) - 8320), (((1 <= ((i_1 * 16) + (((int)threadIdx.x) >> 3))) && (1 <= (((int)blockIdx.y) & 63))) && in_bound));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_2 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)), kernel_flat+((i_2 * 1024) + (((int)threadIdx.x) * 8)));
  }
  tl::cp_async_commit();
  for (int k_iter = 0; k_iter < 17; ++k_iter) {
    tl::cp_async_wait<0>();
    __syncthreads();
    tl::gemm_ss<64, 128, 64, 2, 2, 0, 0, 0, 64, 128, 0, 0>((&(((half_t*)buf_dyn_shmem)[8192])), (&(((half_t*)buf_dyn_shmem)[0])), (&(out_local[0])));
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      bool in_bound_1 = ((((1 <= (((k_iter + 1) / 6) + (((int)blockIdx.y) & 63))) && (1 <= (((i_3 * 16) + (((int)threadIdx.x) >> 3)) + (((k_iter + 1) % 6) >> 1)))) && ((((k_iter + 1) / 6) + (((int)blockIdx.y) & 63)) < 65)) && ((((i_3 * 16) + (((int)threadIdx.x) >> 3)) + (((k_iter + 1) % 6) >> 1)) < 65));
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((((((i_3 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), data+(((((((((k_iter + 1) / 6) * 8192) + (((int)blockIdx.y) * 8192)) + (i_3 * 2048)) + ((((int)threadIdx.x) >> 3) * 128)) + (((k_iter + 1) % 6) * 64)) + ((((int)threadIdx.x) & 7) * 8)) - 8320), (((((1 <= (((i_3 * 16) + (((int)threadIdx.x) >> 3)) + (((k_iter + 1) % 6) >> 1))) && ((((i_3 * 16) + (((int)threadIdx.x) >> 3)) + (((k_iter + 1) % 6) >> 1)) < 65)) && (1 <= (((k_iter + 1) / 6) + (((int)blockIdx.y) & 63)))) && ((((k_iter + 1) / 6) + (((int)blockIdx.y) & 63)) < 65)) && in_bound_1));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 8; ++i_4) {
      tl::cp_async_gs<16>(buf_dyn_shmem+((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_4 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)), kernel_flat+((((k_iter * 8192) + (i_4 * 1024)) + (((int)threadIdx.x) * 8)) + 8192));
    }
    tl::cp_async_commit();
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::gemm_ss<64, 128, 64, 2, 2, 0, 0, 0, 64, 128, 0, 0>((&(((half_t*)buf_dyn_shmem)[8192])), (&(((half_t*)buf_dyn_shmem)[0])), (&(out_local[0])));
  __syncthreads();
  #pragma unroll
  for (int i_5 = 0; i_5 < 32; ++i_5) {
    uint1 __1;
    float2 v_ = *(float2*)(out_local + (i_5 * 2));
    ((half2*)(&(__1.x)))->x = (half_t)(v_.x);
    ((half2*)(&(__1.x)))->y = (half_t)(v_.y);
    *(uint1*)(((half_t*)buf_dyn_shmem) + (((((((((((((i_5 >> 2) * 16) + ((((int)threadIdx.x) >> 6) * 8)) >> 6) * 4096) + (((i_5 & 3) >> 1) * 2048)) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((i_5 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((((i_5 & 15) >> 2) * 16) + ((((int)threadIdx.x) >> 6) * 8)) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_5 & 7) >> 2)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __1;
  }
  __syncthreads();
  #pragma unroll
  for (int i_6 = 0; i_6 < 8; ++i_6) {
    *(uint4*)(out_flat + (((((int)blockIdx.y) * 8192) + (i_6 * 1024)) + (((int)threadIdx.x) * 8))) = *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_6 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
  }
}


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int init() {
    error_buf[0] = '\0';
    
    cudaError_t result_main_kernel = cudaFuncSetAttribute(main_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576);
    if (result_main_kernel != CUDA_SUCCESS) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 24576, cudaGetErrorString(result_main_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(half_t* __restrict__ data, half_t* __restrict__ kernel_flat, half_t* __restrict__ out_flat, cudaStream_t stream=cudaStreamDefault) {
	main_kernel<<<dim3(1, 8192, 1), dim3(128, 1, 1), 24576, stream>>>(data, kernel_flat, out_flat);
	TILELANG_CHECK_LAST_ERROR("main_kernel");

	return 0;
}
