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
  for (int i = 0; i < 16; ++i) {
    *(float2*)(out_local + (i * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  for (int kt = 0; kt < 36; ++kt) {
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      bool inb = ((((1 <= ((kt / 12) + (((int)blockIdx.y) & 63))) && (((kt / 12) + (((int)blockIdx.y) & 63)) < 65)) && (1 <= (((i_1 * 32) + (((int)threadIdx.x) >> 2)) + ((kt % 12) >> 2)))) && ((((i_1 * 32) + (((int)threadIdx.x) >> 2)) + ((kt % 12) >> 2)) < 65));
      uint4 condval;
      if ((((((1 <= (((i_1 * 32) + (((int)threadIdx.x) >> 2)) + ((kt % 12) >> 2))) && ((((i_1 * 32) + (((int)threadIdx.x) >> 2)) + ((kt % 12) >> 2)) < 65)) && (1 <= ((kt / 12) + (((int)blockIdx.y) & 63)))) && (((kt / 12) + (((int)blockIdx.y) & 63)) < 65)) && inb)) {
        condval = *(uint4*)(data + ((((((((kt / 12) * 8192) + (((int)blockIdx.y) * 8192)) + (i_1 * 4096)) + ((((int)threadIdx.x) >> 2) * 128)) + ((kt % 12) * 32)) + ((((int)threadIdx.x) & 3) * 8)) - 8320));
      } else {
        condval = make_uint4(__pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)));
      }
      *(uint4*)(((half_t*)buf_dyn_shmem) + (((((i_1 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 2048)) = condval;
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      *(uint4*)(((half_t*)buf_dyn_shmem) + (((((i_2 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(w_flat + (((((kt * 4096) + (i_2 * 2048)) + ((((int)threadIdx.x) >> 3) * 128)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
    }
    __syncthreads();
    tl::gemm_ss<64, 64, 32, 2, 2, 0, 0, 0, 32, 64, 0, 0>((&(((half_t*)buf_dyn_shmem)[2048])), (&(((half_t*)buf_dyn_shmem)[0])), (&(out_local[0])));
    __syncthreads();
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 16; ++i_3) {
    uint1 __1;
    float2 v_ = *(float2*)(out_local + (i_3 * 2));
    ((half2*)(&(__1.x)))->x = (half_t)(v_.x);
    ((half2*)(&(__1.x)))->y = (half_t)(v_.y);
    *(uint1*)(((half_t*)buf_dyn_shmem) + (((((((((i_3 & 3) >> 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((i_3 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_3 >> 2) * 16)) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __1;
  }
  __syncthreads();
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    *(uint4*)(out_flat + (((((((int)blockIdx.y) * 8192) + (i_4 * 2048)) + ((((int)threadIdx.x) >> 3) * 128)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8))) = *(uint4*)(((half_t*)buf_dyn_shmem) + ((i_4 * 1024) + (((int)threadIdx.x) * 8)));
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
	main_kernel<<<dim3(2, 8192, 1), dim3(128, 1, 1), 8192, stream>>>(data, out_flat, w_flat);
	TILELANG_CHECK_LAST_ERROR("main_kernel");

	return 0;
}
