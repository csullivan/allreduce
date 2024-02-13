#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

using fp8_e4_2_t = __nv_fp8x2_e4m3;

const char* RED = "\033[31m";
const char* GREEN = "\033[32m";
const char* YELLOW = "\033[33m";
const char* BOLD_RED = "\033[1;31m";
const char* RESET = "\033[0m";

struct __align__(4) half4_2 {
  __half2 lo;
  __half2 hi;

  __host__ __device__ half4_2() : lo(), hi() {}
  __host__ __device__ half4_2(__half2 lo, __half2 hi) : lo(lo), hi(hi) {}
  __host__ __device__ explicit half4_2(const __nv_fp8x4_e4m3& fp8x4) {
    __nv_fp8x2_e4m3 lo_part, hi_part;
    lo_part.__x = static_cast<__nv_fp8x2_storage_t>(fp8x4.__x & 0xFFFF);
    hi_part.__x = static_cast<__nv_fp8x2_storage_t>(fp8x4.__x >> 16);
    lo = static_cast<__half2>(lo_part);
    hi = static_cast<__half2>(hi_part);
  }

  __host__ __device__ explicit operator __nv_fp8x4_e4m3() const {
    __nv_fp8x4_e4m3 result;
    __nv_fp8x2_e4m3 lo_part(lo), hi_part(hi);
    result.__x =
        (static_cast<__uint32_t>(hi_part.__x) << 16) | static_cast<__uint32_t>(lo_part.__x);
    return result;
  }
};

struct __align__(4) half4 {
  __half x, y, z, w;

  __host__ __device__ half4() : x(__half(0)), y(__half(0)), z(__half(0)), w(__half(0)) {}

  __host__ __device__ half4(__half x, __half y, __half z, __half w) : x(x), y(y), z(z), w(w) {}

  __host__ __device__ explicit half4(const __nv_fp8x4_e4m3& fp8x4) {
    __nv_fp8x2_e4m3 lo_part, hi_part;
    lo_part.__x = static_cast<__nv_fp8x2_storage_t>(fp8x4.__x & 0xFFFF);
    hi_part.__x = static_cast<__nv_fp8x2_storage_t>((fp8x4.__x >> 16) & 0xFFFF);

    __half2 lo_half2 = static_cast<__half2>(lo_part);
    __half2 hi_half2 = static_cast<__half2>(hi_part);

    x = reinterpret_cast<__half*>(&lo_half2)[0];
    y = reinterpret_cast<__half*>(&lo_half2)[1];
    z = reinterpret_cast<__half*>(&hi_half2)[0];
    w = reinterpret_cast<__half*>(&hi_half2)[1];
  }

  __host__ __device__ explicit operator __nv_fp8x4_e4m3() const {
    __nv_fp8x4_e4m3 result;

    __half2 lo_half2 = *reinterpret_cast<const __half2*>(&x);
    __half2 hi_half2 = *reinterpret_cast<const __half2*>(&z);

    __nv_fp8x2_e4m3 lo_part(lo_half2), hi_part(hi_half2);
    result.__x =
        (static_cast<__uint32_t>(lo_part.__x) | (static_cast<__uint32_t>(hi_part.__x) << 16));
    return result;
  }
};

__global__ void vectorAddPTX(const float* A, const float* B, float* C, int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numElements) {
    // Inline PTX to perform the addition
    // %0, %1, and %2 correspond to C[i], A[i], and B[i] respectively
    asm("{\n\t"
        "add.f32 %0, %1, %2;\n\t"
        "}"
        : "=f"(C[i])
        : "f"(A[i]), "f"(B[i]));
  }
}

int test_ptx() {
  int numElements = 50000;
  size_t size = numElements * sizeof(float);
  float* h_A = new float[numElements];
  float* h_B = new float[numElements];
  float* h_C = new float[numElements];

  // Initialize input vectors
  for (int i = 0; i < numElements; ++i) {
    h_A[i] = static_cast<float>(i);
    h_B[i] = static_cast<float>(i * 2);
  }

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  cudaMalloc((void**)&d_A, size);
  cudaMalloc((void**)&d_B, size);
  cudaMalloc((void**)&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  vectorAddPTX<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << RED << "Failed to launch vectorAddPTX kernel (error code "
              << cudaGetErrorString(err) << ")" << RESET << std::endl;
    return -1;
  }

  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < numElements; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      std::cerr << BOLD_RED << "[FAILED]" << YELLOW << " Result verification failed at element "
                << i << RESET << std::endl;
      return -1;
    }
  }

  std::cout << GREEN << "[PASSED]" << YELLOW << " ptx_vector_add" << RESET << std::endl;

  // Free device global memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Free host memory
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  return 0;
}

__global__ void testFp8Conversion() {
  __half2_raw input = __floats2half2_rn(1.0f, 0.5f);
  __nv_fp8x2_storage_t fp8x2 = __nv_cvt_halfraw2_to_fp8x2(input, __NV_SATFINITE, __NV_E5M2);
  __half2_raw output = __nv_cvt_fp8x2_to_halfraw2(fp8x2, __NV_E5M2);
  assert(input.x == output.x && input.y == output.y);
}

int test_fp8_conversion() {
  testFp8Conversion<<<1, 1>>>();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << BOLD_RED << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    std::cerr << RED << "[FAILED]" << YELLOW << " test_fp8_conversion" << RESET << std::endl;
    return -1;
  }

  cudaDeviceSynchronize();

  std::cout << GREEN << "[PASSED]" << YELLOW << " test_fp8_conversion" << RESET << std::endl;
  return 0;
}
__global__ void testFp8VectorConversion() {
  half2 h_;
  h_.x = 5.75;
  h_.y = 1.33;
  printf("h_.x = %f\n", __half2float(h_.x));
  printf("h_.y = %f\n", __half2float(h_.y));

  fp8_e4_2_t v_ = (fp8_e4_2_t)(h_);
  h_ = (half2)(v_);

  printf("h_.x = %f\n", __half2float(h_.x));
  printf("h_.y = %f\n", __half2float(h_.y));
}

int test_fp8_vector_conversion() {
  testFp8VectorConversion<<<1, 1>>>();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << BOLD_RED << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    std::cerr << RED << "[FAILED]" << YELLOW << " test_fp8_conversion" << RESET << std::endl;
    return -1;
  }

  cudaDeviceSynchronize();

  std::cout << GREEN << "[PASSED]" << YELLOW << " test_fp8_vector_conversion" << RESET << std::endl;
  return 0;
}

__global__ void testFp8x4Half4_2_RoundTrip() {
  // Test values for half4
  __half2 lo;
  lo.x = 3.88;
  lo.y = 1.23;
  __half2 hi;
  hi.x = 5.2;
  hi.y = 4;

  half4_2 original_half4(lo, hi);

  __nv_fp8x4_e4m3 converted_fp8x4 = static_cast<__nv_fp8x4_e4m3>(original_half4);
  half4_2 round_trip_half4 = static_cast<half4_2>(converted_fp8x4);

  printf("h4.lo.x = %f\n", __half2float(original_half4.lo.x));
  printf("h4.lo.y = %f\n", __half2float(original_half4.lo.y));
  printf("h4.hi.x = %f\n", __half2float(original_half4.hi.x));
  printf("h4.hi.y = %f\n\n", __half2float(original_half4.hi.y));

  printf("h4.lo.x = %f\n", __half2float(round_trip_half4.lo.x));
  printf("h4.lo.y = %f\n", __half2float(round_trip_half4.lo.y));
  printf("h4.hi.x = %f\n", __half2float(round_trip_half4.hi.x));
  printf("h4.hi.y = %f\n", __half2float(round_trip_half4.hi.y));
}

__global__ void testFp8x4Half4RoundTrip() {
  // Test values for half4
  __half x = 3.88;
  __half y = 1.23;
  __half z = 5.2;
  __half w = 4;

  half4 original_half4(x, y, z, w);

  __nv_fp8x4_e4m3 converted_fp8x4 = static_cast<__nv_fp8x4_e4m3>(original_half4);
  half4 round_trip_half4 = static_cast<half4>(converted_fp8x4);

  printf("h4.x = %f\n", __half2float(original_half4.x));
  printf("h4.y = %f\n", __half2float(original_half4.y));
  printf("h4.z = %f\n", __half2float(original_half4.z));
  printf("h4.w = %f\n\n", __half2float(original_half4.w));

  printf("h4.x = %f\n", __half2float(round_trip_half4.x));
  printf("h4.y = %f\n", __half2float(round_trip_half4.y));
  printf("h4.z = %f\n", __half2float(round_trip_half4.z));
  printf("h4.w = %f\n", __half2float(round_trip_half4.w));
}

int test_fp8x4_half4_2_conversion() {
  testFp8x4Half4_2_RoundTrip<<<1, 1>>>();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << BOLD_RED << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    std::cerr << RED << "[FAILED]" << YELLOW << " test_fp8x4_half4_2_conversion" << RESET
              << std::endl;
    return -1;
  }

  cudaDeviceSynchronize();

  std::cout << GREEN << "[PASSED]" << YELLOW << " test_fp8x4_half4_2_conversion" << RESET
            << std::endl;
  return 0;
}

int test_fp8x4_half4_conversion() {
  testFp8x4Half4RoundTrip<<<1, 1>>>();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << BOLD_RED << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    std::cerr << RED << "[FAILED]" << YELLOW << " test_fp8x4_half4_conversion" << RESET
              << std::endl;
    return -1;
  }

  cudaDeviceSynchronize();

  std::cout << GREEN << "[PASSED]" << YELLOW << " test_fp8x4_half4_conversion" << RESET
            << std::endl;
  return 0;
}

int main(int argc, char* argv[]) {
  // test_ptx();
  // test_fp8_conversion();
  // test_fp8_vector_conversion();
  // test_fp8x4_half4_2_conversion();
  test_fp8x4_half4_conversion();
  return 0;
}
