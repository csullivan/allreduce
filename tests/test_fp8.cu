#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

#include "cuda_runtime.h"

const char* RED = "\033[31m";
const char* GREEN = "\033[32m";
const char* YELLOW = "\033[33m";
const char* BOLD_RED = "\033[1;31m";
const char* RESET = "\033[0m";

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

int main(int argc, char* argv[]) {
  test_ptx();
  return 0;
}
