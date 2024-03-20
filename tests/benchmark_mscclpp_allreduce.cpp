#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

#include "cuda_runtime.h"
#include "mpi.h"
#include "mscclpp_allreduce.h"

const char* RED = "\033[31m";
const char* BOLD_RED = "\033[1;31m";
const char* GREEN = "\033[32m";
const char* YELLOW = "\033[33m";
const char* RESET = "\033[0m";

// Benchmark function for a single data size.
double benchmark_allreduce(int world_size, int rank, mscclComm_t comm, int data_size) {
  float* d_buffer;
  float* o_buffer;
  cudaMalloc(&d_buffer, data_size * sizeof(float));
  cudaMalloc(&o_buffer, data_size * sizeof(float));

  std::vector<float> h_buffer(data_size);
  std::vector<float> result_buffer(data_size, 0.0f);

  for (int i = 0; i < data_size; i++) {
    // h_buffer[i] = static_cast<float>(drand48());  // Using drand48 for simplicity.
    h_buffer[i] = static_cast<float>(rank + 1);
  }
  cudaMemcpy(d_buffer, h_buffer.data(), data_size * sizeof(float), cudaMemcpyHostToDevice);

  cudaStream_t stream(0);

  size_t num_warmup = 10;
  for (int i = 0; i < num_warmup; ++i) {
    mscclAllReduce(d_buffer, o_buffer, data_size, mscclFloat32, mscclSum, comm, stream);
  }
  cudaDeviceSynchronize();
  cudaMemcpy(result_buffer.data(), o_buffer, data_size * sizeof(float), cudaMemcpyDeviceToHost);
  float expected = 0.0;
  for (int rank = 0; rank < world_size; rank++) {
    expected += (rank + 1);
  }
  for (int i = 0; i < data_size; i++) {
    if (result_buffer[i] != expected) {
      std::cerr << BOLD_RED << "Verification failed in warmup at rank " << rank << ", index " << i
                << RESET << std::endl;
      std::cerr << BOLD_RED << "~~~~~~> " << result_buffer[i] << " != " << expected << RESET
                << std::endl;
      throw std::runtime_error("Verification failed");
    }
  }

  if (rank == 0) {
    std::cerr << GREEN << "[PASSED]" << YELLOW << " verification test" << RESET << std::endl;
  }

  size_t num_iterations = 100;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_iterations; ++i) {
    mscclAllReduce(d_buffer, o_buffer, data_size, mscclFloat32, mscclSum, comm, stream);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;

  cudaFree(d_buffer);
  cudaFree(o_buffer);

  return diff.count() / static_cast<double>(num_iterations);
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int world_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  cudaSetDevice(rank);
  mscclComm_t comm;
  mscclUniqueId id;
  if (rank == 0) {
    mscclGetUniqueId(&id);
  }
  MPI_Bcast(&id, sizeof(mscclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
  mscclCommInitRank(&comm, world_size, id, rank);

  for (int exp = 12; exp < 24; exp++) {
    int data_size = 1 << exp;
    double avg_duration = benchmark_allreduce(world_size, rank, comm, data_size / sizeof(float));
    double data_size_mb = data_size / static_cast<double>(1024 * 1024);
    double bandwidth_mb_s = (data_size_mb / avg_duration);
    if (rank == 0) {
      std::cout << "Data Size: " << data_size_mb << " MiB, Bandwidth: " << bandwidth_mb_s / 1024.0
                << " GiB/s"
                << " Latency: " << avg_duration << " s" << std::endl;
    }
  }

  mscclCommDestroy(comm);
  MPI_Finalize();

  return 0;
}
