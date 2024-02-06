#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

#include "cuda_allreduce.h"
#include "cuda_runtime.h"
#include "mpi.h"
#include "tensorrt_llm/kernels/customAllReduceKernels.h"

const char* RED = "\033[31m";
const char* GREEN = "\033[32m";
const char* YELLOW = "\033[33m";
const char* RESET = "\033[0m";

// Benchmark function for a single data size.
double benchmark_allreduce(int world_size, int rank, ncclComm_t comm, int data_size) {
  float* d_buffer;
  cudaMalloc(&d_buffer, data_size * sizeof(float));

  std::vector<float> h_buffer(data_size);
  for (int i = 0; i < data_size; i++) {
    h_buffer[i] = static_cast<float>(drand48());  // Using drand48 for simplicity.
  }
  cudaMemcpy(d_buffer, h_buffer.data(), data_size * sizeof(float), cudaMemcpyHostToDevice);

  cudaStream_t stream(0);
  CustomAllReduce allReduce(world_size, rank, comm);

  size_t num_warmup = 5;
  for (int i = 0; i < num_warmup; ++i) {
    if (!allReduce.enqueue(d_buffer, d_buffer, data_size, sizeof(float), ncclFloat32, ncclSum,
                           stream)) {
      ncclAllReduce(d_buffer, d_buffer, data_size, ncclFloat32, ncclSum, comm, stream);
    }
    // cudaStreamSynchronize(stream);
  }

  size_t num_iterations = 100;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_iterations; ++i) {
    if (!allReduce.enqueue(d_buffer, d_buffer, data_size, sizeof(float), ncclFloat32, ncclSum,
                           stream)) {
      ncclAllReduce(d_buffer, d_buffer, data_size, ncclFloat32, ncclSum, comm, stream);
    }
    // cudaStreamSynchronize(stream);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;

  cudaFree(d_buffer);

  return diff.count() / static_cast<double>(num_iterations);
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int world_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  cudaSetDevice(rank);
  ncclComm_t comm;
  ncclUniqueId id;
  if (rank == 0) {
    ncclGetUniqueId(&id);
  }
  MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclCommInitRank(&comm, world_size, id, rank);

  for (int exp = 11; exp <= 30; exp++) {
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

  ncclCommDestroy(comm);
  MPI_Finalize();

  return 0;
}
