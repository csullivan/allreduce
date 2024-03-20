#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

#include "cuda_runtime.h"
#include "mpi.h"
#include "mscclpp_allreduce.h"

const char* RED = "\033[31m";
const char* GREEN = "\033[32m";
const char* YELLOW = "\033[33m";
const char* BOLD_RED = "\033[1;31m";
const char* RESET = "\033[0m";

template <typename ALLREDUCE_T>
int test_allreduce(int world_size, int rank, mscclComm_t comm) {
  const int data_size = 16;
  float* d_buffer;
  float h_buffer[data_size];
  float result_buffer[data_size];
  cudaMalloc(&d_buffer, data_size * sizeof(float));
  for (int i = 0; i < data_size; i++) {
    h_buffer[i] = static_cast<float>(rank + 1);
  }
  cudaMemcpy(d_buffer, h_buffer, data_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaStream_t stream(0);
  ALLREDUCE_T allReduce(world_size, rank, comm);
  if (allReduce.enqueue(d_buffer, d_buffer, data_size, sizeof(float), mscclFloat32, mscclSum,
                        stream) != mscclSuccess) {
    std::cerr << BOLD_RED << "Requested All Reduce is unsupported" << std::endl;
    return 1;
  }

  cudaDeviceSynchronize();
  cudaMemcpy(result_buffer, d_buffer, data_size * sizeof(float), cudaMemcpyDeviceToHost);
  float expected = 0.0;
  for (int rank = 0; rank < world_size; rank++) {
    expected += (rank + 1);
  }
  for (int i = 0; i < data_size; i++) {
    if (result_buffer[i] != expected) {
      std::cerr << BOLD_RED << "Verification failed at rank " << rank << ", index " << i << RESET
                << std::endl;
      std::cerr << BOLD_RED << "~~~~~~> " << result_buffer[i] << " != " << expected << RESET
                << std::endl;
      return 1;
    }
  }

  cudaFree(d_buffer);
  return 0;
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
  // Broadcast the unique ID to other processes
  MPI_Bcast(&id, sizeof(mscclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
  cudaSetDevice(rank);
  mscclCommInitRank(&comm, world_size, id, rank);

  if (test_allreduce<MscclppAllReduce>(world_size, rank, comm)) {
    std::cout << RED << "[FAILED]" << YELLOW << " test_allreduce" << RESET << std::endl;
  } else {
    std::cout << GREEN << "[PASSED]" << YELLOW << " test_allreduce" << RESET << std::endl;
  }

  mscclCommDestroy(comm);
  MPI_Finalize();

  return 0;
}
