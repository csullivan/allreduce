#pragma once

#include "cuda_ipc_memory.h"
#include <nccl.h>

namespace tensorrt_llm {
namespace kernels {
    struct AllReduceParams;
}
}

class CustomAllReduce {
public:
  CustomAllReduce(int worldSize, int rank, ncclComm_t ncclComm);
  ~CustomAllReduce();

  void enqueue(float* d_buffer, size_t dataSize);

private:
  int m_world_size;
  int m_rank;
  unsigned int m_flag_value;
  CUDAIpcMemory m_buffer_ptrs;
  CUDAIpcMemory m_input_barriers;
  CUDAIpcMemory m_output_barriers;
  static const size_t IPC_BUFFERS_SIZE = 50331648; 
  static const size_t IPC_BARRIERS_SIZE_PER_GPU = 25 * 4; 

  tensorrt_llm::kernels::AllReduceParams setupParams();
  int selectImplementation(size_t messageSize, int worldSize) noexcept;
};

