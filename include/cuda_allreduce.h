#pragma once

#include <nccl.h>

#include <cstdint>

#include "cuda_ipc_memory.h"

namespace tensorrt_llm {
namespace kernels {
struct AllReduceParams;
}
}  // namespace tensorrt_llm

class CustomAllReduce {
 public:
  CustomAllReduce(int worldSize, int rank, ncclComm_t ncclComm);
  ~CustomAllReduce();

  bool enqueue(void* input, void* output, int64_t num_elements, size_t type_size,
               ncclDataType_t type, ncclRedOp_t op_type, cudaStream_t stream);

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
