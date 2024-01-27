#include "cuda_allreduce.h"

#include <cassert>

#include "tensorrt_llm/kernels/customAllReduceKernels.h"
#include "tensorrt_llm/common/tensor.h"

// #define CUDA_CALL(call)                                                                        \
//   {                                                                                            \
//     cudaError_t cudaStatus = call;                                                             \
//     if (cudaStatus != cudaSuccess) {                                                           \
//       std::cerr << "CUDA error at line " << __LINE__ << ": " << cudaGetErrorString(cudaStatus) \
//                 << std::endl;                                                                  \
//       assert(cudaStatus == cudaSuccess);                                                       \
//     }                                                                                          \
//   }

tensorrt_llm::common::datatype_enum select_type(ncclDataType_t ncclType) {
    switch (ncclType) {
        case ncclFloat32:
            return tensorrt_llm::common::datatype_enum::TYPE_FP32;
        case ncclFloat16:
            return tensorrt_llm::common::datatype_enum::TYPE_FP16;
        case ncclBfloat16:
            return tensorrt_llm::common::datatype_enum::TYPE_BF16;
        default:
            std::cerr << "Unsupported NCCL to TRT_LLM datatype conversion" << std::endl;
            return tensorrt_llm::common::datatype_enum::TYPE_INVALID;
    }
}


CustomAllReduce::CustomAllReduce(int worldSize, int rank, ncclComm_t ncclComm)
    : m_world_size(worldSize),
      m_rank(rank),
      m_flag_value(2112),
      m_buffer_ptrs(IPC_BUFFERS_SIZE, worldSize, rank, ncclComm),
      m_input_barriers(IPC_BARRIERS_SIZE_PER_GPU * worldSize, worldSize, rank, ncclComm),
      m_output_barriers(IPC_BARRIERS_SIZE_PER_GPU * worldSize, worldSize, rank, ncclComm) {}

CustomAllReduce::~CustomAllReduce() {}

tensorrt_llm::kernels::AllReduceParams CustomAllReduce::setupParams() {
  tensorrt_llm::kernels::AllReduceParams params;
  void* const* buffer_ptrs = m_buffer_ptrs.getIpcHandles();
  for (int i = 0; i < m_world_size; ++i) {
    params.peer_comm_buffer_ptrs[i] = buffer_ptrs[i];
  }
  void* const* barrier_ptrs_in = m_input_barriers.getIpcHandles();
  for (int i = 0; i < m_world_size; ++i) {
    params.peer_barrier_ptrs_in[i] = reinterpret_cast<uint32_t*>(barrier_ptrs_in[i]);
    // printf("barrier_ptrs_in[i] = %p\n", params.peer_barrier_ptrs_in[i]);
  }
  void* const* barrier_ptrs_out = m_output_barriers.getIpcHandles();
  for (int i = 0; i < m_world_size; ++i) {
    params.peer_barrier_ptrs_out[i] = reinterpret_cast<uint32_t*>(barrier_ptrs_out[i]);
    // printf("barrier_ptrs_out[i] = %p\n", params.peer_barrier_ptrs_out[i]);
  }
  params.barrier_flag = m_flag_value;
  params.ranks_per_node = m_world_size;
  params.rank = m_rank;
  params.local_rank = m_rank;

  return params;
}

int CustomAllReduce::selectImplementation(size_t messageSize, int worldSize) noexcept {
  using namespace tensorrt_llm::kernels;

  AllReduceStrategyType strategy;
  if (worldSize <= 2) {
    if (messageSize < 16 * 1000 * 1000) {
      strategy = AllReduceStrategyType::ONESHOT;
    } else {
      strategy = AllReduceStrategyType::RING;
    }
  } else if (worldSize > 2 && worldSize <= 4) {
    if (messageSize < 1 * 1000 * 1000) {
      strategy = AllReduceStrategyType::ONESHOT;
    } else if (messageSize < 8 * 1000 * 1000) {
      strategy = AllReduceStrategyType::TWOSHOT;
    } else {
      strategy = AllReduceStrategyType::RING;
    }

  } else if (worldSize > 4) {
    if (messageSize < 500 * 1000) {
      strategy = AllReduceStrategyType::ONESHOT;
    } else if (messageSize < 8 * 1000 * 1000) {
      strategy = AllReduceStrategyType::TWOSHOT;
    } else {
      strategy = AllReduceStrategyType::RING;
    }
  } else {
    strategy = AllReduceStrategyType::RING;
  }
  return static_cast<int>(strategy);
}

bool is_supported(int strategy, ncclDataType_t type, ncclRedOp_t op_type) {
  if (static_cast<tensorrt_llm::kernels::AllReduceStrategyType>(strategy) ==
      tensorrt_llm::kernels::AllReduceStrategyType::RING) {
    return false;
  }
  std::vector<ncclDataType_t> supported_types{ncclFloat16, ncclBfloat16, ncclFloat32};
  bool is_supported_type = false;
  for (auto supported_type : supported_types) {
    if (type == supported_type) {
      is_supported_type = true;
      break;
    }
  }
  if (!is_supported_type) {
    return false;
  }
  if (op_type != ncclSum) {
    return false;
  }

  return true;
}

bool CustomAllReduce::enqueue(void* input, void* output, int64_t num_elements, size_t type_size,
                              ncclDataType_t type, ncclRedOp_t op_type, cudaStream_t stream) {
  size_t messageSize = num_elements * type_size;
  int strategy = selectImplementation(messageSize, m_world_size);
  if (!is_supported(strategy, type, op_type)) {
    return false;
  }

  m_flag_value++;
  auto params = setupParams();
  params.barrier_flag = m_flag_value;

  tensorrt_llm::kernels::invokeMultiGpuBarrier(params, stream);
  cudaMemcpyAsync(params.peer_comm_buffer_ptrs[m_rank], input, messageSize,
                  cudaMemcpyDeviceToDevice, stream);

  tensorrt_llm::common::datatype_enum trt_type = select_type(type);

  tensorrt_llm::kernels::customAllReduce(
      params, output, num_elements, type_size, trt_type,
      static_cast<tensorrt_llm::kernels::AllReduceStrategyType>(strategy), 0);
  return true;
}
