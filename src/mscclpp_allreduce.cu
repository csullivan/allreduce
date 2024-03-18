#include <cassert>
#include <iostream>
#include <mscclpp/core.hpp>
#include <mscclpp/sm_channel.hpp>
#include <mscclpp/sm_channel_device.hpp>
#include <unordered_map>

#include "mscclpp_allreduce.h"

// #define CUDA_CALL(call)                                                                        \
//   {                                                                                            \
//     cudaError_t cudaStatus = call;                                                             \
//     if (cudaStatus != cudaSuccess) {                                                           \
//       std::cerr << "CUDA error at line " << __LINE__ << ": " << cudaGetErrorString(cudaStatus) \
//                 << std::endl;                                                                  \
//       assert(cudaStatus == cudaSuccess);                                                       \
//     }                                                                                          \
//   }

// tensorrt_llm::common::datatype_enum select_type(ncclDataType_t ncclType) {
//     switch (ncclType) {
//         case ncclFloat32:
//             return tensorrt_llm::common::datatype_enum::TYPE_FP32;
//         case ncclFloat16:
//             return tensorrt_llm::common::datatype_enum::TYPE_FP16;
//         case ncclBfloat16:
//             return tensorrt_llm::common::datatype_enum::TYPE_BF16;
//         default:
//             std::cerr << "Unsupported NCCL to TRT_LLM datatype conversion" << std::endl;
//             return tensorrt_llm::common::datatype_enum::TYPE_INVALID;
//     }
// }

struct channelKey {
  const void* sendbuff;
  const void* recvbuff;
  size_t bytes;
  bool operator==(const channelKey& other) const {
    return sendbuff == other.sendbuff && recvbuff == other.recvbuff && bytes == other.bytes;
  }
};

namespace std {
template <>
struct hash<channelKey> {
  std::size_t operator()(const channelKey& k) const {
    return std::hash<const void*>()(k.sendbuff) ^ std::hash<const void*>()(k.recvbuff) ^
           std::hash<size_t>()(k.bytes);
  }
};
}  // namespace std

struct ChannelInfo {
  std::vector<mscclpp::SmChannel> smChannels;
  std::vector<mscclpp::SmChannel> smOutChannels;
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SmChannel>> smChannelDeviceHandles;
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SmChannel>> smOutChannelDeviceHandles;
};

struct ncclComm {
  std::shared_ptr<mscclpp::Communicator> comm;
  std::vector<std::shared_ptr<mscclpp::Connection>> connections;
  std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>> smSemaphores;

  std::unordered_map<channelKey, ChannelInfo> channelInfos;
  std::shared_ptr<char> scratchBuff;
  std::vector<mscclpp::RegisteredMemory> remoteScratchRegMemories;
};

MscclppAllReduce::MscclppAllReduce(int worldSize, int rank, ncclComm_t comm)
    : m_world_size(worldSize), m_rank(rank), m_flag_value(1) {
  mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels = nullptr;
  int rank2 = comm->comm->bootstrap()->getRank();
  std::cout << "MPI rank: " << rank << " MSCCLP rank: " << rank2 << std::endl;
}

MscclppAllReduce::~MscclppAllReduce() {}

bool MscclppAllReduce::enqueue(void* input, void* output, int64_t num_elements, size_t type_size,
                               ncclDataType_t type, ncclRedOp_t op_type, cudaStream_t stream) {
  size_t messageSize = num_elements * type_size;
  // int strategy = selectImplementation(messageSize, m_world_size);
  // if (!is_supported(strategy, type, op_type)) {
  //   return false;
  // }

  // auto params = setupParams();

  return true;
}
