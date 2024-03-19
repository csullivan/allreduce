#include <algorithm>
#include <cassert>
#include <iostream>
#include <unordered_map>

#include "mscclpp/core.hpp"
#include "mscclpp/sm_channel.hpp"
#include "mscclpp/sm_channel_device.hpp"
#include "mscclpp_allreduce.h"

#define NUM_CHANNELS_PER_CONNECTION 64

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
    : m_world_size(worldSize), m_rank(rank), m_flag_value(1), m_comm(comm) {}

MscclppAllReduce::~MscclppAllReduce() {}

static std::vector<mscclpp::SmChannel> setupSmChannels(
    ncclComm_t comm, const std::vector<mscclpp::RegisteredMemory>& remoteMemories, void* src) {
  std::vector<mscclpp::SmChannel> channels;
  std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>>& smSemaphores =
      comm->smSemaphores;
  size_t nConnections = comm->connections.size();
  for (size_t idx = 0; idx < NUM_CHANNELS_PER_CONNECTION; ++idx) {
    for (size_t cid = 0; cid < nConnections; ++cid) {
      if (comm->connections[cid]->transport() == mscclpp::Transport::CudaIpc) {
        channels.emplace_back(smSemaphores[idx * nConnections + cid], remoteMemories[cid], src,
                              nullptr);
      }
    }
  }
  return channels;
}

static std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SmChannel>> setupSmChannelDeviceHandles(
    const std::vector<mscclpp::SmChannel>& smChannels) {
  std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> smChannelDeviceHandles;
  std::transform(
      smChannels.begin(), smChannels.end(), std::back_inserter(smChannelDeviceHandles),
      [](const mscclpp::SmChannel& smChannel) { return mscclpp::deviceHandle(smChannel); });
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SmChannel>> ptr =
      mscclpp::allocSharedCuda<mscclpp::DeviceHandle<mscclpp::SmChannel>>(
          smChannelDeviceHandles.size());
  mscclpp::memcpyCuda<mscclpp::DeviceHandle<mscclpp::SmChannel>>(
      ptr.get(), smChannelDeviceHandles.data(), smChannelDeviceHandles.size(),
      cudaMemcpyHostToDevice);
  return ptr;
}

namespace {

template <typename To, typename From>
__forceinline__ __device__ To bit_cast(const From& src) {
  static_assert(sizeof(To) == sizeof(From), "Size mismatch for bit_cast");

  union {
    From f;
    To t;
  } u;
  u.f = src;
  return u.t;
}

template <typename T>
__forceinline__ __device__ T add_elements(T a, T b) {
  return a + b;
}

template <>
__forceinline__ __device__ __half2 add_elements(__half2 a, __half2 b) {
  return __hadd2(a, b);
}

template <typename T>
__forceinline__ __device__ int4 add_vectors_helper(int4 a, int4 b) {
  int4 ret;
  ret.w = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.w), bit_cast<T, int>(b.w)));
  ret.x = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  ret.z = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.z), bit_cast<T, int>(b.z)));
  return ret;
}

template <typename T>
__forceinline__ __device__ int4 add_vectors(int4 a, int4 b) {
  return add_vectors_helper<T>(a, b);
}

template <>
__forceinline__ __device__ int4 add_vectors<__half>(int4 a, int4 b) {
  return add_vectors_helper<__half2>(a, b);
}

template <typename T>
__forceinline__ __device__ uint2 add_vectors_helper(uint2 a, uint2 b) {
  uint2 ret;
  ret.x = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  return ret;
}

template <typename T>
__forceinline__ __device__ uint2 add_vectors(uint2 a, uint2 b) {
  return add_vectors_helper<T>(a, b);
}

template <>
__forceinline__ __device__ uint2 add_vectors<__half>(uint2 a, uint2 b) {
  return add_vectors_helper<__half2>(a, b);
}

template <typename T>
__forceinline__ __device__ int add_vectors_helper(int a, int b) {
  return bit_cast<int, T>(add_elements(bit_cast<T, int>(a), bit_cast<T, int>(b)));
}

template <typename T>
__forceinline__ __device__ int add_vectors(int a, int b) {
  return add_vectors_helper<T>(a, b);
}

template <>
__forceinline__ __device__ int add_vectors<__half>(int a, int b) {
  return add_vectors_helper<__half2>(a, b);
}

__device__ uint64_t globalFlag = 1;

template <typename TYPE>
__global__ void __launch_bounds__(1024, 1)
    allreduce_simple(mscclpp::SmChannelDeviceHandle* smChans, TYPE* buff, TYPE* scratch,
                     void* resultBuff, int rank, int worldSize, size_t nelems) {
  nelems = nelems / (sizeof(int) / sizeof(TYPE));

  const int nPeers = worldSize - 1;
  const size_t nPkts = nelems / 2;
  const int nelemsPerRank = nelems / worldSize;
  const int nPktsPerRank = nelemsPerRank / 2;
  const uint32_t flag = (uint32_t)globalFlag;
  const int nBlocksPerPeer = gridDim.x / nPeers;
  const int localBlockIdx = blockIdx.x % nBlocksPerPeer;
  const int peerIdx = blockIdx.x / nBlocksPerPeer;
  const int remoteRank = peerIdx < rank ? peerIdx : peerIdx + 1;
  mscclpp::SmChannelDeviceHandle smChan = smChans[peerIdx];
  const int tid = threadIdx.x + localBlockIdx * blockDim.x;

  size_t scratchOffset = rank * nPktsPerRank * sizeof(mscclpp::LLPacket);
  size_t resultOffset = 2 * nPkts * sizeof(mscclpp::LLPacket);  // Fixed result offset
  size_t srcOffset = remoteRank * nelemsPerRank * sizeof(int);
  uint2* src = (uint2*)((char*)buff + rank * nelemsPerRank * sizeof(int));
  uint2* dst = (uint2*)((char*)resultBuff + rank * nelemsPerRank * sizeof(int));

  // Step 1. Write to scratch buffer which exposes memory to peers
  smChan.putPackets(scratchOffset, srcOffset, nelemsPerRank * sizeof(int), tid,
                    blockDim.x * nBlocksPerPeer, flag);

  // Step 2. Get data from scratch buffer, reduce data, and write result back to peer scratch
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPktsPerRank;
       idx += blockDim.x * gridDim.x) {
    uint2 data = make_uint2(0, 0);
    for (int index = 0; index < nPeers; index++) {
      const int remoteRank = index < rank ? index : index + 1;
      mscclpp::LLPacket* dstPkt = (mscclpp::LLPacket*)scratch + remoteRank * nPktsPerRank;
      uint2 val = dstPkt[idx].read(flag);
      data = add_vectors<TYPE>(val, data);
    }
    data = add_vectors<TYPE>(data, src[idx]);
    dst[idx] = data;

    mscclpp::LLPacket packet;
    packet.data1 = data.x;
    packet.flag1 = flag;
    packet.data2 = data.y;
    packet.flag2 = flag;
    size_t offset = resultOffset / sizeof(mscclpp::LLPacket) + (idx + rank * nPktsPerRank);
    for (int index = 0; index < nPeers; index++) {
      smChans[index].write(offset, packet);
    }
  }

  // Step 3. Update local gpus final result from peer scratch buffers
  mscclpp::LLPacket* dstPkt = (mscclpp::LLPacket*)((char*)scratch + resultOffset);
  const int dstOffset = remoteRank * nPktsPerRank;
  uint2* result = (uint2*)((char*)resultBuff + remoteRank * nelemsPerRank * sizeof(int));
  for (int idx = threadIdx.x + localBlockIdx * blockDim.x; idx < nPktsPerRank;
       idx += blockDim.x * nBlocksPerPeer) {
    uint2 data = dstPkt[idx + dstOffset].read(flag);
    result[idx].x = data.x;
    result[idx].y = data.y;
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    globalFlag += 1;
  }
}

}  // namespace

ncclResult_t MscclppAllReduce::enqueue(void* input, void* output, int64_t num_elements,
                                       size_t type_size, ncclDataType_t datatype,
                                       ncclRedOp_t op_type, cudaStream_t stream) {
  size_t bytes = num_elements * type_size;
  if (input == nullptr || output == nullptr || bytes == 0 || m_comm == nullptr) {
    return ncclInvalidArgument;
  }
  // TODO(csullivan): Need to evaluate ross over point with nccl -- likely around 2**20
  if (bytes > (1 << 24)) {
    return ncclInvalidArgument;
  }
  int rank = m_comm->comm->bootstrap()->getRank();
  channelKey key{input, output, bytes};
  mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels = nullptr;

  auto it = m_comm->channelInfos.find(key);
  if (it == m_comm->channelInfos.end()) {
    // setup smChannels (src: input, dst: remote scratch buff)
    std::vector<mscclpp::SmChannel> channels =
        setupSmChannels(m_comm, m_comm->remoteScratchRegMemories, const_cast<void*>(input));
    ChannelInfo channelInfo{channels, {}, setupSmChannelDeviceHandles(channels), nullptr};
    it = m_comm->channelInfos.emplace(key, channelInfo).first;
  }

  smChannels = it->second.smChannelDeviceHandles.get();

  int num_blocks = 105;
  int num_threads = 1024;
  switch (datatype) {
    case ncclFloat16:
      allreduce_simple<<<num_blocks, num_threads, 0, stream>>>(
          smChannels, (half*)input, (half*)m_comm->scratchBuff.get(), (half*)output, m_rank,
          m_world_size, num_elements);
      break;
    case ncclFloat32:
      allreduce_simple<<<num_blocks, num_threads, 0, stream>>>(
          smChannels, (float*)input, (float*)m_comm->scratchBuff.get(), (float*)output, m_rank,
          m_world_size, num_elements);
      break;
    default:
      std::cout << 1 << std::endl;

      return ncclInvalidArgument;
  }
  return ncclSuccess;
}
