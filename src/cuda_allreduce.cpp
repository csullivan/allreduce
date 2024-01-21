#include "cuda_allreduce.h"

CustomAllReduce::CustomAllReduce(int worldSize, int rank)
  : m_world_size(worldSize), m_rank(rank), m_flag_value(2112),
  m_buffer_ptrs(IPC_BUFFERS_SIZE, worldSize, rank),
  m_input_barriers(IPC_BARRIERS_SIZE_PER_GPU * worldSize, worldSize, rank),
  m_output_barriers(IPC_BARRIERS_SIZE_PER_GPU * worldSize, worldSize, rank)
{}

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
    }
    void* const* barrier_ptrs_out = m_output_barriers.getIpcHandles();
    for (int i = 0; i < m_world_size; ++i) {
        params.peer_barrier_ptrs_out[i] = reinterpret_cast<uint32_t*>(barrier_ptrs_out[i]);
    }
    params.barrier_flag = m_flag_value;
    params.ranks_per_node = m_world_size;
    params.rank = m_rank;
    params.local_rank = m_rank;

    return params;
}

tensorrt_llm::kernels::AllReduceStrategyType CustomAllReduce::selectImplementation(size_t messageSize, int worldSize) noexcept
{
    using namespace tensorrt_llm::kernels;
    if (worldSize <= 2)
    {
        if (messageSize < 16 * 1000 * 1000)
        {
            return AllReduceStrategyType::ONESHOT;
        }
    }

    if (worldSize > 2 && worldSize <= 4)
    {
        if (messageSize < 1 * 1000 * 1000)
        {
            return AllReduceStrategyType::ONESHOT;
        }
        if (messageSize < 8 * 1000 * 1000)
        {
            return AllReduceStrategyType::TWOSHOT;
        }
    }

    if (worldSize > 4)
    {
        if (messageSize < 500 * 1000)
        {
            return AllReduceStrategyType::ONESHOT;
        }
        if (messageSize < 8 * 1000 * 1000)
        {
            return AllReduceStrategyType::TWOSHOT;
        }
    }

    return AllReduceStrategyType::RING;
}

void CustomAllReduce::enqueue(float* d_buffer, size_t dataSize) {
   m_flag_value++; // Increment the flag value

    auto params = setupParams();
    params.barrier_flag = m_flag_value;

    size_t messageSize = dataSize * sizeof(float);
    auto strategy = selectImplementation(messageSize, m_world_size);

    cudaStream_t stream(0);
    tensorrt_llm::kernels::invokeMultiGpuBarrier(params, stream);

    cudaMemcpyAsync(
                 params.peer_comm_buffer_ptrs[m_rank], d_buffer, messageSize, cudaMemcpyDeviceToDevice, stream);

    tensorrt_llm::kernels::customAllReduce(params, d_buffer, dataSize, sizeof(float),
                                           tensorrt_llm::common::datatype_enum::TYPE_FP32,
                                           strategy, 0);
}
