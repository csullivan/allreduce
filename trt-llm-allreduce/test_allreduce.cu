#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>
#include "cuda_runtime.h"
#include "mpi.h"

#include "tensorrt_llm/kernels/customAllReduceKernels.h"

const char* RED = "\033[31m";
const char* GREEN = "\033[32m";
const char* YELLOW = "\033[33m";
const char* BOLD_RED = "\033[1;31m";
const char *RESET = "\033[0m";
  

class CUDAIpcMemory {
public:
    CUDAIpcMemory(size_t bufferSize, int worldSize, int rank);
    ~CUDAIpcMemory();

    void* getBufferPtr(int rank) const;
    void* const* getIpcHandles() const; 


private:
    size_t bufferSize;
    int worldSize;
    int rank;
    std::vector<void*> ipcHandles;
    void* localBufferPtr = nullptr;
    char* allHandles = nullptr;

    void allocateAndShare();
    void deallocate();
};

CUDAIpcMemory::CUDAIpcMemory(size_t bufferSize, int worldSize, int rank)
    : bufferSize(bufferSize), worldSize(worldSize), rank(rank), ipcHandles(worldSize) {
    allocateAndShare();
}

CUDAIpcMemory::~CUDAIpcMemory() {
    deallocate();
}

void CUDAIpcMemory::allocateAndShare() {
    
  cudaMalloc(&localBufferPtr, bufferSize);
  cudaMemset(localBufferPtr, 0, bufferSize);

    
  cudaIpcMemHandle_t localHandle;
  cudaIpcGetMemHandle(&localHandle, localBufferPtr);
    char serializedHandle[CUDA_IPC_HANDLE_SIZE];
    memcpy(serializedHandle, localHandle.reserved, CUDA_IPC_HANDLE_SIZE);

    
    allHandles = new char[CUDA_IPC_HANDLE_SIZE * worldSize];
    MPI_Allgather(serializedHandle, CUDA_IPC_HANDLE_SIZE, MPI_BYTE, 
                  allHandles, CUDA_IPC_HANDLE_SIZE, MPI_BYTE, 
                  MPI_COMM_WORLD);

    
    for (int i = 0; i < worldSize; ++i) {
        if (i == rank) {
            ipcHandles[i] = localBufferPtr;
        } else {
          cudaIpcMemHandle_t foreignHandle;
          memcpy(foreignHandle.reserved, &allHandles[i * CUDA_IPC_HANDLE_SIZE], CUDA_IPC_HANDLE_SIZE);
          cudaIpcOpenMemHandle(&ipcHandles[i], foreignHandle, cudaIpcMemLazyEnablePeerAccess);
        }
    }
}

void CUDAIpcMemory::deallocate() {
    for (int i = 0; i < worldSize; ++i) {
        if (i != rank) {
          cudaIpcCloseMemHandle(ipcHandles[i]);
        }
    }
    cudaFree(localBufferPtr);
    delete[] allHandles;
}

void* CUDAIpcMemory::getBufferPtr(int rank) const {
    return ipcHandles.at(rank);
}

void* const* CUDAIpcMemory::getIpcHandles() const {
  return const_cast<void**>(ipcHandles.data());
}

class CustomAllReduce {
public:
  CustomAllReduce(int worldSize, int rank);
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
  tensorrt_llm::kernels::AllReduceStrategyType selectImplementation(size_t messageSize, int worldSize) noexcept;
};

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


int test_one_shot_allreduce(int world_size, int rank) {
    const int data_size = 16;
    float* d_buffer;
    float h_buffer[data_size];
    float result_buffer[data_size];
    cudaMalloc(&d_buffer, data_size * sizeof(float));
    for (int i = 0; i < data_size; i++) {
        h_buffer[i] = static_cast<float>(rank + 1);
    }
    cudaMemcpy(d_buffer, h_buffer, data_size * sizeof(float), cudaMemcpyHostToDevice);


    CustomAllReduce allReduce(world_size, rank);
    allReduce.enqueue(d_buffer, data_size);

    
    cudaDeviceSynchronize();
    cudaMemcpy(result_buffer, d_buffer, data_size * sizeof(float), cudaMemcpyDeviceToHost);
    float expected = 0.0;
    for (int rank = 0; rank < world_size; rank++) {
      expected += (rank + 1);
    }
    for (int i = 0; i < data_size; i++) {
      if (result_buffer[i] != expected) {
        std::cerr << BOLD_RED << "Verification failed at rank " << rank << ", index " << i <<  RESET << std::endl;
        std::cerr << BOLD_RED << "~~~~~~> " << result_buffer[i] << " != " << expected << RESET << std::endl;
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
  
  if (test_one_shot_allreduce(world_size, rank)) {
    std::cout << RED << "[FAILED]" << YELLOW << " test_one_shot_allreduce" << RESET << std::endl;
  } else {
    std::cout << GREEN << "[PASSED]" << YELLOW << " test_one_shot_allreduce" << RESET << std::endl;
  }
  MPI_Finalize();
  
  return 0;
}
