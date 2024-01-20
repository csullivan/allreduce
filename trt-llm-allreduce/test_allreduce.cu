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
  
#define CUDA_CHECK(call)                                                       \
  {                                                                            \
    cudaError_t cudaStatus = call;                                             \
    if (cudaStatus != cudaSuccess) {                                           \
      std::cerr << "CUDA error at line " << __LINE__ << ": "                   \
                << cudaGetErrorString(cudaStatus) << std::endl;                \
      assert(cudaStatus != cudaSuccess);                                       \
    }                                                                          \
  }

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
    
  CUDA_CHECK(cudaMalloc(&localBufferPtr, bufferSize));
  CUDA_CHECK(cudaMemset(localBufferPtr, 0, bufferSize));

    
  cudaIpcMemHandle_t localHandle;
  CUDA_CHECK(cudaIpcGetMemHandle(&localHandle, localBufferPtr));
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
          CUDA_CHECK(cudaIpcOpenMemHandle(&ipcHandles[i], foreignHandle, cudaIpcMemLazyEnablePeerAccess));
        }
    }
}

void CUDAIpcMemory::deallocate() {
    for (int i = 0; i < worldSize; ++i) {
        if (i != rank) {
          CUDA_CHECK(cudaIpcCloseMemHandle(ipcHandles[i]));
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

tensorrt_llm::kernels::AllReduceStrategyType selectImplementation(size_t messageSize, int worldSize) noexcept
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

int test_one_shot_allreduce(int world_size, int rank) {
    const size_t IPC_BUFFERS_SIZE = 50331648; 
    const size_t IPC_BARRIERS_SIZE_PER_GPU = 25 * 4; 
    const int tpSize = world_size;

    
    CUDAIpcMemory BufferPtrs(IPC_BUFFERS_SIZE, world_size, rank);
    CUDAIpcMemory BarriersIn(IPC_BARRIERS_SIZE_PER_GPU * tpSize, world_size, rank);
    CUDAIpcMemory BarriersOut(IPC_BARRIERS_SIZE_PER_GPU * tpSize, world_size, rank);

    void* const* buffer_ptrs = BufferPtrs.getIpcHandles();
    void* const* barrier_ptrs_in = BarriersIn.getIpcHandles();
    void* const* barrier_ptrs_out = BarriersOut.getIpcHandles();
    tensorrt_llm::kernels::AllReduceParams params;

    
    for (int i = 0; i < tpSize; ++i) {
        params.peer_comm_buffer_ptrs[i] = buffer_ptrs[i];
    }
    for (int i = 0; i < tpSize; ++i) {
        params.peer_barrier_ptrs_in[i] = reinterpret_cast<uint32_t*>(barrier_ptrs_in[i]);
    }
    for (int i = 0; i < tpSize; ++i) {
        params.peer_barrier_ptrs_out[i] = reinterpret_cast<uint32_t*>(barrier_ptrs_out[i]);
    }

    params.barrier_flag = 123;;
    params.ranks_per_node = tpSize;
    params.rank = rank;
    params.local_rank = rank;


    
    
    const int data_size = 16;
    float* d_buffer;
    float h_buffer[data_size];
    float result_buffer[data_size];

    
    CUDA_CHECK(cudaMalloc(&d_buffer, data_size * sizeof(float)));

    
    for (int i = 0; i < data_size; i++) {
        h_buffer[i] = static_cast<float>(rank + 1);
    }
    CUDA_CHECK(cudaMemcpy(d_buffer, h_buffer, data_size * sizeof(float), cudaMemcpyHostToDevice));

    
    size_t message_size = data_size * sizeof(float);

    
    auto strategy = selectImplementation(message_size, world_size);

    
    cudaStream_t stream(0);
    tensorrt_llm::kernels::invokeMultiGpuBarrier(params, stream);

    CUDA_CHECK(cudaMemcpyAsync(
                 params.peer_comm_buffer_ptrs[rank], d_buffer, message_size, cudaMemcpyDeviceToDevice, stream));

    tensorrt_llm::kernels::customAllReduce(params, d_buffer, data_size, sizeof(float),
                                           tensorrt_llm::common::datatype_enum::TYPE_FP32,
                                           strategy, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(result_buffer, d_buffer, message_size, cudaMemcpyDeviceToHost));
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

    CUDA_CHECK(cudaFree(d_buffer));
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
