#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>
#include "cuda_runtime.h"
#include "mpi.h"

#include "tensorrt_llm/kernels/customAllReduceKernels.h"

#define CUDA_CHECK(call)                                                       \
  {                                                                            \
    cudaError_t cudaStatus = call;                                             \
    if (cudaStatus != cudaSuccess) {                                           \
      std::cerr << "CUDA error at line " << __LINE__ << ": "                   \
                << cudaGetErrorString(cudaStatus) << std::endl;                \
      assert(cudaStatus != cudaSuccess);                                       \
    }                                                                          \
  }

int test_single_ipc_memory(int world_size, int rank) {
    
    const size_t ipcBufferSize = 50331648; 
    void* localBufferPtr;
    CUDA_CHECK(cudaMalloc(&localBufferPtr, ipcBufferSize));
    CUDA_CHECK(cudaMemset(localBufferPtr, 0, ipcBufferSize));

    
    cudaIpcMemHandle_t localHandle;
    CUDA_CHECK(cudaIpcGetMemHandle(&localHandle, localBufferPtr));

    
    char serializedHandle[CUDA_IPC_HANDLE_SIZE];
    memcpy(serializedHandle, localHandle.reserved, CUDA_IPC_HANDLE_SIZE);

    
    char* allHandles = new char[CUDA_IPC_HANDLE_SIZE * world_size];
    MPI_Allgather(serializedHandle, CUDA_IPC_HANDLE_SIZE, MPI_BYTE,
                  allHandles, CUDA_IPC_HANDLE_SIZE, MPI_BYTE,
                  MPI_COMM_WORLD);

    
    std::vector<void*> ipcHandles(world_size);
    for (int i = 0; i < world_size; ++i) {
        if (i == rank) {
            ipcHandles[i] = localBufferPtr; 
        } else {
          
          cudaIpcMemHandle_t foreignHandle;
          memcpy(foreignHandle.reserved, &allHandles[i * CUDA_IPC_HANDLE_SIZE], CUDA_IPC_HANDLE_SIZE);
          
          CUDA_CHECK(cudaIpcOpenMemHandle(&ipcHandles[i], foreignHandle, cudaIpcMemLazyEnablePeerAccess));
        }
    }

    
    

    
    for (int i = 0; i < world_size; ++i) {
        if (i != rank) {
          CUDA_CHECK(cudaIpcCloseMemHandle(ipcHandles[i]));
        }
    }
    cudaFree(localBufferPtr);

    delete[] allHandles;

    return 0;
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

int test_multiple_ipc_memory(int world_size, int rank) {
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

    
    size_t messageSize = data_size * sizeof(float);

    
    auto strategy = selectImplementation(messageSize, world_size);

    
    cudaStream_t stream(0);
    tensorrt_llm::kernels::invokeMultiGpuBarrier(params, stream);

    // TODO(csullivan): Enable the all reduce
    // CUDA_CHECK(cudaMemcpyAsync(
    //              params.peer_comm_buffer_ptrs[rank], d_buffer, size * sizePerElem, cudaMemcpyDeviceToDevice, stream));

    // // Invoke the all-reduce operation
    // tensorrt_llm::kernels::customAllReduce(params, d_buffer, data_size, sizeof(float),
    //                                        tensorrt_llm::common::datatype_enum::TYPE_FP32,
    //                                        strategy, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // TODO(csullivan): Add assert all_close
    return 0;
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int world_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  cudaSetDevice(rank);
  
  // test_single_ipc_memory(world_size, rank);
  test_multiple_ipc_memory(world_size, rank);
  MPI_Finalize();
  return 0;
}
