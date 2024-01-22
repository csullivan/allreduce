#include <cstring>
#include "cuda_ipc_memory.h"

CUDAIpcMemory::CUDAIpcMemory(size_t bufferSize, int worldSize, int rank, ncclComm_t ncclComm)
  : bufferSize(bufferSize), worldSize(worldSize), rank(rank), ipcHandles(worldSize), ncclComm(ncclComm) {
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

    char* d_allHandles;
    cudaMalloc(&d_allHandles, CUDA_IPC_HANDLE_SIZE * worldSize);
    cudaMemcpy(d_allHandles + rank * CUDA_IPC_HANDLE_SIZE, serializedHandle, CUDA_IPC_HANDLE_SIZE, cudaMemcpyHostToDevice);

    ncclAllGather(d_allHandles + rank * CUDA_IPC_HANDLE_SIZE, d_allHandles, CUDA_IPC_HANDLE_SIZE, ncclChar, ncclComm, 0);

    allHandles = new char[CUDA_IPC_HANDLE_SIZE * worldSize];
    cudaMemcpy(allHandles, d_allHandles, CUDA_IPC_HANDLE_SIZE * worldSize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < worldSize; ++i) {
        if (i == rank) {
            ipcHandles[i] = localBufferPtr;
        } else {
            cudaIpcMemHandle_t foreignHandle;
            memcpy(foreignHandle.reserved, &allHandles[i * CUDA_IPC_HANDLE_SIZE], CUDA_IPC_HANDLE_SIZE);
            cudaIpcOpenMemHandle(&ipcHandles[i], foreignHandle, cudaIpcMemLazyEnablePeerAccess);
        }
    }

    cudaFree(d_allHandles);
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
