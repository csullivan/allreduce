#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <nccl.h>


class CUDAIpcMemory {
public:
    CUDAIpcMemory(size_t bufferSize, int worldSize, int rank, ncclComm_t ncclComm);
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
    ncclComm_t ncclComm;

    void allocateAndShare();
    void deallocate();
};
