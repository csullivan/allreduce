#pragma once

#include <vector>
#include <cuda_runtime.h>
#include "mpi.h"

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

