#include <iostream>
#include <mpi.h>
#include <cuda_runtime.h>
#include "tensorrt_llm/kernels/customAllReduceKernels.h"

// Helper functions for error checking
#define CUDA_CHECK(call) { \
    cudaError_t cudaStatus = call; \
    if (cudaStatus != cudaSuccess) { \
        std::cerr << "CUDA error at line " << __LINE__ << ": " << cudaGetErrorString(cudaStatus) << std::endl; \
        return 1; \
    } \
}
#define MPI_CHECK(call) { \
    int mpiStatus = call; \
    if (mpiStatus != MPI_SUCCESS) { \
        std::cerr << "MPI error at line " << __LINE__ << ": " << mpiStatus << std::endl; \
        return 1; \
    } \
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
int main(int argc, char** argv) {
    // Initialize MPI
    MPI_CHECK(MPI_Init(&argc, &argv));
    int rank, world_size;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    // Set CUDA device
    CUDA_CHECK(cudaSetDevice(rank));

    // Data allocation and initialization
    const int data_size = 10;
    float* d_buffer;
    float h_buffer[data_size];
    float result_buffer[data_size];

    // Allocate GPU buffer
    CUDA_CHECK(cudaMalloc(&d_buffer, data_size * sizeof(float)));

    // Initialize host buffer
    for (int i = 0; i < data_size; i++) {
        h_buffer[i] = static_cast<float>(rank + 1);
    }
    CUDA_CHECK(cudaMemcpy(d_buffer, h_buffer, data_size * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate and set peer communication and barrier buffers
    void* peer_comm_buffer_ptrs[world_size];
    uint32_t* peer_barrier_ptrs_in[world_size];
    uint32_t* peer_barrier_ptrs_out[world_size];
    uint32_t flag_value = 123; // Example flag value

    for (int i = 0; i < world_size; ++i) {
        // Allocate device memory for communication buffers
        CUDA_CHECK(cudaMalloc(&(peer_comm_buffer_ptrs[i]), data_size * sizeof(float)));

        // Allocate device memory for barrier flags
        CUDA_CHECK(cudaMalloc(&(peer_barrier_ptrs_in[i]), sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&(peer_barrier_ptrs_out[i]), sizeof(uint32_t)));

        // Initialize barrier flags to the flag value
        CUDA_CHECK(cudaMemset(peer_barrier_ptrs_in[i], flag_value, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(peer_barrier_ptrs_out[i], flag_value, sizeof(uint32_t)));
    }

    // Create a buffer for deserialize function
    void* buffer_for_deserialize[3 * world_size];
    for (int i = 0; i < world_size; ++i) {
        buffer_for_deserialize[i] = peer_comm_buffer_ptrs[i];
        buffer_for_deserialize[world_size + i] = peer_barrier_ptrs_in[i];
        buffer_for_deserialize[2 * world_size + i] = peer_barrier_ptrs_out[i];
    }

    // Set up AllReduceParams using the deserialize function
    auto params = tensorrt_llm::kernels::AllReduceParams::deserialize(reinterpret_cast<int32_t*>(buffer_for_deserialize), world_size, rank, flag_value);

    size_t messageSize = data_size * sizeof(float);

    // Select the AllReduce strategy
    auto strategy = selectImplementation(messageSize, world_size);

    // Invoke the all-reduce operation
    tensorrt_llm::kernels::customAllReduce(params, d_buffer, data_size, sizeof(float),
                                           tensorrt_llm::common::datatype_enum::TYPE_FP32,
                                           strategy, 0);



    // Synchronize
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy data back and verify results
    CUDA_CHECK(cudaMemcpy(result_buffer, d_buffer, data_size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < data_size; i++) {
        if (result_buffer[i] != 10.0f * world_size) {
            std::cerr << "Verification failed at rank " << rank << ", index " << i << std::endl;
            return 1;
        }
    }

    // Cleanup: Free the allocated device memory for communication buffers and barrier flags
    for (int i = 0; i < world_size; ++i) {
        CUDA_CHECK(cudaFree(peer_comm_buffer_ptrs[i]));
        CUDA_CHECK(cudaFree(peer_barrier_ptrs_in[i]));
        CUDA_CHECK(cudaFree(peer_barrier_ptrs_out[i]));
    }

    // Free data buffer and finalize MPI
    CUDA_CHECK(cudaFree(d_buffer));
    MPI_CHECK(MPI_Finalize());

    std::cout << "Allreduce test passed successfully on rank " << rank << std::endl;
    return 0;
}

