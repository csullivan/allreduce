#include <iostream>
#include <nccl.h>
#include <mpi.h>

// Macro for checking NCCL results and returning an error message
#define CHECK_NCCL(call) { \
    ncclResult_t ncclStatus = call; \
    if (ncclStatus != ncclSuccess) { \
        std::cerr << "NCCL error at line " << __LINE__ << ": " << ncclGetErrorString(ncclStatus) << std::endl; \
        return 1; \
    } \
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Init NCCL
    ncclComm_t comm;
    ncclUniqueId id;
    if(rank == 0) {
        ncclGetUniqueId(&id);
    }
    MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    cudaSetDevice(rank);
    CHECK_NCCL(ncclCommInitRank(&comm, world_size, id, rank));

    // Data
    const int data_size = 10;
    float* d_buffer;
    float h_buffer[data_size];
    float result_buffer[data_size];

    // Prepare data and allocate GPU buffer
    cudaMalloc(&d_buffer, data_size * sizeof(float));
    
    // Initialize host buffer
    for (int i = 0; i < data_size; i++) {
        h_buffer[i] = static_cast<float>(rank + 1);
    }
    cudaMemcpy(d_buffer, h_buffer, data_size * sizeof(float), cudaMemcpyHostToDevice);

    // All-reduce
    CHECK_NCCL(ncclAllReduce(d_buffer, d_buffer, data_size, ncclFloat, ncclSum, comm, 0));

    // Synchronize NCCL operations
    cudaDeviceSynchronize();

    // Copy data back and print
    cudaMemcpy(result_buffer, d_buffer, data_size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < data_size; i++) {
        std::cout << "Rank " << rank << " reduced_data[" << i << "] = " << result_buffer[i] << std::endl;
    }

    cudaFree(d_buffer);

    // Clean up NCCL
    ncclCommDestroy(comm);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}

