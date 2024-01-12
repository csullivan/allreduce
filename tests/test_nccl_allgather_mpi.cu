#include <iostream>
#include <nccl.h>
#include <mpi.h>


int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);  // Detect world size

    // Init NCCL
    ncclComm_t comm;
    ncclUniqueId id;
    if(rank == 0) {
        ncclGetUniqueId(&id);
    }
    // Broadcast the unique ID to other processes
    MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    cudaSetDevice(rank);
    ncclCommInitRank(&comm, world_size, id, rank);

    // Data
    const int data_size = 10;
    float* d_buffer;
    float h_buffer[data_size * world_size]; // Store all-gathered data

    // Prepare data and allocate GPU buffer
    cudaMalloc(&d_buffer, world_size * data_size * sizeof(float));
    
    // Only setting the data for the GPU's corresponding section
    for (int j = rank*data_size; j < (rank+1)*data_size; j++) {
        h_buffer[j] = static_cast<float>(rank);
    }
    cudaMemcpy(d_buffer + rank * data_size, h_buffer + rank * data_size, data_size * sizeof(float), cudaMemcpyHostToDevice);

    // All-gather
    ncclAllGather(d_buffer + rank * data_size, d_buffer, data_size, ncclFloat, comm, 0);

    // Synchronize NCCL operations
    cudaDeviceSynchronize();

    // Copy data back and print
    cudaMemcpy(h_buffer, d_buffer, world_size * data_size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int j = 0; j < world_size * data_size; j++) {
        std::cout << "Rank " << rank << " data[" << j << "] = " << h_buffer[j] << std::endl;
    }

    cudaFree(d_buffer);

    // Clean up NCCL
    ncclCommDestroy(comm);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}

