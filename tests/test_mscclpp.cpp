#include <iostream>

#include "mpi.h"
#include "mscclpp_allreduce.h"
#include "nccl.h"  // This is mscclpp/apps/include/nccl.h

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int world_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  cudaSetDevice(rank);
  mscclComm_t comm;
  mscclUniqueId id;
  if (rank == 0) {
    mscclGetUniqueId(&id);
  }

  // Broadcast the unique ID to other processes
  MPI_Bcast(&id, sizeof(mscclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
  cudaSetDevice(rank);
  mscclCommInitRank(&comm, world_size, id, rank);

  MscclppAllReduce(world_size, rank, comm);

  mscclCommDestroy(comm);
  MPI_Finalize();

  return 0;
}
