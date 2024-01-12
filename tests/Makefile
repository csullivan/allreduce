CC=nvcc
MPI_INCLUDE=/opt/amazon/openmpi/include
MPI_LIB=/opt/amazon/openmpi/lib
NCCL_LIB=/usr/local/cuda-12.1/lib

all: test_allgather test_allreduce

test_allgather: test_nccl_allgather_mpi.cu
	$(CC) -I$(MPI_INCLUDE) -L$(MPI_LIB) -L$(NCCL_LIB) -lmpi -lnccl test_nccl_allgather_mpi.cu -o test_allgather

test_allreduce: test_nccl_allreduce_mpi.cu
	$(CC) -I$(MPI_INCLUDE) -L$(MPI_LIB) -L$(NCCL_LIB) -lmpi -lnccl test_nccl_allreduce_mpi.cu -o test_allreduce

clean:
	rm -f test_allgather test_allreduce

