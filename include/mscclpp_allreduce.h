#pragma once

#include <iostream>

#include "nccl.h"

class MscclppAllReduce {
 public:
  MscclppAllReduce(int worldSize, int rank, ncclComm_t comm);
  ~MscclppAllReduce();

  ncclResult_t enqueue(void* input, void* output, int64_t num_elements, size_t type_size,
                       ncclDataType_t type, ncclRedOp_t op_type, cudaStream_t stream);

 private:
  int m_world_size;
  int m_rank;
  unsigned int m_flag_value;
  ncclComm_t m_comm;
};
