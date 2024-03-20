#pragma once

#include <iostream>

#include "msccl.h"

class MscclppAllReduce {
 public:
  MscclppAllReduce(int worldSize, int rank, mscclComm_t comm);
  ~MscclppAllReduce();

  mscclResult_t enqueue(void* input, void* output, int64_t num_elements, size_t type_size,
                        mscclDataType_t type, mscclRedOp_t op_type, cudaStream_t stream);

 private:
  int m_world_size;
  int m_rank;
  unsigned int m_flag_value;
  mscclComm_t m_comm;
};
