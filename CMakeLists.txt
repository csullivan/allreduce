cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(tensorrt_llm_common)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_language(CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_90 --expt-relaxed-constexpr -Xcompiler -fPIC")

set(CUDA_INCLUDE_DIRS /usr/local/cuda-12.1/targets/x86_64-linux/include/)

include(FetchContent)
FetchContent_Declare(
  tensorrt_llm
  GIT_REPOSITORY git@github.com:NVIDIA/TensorRT-LLM.git
  GIT_TAG v0.5.0
)
FetchContent_MakeAvailable(tensorrt_llm)

### Build the trt-llm common libs
set(SUBPROJECT_SOURCE_DIR ${tensorrt_llm_SOURCE_DIR}/cpp/tensorrt_llm/common)
file(GLOB SUBPROJECT_SRCS "${SUBPROJECT_SOURCE_DIR}/*.cpp")
file(GLOB SUBPROJECT_CU_SRCS "${SUBPROJECT_SOURCE_DIR}/*.cu")

# Don't bring in unnecessary dependencies from trt-llm
list(REMOVE_ITEM SUBPROJECT_SRCS "${SUBPROJECT_SOURCE_DIR}/mpiUtils.cpp")
list(REMOVE_ITEM SUBPROJECT_SRCS "${SUBPROJECT_SOURCE_DIR}/cudaAllocator.cpp")
include_directories(${tensorrt_llm_SOURCE_DIR}/cpp/include/ ${tensorrt_llm_SOURCE_DIR}/cpp ${CUDA_INCLUDE_DIRS})

# Add common sources to a lib
add_library(common_src OBJECT ${SUBPROJECT_SRCS} ${SUBPROJECT_CU_SRCS})
set_property(TARGET common_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET common_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set(SUBPROJECT_BINARY_DIR ${CMAKE_BINARY_DIR}/tensorrt_llm_common_build)

# Add the custom all reduce kernels to a lib
add_library(customAllReduceKernels
    ${tensorrt_llm_SOURCE_DIR}/cpp/tensorrt_llm/kernels/customAllReduceKernels.cu)
target_link_libraries(customAllReduceKernels PRIVATE common_src)

### Build implementation of trt-llm's cuda allreduce
find_package(MPI REQUIRED)
find_package(CUDAToolkit)

macro(find_nccl use_nccl)
  set(NCCL_LIB_NAME nccl_static)
  find_path(NCCL_INCLUDE_DIR NAMES nccl.h HINTS ${use_nccl} ${use_nccl}/include)
  find_library(NCCL_LIBRARY NAMES nccl_static HINTS ${use_nccl} ${use_nccl}/lib)
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(Nccl DEFAULT_MSG NCCL_INCLUDE_DIR NCCL_LIBRARY)
  if (Nccl_FOUND)
    message(STATUS "Found NCCL_LIBRARY: ${NCCL_LIBRARY}")
    message(STATUS "Found NCCL_INCLUDE_DIR: ${NCCL_INCLUDE_DIR}")
    add_library(nccl SHARED IMPORTED)
    set_target_properties(nccl
      PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
      IMPORTED_LOCATION             "${NCCL_LIBRARY}")
  else()
    message(STATUS "NCCL not found")
  endif()
  mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY)
endmacro(find_nccl)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
set(LIBRARY_FILES
    src/cuda_ipc_memory.cpp
    src/cuda_allreduce.cpp
)
add_library(trtllm_allreduce SHARED ${LIBRARY_FILES})
find_nccl(/usr)
target_link_libraries(trtllm_allreduce PRIVATE 
    CUDA::cudart
    customAllReduceKernels 
    nccl
)
set(TEST_SOURCE_FILES
    tests/test_cuda_allreduce.cpp
)
add_executable(test_cuda_allreduce ${TEST_SOURCE_FILES})
target_link_libraries(test_cuda_allreduce PRIVATE 
    trtllm_allreduce
    CUDA::cudart
    MPI::MPI_CXX
)

