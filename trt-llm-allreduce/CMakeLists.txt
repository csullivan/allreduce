cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(tensorrt_llm_common)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_language(CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_90 --expt-relaxed-constexpr")

set(CUDA_INCLUDE_DIRS /usr/local/cuda-12.1/targets/x86_64-linux/include/)

include(FetchContent)
FetchContent_Declare(
  tensorrt_llm
  GIT_REPOSITORY git@github.com:NVIDIA/TensorRT-LLM.git
  GIT_TAG v0.5.0
)
FetchContent_MakeAvailable(tensorrt_llm)

set(SUBPROJECT_SOURCE_DIR ${tensorrt_llm_SOURCE_DIR}/cpp/tensorrt_llm/common)

file(GLOB SUBPROJECT_SRCS "${SUBPROJECT_SOURCE_DIR}/*.cpp")
file(GLOB SUBPROJECT_CU_SRCS "${SUBPROJECT_SOURCE_DIR}/*.cu")

list(REMOVE_ITEM SUBPROJECT_SRCS "${SUBPROJECT_SOURCE_DIR}/mpiUtils.cpp")
list(REMOVE_ITEM SUBPROJECT_SRCS "${SUBPROJECT_SOURCE_DIR}/cudaAllocator.cpp")

include_directories(${tensorrt_llm_SOURCE_DIR}/cpp/include/ ${tensorrt_llm_SOURCE_DIR}/cpp ${CUDA_INCLUDE_DIRS})

add_library(common_src OBJECT ${SUBPROJECT_SRCS} ${SUBPROJECT_CU_SRCS})
set_property(TARGET common_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET common_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set(SUBPROJECT_BINARY_DIR ${CMAKE_BINARY_DIR}/tensorrt_llm_common_build)

add_library(customAllReduceKernels
    ${tensorrt_llm_SOURCE_DIR}/cpp/tensorrt_llm/kernels/customAllReduceKernels.cu)
target_link_libraries(customAllReduceKernels PRIVATE common_src)

find_package(MPI REQUIRED)

add_executable(custom_allreduce_test test_custom_allreduce.cu)
target_link_libraries(custom_allreduce_test PRIVATE customAllReduceKernels MPI::MPI_CXX)

add_executable(test_ipcmem test_ipcmem.cu)
target_link_libraries(test_ipcmem PRIVATE customAllReduceKernels MPI::MPI_CXX)
