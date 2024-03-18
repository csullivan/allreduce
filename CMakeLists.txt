cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(tensorrt_llm_common)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_language(CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_90a --expt-relaxed-constexpr -Xcompiler -fPIC")
set(CMAKE_CUDA_ARCHITECTURES "90a")


find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

include(cmake/TRTLLM.cmake)



include(cmake/MSCCLPP.cmake)

