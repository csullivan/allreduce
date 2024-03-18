include(FetchContent)
FetchContent_Declare(
  mscclpp
  GIT_REPOSITORY https://github.com/csullivan/mscclpp.git
  GIT_TAG csullivan/nccl-interfaces
)

set(USE_CUDA 1)
FetchContent_MakeAvailable(mscclpp)


add_library(mscclpp_allreduce SHARED src/mscclpp_allreduce.cu)
target_include_directories(mscclpp_allreduce PUBLIC
  ${mscclpp_SOURCE_DIR}/include
  ${mscclpp_SOURCE_DIR}/apps/nccl/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(mscclpp_allreduce PRIVATE mscclpp)

if (BUILD_MPI_TESTS)
  find_package(MPI REQUIRED)

  add_executable(test_mscclpp tests/test_mscclpp.cpp)
  target_include_directories(test_mscclpp PUBLIC
    ${mscclpp_SOURCE_DIR}/include
    ${mscclpp_SOURCE_DIR}/apps/nccl/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(test_mscclpp PRIVATE
      mscclpp
      mscclpp_nccl
      mscclpp_allreduce
      CUDA::cudart
      MPI::MPI_CXX
  )
endif()
