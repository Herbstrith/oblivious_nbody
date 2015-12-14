### CMakeLists.txt for CUDA

cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
set(
  CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
  -O3 -gencode arch=compute_35,code=sm_35 -rdc=true
  )

# Specify include directories
include_directories(
    kernels
    utility
    )

# Specify library paths
link_directories(
    )

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    oblivious_nbody
    src/nbody_kernel.cu
    src/nbody_kernel.cuh
    src/main.cpp
    )

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(
    oblivious_nbody
    )