cmake_minimum_required(VERSION 3.18)
if (NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()
project(manifold LANGUAGES CXX CUDA)

set(CMAKE_VERBOSE_MAKEFILE ON)
# OPTION(DEBUG_CMAKE_TARGETS "enable debug output for cmake target properties" OFF)
# if(DEBUG_CMAKE_TARGETS)
#     set(CMAKE_DEBUG_TARGET_PROPERTIES
#             INCLUDE_DIRECTORIES
#         )
# endif()

OPTION( MANIFOLD_USE_OMP
  "Use OpenMP as the Thrust backend instead of CUDA."
  OFF
)
OPTION( MANIFOLD_USE_CPP
  "Use C++ (single-threaded) as the Thrust backend instead of CUDA."
  OFF
)

set(MAINFOLD_FLAGS -Werror -Wall -Wno-sign-compare)
set(MANIFOLD_NVCC_FLAGS -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --extended-lambda)
set(MANIFOLD_NVCC_RELEASE_FLAGS -O3)
set(MANIFOLD_NVCC_DEBUG_FLAGS -G)

IF(MANIFOLD_USE_OMP)
    message("------------------------- Using OpenMP instead of CUDA.")
    find_package(OpenMP REQUIRED)
    set(MANIFOLD_OMP_INCLUDE OpenMP::OpenMP_CXX)
    set(MANIFOLD_NVCC_FLAGS ${MANIFOLD_NVCC_FLAGS} -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -Xcompiler=-fopenmp)
ENDIF(MANIFOLD_USE_OMP)

IF(MANIFOLD_USE_CPP)
    message("------------------------- Using C++ instead of CUDA.")
    set(MANIFOLD_NVCC_FLAGS ${MANIFOLD_NVCC_FLAGS} -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP)
ENDIF(MANIFOLD_USE_CPP)

add_subdirectory(utilities)
add_subdirectory(third_party)
add_subdirectory(collider)
add_subdirectory(polygon)
add_subdirectory(manifold)
add_subdirectory(meshIO)
add_subdirectory(samples)
add_subdirectory(test)
add_subdirectory(tools)