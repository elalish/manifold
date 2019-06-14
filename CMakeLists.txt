cmake_minimum_required(VERSION 3.12)
project(manifold LANGUAGES CXX CUDA)

# set(CMAKE_VERBOSE_MAKEFILE ON)
# OPTION(DEBUG_CMAKE_TARGETS "enable debug output for cmake target properties" ON)
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
set(MANIFOLD_NVCC_FLAGS -DTHRUST_DEBUG)# -Xcompiler=-Werror -Xcompiler=-Wno-macro-redefined)
IF(MANIFOLD_USE_OMP)
    message("------------------------- Using OpenMP instead of CUDA.")
    find_package(OpenMP REQUIRED)
    set(MANIFOLD_OMP_INCLUDE OpenMP::OpenMP_CXX)
    set(MANIFOLD_NVCC_FLAGS -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -Xcompiler=-fopenmp)
ENDIF(MANIFOLD_USE_OMP)
IF(MANIFOLD_USE_CPP)
    message("------------------------- Using C++ instead of CUDA.")
    set(MANIFOLD_NVCC_FLAGS -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP)
ENDIF(MANIFOLD_USE_CPP)

add_subdirectory(utilities)
add_subdirectory(collider)
add_subdirectory(polygon)
add_subdirectory(mesh)
add_subdirectory(third_party)
add_subdirectory(mesh_host)
add_subdirectory(test)
add_subdirectory(tools)