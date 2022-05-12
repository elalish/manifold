cmake_minimum_required(VERSION 3.18)
project(manifold LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)
# OPTION(DEBUG_CMAKE_TARGETS "enable debug output for cmake target properties" OFF)
# if(DEBUG_CMAKE_TARGETS)
#     set(CMAKE_DEBUG_TARGET_PROPERTIES
#             INCLUDE_DIRECTORIES
#         )
# endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(THRUST_BACKEND "CUDA" CACHE STRING
    "Thrust backend, either \"CUDA\", \"OMP\" or \"CPP\"(single-threaded)"
)
if(EMSCRIPTEN)
    message("Building for Emscripten")
    set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} -sALLOW_MEMORY_GROWTH=1)
    set(THRUST_BACKEND "CPP")
endif()

option(BUILD_TEST_CGAL off)
set(DISABLED_DETERMINISM 1)

option(BUILD_SHARED_LIBS off)
set(ASSIMP_INC_DIR 
    ${PROJECT_SOURCE_DIR}/third_party/assimp/include
    ${CMAKE_BINARY_DIR}/third_party/assimp/include)
set(GLM_INC_DIR ${PROJECT_SOURCE_DIR}/third_party/glm)
set(PYBIND11_DIR ${PROJECT_SOURCE_DIR}/third_party/pybind11)

set_property(CACHE THRUST_BACKEND PROPERTY STRINGS CUDA OMP CPP)

if(NOT THRUST_BACKEND STREQUAL "CUDA" AND NOT THRUST_BACKEND STREQUAL "OMP" AND NOT THRUST_BACKEND STREQUAL "CPP")
    message(FATAL_ERROR "Invalid value for THRUST_BACKEND: ${THRUST_BACKEND}. Should be one of \"CUDA\", \"OMP\" or \"CPP\"")
endif()

if(MSVC)
    set(MANIFOLD_FLAGS /DDISABLED_DETERMINISM=${DISABLED_DETERMINISM})
else()
    set(MANIFOLD_FLAGS -Werror -Wall -Wno-sign-compare -Wno-unused -DDISABLED_DETERMINISM=${DISABLED_DETERMINISM})
endif()
set(MANIFOLD_NVCC_FLAGS -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --extended-lambda)
set(MANIFOLD_NVCC_RELEASE_FLAGS -O3)
set(MANIFOLD_NVCC_DEBUG_FLAGS -G)

if(THRUST_BACKEND STREQUAL "CUDA")
    # we cannot set THRUST_INC_DIR when building with CUDA, otherwise the
    # compiler will not use the system CUDA headers which causes incompatibility
    enable_language(CUDA)
else()
    set(THRUST_INC_DIR ${PROJECT_SOURCE_DIR}/third_party/thrust)
    if(THRUST_BACKEND STREQUAL "OMP")
        message("------------------------- Using OpenMP instead of CUDA.")
        find_package(OpenMP REQUIRED)
        set(MANIFOLD_OMP_INCLUDE OpenMP::OpenMP_CXX)
        set(MANIFOLD_FLAGS ${MANIFOLD_FLAGS} -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -fopenmp)
    else()
        message("------------------------- Using C++ instead of CUDA.")
        set(MANIFOLD_FLAGS ${MANIFOLD_FLAGS} -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP)
    endif()
endif()

add_subdirectory(utilities)
add_subdirectory(third_party)
add_subdirectory(collider)
add_subdirectory(polygon)
add_subdirectory(manifold)
add_subdirectory(meshIO)
add_subdirectory(samples)
add_subdirectory(test)
add_subdirectory(tools)
