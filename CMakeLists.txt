cmake_minimum_required(VERSION 3.18)
project(manifold LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)
# OPTION(DEBUG_CMAKE_TARGETS "enable debug output for cmake target properties" OFF)
# if(DEBUG_CMAKE_TARGETS)
#     set(CMAKE_DEBUG_TARGET_PROPERTIES
#             INCLUDE_DIRECTORIES
#         )
# endif()

set(THRUST_BACKEND "CUDA" CACHE STRING
    "Thrust backend, either \"CUDA\", \"OMP\" or \"CPP\"(single-threaded)"
)
if(EMSCRIPTEN)
    message("Building for Emscripten")
    set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} -sALLOW_MEMORY_GROWTH=1)
    set(THRUST_BACKEND "CPP")
endif()

option(BUILD_TEST_CGAL off)

option(BUILD_SHARED_LIBS off)
set(ASSIMP_INC_DIR 
    ${PROJECT_SOURCE_DIR}/third_party/assimp/include
    ${CMAKE_BINARY_DIR}/third_party/assimp/include)
set(GLM_INC_DIR ${PROJECT_SOURCE_DIR}/third_party/glm)

set_property(CACHE THRUST_BACKEND PROPERTY STRINGS CUDA OMP CPP)

option(ASSIMP_FAST_BUILD "build ASSIMP just for tests" ON)
if(ASSIMP_FAST_BUILD)
    option(ASSIMP_INSTALL FALSE)
    option(ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT FALSE)
    option(ASSIMP_BUILD_ALL_EXPORTERS_BY_DEFAULT FALSE)
    foreach(FMT OBJ;PLY;STL;GLTF)
        set(ASSIMP_BUILD_${FMT}_IMPORTER TRUE)
        set(ASSIMP_BUILD_${FMT}_EXPORTER TRUE)
    endforeach()
endif()

if(MSVC)
    set(MANIFOLD_FLAGS)
else()
    set(MANIFOLD_FLAGS -Werror -Wall -Wno-sign-compare -Wno-unused)
endif()
set(MANIFOLD_NVCC_FLAGS -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --extended-lambda)
set(MANIFOLD_NVCC_RELEASE_FLAGS -O3)
set(MANIFOLD_NVCC_DEBUG_FLAGS -G)

set(THRUST_INC_DIR ${PROJECT_SOURCE_DIR}/third_party/thrust)

message("Using ${THRUST_BACKEND} backend")
if (THRUST_BACKEND STREQUAL "CUDA")
    # we cannot set THRUST_INC_DIR when building with CUDA, otherwise the
    # compiler will not use the system CUDA headers which causes incompatibility
    enable_language(CUDA)
    # clear THRUST_INC_DIR, we use the one from nvcc
    set(THRUST_INC_DIR "")
elseif (THRUST_BACKEND STREQUAL "OMP")
    find_package(OpenMP REQUIRED)
    set(MANIFOLD_INCLUDE OpenMP::OpenMP_CXX)
    set(MANIFOLD_FLAGS ${MANIFOLD_FLAGS} -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -fopenmp)
elseif (THRUST_BACKEND STREQUAL "CPP")
    set(MANIFOLD_FLAGS ${MANIFOLD_FLAGS} -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP)
elseif (THRUST_BACKEND STREQUAL "TBB")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(TBB REQUIRED tbb)

    set(MANIFOLD_INCLUDE ${TBB_LINK_LIBRARIES})
    set(MANIFOLD_FLAGS ${MANIFOLD_FLAGS} -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB)
else ()
    message(FATAL_ERROR "Invalid value for THRUST_BACKEND: ${THRUST_BACKEND}. "
    "Should be one of \"CUDA\", \"OMP\", \"CPP\" or \"TBB\"")
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
