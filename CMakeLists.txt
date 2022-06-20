cmake_minimum_required(VERSION 3.18)
project(manifold LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MANIFOLD_USE_CUDA on)
set(MANIFOLD_PAR "TBB" CACHE STRING "Parallel backend, either \"TBB\" or \"OpenMP\" or \"NONE\"")

if(EMSCRIPTEN)
    message("Building for Emscripten")
    set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} -sALLOW_MEMORY_GROWTH=1)
    set(MANIFOLD_USE_CUDA OFF)
    set(MANIFOLD_PAR "NONE")
endif()

option(PYBIND11_FINDPYTHON on)

option(BUILD_TEST_CGAL off)

option(BUILD_SHARED_LIBS off)
set(ASSIMP_INC_DIR
    ${PROJECT_SOURCE_DIR}/third_party/assimp/include
    ${CMAKE_BINARY_DIR}/third_party/assimp/include)
set(GLM_INC_DIR ${PROJECT_SOURCE_DIR}/third_party/glm)
set(PYBIND11_DIR ${PROJECT_SOURCE_DIR}/third_party/pybind11)

option(ASSIMP_FAST_BUILD "build ASSIMP just for tests" ON)

if(ASSIMP_FAST_BUILD)
    set(ASSIMP_INSTALL FALSE)
    set(ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT FALSE)
    set(ASSIMP_BUILD_ALL_EXPORTERS_BY_DEFAULT FALSE)

    foreach(FMT OBJ;PLY;STL;GLTF)
        set(ASSIMP_BUILD_${FMT}_IMPORTER TRUE)
        set(ASSIMP_BUILD_${FMT}_EXPORTER TRUE)
    endforeach()
endif()

set(THRUST_INC_DIR ${PROJECT_SOURCE_DIR}/third_party/thrust)

if(MANIFOLD_USE_CUDA)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)

    # we cannot set THRUST_INC_DIR when building with CUDA, otherwise the
    # compiler will not use the system CUDA headers which causes incompatibility
    # clear THRUST_INC_DIR, we use the one from nvcc
    set(THRUST_INC_DIR "")
    set(MANIFOLD_NVCC_RELEASE_FLAGS -O3 -lineinfo)
    set(MANIFOLD_NVCC_DEBUG_FLAGS -G)
    set(MANIFOLD_NVCC_FLAGS -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --extended-lambda
        "$<$<CONFIG:RELEASE>:${MANIFOLD_NVCC_RELEASE_FLAGS}>"
        "$<$<CONFIG:DEBUG>:${MANIFOLD_NVCC_DEBUG_FLAGS}>")
endif()

if(NOT MSVC)
    set(WARNING_FLAGS -Werror -Wall -Wno-sign-compare -Wno-unused)
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:${WARNING_FLAGS}>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${WARNING_FLAGS}>")
endif()

if(CODE_COVERAGE AND NOT MSVC)
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:-coverage>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-coverage>")
    add_link_options("-coverage")
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
