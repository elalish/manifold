# Copyright 2020 The Manifold Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.18)
project(manifold LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MANIFOLD_EXPORT OFF)
option(MANIFOLD_DEBUG OFF)
option(MANIFOLD_USE_CUDA OFF)
option(MANIFOLD_PYBIND ON)
option(MANIFOLD_CBIND OFF)
set(MANIFOLD_PAR "NONE" CACHE STRING "Parallel backend, either \"TBB\" or \"OpenMP\" or \"NONE\"")

if(EMSCRIPTEN)
  message("Building for Emscripten")
  set(MANIFOLD_FLAGS -fexceptions)
  set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} -sALLOW_MEMORY_GROWTH=1)
endif()

option(PYBIND11_FINDPYTHON ON)

option(BUILD_TEST_CGAL OFF)

option(BUILD_SHARED_LIBS OFF)
set(GLM_INC_DIR ${PROJECT_SOURCE_DIR}/src/third_party/glm)
set(PYBIND11_DIR ${PROJECT_SOURCE_DIR}/bindings/python/third_party/pybind11)
set(THRUST_INC_DIR ${PROJECT_SOURCE_DIR}/src/third_party/thrust)

if(MANIFOLD_CBIND)
  set(MANIFOLD_EXPORT ON)
endif()

if(BUILD_SHARED_LIBS OR MANIFOLD_CBIND)
  # Allow shared libraries to be relocatable (add Place Independent Code flag).
  # Also include when statically linking C bindings to avoid issues with bundling
  # artefacts in host languages using them for FFI.
  add_compile_options(-fPIC)
endif()

if(MANIFOLD_USE_CUDA)
  enable_language(CUDA)
  find_package(CUDA REQUIRED)

  # we cannot set THRUST_INC_DIR when building with CUDA, otherwise the
  # compiler will not use the system CUDA headers which causes incompatibility
  # clear THRUST_INC_DIR, we use the one from nvcc
  set(THRUST_INC_DIR "")
  set(MANIFOLD_NVCC_RELEASE_FLAGS -O3 -lineinfo)
  set(MANIFOLD_NVCC_DEBUG_FLAGS -G)
  set(MANIFOLD_NVCC_FLAGS -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --extended-lambda --expt-relaxed-constexpr
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
  set(COVERAGE_FLAGS -coverage -fno-inline-small-functions -fkeep-inline-functions -fkeep-static-functions)
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:${COVERAGE_FLAGS}>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-coverage>")
  add_link_options("-coverage")
endif()

add_subdirectory(src)
add_subdirectory(samples)
add_subdirectory(test)
add_subdirectory(extras)
add_subdirectory(bindings)
