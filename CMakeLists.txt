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

if(EMSCRIPTEN)
  set(JSBIND_DEFAULT ON)
else()
  set(JSBIND_DEFAULT OFF)
endif()

set(MANIFOLD_PAR "NONE" CACHE STRING "Parallel backend, either \"TBB\" or \"NONE\"")
set(MANIFOLD_FLAGS "" CACHE STRING "Manifold compiler flags")
option(MANIFOLD_CROSS_SECTION "Build CrossSection for 2D support" ON)
option(MANIFOLD_EXPORT "Build mesh export (via assimp) utility library" OFF)
option(MANIFOLD_TEST "Enable testing suite" ON)
# fuzztest is a rather large dependency
option(MANIFOLD_FUZZ "Enable fuzzing tests" OFF)
option(MANIFOLD_DEBUG "Enable debug tracing/timing" OFF)
option(MANIFOLD_EXCEPTIONS "Build manifold with exception enabled" ON)
option(BUILD_SHARED_LIBS "Build shared library" ON)
include(CMakeDependentOption)
cmake_dependent_option(MANIFOLD_CBIND "Build C (FFI) bindings" ON "MANIFOLD_CROSS_SECTION" OFF)
cmake_dependent_option(MANIFOLD_JSBIND "Build js binding" ${JSBIND_DEFAULT} "MANIFOLD_CROSS_SECTION" OFF)
cmake_dependent_option(MANIFOLD_PYBIND "Build python bindings" OFF "MANIFOLD_CROSS_SECTION" OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MANIFOLD_FUZZ)
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "fuzztest only supports clang" )
  endif()
  # we should enable debug checks
  set(MANIFOLD_DEBUG ON)
  # enable fuzztest fuzzing mode
  set(FUZZTEST_FUZZING_MODE ON)
  # address sanitizer required
  set(CMAKE_CXX_FLAGS "$CMAKE_CXX_FLAGS -fsanitize=address")
endif()

option(TRACY_ENABLE "Use tracy profiling" OFF) 
option(TRACY_MEMORY_USAGE "Track memory allocation with tracy (expensive)" OFF)
option(BUILD_TEST_CGAL "Build CGAL performance comparisons" OFF)
if(TRACY_ENABLE)
  option(CMAKE_BUILD_TYPE "Build type" RelWithDebInfo)
  if(TRACY_MEMORY_USAGE)
    set(MANIFOLD_FLAGS ${MANIFOLD_FLAGS} -DTRACY_MEMORY_USAGE)
  endif()
  if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")
  endif()
else()
  option(CMAKE_BUILD_TYPE "Build type" Release)
endif()

if(EMSCRIPTEN)
  message("Building for Emscripten")
  if(MANIFOLD_EXCEPTIONS)
    set(MANIFOLD_FLAGS -fexceptions -D_LIBCUDACXX_HAS_THREAD_API_EXTERNAL -D_LIBCUDACXX_HAS_THREAD_API_CUDA)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -sALLOW_MEMORY_GROWTH=1 -fexceptions -sDISABLE_EXCEPTION_CATCHING=0")
  else()
    set(MANIFOLD_FLAGS -D_LIBCUDACXX_HAS_THREAD_API_EXTERNAL -D_LIBCUDACXX_HAS_THREAD_API_CUDA)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -sALLOW_MEMORY_GROWTH=1")
  endif()
  set(MANIFOLD_PYBIND OFF)
  set(BUILD_SHARED_LIBS OFF)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/manifoldDeps.cmake)

if(MANIFOLD_PYBIND)
  if(Python_VERSION VERSION_LESS 3.12)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  else()
    find_package(Python COMPONENTS Interpreter Development.SABIModule REQUIRED)
  endif()
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if(CMAKE_EXPORT_COMPILE_COMMANDS AND NOT EMSCRIPTEN)
  # for nixos
  set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES 
      ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

if (MSVC)
  set(MANIFOLD_FLAGS ${MANIFOLD_FLAGS} /DNOMINMAX /bigobj)
else()
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(WARNING_FLAGS -Wall -Wno-unused -Wno-array-bounds
      -Wno-stringop-overflow -Wno-alloc-size-larger-than)
  else()
    set(WARNING_FLAGS -Wall -Wno-unused)
  endif()
  if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set(WARNING_FLAGS ${WARNING_FLAGS} -Wno-format)
    endif()
  else()
    set(WARNING_FLAGS ${WARNING_FLAGS} -Werror)
  endif()
  set(MANIFOLD_FLAGS ${MANIFOLD_FLAGS} ${WARNING_FLAGS})
endif()

if(CODE_COVERAGE AND NOT MSVC)
  set(COVERAGE_FLAGS -coverage -fno-inline-small-functions -fkeep-inline-functions -fkeep-static-functions)
  set(MANIFOLD_FLAGS ${MANIFOLD_FLAGS} ${COVERAGE_FLAGS})
  add_link_options("-coverage")
endif()

# installation rpath
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
endif("${isSystemDir}" STREQUAL "-1")


add_subdirectory(src)
add_subdirectory(bindings)

if(MANIFOLD_EXPORT)
  add_subdirectory(meshIO)
endif()

if(MANIFOLD_TEST)
  add_subdirectory(samples)
  add_subdirectory(test)
  add_subdirectory(extras)
endif()

# installation related

set(MANIFOLD_VERSION_MAJOR 2)
set(MANIFOLD_VERSION_MINOR 5)
set(MANIFOLD_VERSION_PATCH 1)
set(MANIFOLD_VERSION "${MANIFOLD_VERSION_MAJOR}.${MANIFOLD_VERSION_MINOR}.${MANIFOLD_VERSION_PATCH}")
set_property(TARGET manifold PROPERTY VERSION "${MANIFOLD_VERSION}")
set_property(TARGET manifold PROPERTY SOVERSION 2)

if(NOT EMSCRIPTEN)
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_BINARY_DIR}/cmake/manifoldConfigVersion.cmake
  VERSION ${MANIFOLD_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(EXPORT manifoldTargets DESTINATION ${CMAKE_INSTALL_DATADIR}/manifold)
configure_file(manifoldConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/manifoldConfig.cmake @ONLY)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/manifoldDeps.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/manifoldConfigVersion.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/manifoldConfig.cmake
  DESTINATION ${CMAKE_INSTALL_DATADIR}/manifold
)

if(MANIFOLD_CROSS_SECTION)
  set(TEMPLATE_OPTIONAL_CLIPPER "Clipper2")
endif()
configure_file(manifold.pc.in ${CMAKE_CURRENT_BINARY_DIR}/manifold.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/manifold.pc
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

endif()

# configuration summary, idea from openscad
# https://github.com/openscad/openscad/blob/master/cmake/Modules/info.cmake
message(STATUS "====================================")
message(STATUS "Manifold Build Configuration Summary")
message(STATUS "====================================")
message(STATUS " ")
if (MXECROSS)
  message(STATUS "Environment: MXE")
elseif (APPLE)
  message(STATUS "Environment: macOS")
elseif (WIN32)
  if (MINGW)
    message(STATUS "Environment: msys2")
  else()
    message(STATUS "Environment: Windows")
  endif()
elseif (LINUX)
  message(STATUS "Environment: Linux")
elseif(UNIX)
  message(STATUS "Environment: Unknown Unix")
else()
  message(STATUS "Environment: Unknown")
endif()
message(STATUS " ")
message(STATUS "CMAKE_VERSION:               ${CMAKE_VERSION}")
message(STATUS "CMAKE_TOOLCHAIN_FILE:        ${CMAKE_TOOLCHAIN_FILE}")
message(STATUS "CMAKE_GENERATOR:             ${CMAKE_GENERATOR}")
message(STATUS "CPACK_CMAKE_GENERATOR:       ${CPACK_CMAKE_GENERATOR}")
message(STATUS "CMAKE_BUILD_TYPE:            ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_PREFIX_PATH:           ${CMAKE_PREFIX_PATH}")
message(STATUS "CMAKE_CXX_COMPILER_ID:       ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION:  ${CMAKE_CXX_COMPILER_VERSION}")
if (APPLE)
message(STATUS "CMAKE_OSX_DEPLOYMENT_TARGET: ${CMAKE_OSX_DEPLOYMENT_TARGET}")
message(STATUS "CMAKE_OSX_ARCHITECTURES:     ${CMAKE_OSX_ARCHITECTURES}")
endif()
message(STATUS "BUILD_SHARED_LIBS:           ${BUILD_SHARED_LIBS}")
message(STATUS " ")
message(STATUS "MANIFOLD_PAR:                ${MANIFOLD_PAR}")
message(STATUS "MANIFOLD_FLAGS:              ${MANIFOLD_FLAGS}")
message(STATUS "MANIFOLD_EXPORT:             ${MANIFOLD_EXPORT}")
message(STATUS "MANIFOLD_TEST:               ${MANIFOLD_TEST}")
message(STATUS "MANIFOLD_FUZZ:               ${MANIFOLD_FUZZ}")
message(STATUS "MANIFOLD_DEBUG:              ${MANIFOLD_DEBUG}")
message(STATUS "MANIFOLD_PYBIND:             ${MANIFOLD_PYBIND}")
message(STATUS "MANIFOLD_CBIND:              ${MANIFOLD_CBIND}")
message(STATUS "MANIFOLD_JSBIND:             ${MANIFOLD_JSBIND}")
message(STATUS "MANIFOLD_EXCEPTIONS:         ${MANIFOLD_EXCEPTIONS}")
message(STATUS " ")
