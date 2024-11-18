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

# Use C++17
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_VERBOSE_MAKEFILE ON)

include(GNUInstallDirs)
include(CMakeDependentOption)

# Define Manifold version
set(MANIFOLD_VERSION_MAJOR 3)
set(MANIFOLD_VERSION_MINOR 0)
set(MANIFOLD_VERSION_PATCH 0)
set(
  MANIFOLD_VERSION
  "${MANIFOLD_VERSION_MAJOR}.${MANIFOLD_VERSION_MINOR}.${MANIFOLD_VERSION_PATCH}"
)

# Primary user facing options
option(MANIFOLD_CROSS_SECTION "Build CrossSection for 2D support" ON)
option(MANIFOLD_DEBUG "Enable debug tracing/timing" OFF)
option(
  MANIFOLD_DOWNLOADS
  "Allow Manifold build to download missing dependencies"
  ON
)
option(MANIFOLD_EXPORT "Build mesh export (via assimp) utility library" OFF)
option(MANIFOLD_PAR "Enable Parallel backend" OFF)
option(
  MANIFOLD_OPTIMIZED
  "Force optimized build, even with debugging enabled"
  OFF
)
option(MANIFOLD_TEST "Enable testing suite" ON)
option(MANIFOLD_PYBIND "Build python bindings" OFF)
# MANIFOLD_CBIND is only available when MANIFOLD_CROSS_SECTION is enabled
cmake_dependent_option(
  MANIFOLD_CBIND
  "Build C (FFI) bindings"
  ON
  "MANIFOLD_CROSS_SECTION"
  OFF
)
# MANIFOLD_JSBIND is only available when building with Emscripten
cmake_dependent_option(
  MANIFOLD_JSBIND
  "Build js binding"
  ON
  "EMSCRIPTEN"
  OFF
)
# These three options can force the build to avoid using the system version of
# the dependency
# This will either use the provided source directory via
# FETCHCONTENT_SOURCE_DIR_XXX, or fetch the source from GitHub.
# Note that the dependency will be built as static dependency to avoid dynamic
# library conflict.
# When the system package is unavailable, the option will be automatically set
# to true.
option(MANIFOLD_USE_BUILTIN_TBB "Use builtin tbb" OFF)
option(MANIFOLD_USE_BUILTIN_CLIPPER2 "Use builtin clipper2" OFF)
option(MANIFOLD_USE_BUILTIN_NANOBIND "Use builtin nanobind" OFF)

# default to Release build
option(CMAKE_BUILD_TYPE "Build type" Release)
# default to building shared library
option(BUILD_SHARED_LIBS "Build shared library" ON)
# Set some option values in the CMake cache
set(MANIFOLD_FLAGS "" CACHE STRING "Manifold compiler flags")

# Development options
option(TRACY_ENABLE "Use tracy profiling" OFF)
option(TRACY_MEMORY_USAGE "Track memory allocation with tracy (expensive)" OFF)
option(MANIFOLD_FUZZ "Enable fuzzing tests" OFF)
mark_as_advanced(TRACY_ENABLE)
mark_as_advanced(TRACY_MEMORY_USAGE)
mark_as_advanced(MANIFOLD_FUZZ)

# Always build position independent code for relocatability
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Various Compiler Flags
if(MANIFOLD_FUZZ)
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "fuzztest only supports clang")
  endif()
  # we should enable debug checks
  set(MANIFOLD_DEBUG ON)
  # enable fuzztest fuzzing mode
  set(FUZZTEST_FUZZING_MODE ON)
  # address sanitizer required
  add_compile_options(-fsanitize=address)
endif()

if(TRACY_ENABLE)
  option(CMAKE_BUILD_TYPE "Build type" RelWithDebInfo)
endif()

if(EMSCRIPTEN)
  message("Building for Emscripten")
  add_link_options(-sALLOW_MEMORY_GROWTH=1)
  add_link_options(-sMAXIMUM_MEMORY=4294967296)
  if(MANIFOLD_PAR)
    set(CMAKE_THREAD_LIBS_INIT "-pthread")
    add_compile_options(-pthread)
    # PTHREAD_POOL_SIZE is set to 4 for OK-ish performance, in general we are
    # not getting too much speedup for very large number of cores, and a lot of
    # CPUs now have 4 cores...
    add_link_options(-sPTHREAD_POOL_SIZE=4)
    # mimalloc is needed for good performance
    add_link_options(-sMALLOC=mimalloc)
    # The default stack size apparently causes problem when parallelization is
    # enabled.
    add_link_options(-sSTACK_SIZE=10MB)
  endif()
  if(MANIFOLD_DEBUG)
    list(APPEND MANIFOLD_FLAGS -fexceptions)
    add_link_options(-fexceptions)
    add_link_options(-sDISABLE_EXCEPTION_CATCHING=0)
  endif()
  set(MANIFOLD_PYBIND OFF)
  set(BUILD_SHARED_LIBS OFF)
endif()

if(CMAKE_EXPORT_COMPILE_COMMANDS AND NOT EMSCRIPTEN)
  # for nixos
  set(
    CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES
    ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES}
  )
endif()

if(MSVC)
  list(APPEND MANIFOLD_FLAGS /DNOMINMAX /bigobj)
else()
  list(
    APPEND
    WARNING_FLAGS
    -Wall
    -Wno-unknown-warning-option
    -Wno-unused
    -Wno-shorten-64-to-32
  )
  if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      list(APPEND WARNING_FLAGS -Wno-format)
    endif()
  elseif(PROJECT_IS_TOP_LEVEL)
    # only do -Werror if we are the top level project
    list(APPEND WARNING_FLAGS -Werror)
  endif()
  list(APPEND MANIFOLD_FLAGS ${WARNING_FLAGS})
  if(
    MANIFOLD_OPTIMIZED
    OR "${CMAKE_BUILD_TYPE}" STREQUAL "Release"
    OR "${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo"
  )
    list(APPEND MANIFOLD_FLAGS -O3)
  endif()
  if("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
    list(APPEND MANIFOLD_FLAGS -fno-omit-frame-pointer)
  endif()
  if(CODE_COVERAGE)
    list(
      APPEND
      MANIFOLD_FLAGS
      -coverage
      -fno-inline-small-functions
      -fkeep-inline-functions
      -fkeep-static-functions
    )
    add_link_options(-coverage)
  endif()
endif()

# RPath settings
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
list(
  FIND
  CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
  ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
  isSystemDir
)
if("${isSystemDir}" STREQUAL "-1")
  set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
endif("${isSystemDir}" STREQUAL "-1")

include(${PROJECT_SOURCE_DIR}/cmake/manifoldDeps.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/configHelper.cmake)

add_subdirectory(src)
add_subdirectory(bindings)
if(MANIFOLD_TEST)
  add_subdirectory(samples)
  add_subdirectory(test)
  add_subdirectory(extras)
endif()

include(${PROJECT_SOURCE_DIR}/cmake/info.cmake)

# note that the path ${CMAKE_CURRENT_BINARY_DIR}/include is included when we
# build the manifold target (as ${PROJECT_SOURCE_DIR}/include), so users can
# include manifold/version.h without installing our library.
configure_file(
  cmake/version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/manifold/version.h
  @ONLY
)
set_source_files_properties(
  ${CMAKE_CURRENT_BINARY_DIR}/include/manifold/version.h
  PROPERTIES GENERATED TRUE
)

# If it's an EMSCRIPTEN build, we're done
if(EMSCRIPTEN)
  return()
endif()

# CMake exports
configure_file(
  cmake/manifoldConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/manifoldConfig.cmake
  @ONLY
)
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/manifoldConfigVersion.cmake
  VERSION ${MANIFOLD_VERSION}
  COMPATIBILITY SameMajorVersion
)

# Location of inputs for CMake find_package - see:
# https://cmake.org/cmake/help/latest/command/find_package.html
set(EXPORT_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake)

install(
  EXPORT manifoldTargets
  NAMESPACE manifold::
  DESTINATION ${EXPORT_INSTALL_DIR}/manifold
)
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/manifoldConfigVersion.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/manifoldConfig.cmake
  DESTINATION ${EXPORT_INSTALL_DIR}/manifold
)

# install public headers
set(
  MANIFOLD_PUBLIC_HDRS
  common.h
  iters.h
  linalg.h
  manifold.h
  optional_assert.h
  polygon.h
  vec_view.h
  $<$<BOOL:${MANIFOLD_CROSS_SECTION}>:cross_section.h>
  $<$<BOOL:${MANIFOLD_EXPORT}>:meshIO.h>
)
list(TRANSFORM MANIFOLD_PUBLIC_HDRS PREPEND include/manifold/)
list(
  APPEND
  MANIFOLD_PUBLIC_HDRS
  ${CMAKE_CURRENT_BINARY_DIR}/include/manifold/version.h
)

install(
  FILES ${MANIFOLD_PUBLIC_HDRS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/manifold
)

# PkgConfig file
if(MANIFOLD_PAR)
  set(TEMPLATE_OPTIONAL_TBB "tbb")
endif()
if(MANIFOLD_CROSS_SECTION)
  set(TEMPLATE_OPTIONAL_CLIPPER "Clipper2")
endif()
configure_file(
  cmake/manifold.pc.in
  ${CMAKE_CURRENT_BINARY_DIR}/manifold.pc
  @ONLY
)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/manifold.pc
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)
