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

# Define Manifold version
set(MANIFOLD_VERSION_MAJOR 2)
set(MANIFOLD_VERSION_MINOR 5)
set(MANIFOLD_VERSION_PATCH 1)
set(MANIFOLD_VERSION "${MANIFOLD_VERSION_MAJOR}.${MANIFOLD_VERSION_MINOR}.${MANIFOLD_VERSION_PATCH}")

# Correct MANIFOLD_PAR values to on/off (previous NONE/TBB values should still
# work this way)
if (DEFINED MANIFOLD_PAR)
  if ("${MANIFOLD_PAR}" STREQUAL "" OR "${MANIFOLD_PAR}" STREQUAL "NONE" OR "${MANIFOLD_PAR}" STREQUAL "OFF")
    set(MANIFOLD_PAR OFF)
  else()
    set(MANIFOLD_PAR ON)
  endif ()
endif ()

# Primary user facing options
option(MANIFOLD_CROSS_SECTION "Build CrossSection for 2D support" ON)
option(MANIFOLD_DEBUG "Enable debug tracing/timing" OFF)
option(MANIFOLD_DOWNLOADS "Allow Manifold build to download missing dependencies" ON) 
option(MANIFOLD_EXCEPTIONS "Build manifold with exception enabled" ON)
option(MANIFOLD_EXPORT "Build mesh export (via assimp) utility library" OFF)
option(MANIFOLD_FUZZ "Enable fuzzing tests" OFF) # fuzztest is a rather large dependency
option(MANIFOLD_PAR "Enable Parallel backend" OFF)
option(MANIFOLD_TEST "Enable testing suite" ON)
option(BUILD_SHARED_LIBS "Build shared library" ON)
include(CMakeDependentOption)
cmake_dependent_option(MANIFOLD_CBIND "Build C (FFI) bindings" ON "MANIFOLD_CROSS_SECTION" OFF)
option(MANIFOLD_PYBIND "Build python bindings" OFF)
cmake_dependent_option(MANIFOLD_JSBIND "Build js binding" ON "EMSCRIPTEN" OFF)

# Set some option values in the CMake cache
if(NOT MANIFOLD_FLAGS)
  set(MANIFOLD_FLAGS "")
endif()
set(MANIFOLD_FLAGS "" CACHE STRING "Manifold compiler flags")


# Development options
option(TRACY_ENABLE "Use tracy profiling" OFF)
option(TRACY_MEMORY_USAGE "Track memory allocation with tracy (expensive)" OFF)
option(BUILD_TEST_CGAL "Build CGAL performance comparisons" OFF)
mark_as_advanced(TRACY_ENABLE)
mark_as_advanced(TRACY_MEMORY_USAGE)
mark_as_advanced(BUILD_TEST_CGAL)

# Define some paths for CMake
include(GNUInstallDirs)

# Always build position independent code for relocatability
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Various Compiler Flags
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

# RPath settings
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
  set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
endif("${isSystemDir}" STREQUAL "-1")


# Dependencies
include(FetchContent)

# GLM is required in all configurations.
find_package(glm QUIET)
if(NOT glm_FOUND)
  if (NOT MANIFOLD_DOWNLOADS)
    message(FATAL_ERROR "glm not found, and dependency downloading disabled.  Please install glm and reconfigure.\n")
  endif()
  message(STATUS "glm not found, downloading from source")
  set(GLM_BUILD_INSTALL "ON" CACHE STRING "")
  FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
    GIT_PROGRESS TRUE
    )
  FetchContent_MakeAvailable(glm)
  if(NOT EMSCRIPTEN)
    install(TARGETS glm)
  endif()
endif()

# If we're building parallel, we need the requisite libraries
if(MANIFOLD_PAR)
  find_package(Threads REQUIRED)
  find_package(TBB QUIET)
  find_package(PkgConfig QUIET)
  if(NOT TBB_FOUND AND PKG_CONFIG_FOUND)
    pkg_check_modules(TBB tbb)
  endif()
  if(NOT TBB_FOUND)
    if (NOT MANIFOLD_DOWNLOADS)
      message(FATAL_ERROR "Parallel mode enabled, but TBB was not found and dependency downloading is disabled.  Please install TBB and reconfigure.\n")
    endif()

    # TODO - this isn't recommended by the TBB upstream - should we be
    # doing it?
    message(STATUS "TBB not found, downloading from source")
    set(TBB_TEST OFF CACHE INTERNAL "" FORCE)
    set(TBB_STRICT OFF CACHE INTERNAL "" FORCE)
    FetchContent_Declare(TBB
      GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
      GIT_TAG        v2021.11.0
      GIT_PROGRESS   TRUE
      )
    FetchContent_MakeAvailable(TBB)
    set_property(DIRECTORY ${tbb_SOURCE_DIR} PROPERTY EXCLUDE_FROM_ALL YES)
    # note: we do want to install tbb to the user machine when built from
    # source
    if(NOT EMSCRIPTEN)
      install(TARGETS tbb)
    endif()
  endif()
endif()

# If we're building cross_section, we need Clipper2
if(MANIFOLD_CROSS_SECTION)
  find_package(Clipper2 QUIET)
  if(NOT Clipper2_FOUND AND PKG_CONFIG_FOUND)
    pkg_check_modules(Clipper2 Clipper2)
  endif()
  if(Clipper2_FOUND)
    add_library(Clipper2 SHARED IMPORTED)
    set_property(TARGET Clipper2 PROPERTY
      IMPORTED_LOCATION ${Clipper2_LINK_LIBRARIES})
    if(WIN32)
      set_property(TARGET Clipper2 PROPERTY
	IMPORTED_IMPLIB ${Clipper2_LINK_LIBRARIES})
    endif()
    target_include_directories(Clipper2 INTERFACE ${Clipper2_INCLUDE_DIRS})
  else()
    if (NOT MANIFOLD_DOWNLOADS)
      message(FATAL_ERROR "cross_section enabled, but Clipper2 was not found and dependency downloading is disabled.  Please install Clipper2 and reconfigure.\n")
    endif()
    message(STATUS "clipper2 not found, downloading from source")
    set(CLIPPER2_UTILS OFF)
    set(CLIPPER2_EXAMPLES OFF)
    set(CLIPPER2_TESTS OFF)
    set(CLIPPER2_USINGZ "OFF" CACHE STRING "Preempt cache default of USINGZ (we only use 2d)")
    FetchContent_Declare(Clipper2
      GIT_REPOSITORY https://github.com/AngusJohnson/Clipper2.git
      GIT_TAG ff378668baae3570e9d8070aa9eb339bdd5a6aba
      GIT_PROGRESS TRUE
      SOURCE_SUBDIR CPP
      )
    FetchContent_MakeAvailable(Clipper2)
    if(NOT EMSCRIPTEN)
      install(TARGETS Clipper2)
    endif()
  endif()
endif()

# If we're supporting mesh I/O, we need assimp
if(MANIFOLD_EXPORT)
  find_package(assimp REQUIRED)
endif()

add_subdirectory(src)
add_subdirectory(include)
add_subdirectory(bindings)
add_subdirectory(samples)
add_subdirectory(test)
add_subdirectory(extras)

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
message(STATUS "MANIFOLD_CBIND:              ${MANIFOLD_CBIND}")
message(STATUS "MANIFOLD_PYBIND:             ${MANIFOLD_PYBIND}")
message(STATUS "MANIFOLD_JSBIND:             ${MANIFOLD_JSBIND}")
message(STATUS "MANIFOLD_EXCEPTIONS:         ${MANIFOLD_EXCEPTIONS}")
message(STATUS " ")

# If it's an EMSCRIPTEN build, we're done
if(EMSCRIPTEN)
  return()
endif()


# CMake exports
configure_file(manifoldConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/manifoldConfig.cmake @ONLY)
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_BINARY_DIR}/cmake/manifoldConfigVersion.cmake
  VERSION ${MANIFOLD_VERSION}
  COMPATIBILITY SameMajorVersion
  )

# Location of inputs for CMake find_package - see:
# https://cmake.org/cmake/help/latest/command/find_package.html
set(EXPORT_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake)

install(EXPORT manifoldTargets
  NAMESPACE manifold::
  DESTINATION ${EXPORT_INSTALL_DIR}/manifold
  )
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/manifoldConfigVersion.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/manifoldConfig.cmake
  DESTINATION ${EXPORT_INSTALL_DIR}/manifold
  )


# PkgConfig file
if(MANIFOLD_CROSS_SECTION)
  set(TEMPLATE_OPTIONAL_CLIPPER "Clipper2")
endif()
configure_file(manifold.pc.in ${CMAKE_CURRENT_BINARY_DIR}/manifold.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/manifold.pc
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)


# Local Variables:
# tab-width: 8
# mode: cmake
# indent-tabs-mode: t
# End:
# ex: shiftwidth=2 tabstop=8
