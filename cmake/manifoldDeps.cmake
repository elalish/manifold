# Copyright 2024 The Manifold Authors.
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
include(FetchContent)

function(logmissingdep PKG)
  # if this is already in FETCHCONTENT_SOURCE_DIR_X, we don't have to
  # download...
  if(DEFINED FETCHCONTENT_SOURCE_DIR_${PKG})
    return()
  endif()
  if(NOT MANIFOLD_DOWNLOADS)
    if(ARGC EQUAL 3)
      set(MSG "${ARGV2} enabled, but ${PKG} was not found ")
      string(APPEND MSG "and dependency downloading is disabled. ")
    else()
      set(MSG "${PKG} not found, and dependency downloading disabled. ")
    endif()
    string(APPEND MSG "Please install ${PKG} and reconfigure.\n")
    message(FATAL_ERROR ${MSG})
  else()
    message(STATUS "${PKG} not found, downloading from source")
  endif()
endfunction()

set(OLD_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
# we build fetched dependencies as static library
set(BUILD_SHARED_LIBS OFF)

# If we're building parallel, we need tbb
if(MANIFOLD_PAR)
  find_package(Threads REQUIRED)
  if(NOT MANIFOLD_USE_BUILTIN_TBB)
    find_package(TBB QUIET)
    find_package(PkgConfig QUIET)
    if(NOT TBB_FOUND AND PKG_CONFIG_FOUND)
      pkg_check_modules(TBB tbb)
    endif()
  endif()
  if(TBB_FOUND)
    if(NOT TARGET TBB::tbb)
      if(NOT TARGET tbb)
        add_library(TBB::tbb SHARED IMPORTED)
        set_property(
          TARGET TBB::tbb
          PROPERTY IMPORTED_LOCATION ${TBB_LINK_LIBRARIES}
        )
        target_include_directories(TBB::tbb INTERFACE ${TBB_INCLUDE_DIRS})
      else()
        add_library(TBB::tbb ALIAS tbb)
      endif()
    endif()
  else()
    logmissingdep("TBB" , "Parallel mode")
    set(MANIFOLD_USE_BUILTIN_TBB ON)
    set(TBB_TEST OFF CACHE INTERNAL "" FORCE)
    set(TBB_STRICT OFF CACHE INTERNAL "" FORCE)
    FetchContent_Declare(
      TBB
      GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
      GIT_TAG v2021.11.0
      GIT_PROGRESS TRUE
      EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(TBB)
  endif()
endif()

# If we're building cross_section, we need Clipper2
if(MANIFOLD_CROSS_SECTION)
  if(NOT MANIFOLD_USE_BUILTIN_CLIPPER2)
    find_package(Clipper2 QUIET)
    if(NOT Clipper2_FOUND AND PKG_CONFIG_FOUND)
      pkg_check_modules(Clipper2 Clipper2)
    endif()
  endif()
  if(Clipper2_FOUND)
    add_library(Clipper2 SHARED IMPORTED)
    set_property(
      TARGET Clipper2
      PROPERTY IMPORTED_LOCATION ${Clipper2_LINK_LIBRARIES}
    )
    if(WIN32)
      set_property(
        TARGET Clipper2
        PROPERTY IMPORTED_IMPLIB ${Clipper2_LINK_LIBRARIES}
      )
    endif()
    target_include_directories(Clipper2 INTERFACE ${Clipper2_INCLUDE_DIRS})
  else()
    logmissingdep("Clipper2" , "cross_section")
    set(MANIFOLD_USE_BUILTIN_CLIPPER2 ON)
    set(CLIPPER2_UTILS OFF)
    set(CLIPPER2_EXAMPLES OFF)
    set(CLIPPER2_TESTS OFF)
    set(
      CLIPPER2_USINGZ
      OFF
      CACHE BOOL
      "Preempt cache default of USINGZ (we only use 2d)"
    )
    FetchContent_Declare(
      Clipper2
      GIT_REPOSITORY https://github.com/AngusJohnson/Clipper2.git
      GIT_TAG ff378668baae3570e9d8070aa9eb339bdd5a6aba
      GIT_PROGRESS TRUE
      SOURCE_SUBDIR CPP
      EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(Clipper2)
  endif()
  if(NOT TARGET Clipper2::Clipper2)
    add_library(Clipper2::Clipper2 ALIAS Clipper2)
  endif()
endif()

if(TRACY_ENABLE)
  logmissingdep("tracy" , "TRACY_ENABLE")
  FetchContent_Declare(
    tracy
    GIT_REPOSITORY https://github.com/wolfpld/tracy.git
    GIT_TAG v0.10
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(tracy)
endif()

# If we're supporting mesh I/O, we need assimp
if(MANIFOLD_EXPORT)
  find_package(assimp REQUIRED)
endif()

if(MANIFOLD_PYBIND)
  if(Python_VERSION VERSION_LESS 3.12)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  else()
    find_package(Python COMPONENTS Interpreter Development.SABIModule REQUIRED)
  endif()
  if(Python_VERSION VERSION_GREATER_EQUAL 3.11)
    set(MANIFOLD_PYBIND_STUBGEN ON)
  else()
    # stubgen does not support version less than 3.11
    set(MANIFOLD_PYBIND_STUBGEN OFF)
    message("Python version too old, stub will not be generated")
  endif()

  if(NOT MANIFOLD_USE_BUILTIN_NANOBIND)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m nanobind --version
      OUTPUT_STRIP_TRAILING_WHITESPACE
      OUTPUT_VARIABLE NB_VERSION
    )
  endif()
  # we are fine with 2.0.0
  if(NB_VERSION VERSION_GREATER_EQUAL 2.0.0)
    message("Found nanobind, version ${NB_VERSION}")
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      OUTPUT_VARIABLE nanobind_ROOT
    )
    find_package(nanobind CONFIG REQUIRED)
  else()
    logmissingdep("nanobind" , "MANIFOLD_PYBIND")
    set(MANIFOLD_USE_BUILTIN_NANOBIND ON)
    FetchContent_Declare(
      nanobind
      GIT_REPOSITORY https://github.com/wjakob/nanobind.git
      GIT_TAG
        784efa2a0358a4dc5432c74f5685ee026e20f2b6 # v2.2.0
      GIT_PROGRESS TRUE
      EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(nanobind)
    set(NB_VERSION 2.2.0)
  endif()

  if(NB_VERSION VERSION_LESS 2.1.0)
    message("Nanobind version too old, stub will not be generated")
    set(MANIFOLD_PYBIND_STUBGEN OFF)
  endif()
endif()

if(MANIFOLD_TEST)
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    logmissingdep("GTest" , "MANIFOLD_TEST")
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    # Prevent installation of GTest with your project
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(INSTALL_GMOCK OFF CACHE BOOL "" FORCE)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.14.0
      GIT_SHALLOW TRUE
      GIT_PROGRESS TRUE
      FIND_PACKAGE_ARGS NAMES GTest gtest
      EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(googletest)
  endif()
  if(NOT TARGET GTest::gtest_main)
    add_library(GTest::gtest_main ALIAS gtest_main)
  endif()
endif()

if(MANIFOLD_FUZZ)
  logmissingdep("fuzztest" , "MANIFOLD_FUZZ")
  FetchContent_Declare(
    fuzztest
    GIT_REPOSITORY https://github.com/google/fuzztest.git
    GIT_TAG 2606e04a43e5a7730e437a849604a61f1cb0ff28
    GIT_PROGRESS TRUE
  )
  FetchContent_MakeAvailable(fuzztest)
endif()

set(BUILD_SHARED_LIBS ${OLD_BUILD_SHARED_LIBS})
