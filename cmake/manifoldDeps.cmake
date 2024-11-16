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

# If we're building parallel, we need the requisite libraries
if(MANIFOLD_PAR)
  find_package(Threads REQUIRED)
  find_package(TBB QUIET)
  find_package(PkgConfig QUIET)
  if(NOT TBB_FOUND AND PKG_CONFIG_FOUND)
    pkg_check_modules(TBB tbb)
  endif()
  if(NOT TBB_FOUND)
    logmissingdep("TBB" , "Parallel mode")
    set(TBB_TEST OFF CACHE INTERNAL BOOL FORCE)
    set(TBB_STRICT OFF CACHE INTERNAL BOOL FORCE)
    FetchContent_Declare(
      TBB
      GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
      GIT_TAG v2021.11.0
      GIT_PROGRESS TRUE
      EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(TBB)
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
    set(CLIPPER2_UTILS OFF CACHE INTERNAL BOOL "" FORCE)
    set(CLIPPER2_EXAMPLES OFF CACHE INTERNAL BOOL "" FORCE)
    set(CLIPPER2_TESTS OFF CACHE INTERNAL BOOL "" FORCE)
    set(CLIPPER2_USINGZ OFF CACHE INTERNAL BOOL "" FORCE)
    FetchContent_Declare(
      Clipper2
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

if(TRACY_ENABLE)
  logmissingdep("tracy" , "TRACY_ENABLE")
  FetchContent_Declare(
    tracy
    GIT_REPOSITORY https://github.com/wolfpld/tracy.git
    GIT_TAG v0.10
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
  )
  FetchContent_MakeAvailable(tracy)
endif()

# If we're supporting mesh I/O, we need assimp
if(MANIFOLD_EXPORT)
  find_package(assimp REQUIRED)
endif()

if(MANIFOLD_TEST)
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    logmissingdep("GTest" , "MANIFOLD_TEST")
    set(gtest_force_shared_crt ON CACHE INTERNAL BOOL "" FORCE)
    # Prevent installation of GTest with your project
    set(INSTALL_GTEST OFF CACHE INTERNAL BOOL "" FORCE)
    set(INSTALL_GMOCK OFF CACHE INTERNAL BOOL "" FORCE)

    include(FetchContent)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.14.0
      GIT_SHALLOW TRUE
      GIT_PROGRESS TRUE
      FIND_PACKAGE_ARGS NAMES GTest gtest
    )
    FetchContent_MakeAvailable(googletest)
  endif()
endif()
