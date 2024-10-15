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

if(NOT MANIFOLD_TEST)
  return()
endif()

enable_testing()

find_package(gtest QUIET)
if(NOT gtest_FOUND)
  if(NOT MANIFOLD_DOWNLOADS)
    message(
      WARNING
      "Downloading is disabled, but testing requires googletest - skipping.\n"
    )
    return()
  endif()

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
  )
  FetchContent_MakeAvailable(googletest)
endif()

# put fast/simple tests files first to run earlier
set(
  SOURCE_FILES
  test_main.cpp
  polygon_test.cpp
  properties_test.cpp
  manifold_test.cpp
  boolean_test.cpp
  sdf_test.cpp
  smooth_test.cpp
  hull_test.cpp
  samples_test.cpp
  boolean_complex_test.cpp
)

if(MANIFOLD_CROSS_SECTION)
  list(APPEND SOURCE_FILES cross_section_test.cpp)
endif()

if(MANIFOLD_CBIND AND NOT EMSCRIPTEN)
  list(APPEND SOURCE_FILES manifoldc_test.cpp)
endif()

if(NOT TARGET GTest::gtest_main)
  add_library(GTest::gtest_main ALIAS gtest_main)
endif()

add_executable(manifold_test ${SOURCE_FILES})
target_link_libraries(manifold_test GTest::gtest_main manifold samples)

if(MANIFOLD_FUZZ)
  FetchContent_Declare(
    fuzztest
    GIT_REPOSITORY https://github.com/google/fuzztest.git
    # note that if commit hash is used, it cannot be a shallow clone
    GIT_TAG 2606e04a43e5a7730e437a849604a61f1cb0ff28
    GIT_PROGRESS TRUE
  )
  FetchContent_MakeAvailable(fuzztest)
  fuzztest_setup_fuzzing_flags()
  add_executable(polygon_fuzz polygon_fuzz.cpp)
  target_link_libraries(polygon_fuzz PUBLIC manifold)
  link_fuzztest(polygon_fuzz)
  gtest_discover_tests(polygon_fuzz)
endif()

if(MANIFOLD_CBIND AND NOT EMSCRIPTEN)
  target_link_libraries(manifold_test manifoldc)
endif()

target_compile_options(manifold_test PRIVATE ${MANIFOLD_FLAGS})
target_compile_features(manifold_test PUBLIC cxx_std_17)

add_test(test_all manifold_test)
target_precompile_headers(manifold_test INTERFACE test.h)

if(EMSCRIPTEN)
  list(
    APPEND
    EMSCRIPTEN_LINK_FLAGS
    -sASSERTIONS=1
    -sDEMANGLE_SUPPORT=1
    --bind
    --preload-file
    ${CMAKE_CURRENT_SOURCE_DIR}/polygons@/polygons
  )
  list(JOIN EMSCRIPTEN_LINK_FLAGS " " EMSCRIPTEN_LINK_FLAGS)
  set_target_properties(
    manifold_test
    PROPERTIES LINK_FLAGS ${EMSCRIPTEN_LINK_FLAGS}
  )
endif()

if(MSVC)
  set_target_properties(
    manifold_test
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_target_properties(
    manifold_test
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
endif()
