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
  $<$<BOOL:${MANIFOLD_CROSS_SECTION}>:cross_section_test.cpp>
  $<$<BOOL:${MANIFOLD_CBIND}>:manifoldc_test.cpp>
)

add_executable(manifold_test ${SOURCE_FILES})
target_link_libraries(
  manifold_test
  PRIVATE
    GTest::gtest_main
    manifold
    samples
    $<$<BOOL:${MANIFOLD_CBIND}>:manifoldc>
    $<$<BOOL:${MANIFOLD_PAR}>:TBB::tbb>
)

if(EMSCRIPTEN)
  target_link_options(
    manifold_test
    PRIVATE
      -sASSERTIONS=1
      -sDEMANGLE_SUPPORT=1
      --bind
      --preload-file
      ${CMAKE_CURRENT_SOURCE_DIR}/polygons@/polygons
  )
endif()

target_compile_options(manifold_test PRIVATE ${MANIFOLD_FLAGS})
exportbin(manifold_test)

add_test(test_all manifold_test)

if(MANIFOLD_FUZZ)
  fuzztest_setup_fuzzing_flags()
  add_executable(polygon_fuzz polygon_fuzz.cpp)
  target_link_libraries(polygon_fuzz PUBLIC manifold)
  link_fuzztest(polygon_fuzz)
  gtest_discover_tests(polygon_fuzz)
endif()
