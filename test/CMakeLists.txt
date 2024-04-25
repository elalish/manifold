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

project(manifold_test)

set(CMAKE_CXX_STANDARD 17)
enable_testing()

set(SOURCE_FILES polygon_test.cpp cross_section_test.cpp manifold_test.cpp boolean_test.cpp complex_boolean_test.cpp sdf_test.cpp samples_test.cpp test_main.cpp)

if(MANIFOLD_CBIND AND NOT EMSCRIPTEN)
  list(APPEND SOURCE_FILES manifoldc_test.cpp)
endif()

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# Prevent installation of GTest with your project
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(INSTALL_GMOCK OFF CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE
  FIND_PACKAGE_ARGS NAMES GTest gtest
)
FetchContent_MakeAvailable(googletest)

if(NOT TARGET GTest::gtest_main)
    add_library(GTest::gtest_main ALIAS gtest_main)
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} polygon GTest::gtest_main manifold sdf samples cross_section)

if(MANIFOLD_FUZZ)
  FetchContent_Declare(fuzztest
    GIT_REPOSITORY https://github.com/google/fuzztest.git
    # note that if commit hash is used, it cannot be a shallow clone
    GIT_TAG        2606e04a43e5a7730e437a849604a61f1cb0ff28
    GIT_PROGRESS   TRUE
  )
  FetchContent_MakeAvailable(fuzztest)
  fuzztest_setup_fuzzing_flags()
  add_executable(polygon_fuzz polygon_fuzz.cpp)
  target_link_libraries(polygon_fuzz PUBLIC polygon)
  link_fuzztest(polygon_fuzz)
  gtest_discover_tests(polygon_fuzz)
endif()

if(MANIFOLD_CBIND AND NOT EMSCRIPTEN)
  target_link_libraries(${PROJECT_NAME} manifoldc)
endif()

if(MANIFOLD_EXPORT)
  target_link_libraries(${PROJECT_NAME} meshIO)
  target_compile_options(${PROJECT_NAME} PUBLIC -DMANIFOLD_EXPORT)
endif()

target_compile_options(${PROJECT_NAME} PRIVATE ${MANIFOLD_FLAGS})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

add_test(test_all ${PROJECT_NAME})

if(EMSCRIPTEN)
  set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS
    "-s ASSERTIONS=1 -s DEMANGLE_SUPPORT=1 --bind")
endif()

if(MSVC)
    set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY 
        ${CMAKE_BINARY_DIR}/bin )
    set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_BINARY_DIR}/bin )
endif()
