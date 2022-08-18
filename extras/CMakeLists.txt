# Copyright 2022 The Manifold Authors.
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

project(extras)

add_executable(perfTest perf_test.cpp)
target_link_libraries(perfTest manifold)

target_compile_options(perfTest PRIVATE ${MANIFOLD_FLAGS})
target_compile_features(perfTest PUBLIC cxx_std_14)

if(BUILD_TEST_CGAL)
    add_executable(perfTestCGAL perf_test_cgal.cpp)
    find_package(CGAL REQUIRED COMPONENTS Core)
    target_compile_definitions(perfTestCGAL PRIVATE CGAL_USE_GMPXX)

    # target_compile_definitions(perfTestCGAL PRIVATE CGAL_DEBUG)
    target_link_libraries(perfTestCGAL manifold CGAL::CGAL CGAL::CGAL_Core)

    target_compile_options(perfTestCGAL PRIVATE ${MANIFOLD_FLAGS})
    target_compile_features(perfTestCGAL PUBLIC cxx_std_14)
endif()
