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

if(NOT MANIFOLD_TEST)
  return()
endif()

add_executable(perfTest perf_test.cpp)
target_link_libraries(perfTest manifold)
target_compile_options(perfTest PRIVATE ${MANIFOLD_FLAGS})
target_compile_features(perfTest PUBLIC cxx_std_17)

if(MANIFOLD_PAR AND NOT MSVC)
  add_executable(stlTest stl_test.cpp)
  target_link_libraries(stlTest manifold)
  target_compile_options(stlTest PRIVATE ${MANIFOLD_FLAGS})
  target_compile_features(stlTest PUBLIC cxx_std_17)
endif()

add_executable(largeSceneTest large_scene_test.cpp)
target_link_libraries(largeSceneTest manifold)
target_compile_options(largeSceneTest PRIVATE ${MANIFOLD_FLAGS})
target_compile_features(largeSceneTest PUBLIC cxx_std_17)

if(MANIFOLD_DEBUG)
  add_executable(minimizeTestcase minimize_testcase.cpp)
  target_link_libraries(minimizeTestcase manifold)
  target_compile_options(minimizeTestcase PRIVATE ${MANIFOLD_FLAGS})
  target_compile_features(minimizeTestcase PUBLIC cxx_std_17)
endif()

if(MANIFOLD_EXPORT)
  add_executable(convertFile convert_file.cpp)
  target_link_libraries(convertFile manifold manifold)
  target_compile_options(convertFile PRIVATE ${MANIFOLD_FLAGS})
  target_compile_features(convertFile PUBLIC cxx_std_17)
endif()

if(BUILD_TEST_CGAL)
  add_executable(perfTestCGAL perf_test_cgal.cpp)
  find_package(CGAL REQUIRED COMPONENTS Core)
  find_package(Boost REQUIRED COMPONENTS thread)
  target_compile_definitions(perfTestCGAL PRIVATE CGAL_USE_GMPXX)
  target_link_libraries(perfTestCGAL manifold CGAL::CGAL CGAL::CGAL_Core Boost::thread)
  target_compile_options(perfTestCGAL PRIVATE ${MANIFOLD_FLAGS})
  target_compile_features(perfTestCGAL PUBLIC cxx_std_17)

  add_executable(testHullPerformance test_hull_performance.cpp)
  target_compile_definitions(testHullPerformance PRIVATE CGAL_USE_GMPXX)
  target_link_libraries(testHullPerformance manifold samples CGAL::CGAL CGAL::CGAL_Core Boost::thread)
  target_compile_options(testHullPerformance PRIVATE ${MANIFOLD_FLAGS})
  target_compile_features(testHullPerformance PUBLIC cxx_std_17)
endif()


# Local Variables:
# tab-width: 8
# mode: cmake
# indent-tabs-mode: t
# End:
# ex: shiftwidth=2 tabstop=8
