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
exportbin(perfTest)

if(MANIFOLD_PAR AND NOT MSVC)
  add_executable(stlTest stl_test.cpp)
  target_link_libraries(stlTest PRIVATE manifold TBB::tbb)
  target_compile_options(stlTest PRIVATE ${MANIFOLD_FLAGS})
endif()

add_executable(largeSceneTest large_scene_test.cpp)
target_link_libraries(largeSceneTest manifold)
target_compile_options(largeSceneTest PRIVATE ${MANIFOLD_FLAGS})
exportbin(largeSceneTest)

if(MANIFOLD_DEBUG)
  add_executable(minimizeTestcase minimize_testcase.cpp)
  target_link_libraries(
    minimizeTestcase
    manifold
    $<$<BOOL:${MANIFOLD_PAR}>:TBB::tbb>
  )
  target_compile_options(minimizeTestcase PRIVATE ${MANIFOLD_FLAGS})
  exportbin(minimizeTestcase)
endif()

if(MANIFOLD_EXPORT)
  add_executable(convertFile convert_file.cpp)
  target_link_libraries(convertFile manifold manifold)
  target_compile_options(convertFile PRIVATE ${MANIFOLD_FLAGS})
  exportbin(convertFile)
endif()
