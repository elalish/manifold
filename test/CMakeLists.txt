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

add_subdirectory(third_party)

enable_testing()

set(SOURCE_FILES polygon_test.cpp mesh_test.cpp sdf_test.cpp samples_test.cpp test_main.cpp)
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} polygon GTest::GTest manifold samples samplesGPU)

if(MANIFOLD_EXPORT)
  add_subdirectory(meshIO)
  target_link_libraries(${PROJECT_NAME} meshIO)
  target_compile_options(${PROJECT_NAME} PUBLIC -DMANIFOLD_EXPORT)
endif()

if(MANIFOLD_USE_CUDA)
  set_source_files_properties(mesh_test.cpp sdf_test.cpp PROPERTIES LANGUAGE CUDA)
  set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 61)
endif()

target_compile_options(${PROJECT_NAME} PRIVATE ${MANIFOLD_FLAGS})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

add_test(test_all ${PROJECT_NAME})

if(EMSCRIPTEN)
  set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS
    "-s ASSERTIONS=1 -s DEMANGLE_SUPPORT=1 --bind")
else()
endif()
