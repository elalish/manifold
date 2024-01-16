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

project(python)

add_subdirectory(third_party)
nanobind_add_module(
  manifold3d
  NB_STATIC STABLE_ABI LTO
  manifold3d.cpp)
target_link_libraries(manifold3d PRIVATE manifold sdf polygon)
target_compile_options(manifold3d PRIVATE ${MANIFOLD_FLAGS} -DMODULE_NAME=manifold3d)
target_compile_features(manifold3d PUBLIC cxx_std_17)
set_target_properties(manifold3d PROPERTIES OUTPUT_NAME "manifold3d")

message(Python_EXECUTABLE = ${Python_EXECUTABLE})
add_custom_target(
  autogen_docstrings
  ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_docs.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  BYPRODUCTS autogen_docstrings.inl
)
target_include_directories(manifold3d PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(autogen_docstrings manifold sdf polygon)
add_dependencies(manifold3d autogen_docstrings)

if(SKBUILD)
install(
  TARGETS manifold3d
  LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}
  COMPONENT bindings
)
else()
install(
  TARGETS manifold3d
  LIBRARY DESTINATION ${Python_SITEARCH}
  COMPONENT bindings
)
endif()
