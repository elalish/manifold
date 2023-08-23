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
if(SKBUILD)
set(module_name _manifold3d)
else()
set(module_name manifold3d)
endif()
pybind11_add_module(${module_name} manifold3d.cpp)
target_link_libraries(${module_name} PRIVATE manifold)
target_compile_options(${module_name} PRIVATE ${MANIFOLD_FLAGS}
  -DMODULE_NAME=${module_name})
target_compile_features(${module_name} PUBLIC cxx_std_17)
target_include_directories(${module_name}
    PRIVATE ${PYBIND11_DIR}/include
)
set_target_properties(${module_name} PROPERTIES OUTPUT_NAME "${module_name}")
if(SKBUILD)
install(
  TARGETS ${module_name}
  LIBRARY DESTINATION .
)
else()
install(
  TARGETS ${module_name}
  LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
  COMPONENT bindings
)
endif()
