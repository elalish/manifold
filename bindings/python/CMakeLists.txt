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
pybind11_add_module(manifold3d manifold3d.cpp)
target_link_libraries(manifold3d PRIVATE manifold)
target_compile_options(manifold3d PRIVATE ${MANIFOLD_FLAGS})
target_compile_features(manifold3d PUBLIC cxx_std_17)
target_include_directories(manifold3d
    PRIVATE ${PYBIND11_DIR}/include
)
