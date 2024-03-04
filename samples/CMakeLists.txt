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

project(samples)

add_library(samples src/bracelet.cpp src/knot.cpp src/menger_sponge.cpp
    src/rounded_frame.cpp src/scallop.cpp src/tet_puzzle.cpp
    src/gyroid_module.cpp src/condensed_matter.cpp)

target_include_directories(samples
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(samples
    PUBLIC manifold sdf
)

target_compile_options(samples PRIVATE ${MANIFOLD_FLAGS})
target_compile_features(samples PUBLIC cxx_std_17)
