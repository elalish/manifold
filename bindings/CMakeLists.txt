# Copyright 2022 The Manifold Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(NOT MANIFOLD_CROSS_SECTION)
  return()
endif()

if(MANIFOLD_CBIND)
  add_subdirectory(c)
endif()

if(NOT EMSCRIPTEN AND MANIFOLD_PYBIND)
  add_subdirectory(python)
endif()

if(EMSCRIPTEN AND MANIFOLD_JSBIND)
  add_subdirectory(wasm)
endif()
