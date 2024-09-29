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

project(wasm)

add_executable(manifoldjs bindings.cpp)

set_source_files_properties(bindings.cpp PROPERTIES OBJECT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bindings.js)
target_link_libraries(manifoldjs manifold)
target_compile_options(manifoldjs PRIVATE ${MANIFOLD_FLAGS})
target_link_options(manifoldjs PUBLIC --pre-js ${CMAKE_CURRENT_SOURCE_DIR}/bindings.js --bind -sALLOW_TABLE_GROWTH=1
  -sEXPORTED_RUNTIME_METHODS=addFunction,removeFunction -sMODULARIZE=1 -sEXPORT_ES6=1)

target_compile_features(manifoldjs PUBLIC cxx_std_17)
set_target_properties(manifoldjs PROPERTIES OUTPUT_NAME "manifold")

file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples/built)

# ensure that interface files are copied over when modified
add_custom_target(js_deps ALL
  DEPENDS manifoldjs
  ${CMAKE_CURRENT_SOURCE_DIR}/manifold*.d.ts)

# copy WASM build back here for publishing to npm
add_custom_command(
  TARGET js_deps POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_BINARY_DIR}/manifold.*
  ${CMAKE_CURRENT_SOURCE_DIR})

# copy WASM build and TS declarations for Vite to package into ManifoldCAD.org
add_custom_command(
  TARGET js_deps POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/manifold*
  ${CMAKE_CURRENT_SOURCE_DIR}/examples/built/)

# copy TS declarations to public so they can be accessed by our editor
add_custom_command(
  TARGET js_deps POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/manifold*.d.ts
  ${CMAKE_CURRENT_SOURCE_DIR}/examples/public/)

# Local Variables:
# tab-width: 8
# mode: cmake
# indent-tabs-mode: t
# End:
# ex: shiftwidth=2 tabstop=8
