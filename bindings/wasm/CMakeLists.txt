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

# make sure that we recompile the wasm when bindings.js is being modified
set_source_files_properties(bindings.cpp OBJECT_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/bindings.js)
target_link_libraries(manifoldjs manifold sdf)
target_compile_options(manifoldjs PRIVATE ${MANIFOLD_FLAGS} -fexceptions)
target_link_options(manifoldjs PUBLIC --pre-js ${CMAKE_CURRENT_SOURCE_DIR}/bindings.js --bind -sALLOW_TABLE_GROWTH=1
  -sEXPORTED_RUNTIME_METHODS=addFunction,removeFunction -sMODULARIZE=1)

target_compile_features(manifoldjs PUBLIC cxx_std_14)
set_target_properties(manifoldjs PROPERTIES OUTPUT_NAME "manifold")

file(COPY examples;test;. DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS examples;test)

add_custom_command(
        TARGET manifoldjs POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_BINARY_DIR}/manifold.*
                ${CMAKE_CURRENT_BINARY_DIR}/examples/)
                