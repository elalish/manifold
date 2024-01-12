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

project(sdf)

add_library(${PROJECT_NAME} OBJECT src/sdf.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(${PROJECT_NAME} PUBLIC utilities)

target_compile_options(${PROJECT_NAME} PRIVATE ${MANIFOLD_FLAGS})
install(TARGETS ${PROJECT_NAME} EXPORT manifoldTargets)
install(FILES include/sdf.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
