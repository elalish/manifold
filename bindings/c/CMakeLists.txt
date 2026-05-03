# Copyright 2024 The Manifold Authors.
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

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

add_library(
  manifoldc
  manifoldc.cpp
  conv.cpp
  box.cpp
  cross.cpp
  rect.cpp
)
exportlib(manifoldc)

target_link_libraries(manifoldc PRIVATE manifold)

target_include_directories(
  manifoldc
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_options(manifoldc PRIVATE ${MANIFOLD_FLAGS})

install(TARGETS manifoldc EXPORT manifoldTargets)
install(
  FILES include/manifold/manifoldc.h include/manifold/types.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/manifold
)

# Always-on build-time check: compiles manifold/manifoldc.h with
# MANIFOLD_NO_IOSTREAM defined. Mirrors src/no_iostream_check.cpp's
# role for the C++ public header.
add_library(manifoldc_no_iostream_check OBJECT no_iostream_check.cpp)
target_link_libraries(manifoldc_no_iostream_check PRIVATE manifoldc)
if(NOT MANIFOLD_NO_IOSTREAM)
  target_compile_definitions(
    manifoldc_no_iostream_check PRIVATE MANIFOLD_NO_IOSTREAM
  )
endif()
