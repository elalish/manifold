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

# configuration summary, idea from openscad
# https://github.com/openscad/openscad/blob/master/cmake/Modules/info.cmake

get_target_property(MANIFOLD_COMPILE_OPTIONS manifold COMPILE_OPTIONS)
get_target_property(MANIFOLD_LINK_OPTIONS manifold LINK_OPTIONS)
if(MANIFOLD_LINK_OPTIONS STREQUAL "MANIFOLD_LINK_OPTIONS-NOTFOUND")
  set(MANIFOLD_LINK_OPTIONS "")
endif()

message(STATUS "====================================")
message(STATUS "Manifold Build Configuration Summary")
message(STATUS "====================================")
message(STATUS " ")
if(MXECROSS)
  message(STATUS "Environment: MXE")
elseif(APPLE)
  message(STATUS "Environment: macOS")
elseif(WIN32)
  if(MINGW)
    message(STATUS "Environment: msys2")
  else()
    message(STATUS "Environment: Windows")
  endif()
elseif(LINUX)
  message(STATUS "Environment: Linux")
elseif(UNIX)
  message(STATUS "Environment: Unknown Unix")
else()
  message(STATUS "Environment: Unknown")
endif()
message(STATUS " ")
message(STATUS "CMAKE_VERSION:                 ${CMAKE_VERSION}")
message(STATUS "CMAKE_TOOLCHAIN_FILE:          ${CMAKE_TOOLCHAIN_FILE}")
message(STATUS "CMAKE_GENERATOR:               ${CMAKE_GENERATOR}")
message(STATUS "CPACK_CMAKE_GENERATOR:         ${CPACK_CMAKE_GENERATOR}")
message(STATUS "CMAKE_BUILD_TYPE:              ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_PREFIX_PATH:             ${CMAKE_PREFIX_PATH}")
message(STATUS "CMAKE_CXX_COMPILER_ID:         ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION:    ${CMAKE_CXX_COMPILER_VERSION}")
if(APPLE)
  message(
    STATUS
    "CMAKE_OSX_DEPLOYMENT_TARGET:   ${CMAKE_OSX_DEPLOYMENT_TARGET}"
  )
  message(STATUS "CMAKE_OSX_ARCHITECTURES:       ${CMAKE_OSX_ARCHITECTURES}")
endif()
message(STATUS "BUILD_SHARED_LIBS:             ${BUILD_SHARED_LIBS}")
message(STATUS " ")
message(STATUS "MANIFOLD_VERSION:              ${MANIFOLD_VERSION}")
message(STATUS "MANIFOLD_PAR:                  ${MANIFOLD_PAR}")
message(STATUS "MANIFOLD_CROSS_SECTION:        ${MANIFOLD_CROSS_SECTION}")
message(STATUS "MANIFOLD_EXPORT:               ${MANIFOLD_EXPORT}")
message(STATUS "MANIFOLD_TEST:                 ${MANIFOLD_TEST}")
message(STATUS "MANIFOLD_DEBUG:                ${MANIFOLD_DEBUG}")
message(STATUS "MANIFOLD_CBIND:                ${MANIFOLD_CBIND}")
message(STATUS "MANIFOLD_PYBIND:               ${MANIFOLD_PYBIND}")
message(STATUS "MANIFOLD_JSBIND:               ${MANIFOLD_JSBIND}")
message(STATUS "MANIFOLD_FLAGS:                ${MANIFOLD_FLAGS}")
# these two include global flags added through add_compile_options and
# add_link_options
message(STATUS "MANIFOLD_COMPILE_OPTIONS:      ${MANIFOLD_COMPILE_OPTIONS}")
message(STATUS "MANIFOLD_LINK_OPTIONS:         ${MANIFOLD_LINK_OPTIONS}")
message(STATUS "MANIFOLD_DOWNLOADS:            ${MANIFOLD_DOWNLOADS}")
message(STATUS "MANIFOLD_USE_BUILTIN_TBB:      ${MANIFOLD_USE_BUILTIN_TBB}")
message(
  STATUS
  "MANIFOLD_USE_BUILTIN_CLIPPER2: ${MANIFOLD_USE_BUILTIN_CLIPPER2}"
)
message(
  STATUS
  "MANIFOLD_USE_BUILTIN_NANOBIND: ${MANIFOLD_USE_BUILTIN_NANOBIND}"
)
message(STATUS "MANIFOLD_FUZZ:                 ${MANIFOLD_FUZZ}")
message(STATUS "TRACY_ENABLE:                  ${TRACY_ENABLE}")
message(STATUS "TRACY_MEMORY_USAGE:            ${TRACY_MEMORY_USAGE}")
message(STATUS " ")
