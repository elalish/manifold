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

if(NOT MANIFOLD_PYBIND)
  return()
endif()

project(python)

if(Python_VERSION VERSION_LESS 3.12)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
else()
  find_package(Python COMPONENTS Interpreter Development.SABIModule REQUIRED)
endif()
if(Python_VERSION VERSION_GREATER_EQUAL 3.11)
  set(MANIFOLD_PYBIND_STUBGEN ON)
else()
  # stubgen does not support version less than 3.11
  set(MANIFOLD_PYBIND_STUBGEN OFF)
  message("Python version too old, stub will not be generated")
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --version
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_VERSION)
# we are fine with 2.0.0
if(NB_VERSION VERSION_GREATER_EQUAL 2.0.0)
  message("Found nanobind, version ${NB_VERSION}")
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)
else()
  message(STATUS "nanobind not found, downloading from source")
  include(FetchContent)
  FetchContent_Declare(nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG 9641bb7151f04120013b812789b3ebdfa7e7324f # v2.1.0
    GIT_PROGRESS TRUE
    )
  FetchContent_MakeAvailable(nanobind)
endif()

if(NB_VERSION VERSION_LESS 2.1.0)
  message("Nanobind version too old, stub will not be generated")
  set(MANIFOLD_PYBIND_STUBGEN OFF)
endif()

nanobind_add_module(
  manifold3d
  NB_STATIC STABLE_ABI LTO
  autogen_docstrings.inl
  manifold3d.cpp)

if(MANIFOLD_PYBIND_STUBGEN)
  nanobind_add_stub(
    manifold3d_stub
    MODULE manifold3d
    OUTPUT manifold3d.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:manifold3d>
    DEPENDS manifold3d
    MARKER_FILE py.typed
    PATTERN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/stub_pattern.txt
    )
endif()

target_link_libraries(manifold3d PRIVATE manifold)
target_compile_options(manifold3d PRIVATE ${MANIFOLD_FLAGS} -DMODULE_NAME=manifold3d)
target_compile_features(manifold3d PUBLIC cxx_std_17)
set_target_properties(manifold3d PROPERTIES OUTPUT_NAME "manifold3d")

message(Python_EXECUTABLE = ${Python_EXECUTABLE})
# ideally we should generate a dependency file from python...
set(DOCSTRING_DEPS
  ${CMAKE_SOURCE_DIR}/src/manifold.cpp
  ${CMAKE_SOURCE_DIR}/src/constructors.cpp
  ${CMAKE_SOURCE_DIR}/src/sort.cpp
  ${CMAKE_SOURCE_DIR}/src/cross_section/cross_section.cpp
  ${CMAKE_SOURCE_DIR}/src/polygon.cpp
  ${CMAKE_SOURCE_DIR}/include/manifold/common.h
  )
add_custom_command(
  OUTPUT autogen_docstrings.inl
  DEPENDS ${DOCSTRING_DEPS}
  COMMAND ${Python_EXECUTABLE} ARGS ${CMAKE_CURRENT_SOURCE_DIR}/gen_docs.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
target_include_directories(manifold3d PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

if(SKBUILD)
  set(MANIFOLD_PYBIND_LIBDIR ${SKBUILD_PLATLIB_DIR})
else()
  set(MANIFOLD_PYBIND_LIBDIR ${Python_SITEARCH})
endif()

install(
  TARGETS manifold3d
  LIBRARY DESTINATION ${MANIFOLD_PYBIND_LIBDIR}
  COMPONENT bindings
  EXCLUDE_FROM_ALL
  )
if(MANIFOLD_PYBIND_STUBGEN)
  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/py.typed ${CMAKE_CURRENT_BINARY_DIR}/manifold3d.pyi
    DESTINATION ${MANIFOLD_PYBIND_LIBDIR}
    COMPONENT bindings
    EXCLUDE_FROM_ALL
    )
endif()

# Local Variables:
# tab-width: 8
# mode: cmake
# indent-tabs-mode: t
# End:
# ex: shiftwidth=2 tabstop=8
