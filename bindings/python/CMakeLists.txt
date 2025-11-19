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

nanobind_add_module(manifold3d NB_STATIC STABLE_ABI LTO autogen_docstrings.inl
                    manifold3d.cpp
)

message(Python_EXECUTABLE = ${Python_EXECUTABLE})

if(MANIFOLD_PYBIND_STUBGEN)
  nanobind_add_stub(
    manifold3d_stub
    MODULE manifold3d
    OUTPUT manifold3d.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:manifold3d>
    DEPENDS manifold3d
    PATTERN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/stub_pattern.txt
  )

  add_custom_command(
    TARGET manifold3d_stub
    POST_BUILD
    COMMAND
      "${Python_EXECUTABLE}"
      "${PROJECT_SOURCE_DIR}/bindings/python/stub_postbuild.py"
      "${CMAKE_CURRENT_BINARY_DIR}/manifold3d.pyi"
    VERBATIM
  )
endif()

target_link_libraries(manifold3d PRIVATE manifold)
target_compile_options(
  manifold3d
  PRIVATE ${MANIFOLD_FLAGS} -DMODULE_NAME=manifold3d
)
set_target_properties(manifold3d PROPERTIES OUTPUT_NAME "manifold3d")

# ideally we should generate a dependency file from python...
set(
  DOCSTRING_DEPS
  ${PROJECT_SOURCE_DIR}/src/manifold.cpp
  ${PROJECT_SOURCE_DIR}/src/constructors.cpp
  ${PROJECT_SOURCE_DIR}/src/sort.cpp
  ${PROJECT_SOURCE_DIR}/src/cross_section/cross_section.cpp
  ${PROJECT_SOURCE_DIR}/src/polygon.cpp
  ${PROJECT_SOURCE_DIR}/include/manifold/common.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gen_docs.py
  ${CMAKE_CURRENT_SOURCE_DIR}/docstring_override.txt
)
add_custom_command(
  OUTPUT autogen_docstrings.inl
  DEPENDS ${DOCSTRING_DEPS}
  COMMAND ${Python_EXECUTABLE}
  ARGS ${CMAKE_CURRENT_SOURCE_DIR}/gen_docs.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
target_include_directories(manifold3d PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

if(SKBUILD)
  set(MANIFOLD_PYBIND_LIBDIR ${SKBUILD_PLATLIB_DIR})
else()
  set(MANIFOLD_PYBIND_LIBDIR ${Python_SITEARCH})
endif()

configure_file(pyproject.toml.in
      ${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml @ONLY
      )
configure_file(getwheelname.py.in
      ${CMAKE_CURRENT_BINARY_DIR}/getwheelname.py @ONLY
      )
configure_file(setup.py
      ${CMAKE_CURRENT_BINARY_DIR}/setup.py COPYONLY
      )
execute_process(COMMAND ${Python_EXECUTABLE} getwheelname.py OUTPUT_VARIABLE PY_WHEELFILE
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      )
message(STATUS "Detected Python wheel filename ${PY_WHEELFILE}")

if(MANIFOLD_PYBIND_STUBGEN)
  add_custom_target(python-wheel ALL
        COMMAND ${Python_EXECUTABLE} -m pip wheel . --no-deps
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Building python wheel package"
        DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/manifold3d.pyi"
        )
else()
  add_custom_target(python-wheel ALL
        COMMAND ${Python_EXECUTABLE} -m pip wheel . --no-deps
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Building python wheel package"
        DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_FILE_NAME}"
        )
endif()

install(CODE
         "execute_process(
             COMMAND ${Python_EXECUTABLE} -m pip install --target ${MANIFOLD_PYBIND_LIBDIR} ${PY_WHEELFILE}
                          )"
          )
