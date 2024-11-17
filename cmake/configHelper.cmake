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

function(exportbin TARGET)
  if(MSVC)
    set_target_properties(
      ${TARGET}
      PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
    )
  endif()
endfunction()

function(exportlib TARGET)
  add_library(manifold::${TARGET} ALIAS ${TARGET})
  set_target_properties(
    ${TARGET}
    PROPERTIES VERSION "${MANIFOLD_VERSION}" SOVERSION ${MANIFOLD_VERSION_MAJOR}
  )
  if(MSVC)
    set_target_properties(
      ${TARGET}
      PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
        ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
    )
  endif()
endfunction()
