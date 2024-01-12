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

project(meshIO)

find_package(assimp REQUIRED)

add_library(${PROJECT_NAME} src/meshIO.cpp)

target_sources(${PROJECT_NAME} PUBLIC
    FILE_SET HEADERS
    BASE_DIRS include
    FILES include/meshIO.h)

target_link_libraries(${PROJECT_NAME}
    PUBLIC manifold
    PRIVATE assimp::assimp
)

target_compile_options(${PROJECT_NAME} PRIVATE ${MANIFOLD_FLAGS})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

if(CMAKE_VERSION VERSION_LESS "3.23.0")
    install(FILES include/meshIO.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/manifold)
else()
    install(TARGETS ${PROJECT_NAME}
        EXPORT manifoldTargets FILE_SET HEADERS
        DESTINATION include/manifold)
endif()
