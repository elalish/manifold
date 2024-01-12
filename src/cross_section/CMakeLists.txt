# Copyright 2023 The Manifold Authors.
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

project (cross_section)

set(CLIPPER2_UTILS OFF)
set(CLIPPER2_EXAMPLES OFF)
set(CLIPPER2_TESTS OFF)
set(CLIPPER2_USINGZ "OFF" CACHE STRING "Preempt cache default of USINGZ (we only use 2d)")

find_package(PkgConfig)
if (PKG_CONFIG_FOUND)
    pkg_check_modules(Clipper2 Clipper2)
endif()
message("clipper2_FOUND: ${Clipper2_FOUND}")
if(Clipper2_FOUND)
    add_library(Clipper2 SHARED IMPORTED)
    set_property(TARGET Clipper2 PROPERTY
        IMPORTED_LOCATION ${Clipper2_LINK_LIBRARIES})
    target_include_directories(Clipper2 INTERFACE ${Clipper2_INCLUDE_DIRS})
else()
    include(FetchContent)
    FetchContent_Declare(Clipper2
        GIT_REPOSITORY https://github.com/AngusJohnson/Clipper2.git
        GIT_TAG Clipper2_1.3.0
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
        SOURCE_SUBDIR CPP
    )
    FetchContent_MakeAvailable(Clipper2)
endif()

file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS *.cpp)
add_library(${PROJECT_NAME} OBJECT ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(${PROJECT_NAME}
    PUBLIC utilities
    PRIVATE Clipper2)

target_compile_options(${PROJECT_NAME} PRIVATE ${MANIFOLD_FLAGS})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

install(TARGETS ${PROJECT_NAME} EXPORT manifoldTargets)
install(FILES include/cross_section.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/manifold)
