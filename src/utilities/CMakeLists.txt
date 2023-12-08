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

project(utilities)

add_library(${PROJECT_NAME} INTERFACE)
message("Parallel Backend: ${MANIFOLD_PAR}")

include(FetchContent)
FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 0.9.9.8
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
    FIND_PACKAGE_ARGS NAMES glm
)
FetchContent_MakeAvailable(glm)

FetchContent_Declare(Thrust
    GIT_REPOSITORY https://github.com/NVIDIA/thrust.git
    GIT_TAG 2.1.0
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
find_package(Thrust QUIET)
if(NOT Thrust_FOUND AND NOT DEFINED thrust_SOURCE_DIR)
    FetchContent_Populate(Thrust)
endif()

if (TRACY_ENABLE)
    include(FetchContent)
    FetchContent_Declare(tracy
        GIT_REPOSITORY https://github.com/wolfpld/tracy.git
        GIT_TAG v0.10
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(tracy)
    target_link_libraries(${PROJECT_NAME} INTERFACE TracyClient)
endif()

if(MANIFOLD_PAR STREQUAL "TBB")
    find_package(PkgConfig)
    if (PKG_CONFIG_FOUND)
        pkg_check_modules(TBB tbb)
    endif()
    if(NOT TBB_FOUND)
        message(STATUS "tbb not found, downloading from source")
        include(FetchContent)
        set(TBB_TEST OFF CACHE INTERNAL "" FORCE)
        set(TBB_STRICT OFF CACHE INTERNAL "" FORCE)
        FetchContent_Declare(TBB
            GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
            GIT_TAG        v2021.11.0
            GIT_SHALLOW    TRUE
            GIT_PROGRESS   TRUE
        )
        FetchContent_MakeAvailable(TBB)
        set_property(DIRECTORY ${tbb_SOURCE_DIR} PROPERTY EXCLUDE_FROM_ALL YES)
    endif()
    target_compile_options(${PROJECT_NAME} INTERFACE -DMANIFOLD_PAR='T')
    if(TARGET TBB::tbb)
        target_link_libraries(${PROJECT_NAME} INTERFACE TBB::tbb)
    else()
        target_include_directories(${PROJECT_NAME} INTERFACE ${TBB_INCLUDE_DIRS})
        target_link_libraries(${PROJECT_NAME} INTERFACE ${TBB_LINK_LIBRARIES})
    endif()
elseif(MANIFOLD_PAR STREQUAL "NONE")
    set(MANIFOLD_PAR "CPP")
else()
    message(FATAL_ERROR "Invalid value for MANIFOLD_PAR: ${MANIFOLD_PAR}. "
        "Should be \"TBB\" or \"NONE\"")
endif()

target_include_directories(${PROJECT_NAME} INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} INTERFACE glm::glm)
if(DEFINED thrust_SOURCE_DIR)
target_include_directories(${PROJECT_NAME} INTERFACE
    ${thrust_SOURCE_DIR}
    ${thrust_SOURCE_DIR}/dependencies/libcudacxx/include
)
else()
target_include_directories(${PROJECT_NAME} INTERFACE
    ${_THRUST_INCLUDE_DIR}
    ${_THRUST_INCLUDE_DIR}/dependencies/libcudacxx/include
)
endif()

target_compile_options(${PROJECT_NAME} INTERFACE
    -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_${MANIFOLD_PAR}
)

if(MANIFOLD_DEBUG)
    target_compile_options(${PROJECT_NAME}
        INTERFACE -DMANIFOLD_DEBUG)
endif()

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

install(
  FILES include/public.h
  DESTINATION ${CMAKE_INSTALL_FULL_INCLUDEDIR}/manifold
  COMPONENT devel
)

