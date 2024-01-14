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
    target_compile_options(${PROJECT_NAME} INTERFACE -DMANIFOLD_PAR='T')
    if(TARGET TBB::tbb)
        target_link_libraries(${PROJECT_NAME} INTERFACE TBB::tbb)
    else()
        target_include_directories(${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${TBB_INCLUDE_DIRS}>)
        target_link_libraries(${PROJECT_NAME} INTERFACE ${TBB_LINK_LIBRARIES})
    endif()
elseif(MANIFOLD_PAR STREQUAL "NONE")
    set(MANIFOLD_PAR "CPP")
else()
    message(FATAL_ERROR "Invalid value for MANIFOLD_PAR: ${MANIFOLD_PAR}. "
        "Should be \"TBB\" or \"NONE\"")
endif()

target_include_directories(${PROJECT_NAME} INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(${PROJECT_NAME} INTERFACE glm::glm)

if(NOT DEFINED thrust_SOURCE_DIR)
    set(thrust_SOURCE_DIR ${_THRUST_INCLUDE_DIR})
endif()

target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${thrust_SOURCE_DIR}>
    $<BUILD_INTERFACE:${thrust_SOURCE_DIR}/dependencies/libcudacxx/include>
)

target_compile_options(${PROJECT_NAME} INTERFACE
    -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_${MANIFOLD_PAR}
)

if(MANIFOLD_DEBUG)
    target_compile_options(${PROJECT_NAME}
        INTERFACE -DMANIFOLD_DEBUG)
endif()

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

install(TARGETS ${PROJECT_NAME} EXPORT manifoldTargets)
install(FILES include/public.h include/vec_view.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
