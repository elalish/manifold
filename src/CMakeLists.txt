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

add_library(utilities INTERFACE)
message("Parallel Backend: ${MANIFOLD_PAR}")

include(FetchContent)
if (TRACY_ENABLE)
    include(FetchContent)
    FetchContent_Declare(tracy
        GIT_REPOSITORY https://github.com/wolfpld/tracy.git
        GIT_TAG v0.10
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(tracy)
    target_link_libraries(utilities INTERFACE TracyClient)
endif()

if(MANIFOLD_PAR STREQUAL "TBB")
    target_compile_options(utilities INTERFACE -DMANIFOLD_PAR='T')
    if(TARGET TBB::tbb)
        target_link_libraries(utilities INTERFACE TBB::tbb)
    else()
        target_include_directories(utilities INTERFACE $<BUILD_INTERFACE:${TBB_INCLUDE_DIRS}>)
        target_link_libraries(utilities INTERFACE ${TBB_LINK_LIBRARIES})
    endif()
    if(APPLE)
        if(oneDPL_FOUND)
            target_link_libraries(utilities INTERFACE oneDPL)
        else()
            message(WARNING "oneDPL not found, sequential implementation is used instead")
        endif()
    endif()
elseif(MANIFOLD_PAR STREQUAL "NONE")
    set(MANIFOLD_PAR "CPP")
else()
    message(FATAL_ERROR "Invalid value for MANIFOLD_PAR: ${MANIFOLD_PAR}. "
        "Should be \"TBB\" or \"NONE\"")
endif()

target_include_directories(utilities INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(utilities INTERFACE glm::glm)

if(MANIFOLD_EXCEPTIONS)
    target_compile_options(utilities INTERFACE
        -DMANIFOLD_EXCEPTIONS=1
    )
endif()

if(MANIFOLD_DEBUG)
    target_compile_options(utilities
        INTERFACE -DMANIFOLD_DEBUG)
endif()

target_compile_features(utilities INTERFACE cxx_std_17)

install(TARGETS utilities EXPORT manifoldTargets)

add_library(collider INTERFACE)
target_include_directories(collider INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(collider INTERFACE utilities)
target_compile_options(collider INTERFACE ${MANIFOLD_FLAGS})
target_compile_features(collider INTERFACE cxx_std_17)


if(MANIFOLD_CROSS_SECTION)
	file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS *.cpp)
	add_library(cross_section ${SOURCE_FILES})
	target_include_directories(cross_section PUBLIC
		$<INSTALL_INTERFACE:include>
		$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

	if(TARGET Clipper2::Clipper2)
		target_link_libraries(cross_section
			PUBLIC utilities
			PRIVATE Clipper2::Clipper2)
	else()
		target_link_libraries(cross_section
			PUBLIC utilities
			PRIVATE Clipper2)
	endif()

	target_compile_options(cross_section PRIVATE ${MANIFOLD_FLAGS})
	target_compile_features(cross_section PUBLIC cxx_std_17)

	install(TARGETS cross_section EXPORT manifoldTargets)
endif()


if (MANIFOLD_POLYGON)
	file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS *.cpp)
	add_library(polygon ${SOURCE_FILES})
	target_include_directories(polygon PUBLIC
		$<INSTALL_INTERFACE:include>
		$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
	target_link_libraries(polygon 
		PUBLIC utilities 
		PRIVATE $<BUILD_INTERFACE:collider>
		)

	target_compile_options(polygon PRIVATE ${MANIFOLD_FLAGS})
	target_compile_features(polygon PUBLIC cxx_std_17)

	install(TARGETS polygon EXPORT manifoldTargets)
endif()


file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS *.cpp)
add_library(manifold ${SOURCE_FILES})

target_include_directories(manifold PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(manifold
    PUBLIC utilities
    PRIVATE $<BUILD_INTERFACE:collider> polygon ${MANIFOLD_INCLUDE}
)

target_compile_options(manifold PRIVATE ${MANIFOLD_FLAGS})

target_compile_features(manifold
    PUBLIC cxx_std_17
)

install(TARGETS manifold EXPORT manifoldTargets)

if (MANIFOLD_EXPORT)
	add_library(meshIO src/meshIO.cpp)

	target_include_directories(meshIO PUBLIC
		$<INSTALL_INTERFACE:include>
		$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

	target_link_libraries(meshIO
		PUBLIC manifold
		PRIVATE assimp::assimp
		)

	target_compile_options(meshIO PRIVATE ${MANIFOLD_FLAGS})
	target_compile_features(meshIO PUBLIC cxx_std_17)

	install(TARGETS meshIO EXPORT manifoldTargets)
endif()
