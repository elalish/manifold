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

# For private headers
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

set(MANIFOLD_SRCS
	boolean3.cpp
	boolean_result.cpp
	constructors.cpp
	csg_tree.cpp
	edge_op.cpp
	face_op.cpp
	impl.cpp
	manifold.cpp
	properties.cpp
	quickhull.cpp
	sdf.cpp
	smoothing.cpp
	sort.cpp
	subdivision.cpp
	)
set(MANIFOLD_PRIVATE_HDRS
	boolean3.h
	collider.h
	csg_tree.h
	hashtable.h
	impl.h
	mesh_fixes.h
	quickhull.h
	shared.h
	sparse.h
	svd.h
	utils.h
	vec.h
	)
if(MANIFOLD_CROSS_SECTION)
	list(APPEND MANIFOLD_SRCS cross_section.cpp)
	list(APPEND MANIFOLD_PRIVATE_HDRS ../include/manifold/cross_section.h)
endif()
if(MANIFOLD_EXPORT)
	list(APPEND MANIFOLD_SRCS meshIO.cpp)
	list(APPEND MANIFOLD_PRIVATE_HDRS ../include/manifold/meshIO.h)
endif()
if(MANIFOLD_POLYGON)
	list(APPEND MANIFOLD_SRCS polygon.cpp)
	list(APPEND MANIFOLD_PRIVATE_HDRS ../include/manifold/polygon.h)
endif()

# Dependency libraries
set(MANIFOLD_LIBS glm::glm)
if(MANIFOLD_CROSS_SECTION)
	if(TARGET Clipper2::Clipper2)
		list(APPEND MANIFOLD_LIBS Clipper2::Clipper2)
	else()
		list(APPEND MANIFOLD_LIBS Clipper2)
	endif()
endif()
if(MANIFOLD_PAR STREQUAL "TBB")
	if (NOT APPLE)
		if(TARGET TBB::tbb)
			list(APPEND MANIFOLD_LIBS TBB::tbb)
		else()
			list(APPEND MANIFOLD_LIBS ${TBB_LINK_LIBRARIES})
		endif()
	else()
		if(oneDPL_FOUND)
			list(APPEND MANIFOLD_LIBS oneDPL)
		else()
			message(WARNING "oneDPL not found, sequential implementation is used instead")
		endif()
	endif()
endif()

#Include directories
set(MANIFOLD_INCLUDE_DIRS
	${CMAKE_SOURCE_DIR}/include
	)
if(TBB_INCLUDE_DIRS)
	list(APPEND MANIFOLD_INCLUDE_DIRS ${TBB_INCLUDE_DIRS})
endif()

add_library(manifold ${MANIFOLD_SRCS} ${MANIFOLD_PRIVATE_HDRS})
set_property(TARGET manifold PROPERTY VERSION "${MANIFOLD_VERSION}")
set_property(TARGET manifold PROPERTY SOVERSION ${MANIFOLD_VERSION_MAJOR})
target_link_libraries(manifold PRIVATE ${MANIFOLD_LIBS})
target_compile_options(manifold PRIVATE ${MANIFOLD_FLAGS})
target_include_directories(manifold PRIVATE ${MANIFOLD_INCLUDE_DIRS})
if(MANIFOLD_EXCEPTIONS)
	target_compile_options(manifold PRIVATE -DMANIFOLD_EXCEPTIONS=1)
endif()
if(MANIFOLD_DEBUG)
	target_compile_options(manifold PRIVATE -DMANIFOLD_DEBUG)
endif()
if(MANIFOLD_EXPORT)
	target_link_libraries(meshIO PRIVATE assimp::assimp)
endif()
target_compile_features(manifold PUBLIC cxx_std_17)
if(MANIFOLD_PAR STREQUAL "TBB")
	target_compile_options(manifold INTERFACE -DMANIFOLD_PAR='T')
endif()

install(TARGETS manifold EXPORT manifoldTargets)

# Tracy Support
if (MANIFOLD_DOWNLOADS)
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
		target_link_libraries(manifold INTERFACE TracyClient)
	endif()
endif()

