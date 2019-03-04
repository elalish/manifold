cmake_minimum_required(VERSION 3.12)
# cmake_policy(SET CMP0042 NEW)
# cmake_policy(SET CMP0048 NEW)
project(manifold VERSION 0.1 LANGUAGES CXX CUDA)

set(CMAKE_VERBOSE_MAKEFILE ON)
OPTION(DEBUG_CMAKE_TARGETS "enable debug output for cmake target properties" ON)
if(DEBUG_CMAKE_TARGETS)
    set(CMAKE_DEBUG_TARGET_PROPERTIES
            INCLUDE_DIRECTORIES
        )
endif()

add_subdirectory(third_party)
add_subdirectory(utilities)
add_subdirectory(polygon)
add_subdirectory(collider)
add_subdirectory(mesh)
add_subdirectory(mesh_host)
add_subdirectory(test)
add_subdirectory(tools)