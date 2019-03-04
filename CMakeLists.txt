cmake_minimum_required(VERSION 3.12)
project(manifold VERSION 0.1 LANGUAGES CXX CUDA)

# set(CMAKE_VERBOSE_MAKEFILE ON)
# OPTION(DEBUG_CMAKE_TARGETS "enable debug output for cmake target properties" ON)
# if(DEBUG_CMAKE_TARGETS)
#     set(CMAKE_DEBUG_TARGET_PROPERTIES
#             INCLUDE_DIRECTORIES
#         )
# endif()

find_package(OpenMP REQUIRED)

add_subdirectory(utilities)
add_subdirectory(polygon)
add_subdirectory(collider)
add_subdirectory(mesh)
add_subdirectory(third_party)
add_subdirectory(mesh_host)
add_subdirectory(test)
add_subdirectory(tools)