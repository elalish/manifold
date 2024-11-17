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

set(
  MANIFOLD_SRCS
  boolean3.cpp
  boolean_result.cpp
  constructors.cpp
  csg_tree.cpp
  edge_op.cpp
  face_op.cpp
  impl.cpp
  manifold.cpp
  polygon.cpp
  properties.cpp
  quickhull.cpp
  sdf.cpp
  smoothing.cpp
  sort.cpp
  subdivision.cpp
  # optional source files
  $<$<BOOL:${MANIFOLD_CROSS_SECTION}>:cross_section/cross_section.cpp>
  $<$<BOOL:${MANIFOLD_EXPORT}>:meshIO/meshIO.cpp>
)

set(
  MANIFOLD_PRIVATE_HDRS
  boolean3.h
  collider.h
  csg_tree.h
  hashtable.h
  impl.h
  mesh_fixes.h
  parallel.h
  quickhull.h
  shared.h
  sparse.h
  svd.h
  tri_dist.h
  utils.h
  vec.h
)

# Include directories
set(MANIFOLD_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)

add_library(manifold ${MANIFOLD_SRCS} ${MANIFOLD_PRIVATE_HDRS})

target_link_libraries(manifold PUBLIC $<$<BOOL:${TRACY_ENABLE}>:TracyClient>)

target_link_libraries(
  manifold
  PRIVATE
    # optional dependencies
    $<$<BOOL:${MANIFOLD_PAR}>:TBB::tbb>
    $<$<BOOL:${MANIFOLD_CROSS_SECTION}>:Clipper2::Clipper2>
    $<$<BOOL:${MANIFOLD_EXPORT}>:assimp::assimp>
)

target_compile_options(manifold PRIVATE ${MANIFOLD_FLAGS})
# make sure users use appropriate c++ standard when including our headers
target_compile_features(manifold PUBLIC cxx_std_17)
set(
  OPTIONS
  MANIFOLD_DEBUG
  MANIFOLD_CROSS_SECTION
  MANIFOLD_EXPORT
  TRACY_ENABLE
  TRACY_MEMORY_USAGE
)
foreach(OPT IN LISTS OPTIONS)
  if(${${OPT}})
    target_compile_options(manifold PUBLIC -D${OPT})
  endif()
endforeach()
if(MANIFOLD_PAR)
  target_compile_options(manifold PUBLIC -DMANIFOLD_PAR=1)
else()
  target_compile_options(manifold PUBLIC -DMANIFOLD_PAR=-1)
endif()

target_include_directories(
  manifold
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  PRIVATE ${MANIFOLD_INCLUDE_DIRS}
)

exportlib(manifold)

install(TARGETS manifold EXPORT manifoldTargets)
