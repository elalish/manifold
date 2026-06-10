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
  boolean2.cpp
  boolean2_diagnostics.cpp
  boolean2_offset.cpp
  boolean2_predicates.cpp
  boolean2_winding.cpp
  boolean3.cpp
  boolean_result.cpp
  constructors.cpp
  cross_section.cpp
  cross_section/boolean2/bvh.cpp
  cross_section/boolean2/canonicalize.cpp
  cross_section/boolean2/containment.cpp
  cross_section/boolean2/driver.cpp
  cross_section/boolean2/edge_vert_lists.cpp
  cross_section/boolean2/intersections.cpp
  cross_section/boolean2/vertex_merge.cpp
  csg_tree.cpp
  edge_op.cpp
  execution_impl.cpp
  face_op.cpp
  impl.cpp
  manifold.cpp
  minkowski.cpp
  polygon.cpp
  properties.cpp
  quickhull.cpp
  sdf.cpp
  smoothing.cpp
  sort.cpp
  subdivision.cpp
  tree2d.cpp
)

set(
  MANIFOLD_PRIVATE_HDRS
  boolean2.h
  boolean2_diagnostics.h
  boolean3.h
  collider.h
  csg_tree.h
  execution_impl.h
  hashtable.h
  impl.h
  iters.h
  mesh_fixes.h
  parallel.h
  polygon_internal.h
  quickhull.h
  shared.h
  svd.h
  tree2d.h
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
)

target_compile_options(manifold PRIVATE ${MANIFOLD_FLAGS})
# make sure users use appropriate c++ standard when including our headers
target_compile_features(manifold PUBLIC cxx_std_17)
set(
  OPTIONS
  MANIFOLD_DEBUG
  MANIFOLD_ASSERT
  MANIFOLD_TIMING
  MANIFOLD_NO_IOSTREAM
  TRACY_ENABLE
  TRACY_MEMORY_USAGE
)
foreach(OPT IN LISTS OPTIONS)
  if(${${OPT}})
    target_compile_definitions(manifold PUBLIC ${OPT})
  endif()
endforeach()
if(MANIFOLD_PAR)
  target_compile_definitions(manifold PUBLIC MANIFOLD_PAR=1)
else()
  target_compile_definitions(manifold PUBLIC MANIFOLD_PAR=-1)
endif()

# MANIFOLD_NO_IOSTREAM also defines MANIFOLD_NO_FILESYSTEM (the two
# track together in practice — `<filesystem>` paths are only useful
# if you can `<fstream>` them, which needs iostream). Source code
# uses the macros separately for semantic clarity (NO_IOSTREAM gates
# the iostream-using public API; NO_FILESYSTEM gates filesystem use
# in test_main.cpp's fixture helpers + main()).
if(MANIFOLD_NO_IOSTREAM)
  target_compile_definitions(manifold PUBLIC MANIFOLD_NO_FILESYSTEM)
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
