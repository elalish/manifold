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

set(MANIFOLD_CROSS_SECTION_SRCS)
if(MANIFOLD_CROSS_SECTION)
  if(MANIFOLD_CROSS_SECTION_BACKEND STREQUAL "clipper2")
    list(APPEND MANIFOLD_CROSS_SECTION_SRCS cross_section/cross_section.cpp)
  elseif(MANIFOLD_CROSS_SECTION_BACKEND STREQUAL "boolean2")
    list(
      APPEND
      MANIFOLD_CROSS_SECTION_SRCS
      cross_section/cross_section_boolean2.cpp
      cross_section/boolean2/boolean2.cpp
      cross_section/boolean2/bvh.cpp
      cross_section/boolean2/canonicalize.cpp
      cross_section/boolean2/containment.cpp
      cross_section/boolean2/diagnostics.cpp
      cross_section/boolean2/driver.cpp
      cross_section/boolean2/edge_vert_lists.cpp
      cross_section/boolean2/intersections.cpp
      cross_section/boolean2/offset.cpp
      cross_section/boolean2/predicates.cpp
      cross_section/boolean2/vertex_merge.cpp
      cross_section/boolean2/winding_filter.cpp
    )
  endif()
endif()

set(MANIFOLD_CROSS_SECTION_LIBS)
if(MANIFOLD_CROSS_SECTION AND MANIFOLD_CROSS_SECTION_BACKEND STREQUAL "clipper2")
  list(APPEND MANIFOLD_CROSS_SECTION_LIBS Clipper2::Clipper2)
endif()

set(
  MANIFOLD_SRCS
  boolean3.cpp
  boolean_result.cpp
  constructors.cpp
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
  ${MANIFOLD_CROSS_SECTION_SRCS}
)

set(
  MANIFOLD_PRIVATE_HDRS
  boolean3.h
  collider.h
  cross_section/boolean2/boolean2.h
  cross_section/boolean2/bvh.h
  cross_section/boolean2/canonicalize.h
  cross_section/boolean2/diagnostics.h
  cross_section/boolean2/driver.h
  cross_section/boolean2/edge_vert_lists.h
  cross_section/boolean2/intersections.h
  cross_section/boolean2/parallel_policy.h
  cross_section/boolean2/predicates.h
  cross_section/boolean2/vertex_merge.h
  cross_section/boolean2/winding_filter.h
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
    ${MANIFOLD_CROSS_SECTION_LIBS}
)

target_compile_options(manifold PRIVATE ${MANIFOLD_FLAGS})
# make sure users use appropriate c++ standard when including our headers
target_compile_features(manifold PUBLIC cxx_std_17)
set(
  OPTIONS
  MANIFOLD_DEBUG
  MANIFOLD_ASSERT
  MANIFOLD_TIMING
  MANIFOLD_CROSS_SECTION
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
