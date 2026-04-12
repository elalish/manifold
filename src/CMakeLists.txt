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
  minkowski.cpp
  polygon.cpp
  properties.cpp
  quickhull.cpp
  sdf.cpp
  smoothing.cpp
  sort.cpp
  subdivision.cpp
  tree2d.cpp
  # optional source files
  $<$<BOOL:${MANIFOLD_CROSS_SECTION}>:cross_section/cross_section.cpp>
)

set(
  MANIFOLD_PRIVATE_HDRS
  boolean3.h
  collider.h
  csg_tree.h
  hashtable.h
  impl.h
  iters.h
  mesh_fixes.h
  parallel.h
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

if(MANIFOLD_METAL AND APPLE)
  list(APPEND MANIFOLD_SRCS metal_backend.mm)
  list(APPEND MANIFOLD_PRIVATE_HDRS metal_backend.h)
endif()

add_library(manifold ${MANIFOLD_SRCS} ${MANIFOLD_PRIVATE_HDRS})

target_link_libraries(manifold PUBLIC $<$<BOOL:${TRACY_ENABLE}>:TracyClient>)

target_link_libraries(
  manifold
  PRIVATE
    # optional dependencies
    $<$<BOOL:${MANIFOLD_PAR}>:TBB::tbb>
    $<$<BOOL:${MANIFOLD_CROSS_SECTION}>:Clipper2::Clipper2>
)

# Metal GPU compute backend (macOS / Apple Silicon)
if(MANIFOLD_METAL AND APPLE)
  find_library(METAL_FRAMEWORK Metal)
  find_library(FOUNDATION_FRAMEWORK Foundation)
  if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
    target_link_libraries(manifold PRIVATE ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
    target_compile_options(manifold PUBLIC -DMANIFOLD_METAL)

    # Compile Metal shaders to metallib (if Metal compiler is available)
    set(METAL_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/metal_kernels.metal)
    set(METAL_AIR_FILE ${CMAKE_CURRENT_BINARY_DIR}/metal_kernels.air)
    set(METAL_LIB_FILE ${CMAKE_CURRENT_BINARY_DIR}/metal_kernels.metallib)

    # Check if xcrun metal compiler is actually usable
    execute_process(
      COMMAND xcrun -sdk macosx metal --version
      RESULT_VARIABLE METAL_COMPILER_RESULT
      OUTPUT_QUIET ERROR_QUIET
    )

    if(METAL_COMPILER_RESULT EQUAL 0)
      add_custom_command(
        OUTPUT ${METAL_AIR_FILE}
        COMMAND xcrun -sdk macosx metal -c ${METAL_SHADER_SOURCE} -o ${METAL_AIR_FILE}
        DEPENDS ${METAL_SHADER_SOURCE}
        COMMENT "Compiling Metal shaders"
      )
      add_custom_command(
        OUTPUT ${METAL_LIB_FILE}
        COMMAND xcrun -sdk macosx metallib ${METAL_AIR_FILE} -o ${METAL_LIB_FILE}
        DEPENDS ${METAL_AIR_FILE}
        COMMENT "Linking Metal shader library"
      )
      add_custom_target(metal_shaders ALL DEPENDS ${METAL_LIB_FILE})
      add_dependencies(manifold metal_shaders)

      # Install the metallib alongside the binary
      install(FILES ${METAL_LIB_FILE} DESTINATION ${CMAKE_INSTALL_BINDIR})

      # Also copy to the build output directory so it's found at runtime
      add_custom_command(TARGET metal_shaders POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${METAL_LIB_FILE}
                ${CMAKE_BINARY_DIR}/metal_kernels.metallib
        COMMENT "Copying metallib to build directory"
      )
      message(STATUS "Metal shaders: pre-compiled (metallib)")
    else()
      # No Metal compiler — shaders will be compiled at runtime from source
      # Copy the .metal source to the build directory for runtime compilation
      configure_file(${METAL_SHADER_SOURCE}
                     ${CMAKE_BINARY_DIR}/metal_kernels.metal COPYONLY)
      message(STATUS "Metal shaders: runtime compilation (Metal toolchain not found, run: xcodebuild -downloadComponent MetalToolchain)")
    endif()

    # Enable Obj-C++ compilation for .mm files
    set_source_files_properties(metal_backend.mm PROPERTIES
      COMPILE_FLAGS "-fobjc-arc"
    )

    message(STATUS "Metal GPU compute: enabled")
  else()
    message(STATUS "Metal GPU compute: Metal framework not found, disabled")
    set(MANIFOLD_METAL OFF PARENT_SCOPE)
  endif()
endif()

target_compile_options(manifold PRIVATE ${MANIFOLD_FLAGS})
# make sure users use appropriate c++ standard when including our headers
target_compile_features(manifold PUBLIC cxx_std_17)
set(
  OPTIONS
  MANIFOLD_DEBUG
  MANIFOLD_ASSERT
  MANIFOLD_CROSS_SECTION
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
