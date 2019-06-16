project (tools)

add_executable(loadMesh load_mesh.cpp)
target_link_libraries(loadMesh manifold)

target_compile_options(loadMesh PRIVATE ${MANIFOLD_FLAGS})
target_compile_features(loadMesh PUBLIC cxx_std_11)

if(APPLE)
#   All executables that link to CUDA need this. If you ever get 
#   "CUDA driver version is insufficient for CUDA runtime version", 
#   this is probably what's missing.
  set_property(TARGET loadMesh
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()


add_executable(perfTest perf_test.cpp)
target_link_libraries(perfTest manifold)

target_compile_options(perfTest PRIVATE ${MANIFOLD_FLAGS})
target_compile_features(perfTest PUBLIC cxx_std_11)

if(APPLE)
#   All executables that link to CUDA need this. If you ever get 
#   "CUDA driver version is insufficient for CUDA runtime version", 
#   this is probably what's missing.
  set_property(TARGET perfTest
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()