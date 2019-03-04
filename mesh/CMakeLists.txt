project (mesh)

add_library(${PROJECT_NAME} src/mesh.cu)

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries( ${PROJECT_NAME}
    PUBLIC utilities mesh_host
    PRIVATE collider polygon OpenMP::OpenMP_CXX
)

target_compile_options(${PROJECT_NAME} 
    PRIVATE -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -Xcompiler -fopenmp#-Xcompiler=-Werror
)
target_compile_features(${PROJECT_NAME} 
    PUBLIC cxx_std_11 
    PRIVATE cxx_std_14
)

if(APPLE)
#   All CUDA libraries need this. If you ever get 
#   "CUDA driver version is insufficient for CUDA runtime version", 
#   this is probably what's missing.
  set_property(TARGET ${PROJECT_NAME}
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
