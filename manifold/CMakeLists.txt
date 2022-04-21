project (manifold)

add_library(${PROJECT_NAME} src/manifold.cu src/constructors.cu src/impl.cu src/properties.cu src/sort.cu src/edge_op.cu src/face_op.cu src/smoothing.cu src/boolean3.cu src/boolean_result.cu)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 61)

target_include_directories( ${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include )
target_link_libraries( ${PROJECT_NAME}
    PUBLIC utilities
    PRIVATE collider polygon ${MANIFOLD_OMP_INCLUDE} graphlite
)

target_compile_options(${PROJECT_NAME} 
    PRIVATE ${MANIFOLD_NVCC_FLAGS}
)
target_compile_options(${PROJECT_NAME} 
    PRIVATE "$<$<CONFIG:RELEASE>:${MANIFOLD_NVCC_RELEASE_FLAGS}>" "$<$<CONFIG:DEBUG>:${MANIFOLD_NVCC_DEBUG_FLAGS}>"
)
target_compile_features(${PROJECT_NAME} 
    PUBLIC cxx_std_14 
    PRIVATE cxx_std_17
)

