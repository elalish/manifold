project (collider)

add_library(${PROJECT_NAME} src/collider.cu)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 61)

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries( ${PROJECT_NAME}
    PUBLIC utilities
    PRIVATE ${MANIFOLD_OMP_INCLUDE}
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

