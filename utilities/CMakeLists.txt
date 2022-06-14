project (utilities)

add_library(${PROJECT_NAME} src/detect_cuda.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${THRUST_INC_DIR} ${GLM_INC_DIR})
if(MANIFOLD_USE_CUDA)
    set_source_files_properties(src/detect_cuda.cpp PROPERTIES LANGUAGE CUDA)
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 61)
    target_compile_options(${PROJECT_NAME} 
        PRIVATE ${MANIFOLD_NVCC_FLAGS}
    )
    target_compile_options(${PROJECT_NAME} 
        PRIVATE "$<$<CONFIG:RELEASE>:${MANIFOLD_NVCC_RELEASE_FLAGS}>" "$<$<CONFIG:DEBUG>:${MANIFOLD_NVCC_DEBUG_FLAGS}>"
    )
else()
    target_compile_options(${PROJECT_NAME} 
        PRIVATE ${MANIFOLD_FLAGS}
    )
endif()
target_include_directories(${PROJECT_NAME} INTERFACE ${PROJECT_SOURCE_DIR}/include)
