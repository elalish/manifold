project (utilities)

add_library(${PROJECT_NAME} src/detect_cuda.cpp)

target_include_directories(${PROJECT_NAME} INTERFACE ${THRUST_INC_DIR} ${GLM_INC_DIR})
target_include_directories(${PROJECT_NAME}
    INTERFACE
        ${PROJECT_SOURCE_DIR}/include
)
target_compile_options(${PROJECT_NAME} 
    PRIVATE ${MANIFOLD_FLAGS}
)
