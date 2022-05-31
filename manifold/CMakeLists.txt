project (manifold)

add_library(${PROJECT_NAME} src/manifold.cpp src/constructors.cpp src/impl.cpp
    src/properties.cpp src/sort.cpp src/edge_op.cpp src/face_op.cpp
    src/smoothing.cpp src/boolean3.cpp src/boolean_result.cpp
)

if(THRUST_BACKEND STREQUAL "CUDA")
    set_source_files_properties(
        src/manifold.cpp src/constructors.cpp src/impl.cpp
        src/properties.cpp src/sort.cpp src/edge_op.cpp src/face_op.cpp
        src/smoothing.cpp src/boolean3.cpp src/boolean_result.cpp
        src/manifold_set.cpp

        PROPERTIES LANGUAGE CUDA
    )
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

target_include_directories( ${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include )
target_link_libraries( ${PROJECT_NAME}
    PUBLIC utilities
    PRIVATE collider polygon ${MANIFOLD_INCLUDE} graphlite
)

target_compile_features(${PROJECT_NAME} 
    PUBLIC cxx_std_14 
    PRIVATE cxx_std_17
)

