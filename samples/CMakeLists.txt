project (samples)

add_library(${PROJECT_NAME} src/knot.cpp)

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries( ${PROJECT_NAME}
    PUBLIC manifold
)

target_compile_options(${PROJECT_NAME} PRIVATE ${MANIFOLD_FLAGS})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
