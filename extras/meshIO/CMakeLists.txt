project (meshIO)

add_library(${PROJECT_NAME} src/meshIO.cpp)

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include ${ASSIMP_INC_DIR}
)

target_link_libraries( ${PROJECT_NAME}
    PUBLIC utilities
    PRIVATE assimp
)

target_compile_options(${PROJECT_NAME} PRIVATE ${MANIFOLD_FLAGS})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)
