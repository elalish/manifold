project (mesh_host)

add_library(${PROJECT_NAME} src/mesh_host.cpp)

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries( ${PROJECT_NAME}
    PUBLIC utilities
    PRIVATE assimp
)

target_compile_options(${PROJECT_NAME} PRIVATE -Werror -Wall -Wno-sign-compare)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
