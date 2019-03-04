project (tools)

add_executable(${PROJECT_NAME} load_mesh.cpp)
target_link_libraries(${PROJECT_NAME} mesh)

target_compile_options(${PROJECT_NAME} PRIVATE -Werror -Wall -Wno-sign-compare)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
