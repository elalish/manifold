project (manifold_test)

enable_testing()

add_executable(${PROJECT_NAME} polygon_test.cpp mesh_test.cpp)
target_link_libraries(${PROJECT_NAME} polygon mesh GTest::GTest)

target_compile_options(${PROJECT_NAME} PRIVATE -Werror -Wall -Wno-sign-compare)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

add_test(test_all ${PROJECT_NAME})

file(COPY data DESTINATION . FILES_MATCHING PATTERN "*.ply")