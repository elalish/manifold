project (manifold_test)

enable_testing()

add_executable(${PROJECT_NAME} test_main.cpp polygon_test.cpp mesh_test.cpp samples_test.cpp)
target_link_libraries(${PROJECT_NAME} polygon GTest::GTest manifold meshIO samples)

target_compile_options(${PROJECT_NAME} PRIVATE ${MANIFOLD_FLAGS})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

add_test(test_all ${PROJECT_NAME})

file(COPY data DESTINATION . FILES_MATCHING PATTERN "*.ply")
if(EMSCRIPTEN)
    set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS 
        "-s ASSERTIONS=1 -s DEMANGLE_SUPPORT=1 --preload-file data --bind")
else()
endif()

if(APPLE)
#   All executables that link to CUDA need this. If you ever get 
#   "CUDA driver version is insufficient for CUDA runtime version", 
#   this is probably what's missing.
  set_property(TARGET ${PROJECT_NAME}
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
