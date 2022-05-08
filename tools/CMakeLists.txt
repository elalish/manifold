project (tools)

add_executable(loadMesh load_mesh.cpp)
target_link_libraries(loadMesh manifold meshIO)

target_compile_options(loadMesh PRIVATE ${MANIFOLD_FLAGS})
target_compile_features(loadMesh PUBLIC cxx_std_14)

add_executable(perfTest perf_test.cpp)
target_link_libraries(perfTest manifold)

target_compile_options(perfTest PRIVATE ${MANIFOLD_FLAGS})
target_compile_features(perfTest PUBLIC cxx_std_14)

if(BUILD_TEST_CGAL)
add_executable(perfTestCGAL perf_test_cgal.cpp)
find_package(CGAL REQUIRED COMPONENTS Core)
target_compile_definitions(perfTestCGAL PRIVATE CGAL_USE_GMPXX)
# target_compile_definitions(perfTestCGAL PRIVATE CGAL_DEBUG)
target_link_libraries(perfTestCGAL manifold CGAL::CGAL CGAL::CGAL_Core)

target_compile_options(perfTestCGAL PRIVATE ${MANIFOLD_FLAGS})
target_compile_features(perfTestCGAL PUBLIC cxx_std_14)
endif()

# add_executable(playground playground.cpp)
# target_link_libraries(playground manifold meshIO samples)

# target_compile_options(playground PRIVATE ${MANIFOLD_FLAGS})
# target_compile_features(playground PUBLIC cxx_std_14)
