project(python)

add_subdirectory(third_party)
pybind11_add_module(pymanifold pymanifold.cpp)
target_link_libraries(pymanifold PRIVATE manifold meshIO)
target_compile_options(pymanifold PRIVATE ${MANIFOLD_FLAGS})
target_compile_features(pymanifold PUBLIC cxx_std_14)
target_include_directories(pymanifold
    PRIVATE ${PYBIND11_DIR}/include
)