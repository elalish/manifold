project (wasm)

add_executable(manifoldjs bindings.cpp)
target_link_libraries(manifoldjs manifold)
target_compile_options(manifoldjs PRIVATE ${MANIFOLD_FLAGS} -fexceptions)
target_link_options(manifoldjs PUBLIC --bind)
target_compile_features(manifoldjs PUBLIC cxx_std_14)
set_target_properties(manifoldjs PROPERTIES OUTPUT_NAME "manifold")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/index.html DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
