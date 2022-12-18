project(manifoldc)


add_library(manifoldc manifoldc.cpp conv.cpp)

if(MANIFOLD_USE_CUDA)
    set_source_files_properties(manifoldc.cpp conv.cpp PROPERTIES LANGUAGE CUDA)
    set_property(TARGET manifoldc PROPERTY CUDA_ARCHITECTURES 61)
endif()

target_link_libraries(
  manifoldc
  PRIVATE manifold sdf meshIO graphlite
)

target_include_directories(manifoldc PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_options(manifoldc PRIVATE ${MANIFOLD_FLAGS})
target_compile_features(manifoldc PRIVATE cxx_std_17)
