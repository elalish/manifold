project (manifold)

find_package(Boost COMPONENTS graph REQUIRED)

add_library(${PROJECT_NAME} src/manifold.cu src/manifold_impl.cu src/boolean3.cu src/connected_components.cu)

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries( ${PROJECT_NAME}
    PUBLIC utilities
    PRIVATE collider polygon nvgraph ${MANIFOLD_OMP_INCLUDE} Boost::graph 
)

target_compile_options(${PROJECT_NAME} 
    PRIVATE ${MANIFOLD_NVCC_FLAGS}
)
target_compile_features(${PROJECT_NAME} 
    PUBLIC cxx_std_14 
    PRIVATE cxx_std_14
)

