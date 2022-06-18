project (manifold)

file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS *.cpp)
add_library(${PROJECT_NAME} ${SOURCE_FILES})

if(MANIFOLD_USE_CUDA)
    set_source_files_properties( ${SOURCE_FILES} PROPERTIES LANGUAGE CUDA)
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 61)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include )
target_link_libraries(${PROJECT_NAME}
    PUBLIC utilities
    PRIVATE collider polygon ${MANIFOLD_INCLUDE} graphlite
)

target_compile_features(${PROJECT_NAME} 
    PUBLIC cxx_std_14 
    PRIVATE cxx_std_17
)

