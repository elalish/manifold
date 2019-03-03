project (collider)

add_library(${PROJECT_NAME} src/collider.cu)

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries( ${PROJECT_NAME}
    utilities
)

target_compile_options(${PROJECT_NAME} 
    PRIVATE -Xcompiler -Werror 
)
target_compile_features(${PROJECT_NAME} 
    PUBLIC cxx_std_11
)

if(APPLE)
#   All CUDA libraries need this. If you ever get 
#   "CUDA driver version is insufficient for CUDA runtime version", 
#   this is probably what's missing.
  set_property(TARGET ${PROJECT_NAME}
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()