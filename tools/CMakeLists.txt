project (tools)

add_executable(${PROJECT_NAME} load_mesh.cpp)
target_link_libraries(${PROJECT_NAME} mesh)

target_compile_options(${PROJECT_NAME} PRIVATE -Werror -Wall -Wno-sign-compare)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

if(APPLE)
#   All executables that link to CUDA need this. If you ever get 
#   "CUDA driver version is insufficient for CUDA runtime version", 
#   this is probably what's missing.
  set_property(TARGET ${PROJECT_NAME}
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()