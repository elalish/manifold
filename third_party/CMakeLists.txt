add_subdirectory(google_test)

option(ASSIMP_BUILD_ASSIMP_TOOLS "" OFF)
option(ASSIMP_BUILD_TESTS "" OFF)
option(ASSIMP_INSTALL_PDB "" OFF)
add_subdirectory(assimp)
target_include_directories(assimp
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/assimp/include ${CMAKE_CURRENT_BINARY_DIR}/assimp/include
)
