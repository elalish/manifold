add_subdirectory(google_test)

option(ASSIMP_BUILD_ASSIMP_TOOLS "" OFF)
option(ASSIMP_BUILD_TESTS "" OFF)
option(ASSIMP_INSTALL_PDB "" OFF)
add_subdirectory(assimp)
target_include_directories(assimp
    PUBLIC ${ASSIMP_SOURCE_DIR}/include ${ASSIMP_BINARY_DIR}/include
)
