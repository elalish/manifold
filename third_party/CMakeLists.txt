add_subdirectory(google_test)
add_subdirectory(stacktrace)

option(ASSIMP_BUILD_ASSIMP_TOOLS "" OFF)
option(ASSIMP_BUILD_TESTS "" OFF)
option(ASSIMP_INSTALL_PDB "" OFF)
add_subdirectory(assimp)