add_subdirectory(google_test)
add_subdirectory(graphlite)

if(EMSCRIPTEN)
    set(BUILD_SHARED_LIBS off)
    set(ASSIMP_BUILD_ASSIMP_TOOLS off)
    set(ASSIMP_BUILD_TESTS off)
    add_subdirectory(assimp)
endif()


