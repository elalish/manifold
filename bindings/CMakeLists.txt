
if(NOT EMSCRIPTEN)
    add_subdirectory(python)
endif()

if(EMSCRIPTEN)
    add_subdirectory(wasm)
endif()
