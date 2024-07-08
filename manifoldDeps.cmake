include(FetchContent)
include(GNUInstallDirs)
find_package(PkgConfig QUIET)
find_package(Clipper2 QUIET)
if(MANIFOLD_PAR STREQUAL "TBB")
    find_package(TBB QUIET)
    if(APPLE)
        find_package(oneDPL QUIET)
    endif()
endif()
if (PKG_CONFIG_FOUND)
    if (NOT Clipper2_FOUND)
        pkg_check_modules(Clipper2 Clipper2)
    endif()
    if(MANIFOLD_PAR STREQUAL "TBB" AND NOT TBB_FOUND)
        pkg_check_modules(TBB tbb)
    endif()
endif()
if(Clipper2_FOUND)
    add_library(Clipper2 SHARED IMPORTED)
    set_property(TARGET Clipper2 PROPERTY
        IMPORTED_LOCATION ${Clipper2_LINK_LIBRARIES})
    if(WIN32)
        set_property(TARGET Clipper2 PROPERTY
            IMPORTED_IMPLIB ${Clipper2_LINK_LIBRARIES})
    endif()
    target_include_directories(Clipper2 INTERFACE ${Clipper2_INCLUDE_DIRS})
else()
    message(STATUS "clipper2 not found, downloading from source")
    set(CLIPPER2_UTILS OFF)
    set(CLIPPER2_EXAMPLES OFF)
    set(CLIPPER2_TESTS OFF)
    set(CLIPPER2_USINGZ "OFF" CACHE STRING "Preempt cache default of USINGZ (we only use 2d)")
    FetchContent_Declare(Clipper2
        GIT_REPOSITORY https://github.com/AngusJohnson/Clipper2.git
        GIT_TAG ff378668baae3570e9d8070aa9eb339bdd5a6aba
        GIT_PROGRESS TRUE
        SOURCE_SUBDIR CPP
    )
    FetchContent_MakeAvailable(Clipper2)
    if(NOT EMSCRIPTEN)
        install(TARGETS Clipper2)
    endif()
endif()

find_package(glm QUIET)
if(NOT glm_FOUND)
    message(STATUS "glm not found, downloading from source")
    set(GLM_BUILD_INSTALL "ON" CACHE STRING "")
    FetchContent_Declare(glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG 1.0.1
        GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(glm)
    if(NOT EMSCRIPTEN)
         install(TARGETS glm)
    endif()
endif()

if(MANIFOLD_PAR STREQUAL "TBB" AND NOT TBB_FOUND)
    message(STATUS "tbb not found, downloading from source")
    include(FetchContent)
    set(TBB_TEST OFF CACHE INTERNAL "" FORCE)
    set(TBB_STRICT OFF CACHE INTERNAL "" FORCE)
    FetchContent_Declare(TBB
        GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
        GIT_TAG        v2021.11.0
        GIT_PROGRESS   TRUE
    )
    FetchContent_MakeAvailable(TBB)
    set_property(DIRECTORY ${tbb_SOURCE_DIR} PROPERTY EXCLUDE_FROM_ALL YES)
    # note: we do want to install tbb to the user machine when built from
    # source
    if(NOT EMSCRIPTEN)
        install(TARGETS tbb)
    endif()
endif()

if(MANIFOLD_EXPORT)
    find_package(assimp REQUIRED)
endif()
