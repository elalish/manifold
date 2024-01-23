include(FetchContent)
include(GNUInstallDirs)
find_package(PkgConfig)
if (PKG_CONFIG_FOUND)
    pkg_check_modules(Clipper2 Clipper2)
    if(MANIFOLD_PAR STREQUAL "TBB")
        pkg_check_modules(TBB tbb)
    endif()
endif()
if(Clipper2_FOUND)
    add_library(Clipper2 SHARED IMPORTED)
    set_property(TARGET Clipper2 PROPERTY
        IMPORTED_LOCATION ${Clipper2_LINK_LIBRARIES})
    target_include_directories(Clipper2 INTERFACE ${Clipper2_INCLUDE_DIRS})
else()
    message(STATUS "clipper2 not found, downloading from source")
    set(CLIPPER2_UTILS OFF)
    set(CLIPPER2_EXAMPLES OFF)
    set(CLIPPER2_TESTS OFF)
    set(CLIPPER2_USINGZ "OFF" CACHE STRING "Preempt cache default of USINGZ (we only use 2d)")
    FetchContent_Declare(Clipper2
        GIT_REPOSITORY https://github.com/AngusJohnson/Clipper2.git
        GIT_TAG Clipper2_1.3.0
        GIT_PROGRESS TRUE
        SOURCE_SUBDIR CPP
    )
    FetchContent_MakeAvailable(Clipper2)
    if(NOT EMSCRIPTEN)
        set_target_properties(Clipper2 PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES
            "$<INSTALL_INTERFACE:include>$<BUILD_INTERFACE:${Clipper2_SOURCE_DIR}/Clipper2Lib/include>")
        install(TARGETS Clipper2 EXPORT clipper2Targets)
        install(EXPORT clipper2Targets DESTINATION ${CMAKE_INSTALL_DATADIR}/clipper2)
    endif()
endif()

FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG b06b775c1c80af51a1183c0e167f9de3b2351a79
    GIT_PROGRESS TRUE
    FIND_PACKAGE_ARGS NAMES glm
)
FetchContent_MakeAvailable(glm)

if(NOT glm_FOUND)
    message(STATUS "glm not found, downloading from source")
    if(NOT EMSCRIPTEN)
        install(TARGETS glm EXPORT glmTargets)
        install(TARGETS glm-header-only EXPORT glmTargets)
        install(EXPORT glmTargets DESTINATION ${CMAKE_INSTALL_DATADIR}/glm)
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
