include(FetchContent)
find_package(PkgConfig)
if (PKG_CONFIG_FOUND)
    pkg_check_modules(Clipper2 Clipper2)
endif()
if(Clipper2_FOUND)
    add_library(Clipper2 SHARED IMPORTED)
    set_property(TARGET Clipper2 PROPERTY
        IMPORTED_LOCATION ${Clipper2_LINK_LIBRARIES})
    target_include_directories(Clipper2 INTERFACE ${Clipper2_INCLUDE_DIRS})
else()
    FetchContent_Declare(Clipper2
        GIT_REPOSITORY https://github.com/AngusJohnson/Clipper2.git
        GIT_TAG Clipper2_1.3.0
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
        SOURCE_SUBDIR CPP
    )
    FetchContent_MakeAvailable(Clipper2)
    set_target_properties(Clipper2 PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES
        "$<INSTALL_INTERFACE:include>$<BUILD_INTERFACE:${Clipper2_SOURCE_DIR}/Clipper2Lib/include>")
    install(TARGETS Clipper2 EXPORT clipper2Targets)
    install(EXPORT clipper2Targets DESTINATION ${CMAKE_INSTALL_DATADIR}/clipper2)
endif()

FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG b06b775c1c80af51a1183c0e167f9de3b2351a79
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
    FIND_PACKAGE_ARGS NAMES glm
)
FetchContent_MakeAvailable(glm)

if(NOT glm_FOUND)
    install(TARGETS glm EXPORT glmTargets)
    install(TARGETS glm-header-only EXPORT glmTargets)
    install(EXPORT glmTargets DESTINATION ${CMAKE_INSTALL_DATADIR}/glm)
endif()
