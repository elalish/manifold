project(manifoldc)

add_library(${PROJECT_NAME} manifoldc.cpp conv.cpp)

if(MANIFOLD_EXPORT)
    target_link_libraries(${PROJECT_NAME} PRIVATE meshIO)
    target_compile_options(${PROJECT_NAME} PUBLIC -DMANIFOLD_EXPORT)
endif()

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE manifold sdf cross_section
)

target_include_directories(${PROJECT_NAME} PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_compile_options(${PROJECT_NAME} PRIVATE ${MANIFOLD_FLAGS})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

install(TARGETS ${PROJECT_NAME} EXPORT manifoldTargets)
install(FILES include/manifoldc.h include/types.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
