project(manifoldc)

# this is a shared library for FFI bindings
add_library(${PROJECT_NAME} SHARED manifoldc.cpp conv.cpp)

if(MSVC)
    set_target_properties(${PROJECT_NAME} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY 
        ${CMAKE_BINARY_DIR}/bin )
    set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_BINARY_DIR}/bin )
endif()

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
