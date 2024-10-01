# this is a shared library for FFI bindings
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  )

add_library(manifoldc SHARED
  manifoldc.cpp
  conv.cpp
  box.cpp
  cross.cpp
  rect.cpp
  )

if(MSVC)
  set_target_properties(manifoldc PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
  set_target_properties(manifoldc PROPERTIES RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR}/bin )
  set_target_properties(manifoldc PROPERTIES LIBRARY_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR}/bin )
endif()

if(MANIFOLD_EXPORT)
  target_sources(manifoldc PRIVATE meshIOc.cpp)
  target_compile_options(manifoldc PUBLIC -DMANIFOLD_EXPORT)
endif()

target_link_libraries(manifoldc PRIVATE manifold)

target_include_directories(manifoldc PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/bindings/c/include>
  )
target_compile_options(manifoldc PRIVATE ${MANIFOLD_FLAGS})
target_compile_features(manifoldc PRIVATE cxx_std_17)

install(TARGETS manifoldc EXPORT manifoldTargets)
install(FILES
  include/manifold/manifoldc.h
  include/manifold/types.h DESTINATION
  ${CMAKE_INSTALL_INCLUDEDIR}/manifold)

# Local Variables:
# tab-width: 8
# mode: cmake
# indent-tabs-mode: t
# End:
# ex: shiftwidth=2 tabstop=8
