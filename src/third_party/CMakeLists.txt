add_subdirectory(graphlite)

set(CLIPPER2_UTILS OFF)
set(CLIPPER2_EXAMPLES OFF)
set(CLIPPER2_TESTS OFF)
add_definitions(-D_HAS_EXCEPTIONS=0) # disable exceptions for STL

add_subdirectory(clipper2/CPP)
