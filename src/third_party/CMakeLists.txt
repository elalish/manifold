add_subdirectory(glm)
add_subdirectory(graphlite)

set(CLIPPER2_UTILS OFF)
set(CLIPPER2_EXAMPLES OFF)
set(CLIPPER2_TESTS OFF)
set(CLIPPER2_USINGZ "OFF" CACHE STRING "Preempt cache default of USINGZ (we only use 2d)")
add_definitions(-D_HAS_EXCEPTIONS=0) # disable exceptions for STL
add_subdirectory(clipper2/CPP)

file(GLOB_RECURSE QH_SOURCE_FILES CONFIGURE_DEPENDS quickhull/*.cpp)
add_library(quickhull ${QH_SOURCE_FILES})
target_include_directories(quickhull PUBLIC quickhull)
target_compile_features(quickhull PUBLIC cxx_std_17)
