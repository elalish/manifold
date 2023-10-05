set(CLIPPER2_UTILS OFF)
set(CLIPPER2_EXAMPLES OFF)
set(CLIPPER2_TESTS OFF)
set(CLIPPER2_USINGZ "OFF" CACHE STRING "Preempt cache default of USINGZ (we only use 2d)")
add_subdirectory(clipper2/CPP)

add_library(quickhull quickhull/QuickHull.cpp)
target_include_directories(quickhull PUBLIC quickhull)
target_compile_features(quickhull PUBLIC cxx_std_17)
