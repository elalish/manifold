add_library(quickhull OBJECT quickhull/QuickHull.cpp)
target_include_directories(quickhull PUBLIC
    $<INSTALL_INTERFACE:quickhull>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/quickhull>)

target_compile_features(quickhull PUBLIC cxx_std_17)
install(TARGETS quickhull EXPORT manifoldTargets)

