add_library(quickhull OBJECT quickhull/QuickHull.cpp)
target_include_directories(quickhull PUBLIC quickhull)
target_compile_features(quickhull PUBLIC cxx_std_17)
