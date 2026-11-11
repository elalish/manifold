cmake_minimum_required(VERSION 3.18)
project(fillet_benchmark LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set(MANIFOLD_TEST OFF CACHE BOOL "" FORCE)
set(MANIFOLD_CBIND OFF CACHE BOOL "" FORCE)
option(FILLET_BENCHMARK_PAR "Benchmark the TBB parallel backend" OFF)
set(MANIFOLD_PAR ${FILLET_BENCHMARK_PAR} CACHE BOOL "" FORCE)
set(MANIFOLD_DEBUG OFF CACHE BOOL "" FORCE)
set(MANIFOLD_ASSERT OFF CACHE BOOL "" FORCE)
set(MANIFOLD_DOWNLOADS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
get_filename_component(REPO "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
add_subdirectory("${REPO}" manifold)
set(CLIPPER2_SOURCE "${REPO}/build-bench/_deps/clipper2-src/CPP/Clipper2Lib" CACHE PATH "Clipper2Lib source directory")
if(NOT EXISTS "${CLIPPER2_SOURCE}/src/clipper.offset.cpp")
  message(FATAL_ERROR "Set CLIPPER2_SOURCE to a local Clipper2Lib directory")
endif()
add_library(benchmark_clipper STATIC
  "${CLIPPER2_SOURCE}/src/clipper.engine.cpp"
  "${CLIPPER2_SOURCE}/src/clipper.offset.cpp"
  "${CLIPPER2_SOURCE}/src/clipper.rectclip.cpp")
target_include_directories(benchmark_clipper PUBLIC "${CLIPPER2_SOURCE}/include")
add_executable(fillet_benchmark main.cpp)
target_include_directories(fillet_benchmark PRIVATE "${REPO}/test")
target_link_libraries(fillet_benchmark PRIVATE manifold benchmark_clipper)
target_compile_options(fillet_benchmark PRIVATE -ffp-contract=off)

add_executable(fillet_stress stress.cpp)
target_include_directories(fillet_stress PRIVATE "${REPO}/test")
target_link_libraries(fillet_stress PRIVATE manifold)
target_compile_options(fillet_stress PRIVATE -ffp-contract=off)

# Instrument a generated copy; the production library and source stay intact.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(INSTRUMENTED_FILLET "${CMAKE_CURRENT_BINARY_DIR}/fillet2d_instrumented.cpp")
add_custom_command(
  OUTPUT "${INSTRUMENTED_FILLET}"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/instrument_rejections.py"
    "${REPO}/src/fillet2d.cpp" "${INSTRUMENTED_FILLET}"
  DEPENDS "${REPO}/src/fillet2d.cpp" instrument_rejections.py
  VERBATIM)
add_executable(fillet_rejections rejection_main.cpp "${INSTRUMENTED_FILLET}")
target_include_directories(fillet_rejections PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}" "${REPO}/src" "${REPO}/test")
target_compile_options(fillet_rejections PRIVATE -ffp-contract=off)
# The instrumented object defines CrossSection::Fillet; the static archive's
# uninstrumented fillet2d object is consequently not extracted by the linker.
target_link_libraries(fillet_rejections PRIVATE manifold benchmark_clipper)

add_executable(fillet_rejection_probe rejection_probe.cpp "${INSTRUMENTED_FILLET}")
target_include_directories(fillet_rejection_probe PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}" "${REPO}/src" "${REPO}/test")
target_compile_options(fillet_rejection_probe PRIVATE -ffp-contract=off)
target_link_libraries(fillet_rejection_probe PRIVATE manifold)

if(FILLET_BENCHMARK_PAR)
  add_executable(offset_parallel offset_parallel.cpp)
  target_include_directories(offset_parallel PRIVATE "${REPO}/src" "${REPO}/test")
  target_compile_options(offset_parallel PRIVATE -ffp-contract=off)
  target_link_libraries(offset_parallel PRIVATE manifold benchmark_clipper TBB::tbb)
endif()
