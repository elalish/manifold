project (utilities)

file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS *.cpp)
add_library(${PROJECT_NAME} ${SOURCE_FILES})

message("CUDA Support: ${MANIFOLD_USE_CUDA}")
message("Parallel Backend: ${MANIFOLD_PAR}")

target_include_directories(${PROJECT_NAME} INTERFACE ${PROJECT_SOURCE_DIR}/include)

if (MANIFOLD_PAR STREQUAL "OMP")
    find_package(OpenMP REQUIRED)
    target_include_directories(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
    target_compile_options(${PROJECT_NAME} PUBLIC -DMANIFOLD_PAR='O' -fopenmp)
    target_link_options(${PROJECT_NAME} PUBLIC -fopenmp)
elseif (MANIFOLD_PAR STREQUAL "TBB")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(TBB REQUIRED tbb)
    target_include_directories(${PROJECT_NAME} PUBLIC ${TBB_INCLUDE_DIRS})
    target_compile_options(${PROJECT_NAME} PUBLIC -DMANIFOLD_PAR='T')
    target_link_libraries(${PROJECT_NAME} PUBLIC ${TBB_LINK_LIBRARIES})
elseif (MANIFOLD_PAR STREQUAL "NONE")
    set(MANIFOLD_PAR "CPP")
else ()
    message(FATAL_ERROR "Invalid value for MANIFOLD_PAR: ${MANIFOLD_PAR}. "
        "Should be one of \"TBB\", \"OMP\", \"NONE\"")
endif()

target_include_directories(${PROJECT_NAME} PUBLIC ${THRUST_INC_DIR} ${GLM_INC_DIR})
if(MANIFOLD_USE_CUDA)
    set_source_files_properties(${SOURCE_FILES} PROPERTIES LANGUAGE CUDA)
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 61)
    target_compile_options(${PROJECT_NAME} 
        PUBLIC ${MANIFOLD_FLAGS} -DMANIFOLD_USE_CUDA
        "$<$<COMPILE_LANGUAGE:CUDA>:${MANIFOLD_NVCC_FLAGS}>"
    )
else()
    target_compile_options(${PROJECT_NAME} 
        PUBLIC ${MANIFOLD_FLAGS}
        -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_${MANIFOLD_PAR}
    )
endif()

