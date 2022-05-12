add_subdirectory(graphlite)

############################################################################
#                         BUILD ASSIMP LIBRARY                             #
############################################################################
option(ASSIMP_BUILD_ASSIMP_TOOLS off)
option(ASSIMP_BUILD_TESTS off)
add_subdirectory(assimp)

############################################################################
#                           Build Google Test                              #
############################################################################
# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# Prevent installation of GTest with your project
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(INSTALL_GMOCK OFF CACHE BOOL "" FORCE)

add_subdirectory(google_test)
# Add aliases for GTest and GMock libraries
if(NOT TARGET GTest::GTest)
    add_library(GTest::GTest ALIAS gtest)
    add_library(GTest::Main ALIAS gtest_main)
endif()

if(NOT TARGET GTest::GMock)
    add_library(GMock::GMock ALIAS gmock)
    add_library(GMock::Main ALIAS gmock_main)
endif()

