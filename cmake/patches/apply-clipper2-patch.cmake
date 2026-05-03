# Idempotent application of the Clipper2 carry-patch. Runs as the
# PATCH_COMMAND of FetchContent_Declare(Clipper2 ...). Re-runs of
# `cmake` (e.g., when CMakeLists.txt files in the parent project
# are edited) reinvoke the patch step; without this idempotency
# wrapper, `git apply` fails on already-patched files.
#
# Invoked via: cmake -DPATCH_FILE=<path> -DSOURCE_DIR=<src> -P this.cmake

if(NOT PATCH_FILE OR NOT SOURCE_DIR)
  message(FATAL_ERROR "PATCH_FILE and SOURCE_DIR are required")
endif()

# `git apply --reverse --check` succeeds iff the patch is already applied.
execute_process(
  COMMAND git apply --reverse --check --ignore-whitespace --whitespace=nowarn "${PATCH_FILE}"
  WORKING_DIRECTORY "${SOURCE_DIR}"
  RESULT_VARIABLE _already_applied
  OUTPUT_QUIET
  ERROR_QUIET
)
if(_already_applied EQUAL 0)
  return()
endif()

execute_process(
  COMMAND git apply --ignore-whitespace --whitespace=nowarn "${PATCH_FILE}"
  WORKING_DIRECTORY "${SOURCE_DIR}"
  RESULT_VARIABLE _apply_result
)
if(NOT _apply_result EQUAL 0)
  message(FATAL_ERROR "Failed to apply Clipper2 carry-patch: ${PATCH_FILE}")
endif()
