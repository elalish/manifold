#!/usr/bin/env bash

# Shared determinism test/filter definitions for CI export+compare.

DETERMINISM_GTEST_FILTER="Boolean.DeterminismSimpleSubtract:Boolean.DeterminismSimpleUnion:Boolean.DeterminismSimpleIntersect:Boolean.MultiCoplanar:Boolean.NonIntersecting:Boolean.AlmostCoplanar"

DETERMINISM_OBJ_FILES=(
  det_simple_subtract.obj
  det_simple_union.obj
  det_simple_intersect.obj
  det_multi_coplanar.obj
  det_non_overlap.obj
  det_nearly_coplanar.obj
)

