#!/usr/bin/env bash

# Shared sanitizer test/filter definitions.
# Keep core small for PR CI stability, and tune over time with runtime data.

SANITIZER_GTEST_FILTER_CORE="Boolean.DeterminismSimpleSubtract:Boolean.DeterminismSimpleUnion:Boolean.DeterminismSimpleIntersect:Boolean.MultiCoplanar:Boolean.NonIntersecting:Manifold.MeshDeterminism"

# Broader set for scheduled/manual runs.
SANITIZER_GTEST_FILTER_EXTENDED="${SANITIZER_GTEST_FILTER_CORE}:Boolean.AlmostCoplanar"

# Default filter used by workflow/test helpers.
SANITIZER_GTEST_FILTER="${SANITIZER_GTEST_FILTER_CORE}"
