#pragma once
#include <stddef.h>

typedef struct ManifoldManifold ManifoldManifold;
typedef struct ManifoldSimplePolygon ManifoldSimplePolygon;
typedef struct ManifoldPolygons ManifoldPolygons;
typedef struct ManifoldMesh ManifoldMesh;
typedef struct ManifoldMeshGL ManifoldMeshGL;
typedef struct ManifoldCurvature ManifoldCurvature;
typedef struct ManifoldComponents ManifoldComponents;
typedef struct ManifoldBox ManifoldBox;
typedef struct ManifoldMaterial ManifoldMaterial;
typedef struct ManifoldExportOptions ManifoldExportOptions;

typedef struct ManifoldManifoldPair {
  ManifoldManifold* first;
  ManifoldManifold* second;
} ManifoldManifoldPair;

typedef struct ManifoldVec2 {
  float x;
  float y;
} ManifoldVec2;

typedef struct ManifoldVec3 {
  float x;
  float y;
  float z;
} ManifoldVec3;

typedef struct ManifoldIVec3 {
  int x;
  int y;
  int z;
} ManifoldIVec3;

typedef struct ManifoldVec4 {
  float x;
  float y;
  float z;
  float w;
} ManifoldVec4;

typedef struct ManifoldProperties {
  float surface_area;
  float volume;
} ManifoldProperties;

typedef struct ManifoldPolyVert {
  ManifoldVec2 pos;
  int idx;

} ManifoldPolyVert;

typedef struct ManifoldCurvatureBounds {
  float max_mean_curvature;
  float min_mean_curvature;
  float max_gaussian_curvature;
  float min_gaussian_curvature;
} ManifoldCurvatureBounds;

typedef enum ManifoldError {
  MANIFOLD_NO_ERROR,
  MANIFOLD_NON_FINITE_VERTEX,
  MANIFOLD_NOT_MANIFOLD,
  MANIFOLD_VERTEX_INDEX_OUT_OF_BOUNDS,
  MANIFOLD_PROPERTIES_WRONG_LENGTH,
  MANIFOLD_MISSING_POSITION_PROPERTIES,
  MANIFOLD_MERGE_VECTORS_DIFFERENT_LENGTHS,
  MANIFOLD_MERGE_INDEX_OUT_OF_BOUNDS,
  MANIFOLD_TRANSFORM_WRONG_LENGTH,
  MANIFOLD_RUN_INDEX_WRONG_LENGTH,
  MANIFOLD_FACE_ID_WRONG_LENGTH,
} ManifoldError;
