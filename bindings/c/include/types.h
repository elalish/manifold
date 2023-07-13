#pragma once
#include <stddef.h>

// opaque pointers

typedef struct ManifoldManifold ManifoldManifold;
typedef struct ManifoldManifoldVec ManifoldManifoldVec;
typedef struct ManifoldCrossSection ManifoldCrossSection;
typedef struct ManifoldCrossSectionVec ManifoldCrossSectionVec;
typedef struct ManifoldSimplePolygon ManifoldSimplePolygon;
typedef struct ManifoldPolygons ManifoldPolygons;
typedef struct ManifoldMesh ManifoldMesh;
typedef struct ManifoldMeshGL ManifoldMeshGL;
typedef struct ManifoldBox ManifoldBox;
typedef struct ManifoldRect ManifoldRect;

#ifdef MANIFOLD_EXPORT
typedef struct ManifoldMaterial ManifoldMaterial;
typedef struct ManifoldExportOptions ManifoldExportOptions;
#endif

// structs

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

// enums

typedef enum ManifoldOpType {
  MANIFOLD_ADD,
  MANIFOLD_SUBTRACT,
  MANIFOLD_INTERSECT
} ManifoldOpType;

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
  MANIFOLD_INVALID_CONSTRUCTION,
} ManifoldError;

typedef enum ManifoldFillRule {
  MANIFOLD_FILL_RULE_EVEN_ODD,
  MANIFOLD_FILL_RULE_NON_ZERO,
  MANIFOLD_FILL_RULE_POSITIVE,
  MANIFOLD_FILL_RULE_NEGATIVE
} ManifoldFillRule;

typedef enum ManifoldJoinType {
  MANIFOLD_JOIN_TYPE_SQUARE,
  MANIFOLD_JOIN_TYPE_ROUND,
  MANIFOLD_JOIN_TYPE_MITER,
} ManifoldJoinType;
