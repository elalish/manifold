#include <conv.h>
#include <manifold.h>
#include <meshIO.h>
#include <sdf.h>

#include "include/types.h"
#include "public.h"
#include "types.h"

ManifoldManifold *to_c(manifold::Manifold *m) {
  return reinterpret_cast<ManifoldManifold *>(m);
}

ManifoldSimplePolygon *to_c(manifold::SimplePolygon *m) {
  return reinterpret_cast<ManifoldSimplePolygon *>(m);
}

ManifoldPolygons *to_c(manifold::Polygons *m) {
  return reinterpret_cast<ManifoldPolygons *>(m);
}

ManifoldMesh *to_c(manifold::Mesh *m) {
  return reinterpret_cast<ManifoldMesh *>(m);
}

ManifoldMeshGL *to_c(manifold::MeshGL *m) {
  return reinterpret_cast<ManifoldMeshGL *>(m);
}

ManifoldCurvature *to_c(manifold::Curvature *m) {
  return reinterpret_cast<ManifoldCurvature *>(m);
}

ManifoldComponents *to_c(manifold::Components *components) {
  return reinterpret_cast<ManifoldComponents *>(components);
}

ManifoldError to_c(manifold::Manifold::Error error) {
  ManifoldError e = MANIFOLD_NO_ERROR;
  switch (error) {
    case Manifold::Error::NoError:
      e = MANIFOLD_NO_ERROR;
      break;
    case Manifold::Error::NonFiniteVertex:
      e = MANIFOLD_NON_FINITE_VERTEX;
      break;
    case Manifold::Error::NotManifold:
      e = MANIFOLD_NOT_MANIFOLD;
      break;
    case Manifold::Error::VertexOutOfBounds:
      e = MANIFOLD_VERTEX_INDEX_OUT_OF_BOUNDS;
      break;
    case Manifold::Error::PropertiesWrongLength:
      e = MANIFOLD_PROPERTIES_WRONG_LENGTH;
      break;
    case Manifold::Error::MissingPositionProperties:
      e = MANIFOLD_MISSING_POSITION_PROPERTIES;
      break;
    case Manifold::Error::MergeVectorsDifferentLengths:
      e = MANIFOLD_MERGE_VECTORS_DIFFERENT_LENGTHS;
      break;
    case Manifold::Error::MergeIndexOutOfBounds:
      e = MANIFOLD_MERGE_INDEX_OUT_OF_BOUNDS;
      break;
    case Manifold::Error::TransformWrongLength:
      e = MANIFOLD_TRANSFORM_WRONG_LENGTH;
      break;
    case Manifold::Error::RunIndexWrongLength:
      e = MANIFOLD_RUN_INDEX_WRONG_LENGTH;
      break;
    case Manifold::Error::FaceIDWrongLength:
      e = MANIFOLD_FACE_ID_WRONG_LENGTH;
      break;
  };
  return e;
}

ManifoldBox *to_c(manifold::Box *m) {
  return reinterpret_cast<ManifoldBox *>(m);
}

ManifoldMaterial *to_c(manifold::Material *m) {
  return reinterpret_cast<ManifoldMaterial *>(m);
}

ManifoldExportOptions *to_c(manifold::ExportOptions *m) {
  return reinterpret_cast<ManifoldExportOptions *>(m);
}

ManifoldVec3 to_c(glm::vec3 v) { return {v.x, v.y, v.z}; }

ManifoldIVec3 to_c(glm::ivec3 v) { return {v.x, v.y, v.z}; }

ManifoldProperties to_c(manifold::Properties p) {
  return {p.surfaceArea, p.volume};
}

const manifold::Manifold *from_c(ManifoldManifold *m) {
  return reinterpret_cast<manifold::Manifold const *>(m);
}

const manifold::SimplePolygon *from_c(ManifoldSimplePolygon *m) {
  return reinterpret_cast<manifold::SimplePolygon const *>(m);
}

const manifold::Polygons *from_c(ManifoldPolygons *m) {
  return reinterpret_cast<manifold::Polygons const *>(m);
}

const manifold::Mesh *from_c(ManifoldMesh *m) {
  return reinterpret_cast<manifold::Mesh const *>(m);
}

const manifold::MeshGL *from_c(ManifoldMeshGL *m) {
  return reinterpret_cast<manifold::MeshGL const *>(m);
}

const manifold::Curvature *from_c(ManifoldCurvature *m) {
  return reinterpret_cast<manifold::Curvature const *>(m);
}

const manifold::Components *from_c(ManifoldComponents *components) {
  return reinterpret_cast<manifold::Components *>(components);
}

const manifold::Box *from_c(ManifoldBox *m) {
  return reinterpret_cast<manifold::Box const *>(m);
}

manifold::Material *from_c(ManifoldMaterial *mat) {
  return reinterpret_cast<manifold::Material *>(mat);
}

manifold::ExportOptions *from_c(ManifoldExportOptions *options) {
  return reinterpret_cast<manifold::ExportOptions *>(options);
}

glm::vec3 from_c(ManifoldVec3 v) { return glm::vec3(v.x, v.y, v.z); }

glm::ivec3 from_c(ManifoldIVec3 v) { return glm::ivec3(v.x, v.y, v.z); }

glm::vec4 from_c(ManifoldVec4 v) { return glm::vec4(v.x, v.y, v.z, v.w); }

std::vector<glm::vec3> vector_of_vec_array(ManifoldVec3 *vs, size_t length) {
  auto vec = std::vector<glm::vec3>();
  for (int i = 0; i < length; ++i) {
    vec.push_back(from_c(vs[i]));
  }
  return vec;
}

std::vector<glm::ivec3> vector_of_vec_array(ManifoldIVec3 *vs, size_t length) {
  auto vec = std::vector<glm::ivec3>();
  for (int i = 0; i < length; ++i) {
    vec.push_back(from_c(vs[i]));
  }
  return vec;
}

std::vector<glm::vec4> vector_of_vec_array(ManifoldVec4 *vs, size_t length) {
  auto vec = std::vector<glm::vec4>();
  for (int i = 0; i < length; ++i) {
    vec.push_back(from_c(vs[i]));
  }
  return vec;
}
