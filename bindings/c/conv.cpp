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
  ManifoldError e = NO_ERROR;
  switch (error) {
    case Manifold::Error::NO_ERROR:
      e = NO_ERROR;
      break;
    case Manifold::Error::NON_FINITE_VERTEX:
      e = NON_FINITE_VERTEX;
      break;
    case Manifold::Error::NOT_MANIFOLD:
      e = NOT_MANIFOLD;
      break;
    case Manifold::Error::VERTEX_INDEX_OUT_OF_BOUNDS:
      e = VERTEX_INDEX_OUT_OF_BOUNDS;
      break;
    case Manifold::Error::PROPERTIES_WRONG_LENGTH:
      e = PROPERTIES_WRONG_LENGTH;
      break;
    case Manifold::Error::TRI_PROPERTIES_WRONG_LENGTH:
      e = TRI_PROPERTIES_WRONG_LENGTH;
      break;
    case Manifold::Error::TRI_PROPERTIES_OUT_OF_BOUNDS:
      e = TRI_PROPERTIES_OUT_OF_BOUNDS;
      break;
  };
  return e;
}

ManifoldBox *to_c(manifold::Box *m) {
  return reinterpret_cast<ManifoldBox *>(m);
}

ManifoldMeshRelation *to_c(manifold::MeshRelation *m) {
  return reinterpret_cast<ManifoldMeshRelation *>(m);
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

const manifold::MeshRelation *from_c(ManifoldMeshRelation *m) {
  return reinterpret_cast<manifold::MeshRelation const *>(m);
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

std::vector<glm::vec3> vector_of_array(ManifoldVec3 *vs, size_t length) {
  auto vec = std::vector<glm::vec3>();
  for (int i = 0; i < length; ++i) {
    vec.push_back(from_c(vs[i]));
  }
  return vec;
}

std::vector<float> vector_of_array(float *fs, size_t length) {
  auto vec = std::vector<float>();
  for (int i = 0; i < length; ++i) {
    vec.push_back(fs[i]);
  }
  return vec;
}

std::vector<glm::ivec3> vector_of_array(ManifoldIVec3 *vs, size_t length) {
  auto vec = std::vector<glm::ivec3>();
  for (int i = 0; i < length; ++i) {
    vec.push_back(from_c(vs[i]));
  }
  return vec;
}

std::vector<glm::vec4> vector_of_array(ManifoldVec4 *vs, size_t length) {
  auto vec = std::vector<glm::vec4>();
  for (int i = 0; i < length; ++i) {
    vec.push_back(from_c(vs[i]));
  }
  return vec;
}
