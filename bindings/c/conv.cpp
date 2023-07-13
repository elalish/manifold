#include <conv.h>
#include <manifold.h>
#include <sdf.h>

#include <vector>

#include "cross_section.h"
#include "include/types.h"
#include "public.h"
#include "types.h"

ManifoldManifold *to_c(manifold::Manifold *m) {
  return reinterpret_cast<ManifoldManifold *>(m);
}

ManifoldManifoldVec *to_c(ManifoldVec *ms) {
  return reinterpret_cast<ManifoldManifoldVec *>(ms);
}

ManifoldCrossSection *to_c(manifold::CrossSection *cs) {
  return reinterpret_cast<ManifoldCrossSection *>(cs);
}

ManifoldCrossSectionVec *to_c(CrossSectionVec *csv) {
  return reinterpret_cast<ManifoldCrossSectionVec *>(csv);
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

ManifoldOpType to_c(manifold::OpType optype) {
  ManifoldOpType op = MANIFOLD_ADD;
  switch (optype) {
    case manifold::OpType::Add:
      break;
    case manifold::OpType::Subtract:
      op = MANIFOLD_SUBTRACT;
      break;
    case manifold::OpType::Intersect:
      op = MANIFOLD_INTERSECT;
      break;
  };
  return op;
}

ManifoldError to_c(manifold::Manifold::Error error) {
  ManifoldError e = MANIFOLD_NO_ERROR;
  switch (error) {
    case Manifold::Error::NoError:
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
    case Manifold::Error::InvalidConstruction:
      e = MANIFOLD_INVALID_CONSTRUCTION;
      break;
  };
  return e;
}

ManifoldBox *to_c(manifold::Box *m) {
  return reinterpret_cast<ManifoldBox *>(m);
}

ManifoldRect *to_c(manifold::Rect *m) {
  return reinterpret_cast<ManifoldRect *>(m);
}

ManifoldVec2 to_c(glm::vec2 v) { return {v.x, v.y}; }

ManifoldVec3 to_c(glm::vec3 v) { return {v.x, v.y, v.z}; }

ManifoldIVec3 to_c(glm::ivec3 v) { return {v.x, v.y, v.z}; }

ManifoldProperties to_c(manifold::Properties p) {
  return {p.surfaceArea, p.volume};
}

const manifold::Manifold *from_c(ManifoldManifold *m) {
  return reinterpret_cast<manifold::Manifold const *>(m);
}

ManifoldVec *from_c(ManifoldManifoldVec *ms) {
  return reinterpret_cast<ManifoldVec *>(ms);
}

const manifold::CrossSection *from_c(ManifoldCrossSection *cs) {
  return reinterpret_cast<manifold::CrossSection *>(cs);
}

CrossSectionVec *from_c(ManifoldCrossSectionVec *csv) {
  return reinterpret_cast<CrossSectionVec *>(csv);
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

OpType from_c(ManifoldOpType optype) {
  auto op = OpType::Add;
  switch (optype) {
    case MANIFOLD_ADD:
      break;
    case MANIFOLD_SUBTRACT:
      op = OpType::Subtract;
      break;
    case MANIFOLD_INTERSECT:
      op = OpType::Intersect;
      break;
  };
  return op;
}

CrossSection::FillRule from_c(ManifoldFillRule fillrule) {
  auto fr = CrossSection::FillRule::EvenOdd;
  switch (fillrule) {
    case MANIFOLD_FILL_RULE_EVEN_ODD:
      break;
    case MANIFOLD_FILL_RULE_NON_ZERO:
      fr = CrossSection::FillRule::NonZero;
      break;
    case MANIFOLD_FILL_RULE_POSITIVE:
      fr = CrossSection::FillRule::Positive;
      break;
    case MANIFOLD_FILL_RULE_NEGATIVE:
      fr = CrossSection::FillRule::Negative;
      break;
  };
  return fr;
}

CrossSection::JoinType from_c(ManifoldJoinType join_type) {
  auto jt = CrossSection::JoinType::Square;
  switch (join_type) {
    case MANIFOLD_JOIN_TYPE_SQUARE:
      break;
    case MANIFOLD_JOIN_TYPE_ROUND:
      jt = CrossSection::JoinType::Round;
      break;
    case MANIFOLD_JOIN_TYPE_MITER:
      jt = CrossSection::JoinType::Miter;
      break;
  };
  return jt;
}

const manifold::Box *from_c(ManifoldBox *m) {
  return reinterpret_cast<manifold::Box const *>(m);
}

const manifold::Rect *from_c(ManifoldRect *m) {
  return reinterpret_cast<manifold::Rect const *>(m);
}

glm::vec2 from_c(ManifoldVec2 v) { return glm::vec2(v.x, v.y); }

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
