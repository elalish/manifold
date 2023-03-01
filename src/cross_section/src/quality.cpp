#include "quality.h"

#include "public.h"

using namespace manifold;

int Quality::circularSegments_ = 0;
float Quality::circularAngle_ = 10.0f;
float Quality::circularEdgeLength_ = 1.0f;

namespace manifold {

void Quality::SetMinCircularAngle(float angle) {
  if (angle <= 0) return;
  circularAngle_ = angle;
}

void Quality::SetMinCircularEdgeLength(float length) {
  if (length <= 0) return;
  circularEdgeLength_ = length;
}

void Quality::SetCircularSegments(int number) {
  if (number < 3 && number != 0) return;
  circularSegments_ = number;
}

int Quality::GetCircularSegments(float radius) {
  if (circularSegments_ > 0) return circularSegments_;
  int nSegA = 360.0f / circularAngle_;
  int nSegL = 2.0f * radius * glm::pi<float>() / circularEdgeLength_;
  int nSeg = fmin(nSegA, nSegL) + 3;
  nSeg -= nSeg % 4;
  return nSeg;
}
}  // namespace manifold
