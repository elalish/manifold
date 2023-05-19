
#pragma once

extern "C" {
#include "libqhull/libqhull.h"
}

#include "public.h"
#include <glm/glm.hpp>
#include <vector>
#include <queue>
#include <map>
#include <algorithm>

namespace manifold {

Mesh computeConvexHull3D(const Mesh& mesh1, const Mesh& mesh2);
// For MeshGL
void computeConvexHull3D(const std::vector<float>& vertPos, std::vector<float>& resultVertPos, std::vector<uint32_t>& resultTriVerts);

SimplePolygon computeConvexHull2D(const SimplePolygon& allPoints);

}
