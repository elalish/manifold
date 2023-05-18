
#pragma once

extern "C" {
#include "libqhull/libqhull.h"
}

#include "public.h"
#include <glm/glm.hpp>
#include <vector>
#include <queue>
#include <map>

namespace manifold {

Mesh computeConvexHull3D(const Mesh& mesh1, const Mesh& mesh2);
SimplePolygon computeConvexHull2D(const SimplePolygon& allPoints);

}
