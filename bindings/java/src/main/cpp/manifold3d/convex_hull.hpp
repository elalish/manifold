#pragma once

#include <glm/glm.hpp>

#include <quickhull.hpp>
#include <array>
#include <iterator>
#include <vector>
#include <unordered_map>


#include <limits>
#include <random>
#include <cstdlib>

#include "manifold.h"
#include "cross_section.h"

namespace ConvexHull {

struct vec3_hash {
    size_t operator()(const std::array<float, 3>& vec) const {
        std::hash<float> hash_fn;
        return hash_fn(vec[0]) ^ hash_fn(vec[1]) ^ hash_fn(vec[2]);
    }
};

struct vec2_hash {
    size_t operator()(const std::array<float, 2>& vec) const {
        std::hash<float> hash_fn;
        return hash_fn(vec[0]) ^ hash_fn(vec[1]);
    }
};

void QuickHull3D(const std::vector<glm::vec3>& inputVerts, std::vector<glm::ivec3>& triVerts, std::vector<glm::vec3>& vertPos, const float precision) {

    constexpr std::size_t dim = 3;
    using PointType = std::array<float, dim>;
    using Points = std::vector<PointType>;

    Points points(inputVerts.size());

    for (std::size_t i = 0; i < inputVerts.size(); ++i) {
        auto pt = inputVerts[i];
        points[i] = {pt.x, pt.y, pt.z};
    }

    QuickHull::quick_hull<typename Points::const_iterator> quickhull{dim, precision};
    quickhull.add_points(std::cbegin(points), std::cend(points));
    auto initial_simplex = quickhull.get_affine_basis();

    quickhull.create_initial_simplex(std::cbegin(initial_simplex), std::prev(std::cend(initial_simplex)));
    quickhull.create_convex_hull();

    std::unordered_map<std::array<float, 3>, int, vec3_hash> vertIndices;

    for (const auto& facet : quickhull.facets_) {
        for(const auto& vertex : facet.vertices_) {

            auto vert = *vertex;
            std::array<float, 3> arrVertex = {vert[0], vert[1], vert[2]};

            if(vertIndices.count(arrVertex) == 0) {
                vertIndices[arrVertex] = vertPos.size();
                vertPos.push_back(glm::vec3(arrVertex[0], arrVertex[1], arrVertex[2]));
            }
        }
    }

    for (const auto& facet : quickhull.facets_) {

        auto vert = *facet.vertices_[0];
        int firstVertIndex = vertIndices[{vert[0], vert[1], vert[2]}];
        for(std::size_t i = 1; i < facet.vertices_.size() - 1; ++i) {
            auto currVert = *facet.vertices_[i];
            auto nextVert = *facet.vertices_[i+1];
            int secondVertIndex = vertIndices[{currVert[0], currVert[1], currVert[2]}];
            int thirdVertIndex = vertIndices[{nextVert[0], nextVert[1], nextVert[2]}];
            triVerts.push_back(glm::ivec3(firstVertIndex, secondVertIndex, thirdVertIndex));
        }
    }
}

int findMinYPointIndex(const std::vector<glm::vec2>& points) {
    int minYPointIndex = 0;
    for (size_t i = 1; i < points.size(); i++) {
        if ((points[i].y < points[minYPointIndex].y) ||
            (points[i].y == points[minYPointIndex].y && points[i].x > points[minYPointIndex].x)) {
            minYPointIndex = i;
        }
    }
    return minYPointIndex;
}

std::vector<glm::vec2> sortPointsCounterClockwise(const std::vector<glm::vec2>& points) {
    std::vector<glm::vec2> sortedPoints(points);

    // Find the bottom-most point (or one of them, if multiple)
    int minYPointIndex = findMinYPointIndex(sortedPoints);

    // Sort the points by angle from the line horizontal to minYPoint, counter-clockwise
    glm::vec2 minYPoint = points[minYPointIndex];
    std::sort(sortedPoints.begin(), sortedPoints.end(),
        [minYPoint](const glm::vec2& p1, const glm::vec2& p2) -> bool {
            double angle1 = atan2(p1.y - minYPoint.y, p1.x - minYPoint.x);
            double angle2 = atan2(p2.y - minYPoint.y, p2.x - minYPoint.x);
            if (angle1 < 0) angle1 += 2 * 3.141592653589;
            if (angle2 < 0) angle2 += 2 * 3.141592653589;
            return angle1 < angle2;
        }
    );

    return sortedPoints;
}

manifold::SimplePolygon QuickHull2D(const manifold::SimplePolygon& inputVerts, const float precision) {

    constexpr std::size_t dim = 2;
    using PointType = std::array<float, dim>;
    using Points = std::vector<PointType>;

    Points points(inputVerts.size()); // input

    for (std::size_t i = 0; i < inputVerts.size(); ++i) {
        auto pt = inputVerts[i];
        points[i] = {pt.x, pt.y};
    }

    QuickHull::quick_hull<typename Points::const_iterator> quickhull{dim, precision};
    quickhull.add_points(std::cbegin(points), std::cend(points));
    auto initial_simplex = quickhull.get_affine_basis();

    quickhull.create_initial_simplex(std::cbegin(initial_simplex), std::prev(std::cend(initial_simplex)));
    quickhull.create_convex_hull();

    std::unordered_map<std::array<float, 2>, int, vec2_hash> vertIndices;
    manifold::SimplePolygon ret;

    for (const auto& facet : quickhull.facets_) {
        for(const auto& vertex : facet.vertices_) {

            auto vert = *vertex;
            std::array<float, 2> arrVertex = {vert[0], vert[1]};

            if(vertIndices.count(arrVertex) == 0) {
                vertIndices[arrVertex] = ret.size();
                ret.push_back(glm::vec3(arrVertex[0], arrVertex[1], arrVertex[2]));
            }
        }
    }
    return sortPointsCounterClockwise(ret);
}

manifold::Manifold ConvexHull(const manifold::Manifold manifold, const float precision = 0.0001) {
    manifold::Mesh inputMesh = manifold.GetMesh();
    manifold::Mesh outputMesh;
    QuickHull3D(inputMesh.vertPos, outputMesh.triVerts, outputMesh.vertPos, precision);
    //orientMesh(outputMesh);
    return manifold::Manifold(outputMesh);
}

manifold::Manifold ConvexHull(const manifold::Manifold manifold, const manifold::Manifold other, const float precision = 0.0001) {
    manifold::Mesh inputMesh1 = manifold.GetMesh();
    manifold::Mesh inputMesh2 = other.GetMesh();

    // Combine vertices from input meshes
    std::vector<glm::vec3> combinedVerts;
    for (auto& vert: inputMesh1.vertPos) {
        combinedVerts.push_back(vert);
    }

    for (auto& vert: inputMesh2.vertPos) {
        combinedVerts.push_back(vert);
    }

    manifold::Mesh outputMesh;

    QuickHull3D(combinedVerts, outputMesh.triVerts, outputMesh.vertPos, precision);

    //orientMesh(outputMesh);

    return manifold::Manifold(outputMesh);
}

manifold::CrossSection ConvexHull(const manifold::CrossSection& cross_section, const float precision = 0.0001) {
    manifold::SimplePolygon hullPoints;
    for (auto& poly: cross_section.ToPolygons()) {
        for (auto& pt: poly) {
            hullPoints.push_back(glm::vec2(pt.x, pt.y));
        }
    }
    manifold::SimplePolygon res = QuickHull2D(hullPoints, precision);
    return manifold::CrossSection(res);
}


manifold::CrossSection ConvexHull(const manifold::CrossSection& cross_section, const manifold::CrossSection& other, const float precision = 0.0001) {
    manifold::SimplePolygon hullPoints;

    for (auto& poly: cross_section.ToPolygons()) {
        for (auto& pt: poly) {
            hullPoints.push_back(glm::vec2(pt.x, pt.y));
        }
    }

    for (auto& poly: other.ToPolygons()) {
        for (auto& pt: poly) {
            hullPoints.push_back(glm::vec2(pt.x, pt.y));
        }
    }

    manifold::SimplePolygon res = QuickHull2D(hullPoints, precision);
    return manifold::CrossSection(res);
}

}
