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

void QuickHull(const std::vector<glm::vec3>& inputVerts, std::vector<glm::ivec3>& triVerts, std::vector<glm::vec3>& vertPos) {

    using F = float;
    constexpr std::size_t dim = 3;
    using PointType = std::array<float, dim>;
    using Points = std::vector<PointType>;

    Points points(inputVerts.size()); // input

    for (std::size_t i = 0; i < inputVerts.size(); ++i) {
        auto pt = inputVerts[i];
        points[i] = {pt.x, pt.y, pt.z};
    }

    for (auto& pt: points) {
        std::cout << pt[0] << " " << pt[1] << " " << pt[2] << "; ";
    }
    std::cout << std::endl;

    const F eps = 0.0001;
    QuickHull::quick_hull<typename Points::const_iterator> quickhulltmp{dim, eps};
    quickhulltmp.add_points(std::cbegin(points), std::cend(points));
    auto initial_simplex = quickhulltmp.get_affine_basis();
    //if (initial_simplex.size() < dim + 1) {
    //    return EXIT_FAILURE; // degenerated input set
    //}
    quickhulltmp.create_initial_simplex(std::cbegin(initial_simplex), std::prev(std::cend(initial_simplex)));
    quickhulltmp.create_convex_hull();
    //if (!quickhulltmp.check()) {
    //    return EXIT_FAILURE; // resulted structure is not convex (generally due to precision errors)
    //}

    std::unordered_map<std::array<float, 3>, int, vec3_hash> vertIndices;

    for (const auto& facet : quickhulltmp.facets_) {
        for(const auto& vertex : facet.vertices_) {

            std::array<float, 3> arrVertex;

            std::size_t i = 0;
            for (auto & coordinate_ : *vertex) {
                arrVertex[i] = coordinate_;
                i++;
            }

            if(vertIndices.count(arrVertex) == 0) {
                vertIndices[arrVertex] = vertPos.size();
                vertPos.push_back(glm::vec3(arrVertex[0], arrVertex[1], arrVertex[2]));
            }
        }
    }

    for (const auto& facet : quickhulltmp.facets_) {
        //if(facet.vertices_.size() < 3) continue;  // Skip degenerate facets

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

    std::cout << "N verts: " << vertPos.size() << std::endl;
    std::cout << "N tri: " << triVerts.size() << std::endl;
}

manifold::Manifold ConvexHull(const manifold::Manifold manifold) {
    manifold::Mesh inputMesh = manifold.GetMesh();
    manifold::Mesh outputMesh;
    QuickHull(inputMesh.vertPos, outputMesh.triVerts, outputMesh.vertPos);
    //orientMesh(outputMesh);
    return manifold::Manifold(outputMesh);
}

manifold::Manifold ConvexHull(const manifold::Manifold manifold, const manifold::Manifold other) {
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

    QuickHull(combinedVerts, outputMesh.triVerts, outputMesh.vertPos);

    //orientMesh(outputMesh);

    return manifold::Manifold(outputMesh);
}

}
