#pragma once

#include <glm/glm.hpp>
#include <vector>
#include "polygon.h"
#include "manifold.h"
#include "buffer_utils.hpp"

namespace MeshUtils {

std::vector<glm::ivec3> TriangulateFaces(const std::vector<glm::vec3>& vertices, const std::vector<std::vector<uint32_t>>& faces, float precision) {
    std::vector<glm::ivec3> result;
    for (const auto& face : faces) {
        // If the face only has 3 vertices, no need to triangulate, just add it to result
        if (face.size() == 3) {
            result.push_back(glm::ivec3(face[0], face[1], face[2]));
            continue;
        }

        // Compute face normal
        glm::vec3 normal = glm::cross(vertices[face[1]] - vertices[face[0]], vertices[face[2]] - vertices[face[0]]);
        normal = glm::normalize(normal);

        // Compute reference right vector
        glm::vec3 right = glm::normalize(vertices[face[1]] - vertices[face[0]]);

        // Compute up vector
        glm::vec3 up = glm::cross(right, normal);

        // Project vertices onto plane
        std::vector<glm::vec2> face2D;
        for (const auto& index : face) {
            glm::vec3 local = vertices[index] - vertices[face[0]];
            face2D.push_back(glm::vec2(glm::dot(local, right), glm::dot(local, up)));
        }

        // Triangulate and remap the triangulated vertices back to the original indices
        std::vector<glm::ivec3> triVerts = manifold::Triangulate({face2D}, precision);
        for (auto& tri : triVerts) {
            tri.x = face[tri.x];
            tri.y = face[tri.y];
            tri.z = face[tri.z];
        }

        // Append to result
        result.insert(result.end(), triVerts.begin(), triVerts.end());
    }
    return result;
}

manifold::Manifold Polyhedron(const std::vector<glm::vec3>& vertices, const std::vector<std::vector<uint32_t>>& faces) {
    manifold::Mesh mesh;
    mesh.triVerts = TriangulateFaces(vertices, faces, 0.0001);
    mesh.vertPos = vertices;

    return manifold::Manifold(mesh);
}

manifold::Manifold Polyhedron(double* vertices, std::size_t nVertices, int* faceBuf, int* faceLengths, std::size_t nFaces) {

    std::vector<glm::vec3> verts = BufferUtils::createDoubleVec3Vector(vertices, nVertices*3);

    std::vector<std::vector<uint32_t>> faces;
    for (std::size_t faceIdx = 0, faceBufIndex = 0; faceIdx < nFaces; faceIdx++) {
        std::size_t faceLength = (std::size_t) faceLengths[faceIdx];
        std::vector<uint32_t> face;
        for (size_t j = 0; j < faceLength; j++) {
            face.push_back((uint32_t) faceBuf[faceBufIndex]);
            faceBufIndex++;
        }
        faces.push_back(face);
    }

    return Polyhedron(verts, faces);
}

}
