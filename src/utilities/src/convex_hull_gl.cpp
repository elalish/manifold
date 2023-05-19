#include "convex_hull.h"

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

using namespace manifold;

std::vector<std::vector<int>> buildAdjacency(const std::vector<uint32_t> &triVerts) {
    std::map<std::pair<int, int>, std::vector<int>> edgeToFace;
    std::vector<std::vector<int>> adjacency(triVerts.size() / 3);

    for (int i = 0; i < triVerts.size() / 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::pair<int, int> edge(triVerts[i*3+j], triVerts[i*3+(j+1)%3]);
            if (edge.first > edge.second) std::swap(edge.first, edge.second);

            edgeToFace[edge].push_back(i);
        }
    }

    for (const auto &pair : edgeToFace) {
        for (int i = 0; i < pair.second.size(); ++i) {
            for (int j = i+1; j < pair.second.size(); ++j) {
                adjacency[pair.second[i]].push_back(pair.second[j]);
                adjacency[pair.second[j]].push_back(pair.second[i]);
            }
        }
    }

    return adjacency;
}

std::vector<uint32_t> orientMesh(const std::vector<uint32_t> &triVerts) {
    std::vector<uint32_t> orientedTriVerts = triVerts; // copy input to output
    std::vector<std::vector<int>> adjacency = buildAdjacency(orientedTriVerts);
    std::vector<bool> visited(orientedTriVerts.size() / 3, false);
    std::queue<int> q;

    q.push(0);
    visited[0] = true;

    while (!q.empty()) {
        int currentFaceIndex = q.front();
        q.pop();

        for (int neighbour : adjacency[currentFaceIndex]) {
            if (visited[neighbour]) continue;

            for (int i = 0; i < 3; ++i) {
                if ((orientedTriVerts[neighbour*3+i] == orientedTriVerts[currentFaceIndex*3] && orientedTriVerts[neighbour*3+(i+1)%3] == orientedTriVerts[currentFaceIndex*3+1]) ||
                    (orientedTriVerts[neighbour*3+i] == orientedTriVerts[currentFaceIndex*3+1] && orientedTriVerts[neighbour*3+(i+1)%3] == orientedTriVerts[currentFaceIndex*3+2]) ||
                    (orientedTriVerts[neighbour*3+i] == orientedTriVerts[currentFaceIndex*3+2] && orientedTriVerts[neighbour*3+(i+1)%3] == orientedTriVerts[currentFaceIndex*3])) {
                    std::swap(orientedTriVerts[neighbour*3+(i+1)%3], orientedTriVerts[neighbour*3+(i+2)%3]);
                    break;
                }
            }

            visited[neighbour] = true;
            q.push(neighbour);
        }
    }

    return orientedTriVerts;
}


void computeConvexHull3D(const std::vector<float>& vertPos, std::vector<float>& resultVertPos, std::vector<uint32_t>& resultTriVerts) {
    // Convert vertices to coordinate array for Qhull
    int dim = 3;
    int n = vertPos.size() / dim;

    coordT* points = new coordT[n*dim];
    for(int i = 0; i < n; i++) {
        points[i*dim] = vertPos[i*dim];
        points[i*dim+1] = vertPos[i*dim+1];
        points[i*dim+2] = vertPos[i*dim+2];
    }

    boolT ismalloc = false;  // True if qhull should free points in qh_freeqhull() or reallocation
    char flags[] = "qhull Qt";  // option flags for qhull, see qh_opt.htm

    int exitcode = qh_new_qhull(dim, n, points, ismalloc, flags, NULL, NULL);
    if(exitcode != 0) {
        std::cout << "Convex Hull failled! Exiting the function." << std::endl;
        delete[] points;
        return;
    }

    std::map<int, int> vertexIndexMap;
    // Iterate over the facets of the hull
    facetT *facet;
    vertexT *vertex, **vertexp;
    FORALLfacets {
        std::array<uint32_t, 3> tri;
        int i = 0;

        FOREACHvertex_(facet->vertices) {
            int id = qh_pointid(vertex->point);

            // Check if the vertex is already added
            if(vertexIndexMap.find(id) == vertexIndexMap.end()) {
                resultVertPos.push_back(vertPos[id*dim]);
                resultVertPos.push_back(vertPos[id*dim+1]);
                resultVertPos.push_back(vertPos[id*dim+2]);
                vertexIndexMap[id] = resultVertPos.size() / dim - 1;
            }

            // Add the index to the triangle
            tri[i] = vertexIndexMap[id];
            i++;
        }

        resultTriVerts.push_back(tri[0]);
        resultTriVerts.push_back(tri[1]);
        resultTriVerts.push_back(tri[2]);
    }

    qh_freeqhull(!qh_ALL);
    delete[] points;

    // Orient faces by right-hand rule.
    resultTriVerts = orientMesh(resultTriVerts);
}

}
