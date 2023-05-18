#include "convex_hull.h"

#include "public.h"
#include <glm/glm.hpp>
#include <vector>
#include <queue>
#include <map>


namespace manifold {

using namespace manifold;

std::vector<std::vector<int>> buildAdjacency(const Mesh &mesh) {
    std::map<std::pair<int, int>, std::vector<int>> edgeToFace;
    std::vector<std::vector<int>> adjacency(mesh.triVerts.size());

    for (int i = 0; i < mesh.triVerts.size(); ++i) {
        glm::ivec3 face = mesh.triVerts[i];
        for (int j = 0; j < 3; ++j) {
            std::pair<int, int> edge(face[j], face[(j+1)%3]);
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

void orientMesh(Mesh &mesh) {
    std::vector<std::vector<int>> adjacency = buildAdjacency(mesh);
    std::vector<bool> visited(mesh.triVerts.size(), false);
    std::queue<int> q;

    q.push(0);
    visited[0] = true;

    while (!q.empty()) {
        int currentFaceIndex = q.front();
        q.pop();
        glm::ivec3 currentFace = mesh.triVerts[currentFaceIndex];

        for (int neighbour : adjacency[currentFaceIndex]) {
            if (visited[neighbour]) continue;

            glm::ivec3 neighbourFace = mesh.triVerts[neighbour];

            for (int i = 0; i < 3; ++i) {
                if ((neighbourFace[i] == currentFace[0] && neighbourFace[(i+1)%3] == currentFace[1]) ||
                    (neighbourFace[i] == currentFace[1] && neighbourFace[(i+1)%3] == currentFace[2]) ||
                    (neighbourFace[i] == currentFace[2] && neighbourFace[(i+1)%3] == currentFace[0])) {
                    std::swap(mesh.triVerts[neighbour][(i+1)%3], mesh.triVerts[neighbour][(i+2)%3]);
                    break;
                }
            }

            visited[neighbour] = true;
            q.push(neighbour);
        }
    }
}

Mesh computeConvexHull3D(const Mesh& mesh1, const Mesh& mesh2) {
  std::vector<glm::vec3> combinedVerts = mesh1.vertPos;
  combinedVerts.insert(combinedVerts.end(), mesh2.vertPos.begin(), mesh2.vertPos.end());

  // Convert vertices to coordinate array for Qhull
  int dim = 3;
  int n = combinedVerts.size();

  coordT* points = new coordT[n*dim];
  for(int i = 0; i < n; i++) {
    points[i*dim] = combinedVerts[i].x;
    points[i*dim+1] = combinedVerts[i].y;
    points[i*dim+2] = combinedVerts[i].z;
  }

  boolT ismalloc = false;  // True if qhull should free points in qh_freeqhull() or reallocation
  char flags[] = "qhull Qt";  // option flags for qhull, see qh_opt.htm

  int exitcode = qh_new_qhull(dim, n, points, ismalloc, flags, NULL, NULL);
  if(exitcode != 0) {
    std::cout << "Convex Hull failled! Returning first mesh." << std::endl;
    return mesh1;
  }

  // Create a new mesh for the convex hull
  Mesh convexHull;

  std::map<int, int> vertexIndexMap;
  //// Iterate over the facets of the hull
  facetT *facet;
  vertexT *vertex, **vertexp;
  FORALLfacets {
    glm::ivec3 tri;
    int i = 0;

    FOREACHvertex_(facet->vertices) {
      int id = qh_pointid(vertex->point);

      // Check if the vertex is already added
      if(vertexIndexMap.find(id) == vertexIndexMap.end()) {
        convexHull.vertPos.push_back(combinedVerts[id]);
        vertexIndexMap[id] = convexHull.vertPos.size() - 1;
      }

      // Add the index to the triangle
      tri[i] = vertexIndexMap[id];
      i++;
    }

    convexHull.triVerts.push_back(tri);
  }

  qh_freeqhull(!qh_ALL);
  delete[] points;

  // Orient faces by right-hand rule.
  orientMesh(convexHull);

  return convexHull;
}

SimplePolygon computeConvexHull2D(const SimplePolygon& polygon1, const SimplePolygon& polygon2) {
  SimplePolygon combinedPoints = polygon1;
  combinedPoints.insert(combinedPoints.end(), polygon2.begin(), polygon2.end());

  // Convert points to coordinate array for Qhull
  int dim = 2;  // We're now in 2D
  int n = combinedPoints.size();

  coordT* points = new coordT[n*dim];
  for(int i = 0; i < n; i++) {
    points[i*dim] = combinedPoints[i].x;
    points[i*dim+1] = combinedPoints[i].y;
  }

  boolT ismalloc = false;
  char flags[] = "qhull Qt";

  int exitcode = qh_new_qhull(dim, n, points, ismalloc, flags, NULL, NULL);
  if(exitcode != 0) {
    std::cout << "Convex Hull failed! Returning first polygon." << std::endl;
    return polygon1;
  }

  // Create a new polygon for the convex hull
  SimplePolygon convexHull;

  std::map<int, int> vertexIndexMap;

  vertexT *vertex, **vertexp;
  FORALLvertices {
    int id = qh_pointid(vertex->point);

    // Check if the vertex is already added
    if(vertexIndexMap.find(id) == vertexIndexMap.end()) {
      convexHull.push_back(combinedPoints[id]);
      vertexIndexMap[id] = convexHull.size() - 1;
    }
  }

  qh_freeqhull(!qh_ALL);
  delete[] points;

  return convexHull;
}

}
