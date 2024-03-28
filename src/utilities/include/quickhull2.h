//
// LICENCE:
//  The MIT License (MIT)
//
//  Copyright (c) 2016 Karim Naaji, karim.naaji@gmail.com
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
//  sell copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE
//
// REFERENCES:
//  [1] http://box2d.org/files/GDC2014/DirkGregorius_ImplementingQuickHull.pdf
//  [2] http://www.cs.smith.edu/~orourke/books/compgeom.html
//  [3] http://www.flipcode.com/archives/The_Half-Edge_Data_Structure.shtml
//  [4] http://doc.cgal.org/latest/HalfedgeDS/index.html
//  [5] http://thomasdiewald.com/blog/?p=1888
//  [6]
//  https://fgiesen.wordpress.com/2012/02/21/half-edge-based-mesh-representations-theory/
//
// HOWTO:
//  #define QUICKHULL_IMPLEMENTATION
//  #define QUICKHULL_DEBUG // Only if assertions need to be checked
//  #include "quickhull.h"
//
// HISTORY:
//  - 1.0.1 (2016-11-01): Various improvements over epsilon issues and
//  degenerate faces
//                        Debug functionalities to test final results
//                        dynamically API to export hull meshes in OBJ files
//  - 1.0   (2016-09-10): Initial
//
// TODO:
//  - use float* from public interface
//  - reduce memory usage

#ifndef QUICKHULL_H
#define QUICKHULL_H

// ------------------------------------------------------------------------------------------------
// QUICKHULL PUBLIC API
//

typedef struct qh_vertex {
  union {
    float v[3];
    struct {
      float x;
      float y;
      float z;
    };
  };
} qh_vertex_t;

typedef qh_vertex_t qh_vec3_t;

typedef struct qh_mesh {
  qh_vertex_t* vertices;
  qh_vec3_t* normals;
  unsigned int* indices;
  unsigned int* normalindices;
  unsigned int nindices;
  unsigned int nvertices;
  unsigned int nnormals;
} qh_mesh_t;

qh_mesh_t qh_quickhull3d(qh_vertex_t const* vertices, unsigned int nvertices);

void qh_mesh_export(qh_mesh_t const* mesh, char const* filename);

void qh_free_mesh(qh_mesh_t mesh);

//
// END QUICKHULL PUBLIC API
// ------------------------------------------------------------------------------------------------

#endif  // QUICKHULL_H

#ifdef QUICKHULL_IMPLEMENTATION

#include <math.h>    // sqrt & fabs
#include <stdio.h>   // FILE
#include <string.h>  // memcpy

// Quickhull helpers, define your own if needed
#ifndef QUICKHULL_HELPERS
#include <stdlib.h>  // malloc, free, realloc
#define QUICKHULL_HELPERS 1
#define QH_MALLOC(T, N) ((T*)malloc(N * sizeof(T)))
#define QH_REALLOC(T, P, N) ((T*)realloc(P, sizeof(T) * N))
#define QH_FREE(T) free(T)
#define QH_SWAP(T, A, B) \
  {                      \
    T tmp = B;           \
    B = A;               \
    A = tmp;             \
  }
#ifdef QUICKHULL_DEBUG
#define QH_ASSERT(STMT) \
  if (!(STMT)) {        \
    *(int*)0 = 0;       \
  }
#define QH_LOG(FMT, ...) printf(FMT, ##__VA_ARGS__)
#else
#define QH_ASSERT(STMT)
#define QH_LOG(FMT, ...)
#endif  // QUICKHULL_DEBUG
#endif  // QUICKHULL_HELPERS

#ifndef QH_FLT_MAX
#define QH_FLT_MAX 1e+37F
#endif

#ifndef QH_FLT_EPS
#define QH_FLT_EPS 1E-5F
#endif

#ifndef QH_VERTEX_SET_SIZE
#define QH_VERTEX_SET_SIZE 128
#endif

typedef long qh_index_t;

typedef struct qh_half_edge {
  qh_index_t opposite_he;    // index of the opposite half edge
  qh_index_t next_he;        // index of the next half edge
  qh_index_t previous_he;    // index of the previous half edge
  qh_index_t he;             // index of the current half edge
  qh_index_t to_vertex;      // index of the next vertex
  qh_index_t adjacent_face;  // index of the ajacent face
} qh_half_edge_t;

typedef struct qh_index_set {
  qh_index_t* indices;
  unsigned int size;
  unsigned int capacity;
} qh_index_set_t;

typedef struct qh_face {
  qh_index_set_t iset;
  qh_vec3_t normal;
  qh_vertex_t centroid;
  qh_index_t edges[3];
  qh_index_t face;
  float sdist;
  int visitededges;
} qh_face_t;

typedef struct qh_index_stack {
  qh_index_t* begin;
  unsigned int size;
} qh_index_stack_t;

typedef struct qh_context {
  qh_face_t* faces;
  qh_half_edge_t* edges;
  qh_vertex_t* vertices;
  qh_vertex_t centroid;
  qh_index_stack_t facestack;
  qh_index_stack_t scratch;
  qh_index_stack_t horizonedges;
  qh_index_stack_t newhorizonedges;
  char* valid;
  unsigned int nedges;
  unsigned int nvertices;
  unsigned int nfaces;

#ifdef QUICKHULL_DEBUG
  unsigned int maxfaces;
  unsigned int maxedges;
#endif
} qh_context_t;

void qh__find_6eps(qh_vertex_t* vertices, unsigned int nvertices,
                   qh_index_t* eps) {
  qh_vertex_t* ptr = vertices;

  float minxy = +QH_FLT_MAX;
  float minxz = +QH_FLT_MAX;
  float minyz = +QH_FLT_MAX;

  float maxxy = -QH_FLT_MAX;
  float maxxz = -QH_FLT_MAX;
  float maxyz = -QH_FLT_MAX;

  int i = 0;
  for (i = 0; i < 6; ++i) {
    eps[i] = 0;
  }

  for (i = 0; i < nvertices; ++i) {
    if (ptr->z < minxy) {
      eps[0] = i;
      minxy = ptr->z;
    }
    if (ptr->y < minxz) {
      eps[1] = i;
      minxz = ptr->y;
    }
    if (ptr->x < minyz) {
      eps[2] = i;
      minyz = ptr->x;
    }
    if (ptr->z > maxxy) {
      eps[3] = i;
      maxxy = ptr->z;
    }
    if (ptr->y > maxxz) {
      eps[4] = i;
      maxxz = ptr->y;
    }
    if (ptr->x > maxyz) {
      eps[5] = i;
      maxyz = ptr->x;
    }
    ptr++;
  }
}

float qh__vertex_segment_length2(qh_vertex_t* p, qh_vertex_t* a,
                                 qh_vertex_t* b) {
  float dx = b->x - a->x;
  float dy = b->y - a->y;
  float dz = b->z - a->z;

  float d = dx * dx + dy * dy + dz * dz;

  float x = a->x;
  float y = a->y;
  float z = a->z;

  if (d != 0) {
    float t =
        ((p->x - a->x) * dx + (p->y - a->y) * dy + (p->z - a->z) * dz) / d;

    if (t > 1) {
      x = b->x;
      y = b->y;
      z = b->z;
    } else if (t > 0) {
      x += dx * t;
      y += dy * t;
      z += dz * t;
    }
  }

  dx = p->x - x;
  dy = p->y - y;
  dz = p->z - z;

  return dx * dx + dy * dy + dz * dz;
}

void qh__vec3_sub(qh_vec3_t* a, qh_vec3_t* b) {
  a->x -= b->x;
  a->y -= b->y;
  a->z -= b->z;
}

void qh__vec3_add(qh_vec3_t* a, qh_vec3_t* b) {
  a->x += b->x;
  a->y += b->y;
  a->z += b->z;
}

void qh__vec3_multiply(qh_vec3_t* a, float v) {
  a->x *= v;
  a->y *= v;
  a->z *= v;
}

int qh__vertex_equals_epsilon(qh_vertex_t* a, qh_vertex_t* b, float epsilon) {
  return fabs(a->x - b->x) <= epsilon && fabs(a->y - b->y) <= epsilon &&
         fabs(a->z - b->z) <= epsilon;
}

float qh__vec3_length2(qh_vec3_t* v) {
  return v->x * v->x + v->y * v->y + v->z * v->z;
}

float qh__vec3_dot(qh_vec3_t* v1, qh_vec3_t* v2) {
  return v1->x * v2->x + v1->y * v2->y + v1->z * v2->z;
}

void qh__vec3_normalize(qh_vec3_t* v) {
  qh__vec3_multiply(v, 1.f / sqrt(qh__vec3_length2(v)));
}

void qh__find_2dps_6eps(qh_vertex_t* vertices, qh_index_t* eps, int* ii,
                        int* jj) {
  int i, j;
  float max = -QH_FLT_MAX;

  for (i = 0; i < 6; ++i) {
    for (j = 0; j < 6; ++j) {
      qh_vertex_t d;
      float d2;

      if (i == j) {
        continue;
      }

      d = vertices[eps[i]];
      qh__vec3_sub(&d, &vertices[eps[j]]);
      d2 = qh__vec3_length2(&d);

      if (d2 > max) {
        *ii = i;
        *jj = j;
        max = d2;
      }
    }
  }
}

qh_vec3_t qh__vec3_cross(qh_vec3_t* v1, qh_vec3_t* v2) {
  qh_vec3_t cross;

  cross.x = v1->y * v2->z - v1->z * v2->y;
  cross.y = v1->z * v2->x - v1->x * v2->z;
  cross.z = v1->x * v2->y - v1->y * v2->x;

  return cross;
}

qh_vertex_t qh__face_centroid(qh_index_t vertices[3], qh_context_t* context) {
  qh_vertex_t centroid;
  int i;

  centroid.x = centroid.y = centroid.z = 0.0;
  for (i = 0; i < 3; ++i) {
    qh__vec3_add(&centroid, context->vertices + vertices[i]);
  }

  qh__vec3_multiply(&centroid, 1.0 / 3.0);

  return centroid;
}

float qh__dist_point_plane(qh_vertex_t* v, qh_vec3_t* normal, float sdist) {
  return fabs(qh__vec3_dot(v, normal) - sdist);
}

void qh__init_half_edge(qh_half_edge_t* half_edge) {
  half_edge->adjacent_face = -1;
  half_edge->he = -1;
  half_edge->next_he = -1;
  half_edge->opposite_he = -1;
  half_edge->to_vertex = -1;
  half_edge->previous_he = -1;
}

qh_half_edge_t* qh__next_edge(qh_context_t* context) {
  qh_half_edge_t* edge = context->edges + context->nedges;

  qh__init_half_edge(edge);

  edge->he = context->nedges;
  context->nedges++;

  QH_ASSERT(context->nedges < context->maxedges);

  return edge;
}

qh_face_t* qh__next_face(qh_context_t* context) {
  qh_face_t* face = context->faces + context->nfaces;

  face->face = context->nfaces;
  face->iset.indices = NULL;
  context->valid[context->nfaces] = 1;
  context->nfaces++;

  QH_ASSERT(context->nfaces < context->maxfaces);

  return face;
}

qh_vec3_t qh__edge_vec3(qh_half_edge_t* edge, qh_context_t* context) {
  qh_half_edge_t prevhe = context->edges[edge->previous_he];
  qh_vec3_t v0, v1;

  v0 = context->vertices[prevhe.to_vertex];
  v1 = context->vertices[edge->to_vertex];

  qh__vec3_sub(&v1, &v0);
  qh__vec3_normalize(&v1);

  return v1;
}

void qh__face_init(qh_face_t* face, qh_index_t vertices[3],
                   qh_context_t* context) {
  qh_half_edge_t* e0 = qh__next_edge(context);
  qh_half_edge_t* e1 = qh__next_edge(context);
  qh_half_edge_t* e2 = qh__next_edge(context);
  qh_vec3_t v0, v1;
  qh_vertex_t centroid, normal;

  e2->to_vertex = vertices[0];
  e0->to_vertex = vertices[1];
  e1->to_vertex = vertices[2];

  e0->next_he = e1->he;
  e2->previous_he = e1->he;
  face->edges[1] = e1->he;

  e1->next_he = e2->he;
  e0->previous_he = e2->he;
  face->edges[2] = e2->he;
  v1 = qh__edge_vec3(e2, context);

  e2->next_he = e0->he;
  e1->previous_he = e0->he;
  face->edges[0] = e0->he;
  v0 = qh__edge_vec3(e0, context);

  e2->adjacent_face = face->face;
  e1->adjacent_face = face->face;
  e0->adjacent_face = face->face;

  qh__vec3_multiply(&v1, -1.f);
  normal = qh__vec3_cross(&v0, &v1);

  qh__vec3_normalize(&normal);
  centroid = qh__face_centroid(vertices, context);
  face->centroid = centroid;
  face->sdist = qh__vec3_dot(&normal, &centroid);
  face->normal = normal;
  face->iset.indices = QH_MALLOC(qh_index_t, QH_VERTEX_SET_SIZE);
  face->iset.capacity = QH_VERTEX_SET_SIZE;
  face->iset.size = 0;
  face->visitededges = 0;
}

void qh__tetrahedron_basis(qh_context_t* context, qh_index_t vertices[3]) {
  qh_index_t eps[6];
  int i, j, k, l = 0;
  float max = -QH_FLT_MAX;

  qh__find_6eps(context->vertices, context->nvertices, eps);
  qh__find_2dps_6eps(context->vertices, eps, &j, &k);

  for (i = 0; i < 6; ++i) {
    float d2;

    if (i == j || i == k) {
      continue;
    }

    d2 = qh__vertex_segment_length2(context->vertices + eps[i],
                                    context->vertices + eps[j],
                                    context->vertices + eps[k]);

    if (d2 > max) {
      max = d2;
      l = i;
    }
  }

  vertices[0] = eps[j];
  vertices[1] = eps[k];
  vertices[2] = eps[l];
}

void qh__push_stack(qh_index_stack_t* stack, qh_index_t index) {
  stack->begin[stack->size] = index;
  stack->size++;
}

qh_index_t qh__pop_stack(qh_index_stack_t* stack) {
  qh_index_t top = -1;

  if (stack->size > 0) {
    top = stack->begin[stack->size - 1];
    stack->size--;
  }

  return top;
}

qh_index_t qh__furthest_point_from_plane(qh_context_t* context,
                                         qh_index_t* indices, int nindices,
                                         qh_vec3_t* normal, float sdist) {
  int i, j = 0;
  float max = -QH_FLT_MAX;

  for (i = 0; i < nindices; ++i) {
    qh_index_t index = indices ? *(indices + i) : i;
    float dist = qh__dist_point_plane(context->vertices + index, normal, sdist);

    if (dist > max) {
      j = i;
      max = dist;
    }
  }

  return j;
}

int qh__face_can_see_vertex(qh_face_t* face, qh_vertex_t* v) {
  qh_vec3_t tov = *v;

  qh__vec3_sub(&tov, &face->centroid);
  return qh__vec3_dot(&tov, &face->normal) > 0;
}

int qh__face_can_see_vertex_epsilon(qh_context_t* context, qh_face_t* face,
                                    qh_vertex_t* v, float epsilon) {
  float dot;
  qh_vec3_t tov = *v;

  qh__vec3_sub(&tov, &face->centroid);
  dot = qh__vec3_dot(&tov, &face->normal);

  if (dot > epsilon) {
    return 1;
  } else {
    dot = fabsf(dot);

    if (dot <= epsilon && dot >= 0) {
      qh_vec3_t n = face->normal;

      // allow epsilon degeneration along the face normal
      qh__vec3_multiply(&n, epsilon);
      qh__vec3_add(v, &n);

      return 1;
    }
  }

  return 0;
}

static inline void qh__assert_half_edge(qh_half_edge_t* edge,
                                        qh_context_t* context) {
  QH_ASSERT(edge->opposite_he != -1);
  QH_ASSERT(edge->he != -1);
  QH_ASSERT(edge->adjacent_face != -1);
  QH_ASSERT(edge->next_he != -1);
  QH_ASSERT(edge->previous_he != -1);
  QH_ASSERT(edge->to_vertex != -1);
  QH_ASSERT(context->edges[edge->opposite_he].to_vertex != edge->to_vertex);
}

static inline void qh__assert_face(qh_face_t* face, qh_context_t* context) {
  int i;

  for (i = 0; i < 3; ++i) {
    qh__assert_half_edge(context->edges + face->edges[i], context);
  }

  QH_ASSERT(context->valid[face->face]);
}

#ifdef QUICKHULL_DEBUG

void qh__log_face(qh_context_t* context, qh_face_t const* face) {
  QH_LOG("Face %ld:\n", face->face);
  for (int i = 0; i < 3; ++i) {
    qh_half_edge_t edge = context->edges[face->edges[i]];
    QH_LOG("\te%d %ld\n", i, edge.he);
    QH_LOG("\t\te%d.opposite_he %ld\n", i, edge.opposite_he);
    QH_LOG("\t\te%d.next_he %ld\n", i, edge.next_he);
    QH_LOG("\t\te%d.previous_he %ld\n", i, edge.previous_he);
    QH_LOG("\t\te%d.to_vertex %ld\n", i, edge.to_vertex);
    QH_LOG("\t\te%d.adjacent_face %ld\n", i, edge.adjacent_face);
  }
  QH_LOG("\tnormal %f %f %f\n", face->normal.x, face->normal.y, face->normal.z);
  QH_LOG("\tsdist %f\n", face->sdist);
  QH_LOG("\tcentroid %f %f %f\n", face->centroid.x, face->centroid.y,
         face->centroid.z);
}

#endif

int qh__test_hull(qh_context_t* context, float epsilon, int testiset) {
  unsigned int i, j, k;

  for (i = 0; i < context->nvertices; ++i) {
    qh_index_t vindex = i;
    char valid = 1;

    for (j = 0; j < context->nfaces; ++j) {
      if (!context->valid[j]) {
        continue;
      }
      qh_face_t* face = context->faces + j;

      qh_half_edge_t* e0 = context->edges + face->edges[0];
      qh_half_edge_t* e1 = context->edges + face->edges[1];
      qh_half_edge_t* e2 = context->edges + face->edges[2];

      if (e0->to_vertex == vindex || e1->to_vertex == vindex ||
          e2->to_vertex == vindex) {
        valid = 0;
        break;
      }

      if (testiset) {
        for (k = 0; k < face->iset.size; ++k) {
          if (vindex == face->iset.indices[k]) {
            valid = 0;
          }
        }
      }
    }

    if (!valid) {
      continue;
    }

    for (j = 0; j < context->nfaces; ++j) {
      if (!context->valid[j]) {
        continue;
      }
      qh_face_t* face = context->faces + j;

      qh_vertex_t vertex = context->vertices[vindex];
      qh__vec3_sub(&vertex, &face->centroid);
      if (qh__vec3_dot(&face->normal, &vertex) > epsilon) {
#ifdef QUICKHULL_DEBUG
        qh__log_face(context, face);
#endif
        return 0;
      }
    }
  }

  return 1;
}

#ifdef QUICKHULL_DEBUG
void qh__build_hull(qh_context_t* context, float epsilon, unsigned int step,
                    unsigned int* failurestep)
#else
void qh__build_hull(qh_context_t* context, float epsilon)
#endif
{
  qh_index_t topface = qh__pop_stack(&context->facestack);
  int i, j, k;

#ifdef QUICKHULL_DEBUG
  unsigned int iteration = 0;
#endif

  while (topface != -1) {
    qh_face_t* face = context->faces + topface;
    qh_index_t fvi, apex;
    qh_vertex_t* fv;
    int reversed = 0;

#ifdef QUICKHULL_DEBUG
    if (!context->valid[topface] || face->iset.size == 0 || iteration == step)
#else
    if (!context->valid[topface] || face->iset.size == 0)
#endif
    {
      topface = qh__pop_stack(&context->facestack);
      continue;
    }

#ifdef QUICKHULL_DEBUG
    if (failurestep != NULL && !qh__test_hull(context, epsilon, 1)) {
      if (*failurestep == 0) {
        *failurestep = iteration;
        break;
      }
    }

    iteration++;
#endif

    fvi = qh__furthest_point_from_plane(context, face->iset.indices,
                                        face->iset.size, &face->normal,
                                        face->sdist);
    fv = context->vertices + *(face->iset.indices + fvi);

    qh__assert_face(face, context);

    // Reset visited flag for faces
    {
      for (i = 0; i < context->nfaces; ++i) {
        context->faces[i].visitededges = 0;
      }
    }

    // Find horizon edge
    {
      qh_index_t tovisit = topface;
      qh_face_t* facetovisit = context->faces + tovisit;

      // Release scratch
      context->scratch.size = 0;

      while (tovisit != -1) {
        if (facetovisit->visitededges >= 3) {
          context->valid[tovisit] = 0;
          tovisit = qh__pop_stack(&context->scratch);
          facetovisit = context->faces + tovisit;
        } else {
          qh_index_t edgeindex = facetovisit->edges[facetovisit->visitededges];
          qh_half_edge_t* edge;
          qh_half_edge_t* oppedge;
          qh_face_t* adjface;

          facetovisit->visitededges++;

          edge = context->edges + edgeindex;
          oppedge = context->edges + edge->opposite_he;
          adjface = context->faces + oppedge->adjacent_face;

          if (!context->valid[oppedge->adjacent_face]) {
            continue;
          }

          qh__assert_half_edge(oppedge, context);
          qh__assert_half_edge(edge, context);
          qh__assert_face(adjface, context);

          if (!qh__face_can_see_vertex(adjface, fv)) {
            qh__push_stack(&context->horizonedges, edge->he);
          } else {
            context->valid[tovisit] = 0;
            qh__push_stack(&context->scratch, adjface->face);
          }
        }
      }
    }

    apex = face->iset.indices[fvi];

    // Sort horizon edges in CCW order
    {
      qh_vertex_t triangle[3];
      int vindex = 0;
      qh_vec3_t v0, v1, toapex;
      qh_vertex_t n;

      for (i = 0; i < context->horizonedges.size; ++i) {
        qh_index_t he0 = context->horizonedges.begin[i];
        qh_index_t he0vert = context->edges[he0].to_vertex;
        qh_index_t phe0 = context->edges[he0].previous_he;
        qh_index_t phe0vert = context->edges[phe0].to_vertex;

        for (j = i + 2; j < context->horizonedges.size; ++j) {
          qh_index_t he1 = context->horizonedges.begin[j];
          qh_index_t he1vert = context->edges[he1].to_vertex;
          qh_index_t phe1 = context->edges[he1].previous_he;
          qh_index_t phe1vert = context->edges[phe1].to_vertex;

          if (phe1vert == he0vert || phe0vert == he1vert) {
            QH_SWAP(qh_index_t, context->horizonedges.begin[j],
                    context->horizonedges.begin[i + 1]);
            break;
          }
        }

        if (vindex < 3) {
          triangle[vindex++] = context->vertices[context->edges[he0].to_vertex];
        }
      }

      if (vindex == 3) {
        // Detect first triangle face ordering
        v0 = triangle[0];
        v1 = triangle[2];

        qh__vec3_sub(&v0, &triangle[1]);
        qh__vec3_sub(&v1, &triangle[1]);

        n = qh__vec3_cross(&v0, &v1);

        // Get the vector to the apex
        toapex = triangle[0];

        qh__vec3_sub(&toapex, context->vertices + apex);

        reversed = qh__vec3_dot(&n, &toapex) < 0.f;
      }
    }

    // Create new faces
    {
      qh_index_t top = qh__pop_stack(&context->horizonedges);
      qh_index_t last = qh__pop_stack(&context->horizonedges);
      qh_index_t first = top;
      int looped = 0;

      QH_ASSERT(context->newhorizonedges.size == 0);

      // Release scratch
      context->scratch.size = 0;

      while (!looped) {
        qh_half_edge_t* prevhe;
        qh_half_edge_t* nexthe;
        qh_half_edge_t* oppedge;
        qh_vec3_t normal;
        qh_vertex_t fcentroid;
        qh_index_t verts[3];
        qh_face_t* newface;

        if (last == -1) {
          looped = 1;
          last = first;
        }

        prevhe = context->edges + last;
        nexthe = context->edges + top;

        if (reversed) {
          QH_SWAP(qh_half_edge_t*, prevhe, nexthe);
        }

        verts[0] = prevhe->to_vertex;
        verts[1] = nexthe->to_vertex;
        verts[2] = apex;

        context->valid[nexthe->adjacent_face] = 0;

        oppedge = context->edges + nexthe->opposite_he;
        newface = qh__next_face(context);

        qh__face_init(newface, verts, context);

        oppedge->opposite_he = context->edges[newface->edges[0]].he;
        context->edges[newface->edges[0]].opposite_he = oppedge->he;

        qh__push_stack(&context->scratch, newface->face);
        qh__push_stack(&context->newhorizonedges, newface->edges[0]);

        top = last;
        last = qh__pop_stack(&context->horizonedges);
      }
    }

    // Attach point sets to newly created faces
    {
      for (k = 0; k < context->nfaces; ++k) {
        qh_face_t* f = context->faces + k;

        if (context->valid[k] || f->iset.size == 0) {
          continue;
        }

        if (f->visitededges == 3) {
          context->valid[k] = 0;
        }

        for (i = 0; i < f->iset.size; ++i) {
          qh_index_t vertex = f->iset.indices[i];
          qh_vertex_t* v = context->vertices + vertex;
          qh_face_t* dface = NULL;

          for (j = 0; j < context->scratch.size; ++j) {
            qh_face_t* newface = context->faces + context->scratch.begin[j];
            qh_half_edge_t* e0 = context->edges + newface->edges[0];
            qh_half_edge_t* e1 = context->edges + newface->edges[1];
            qh_half_edge_t* e2 = context->edges + newface->edges[2];
            qh_vertex_t cv;

            if (e0->to_vertex == vertex || e1->to_vertex == vertex ||
                e2->to_vertex == vertex) {
              continue;
            }

            if (qh__face_can_see_vertex_epsilon(
                    context, newface, context->vertices + vertex, epsilon)) {
              dface = newface;
              break;
            }
          }

          if (dface) {
            if (dface->iset.size + 1 >= dface->iset.capacity) {
              dface->iset.capacity *= 2;
              dface->iset.indices = QH_REALLOC(qh_index_t, dface->iset.indices,
                                               dface->iset.capacity);
            }

            dface->iset.indices[dface->iset.size++] = vertex;
          }
        }

        f->iset.size = 0;
      }
    }

    // Link new faces together
    {
      for (i = 0; i < context->newhorizonedges.size; ++i) {
        qh_index_t phe0, nhe1;
        qh_half_edge_t* he0;
        qh_half_edge_t* he1;
        int ii;

        if (reversed) {
          ii = (i == 0) ? context->newhorizonedges.size - 1 : (i - 1);
        } else {
          ii = (i + 1) % context->newhorizonedges.size;
        }

        phe0 = context->edges[context->newhorizonedges.begin[i]].previous_he;
        nhe1 = context->edges[context->newhorizonedges.begin[ii]].next_he;

        he0 = context->edges + phe0;
        he1 = context->edges + nhe1;

        QH_ASSERT(he1->to_vertex == apex);
        QH_ASSERT(he0->opposite_he == -1);
        QH_ASSERT(he1->opposite_he == -1);

        he0->opposite_he = he1->he;
        he1->opposite_he = he0->he;
      }

      context->newhorizonedges.size = 0;
    }

    // Push new face to stack
    {
      for (i = 0; i < context->scratch.size; ++i) {
        qh_face_t* face = context->faces + context->scratch.begin[i];

        if (face->iset.size > 0) {
          qh__push_stack(&context->facestack, face->face);
        }
      }

      // Release scratch
      context->scratch.size = 0;
    }

    topface = qh__pop_stack(&context->facestack);

    // TODO: push all non-valid faces for reuse in face stack memory pool
  }
}

void qh_mesh_export(qh_mesh_t const* mesh, char const* filename) {
  FILE* objfile = fopen(filename, "wt");
  fprintf(objfile, "o\n");

  for (int i = 0; i < mesh->nvertices; ++i) {
    qh_vertex_t v = mesh->vertices[i];
    fprintf(objfile, "v %f %f %f\n", v.x, v.y, v.z);
  }

  for (int i = 0; i < mesh->nnormals; ++i) {
    qh_vec3_t n = mesh->normals[i];
    fprintf(objfile, "vn %f %f %f\n", n.x, n.y, n.z);
  }

  for (int i = 0, j = 0; i < mesh->nindices; i += 3, j++) {
    fprintf(objfile, "f %u/%u %u/%u %u/%u\n", mesh->indices[i + 0] + 1,
            mesh->normalindices[j] + 1, mesh->indices[i + 1] + 1,
            mesh->normalindices[j] + 1, mesh->indices[i + 2] + 1,
            mesh->normalindices[j] + 1);
  }

  fclose(objfile);
}

qh_face_t* qh__build_tetrahedron(qh_context_t* context, float epsilon) {
  int i, j;
  qh_index_t vertices[3];
  qh_index_t apex;
  qh_face_t* faces;
  qh_vertex_t normal, centroid, vapex, tcentroid;

  // Get the initial tetrahedron basis (first face)
  qh__tetrahedron_basis(context, &vertices[0]);

  // Find apex from the tetrahedron basis
  {
    float sdist;
    qh_vec3_t v0, v1;

    v0 = context->vertices[vertices[1]];
    v1 = context->vertices[vertices[2]];

    qh__vec3_sub(&v0, context->vertices + vertices[0]);
    qh__vec3_sub(&v1, context->vertices + vertices[0]);

    normal = qh__vec3_cross(&v0, &v1);
    qh__vec3_normalize(&normal);

    centroid = qh__face_centroid(vertices, context);
    sdist = qh__vec3_dot(&normal, &centroid);

    apex = qh__furthest_point_from_plane(context, NULL, context->nvertices,
                                         &normal, sdist);
    vapex = context->vertices[apex];

    qh__vec3_sub(&vapex, &centroid);

    // Whether the face is looking towards the apex
    if (qh__vec3_dot(&vapex, &normal) > 0) {
      QH_SWAP(qh_index_t, vertices[1], vertices[2]);
    }
  }

  faces = qh__next_face(context);
  qh__face_init(&faces[0], vertices, context);

  // Build faces from the tetrahedron basis to the apex
  {
    qh_index_t facevertices[3];
    for (i = 0; i < 3; ++i) {
      qh_half_edge_t* edge = context->edges + faces[0].edges[i];
      qh_half_edge_t prevedge = context->edges[edge->previous_he];
      qh_face_t* face = faces + i + 1;
      qh_half_edge_t* e0;

      facevertices[0] = edge->to_vertex;
      facevertices[1] = prevedge.to_vertex;
      facevertices[2] = apex;

      qh__next_face(context);
      qh__face_init(face, facevertices, context);

      e0 = context->edges + faces[i + 1].edges[0];
      edge->opposite_he = e0->he;
      e0->opposite_he = edge->he;
    }
  }

  // Attach half edges to faces tied to the apex
  {
    for (i = 0; i < 3; ++i) {
      qh_face_t* face;
      qh_face_t* nextface;
      qh_half_edge_t* e1;
      qh_half_edge_t* e2;

      j = (i + 2) % 3;

      face = faces + i + 1;
      nextface = faces + j + 1;

      e1 = context->edges + face->edges[1];
      e2 = context->edges + nextface->edges[2];

      QH_ASSERT(e1->opposite_he == -1);
      QH_ASSERT(e2->opposite_he == -1);

      e1->opposite_he = e2->he;
      e2->opposite_he = e1->he;

      qh__assert_half_edge(e1, context);
      qh__assert_half_edge(e2, context);
    }
  }

  // Create initial point set; every point is
  // attached to the first face it can see
  {
    for (i = 0; i < context->nvertices; ++i) {
      qh_vertex_t* v;
      qh_face_t* dface = NULL;

      if (vertices[0] == i || vertices[1] == i || vertices[2] == i) {
        continue;
      }

      for (j = 0; j < 4; ++j) {
        if (qh__face_can_see_vertex_epsilon(context, context->faces + j,
                                            context->vertices + i, epsilon)) {
          dface = context->faces + j;
          break;
        }
      }

      if (dface) {
        int valid = 1;

        for (int j = 0; j < 3; ++j) {
          qh_half_edge_t* e = context->edges + dface->edges[j];
          if (i == e->to_vertex) {
            valid = 0;
            break;
          }
        }

        if (!valid) {
          continue;
        }

        if (dface->iset.size + 1 >= dface->iset.capacity) {
          dface->iset.capacity *= 2;
          dface->iset.indices =
              QH_REALLOC(qh_index_t, dface->iset.indices, dface->iset.capacity);
        }

        dface->iset.indices[dface->iset.size++] = i;
      }
    }
  }

  // Add initial tetrahedron faces to the face stack
  tcentroid.x = tcentroid.y = tcentroid.z = 0.0;
  for (i = 0; i < 4; ++i) {
    context->valid[i] = 1;
    qh__assert_face(context->faces + i, context);
    qh__push_stack(&context->facestack, i);
    qh__vec3_add(&tcentroid, &context->faces[i].centroid);
  }

  // Assign the tetrahedron centroid
  qh__vec3_multiply(&tcentroid, 0.25);
  context->centroid = tcentroid;

  QH_ASSERT(context->nedges == context->nfaces * 3);
  QH_ASSERT(context->nfaces == 4);
  QH_ASSERT(context->facestack.size == 4);

  return faces;
}

void qh__remove_vertex_duplicates(qh_context_t* context, float epsilon) {
  int i, j, k;
  for (i = 0; i < context->nvertices; ++i) {
    qh_vertex_t* v = context->vertices + i;
    if (v->x == 0) v->x = 0;
    if (v->y == 0) v->y = 0;
    if (v->z == 0) v->z = 0;
    for (j = i + 1; j < context->nvertices; ++j) {
      if (qh__vertex_equals_epsilon(context->vertices + i,
                                    context->vertices + j, epsilon)) {
        for (k = j; k < context->nvertices - 1; ++k) {
          context->vertices[k] = context->vertices[k + 1];
        }
        context->nvertices--;
      }
    }
  }
}

void qh__init_context(qh_context_t* context, qh_vertex_t const* vertices,
                      unsigned int nvertices) {
  // TODO:
  // size_t nedges = 3 * nvertices - 6;
  // size_t nfaces = 2 * nvertices - 4;
  unsigned int nfaces = nvertices * (nvertices - 1);
  unsigned int nedges = nfaces * 3;

  context->edges = QH_MALLOC(qh_half_edge_t, nedges);
  context->faces = QH_MALLOC(qh_face_t, nfaces);
  context->facestack.begin = QH_MALLOC(qh_index_t, nfaces);
  context->scratch.begin = QH_MALLOC(qh_index_t, nfaces);
  context->horizonedges.begin = QH_MALLOC(qh_index_t, nedges);
  context->newhorizonedges.begin = QH_MALLOC(qh_index_t, nedges);
  context->valid = QH_MALLOC(char, nfaces);

  context->vertices = QH_MALLOC(qh_vertex_t, nvertices);
  memcpy(context->vertices, vertices, sizeof(qh_vertex_t) * nvertices);

  context->nvertices = nvertices;
  context->nedges = 0;
  context->nfaces = 0;
  context->facestack.size = 0;
  context->scratch.size = 0;
  context->horizonedges.size = 0;
  context->newhorizonedges.size = 0;

#ifdef QUICKHULL_DEBUG
  context->maxfaces = nfaces;
  context->maxedges = nedges;
#endif
}

void qh__free_context(qh_context_t* context) {
  int i;

  for (i = 0; i < context->nfaces; ++i) {
    QH_FREE(context->faces[i].iset.indices);
    context->faces[i].iset.size = 0;
  }

  context->nvertices = 0;
  context->nfaces = 0;

  QH_FREE(context->edges);

  QH_FREE(context->faces);
  QH_FREE(context->facestack.begin);
  QH_FREE(context->scratch.begin);
  QH_FREE(context->horizonedges.begin);
  QH_FREE(context->newhorizonedges.begin);
  QH_FREE(context->vertices);
  QH_FREE(context->valid);
}

void qh_free_mesh(qh_mesh_t mesh) {
  QH_FREE(mesh.vertices);
  QH_FREE(mesh.indices);
  QH_FREE(mesh.normalindices);
  QH_FREE(mesh.normals);
}

float qh__compute_epsilon(qh_vertex_t const* vertices, unsigned int nvertices) {
  float epsilon;
  unsigned int i;

  float maxxi = -QH_FLT_MAX;
  float maxyi = -QH_FLT_MAX;

  for (i = 0; i < nvertices; ++i) {
    float fxi = fabsf(vertices[i].x);
    float fyi = fabsf(vertices[i].y);

    if (fxi > maxxi) {
      maxxi = fxi;
    }
    if (fyi > maxyi) {
      maxyi = fyi;
    }
  }

  epsilon = 2 * (maxxi + maxyi) * QH_FLT_EPS;

  return epsilon;
}

qh_mesh_t qh_quickhull3d(qh_vertex_t const* vertices, unsigned int nvertices) {
  qh_mesh_t m;
  qh_context_t context;
  unsigned int* indices;
  unsigned int nfaces = 0, i, index, nindices;
  float epsilon;

  epsilon = qh__compute_epsilon(vertices, nvertices);

  qh__init_context(&context, vertices, nvertices);

  qh__remove_vertex_duplicates(&context, epsilon);

  // Build the initial tetrahedron
  qh__build_tetrahedron(&context, epsilon);

// Build the convex hull
#ifdef QUICKHULL_DEBUG
  qh__build_hull(&context, epsilon, -1, NULL);
#else
  qh__build_hull(&context, epsilon);
#endif

  // QH_ASSERT(qh__test_hull(&context, epsilon));

  for (i = 0; i < context.nfaces; ++i) {
    if (context.valid[i]) {
      nfaces++;
    }
  }

  nindices = nfaces * 3;

  m.normals = QH_MALLOC(qh_vertex_t, nfaces);
  m.normalindices = QH_MALLOC(unsigned int, nfaces);
  m.vertices = QH_MALLOC(qh_vertex_t, nindices);
  m.indices = QH_MALLOC(unsigned int, nindices);
  m.nindices = nindices;
  m.nnormals = nfaces;
  m.nvertices = 0;

  {
    index = 0;
    for (i = 0; i < context.nfaces; ++i) {
      if (!context.valid[i]) {
        continue;
      }
      m.normals[index] = context.faces[i].normal;
      index++;
    }

    index = 0;
    for (i = 0; i < context.nfaces; ++i) {
      if (!context.valid[i]) {
        continue;
      }
      m.normalindices[index] = index;
      index++;
    }

    index = 0;
    for (i = 0; i < context.nfaces; ++i) {
      if (!context.valid[i]) {
        continue;
      }
      m.indices[index + 0] = context.faces[i].iset.indices[0];
      // m.indices[index + 0] = context.facestack.begin[index + 0];
      // m.indices[index + 1] = context.facestack.begin[index + 1];
      // m.indices[index + 2] = context.facestack.begin[index + 2];
      m.indices[index + 1] = context.faces[i].iset.indices[1];
      m.indices[index + 2] = context.faces[i].iset.indices[2];
      index += 3;
    }

    for (i = 0; i < context.nfaces; ++i) {
      if (!context.valid[i]) {
        continue;
      }
      qh_half_edge_t e0 = context.edges[context.faces[i].edges[0]];
      qh_half_edge_t e1 = context.edges[context.faces[i].edges[1]];
      qh_half_edge_t e2 = context.edges[context.faces[i].edges[2]];

      m.vertices[m.nvertices++] = context.vertices[e0.to_vertex];
      m.vertices[m.nvertices++] = context.vertices[e1.to_vertex];
      m.vertices[m.nvertices++] = context.vertices[e2.to_vertex];
    }
  }

  qh__free_context(&context);

  return m;
}

#endif  // QUICKHULL_IMPLEMENTATION