/*
 Copyright (c) 2017-2021 Leo McCormack

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
*/
/*
 * Filename:
 *     convhull_3d.h
 * Description:
 *     A header only C implementation of the 3-D quickhull algorithm.
 *     The code is largely derived from the "computational-geometry-toolbox"
 *     by George Papazafeiropoulos (c) 2014, originally distributed under
 *     the BSD (2-clause) license.
 *     To include this implementation in a project, simply add this:
 *         #define CONVHULL_3D_ENABLE
 *         #include "convhull_3d.h"
 *     By default, the algorithm uses double floating point precision. To
 *     use single precision (less accurate but quicker), also add this:
 *         #define CONVHULL_3D_USE_SINGLE_PRECISION
 *     If your project has CBLAS linked, then you can also speed things up
 *     a tad by adding this:
 *         #define CONVHULL_3D_USE_CBLAS
 *     The code is C++ compiler safe.
 *     Reference: "The Quickhull Algorithm for Convex Hull, C. Bradford
 *                 Barber, David P. Dobkin and Hannu Huhdanpaa, Geometry
 *                 Center Technical Report GCG53, July 30, 1993"
 * Dependencies:
 *     cblas (optional for speed ups, especially for very large meshes)
 *     (Available in e.g. Apple Accelerate Framework, or Intel MKL)
 * Author, date created:
 *     Leo McCormack, 02.10.2017
 */

/**********
 * PUBLIC:
 *********/

#ifndef CONVHULL_3D_INCLUDED
#define CONVHULL_3D_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

#ifdef CONVHULL_3D_USE_SINGLE_PRECISION
typedef float CH_FLOAT;
#else
typedef double CH_FLOAT;
#endif
typedef struct _ch_vertex {
  union {
    CH_FLOAT v[3];
    struct {
      CH_FLOAT x, y, z;
    };
  };
} ch_vertex;
typedef ch_vertex ch_vec3;

/* builds the 3-D convexhull */
void convhull_3d_build(                              /* input arguments */
                       ch_vertex* const in_vertices, /* vector of input
                                                        vertices; nVert x 1 */
                       const int nVert,              /* number of vertices */
                       /* output arguments */
                       int** out_faces, /* & of empty int*, output face indices;
                                           flat: nOut_faces x 3 */
                       int* nOut_faces); /* & of int, number of output face
                                            indices */

/* exports the vertices, face indices, and face normals, as an 'obj' file, ready
 * for GPU (for 3d convexhulls only) */
void convhull_3d_export_obj(                           /* input arguments */
                            ch_vertex* const vertices, /* vector of input
                                                          vertices; nVert x 1 */
                            const int nVert,           /* number of vertices */
                            int* const
                                faces, /* face indices; flat: nFaces x 3 */
                            const int nFaces, /* number of faces in hull */
                            const int
                                keepOnlyUsedVerticesFLAG, /* 0: exports
                                                             in_vertices, 1:
                                                             exports only used
                                                             vertices  */
                            char* const obj_filename); /* obj filename, WITHOUT
                                                          extension */

/* exports the vertices, face indices, and face normals, as an 'm' file, for
 * MatLab verification (for 3d convexhulls only) */
void convhull_3d_export_m(                           /* input arguments */
                          ch_vertex* const vertices, /* vector of input
                                                        vertices; nVert x 1 */
                          const int nVert,           /* number of vertices */
                          int* const faces, /* face indices; flat: nFaces x 3 */
                          const int nFaces, /* number of faces in hull */
                          char* const
                              m_filename); /* m filename, WITHOUT extension */

/* reads an 'obj' file and extracts only the vertices (for 3d convexhulls only)
 */
void extract_vertices_from_obj_file(/* input arguments */
                                    char* const
                                        obj_filename, /* obj filename, WITHOUT
                                                         extension */
                                    /* output arguments */
                                    ch_vertex**
                                        out_vertices, /* & of empty ch_vertex*,
                                                         output vertices;
                                                         out_nVert x 1 */
                                    int* out_nVert);  /* & of int, number of
                                                         vertices */

/**** NEW! ****/

/* builds the N-Dimensional convexhull of a grid of points */
void convhull_nd_build(/* input arguments */
                       CH_FLOAT* const
                           in_points,     /* Matrix of points in 'd' dimensions;
                                             FLAT: nPoints x d */
                       const int nPoints, /* number of points */
                       const int d,       /* Number of dimensions */
                       /* output arguments */
                       int** out_faces,   /* (&) output face indices; FLAT:
                                             nOut_faces x d */
                       CH_FLOAT** out_cf, /* (&) contains the coefficients of
                                             the planes (set to NULL if not
                                             wanted); FLAT: nOut_faces x d */
                       CH_FLOAT** out_df, /* (&) contains the constant terms of
                                             the planes (set to NULL if not
                                             wanted); nOut_faces x 1 */
                       int* nOut_faces); /* (&) number of output face indices */

/* Computes the Delaunay triangulation (mesh) of an arrangement of points in
 * N-dimensional space */
void delaunay_nd_mesh(/* input Arguments */
                      const float*
                          points, /* The input points; FLAT: nPoints x nd */
                      const int nPoints, /* Number of points */
                      const int nd,      /* The number of dimensions */
                      /* output Arguments */
                      int** Mesh,  /* (&) the indices defining the Delaunay
                                      triangulation of the points; FLAT: nMesh x
                                      (nd+1) */
                      int* nMesh); /* (&) Number of triangulations */

/**** CUSTOM ALLOCATOR VERSIONS ****/

/* builds the 3-D convexhull */
void convhull_3d_build_alloc(/* input arguments */
                             ch_vertex* const
                                 in_vertices, /* vector of input vertices; nVert
                                                 x 1 */
                             const int nVert, /* number of vertices */
                             /* output arguments */
                             int**
                                 out_faces,   /* & of empty int*, output face
                                                 indices; flat: nOut_faces x 3 */
                             int* nOut_faces, /* & of int, number of output face
                                                 indices */
                             void* allocator); /* & of an allocator */

/* builds the N-Dimensional convexhull of a grid of points */
void convhull_nd_build_alloc(/* input arguments */
                             CH_FLOAT* const
                                 in_points,     /* Matrix of points in 'd'
                                                   dimensions; FLAT: nPoints x d */
                             const int nPoints, /* number of points */
                             const int d,       /* Number of dimensions */
                             /* output arguments */
                             int** out_faces, /* (&) output face indices; FLAT:
                                                 nOut_faces x d */
                             CH_FLOAT**
                                 out_cf, /* (&) contains the coefficients of the
                                            planes (set to NULL if not wanted);
                                            FLAT: nOut_faces x d */
                             CH_FLOAT**
                                 out_df, /* (&) contains the constant terms of
                                            the planes (set to NULL if not
                                            wanted); nOut_faces x 1 */
                             int* nOut_faces,  /* (&) number of output face
                                                  indices */
                             void* allocator); /* & of an allocator */

/* Computes the Delaunay triangulation (mesh) of an arrangement of points in
 * N-dimensional space */
void delaunay_nd_mesh_alloc(                     /* input Arguments */
                            const float* points, /* The input points; FLAT:
                                                    nPoints x nd */
                            const int nPoints,   /* Number of points */
                            const int nd,        /* The number of dimensions */
                            /* output Arguments */
                            int** Mesh, /* (&) the indices defining the Delaunay
                                           triangulation of the points; FLAT:
                                           nMesh x (nd+1) */
                            int* nMesh, /* (&) Number of triangulations */
                            void* allocator); /* & of an allocator */

/* reads an 'obj' file and extracts only the vertices (for 3d convexhulls only)
 */
void extract_vertices_from_obj_file_alloc(/* input arguments */
                                          char* const
                                              obj_filename, /* obj filename,
                                                               WITHOUT extension
                                                             */
                                          /* output arguments */
                                          ch_vertex**
                                              out_vertices, /* & of empty
                                                               ch_vertex*,
                                                               output vertices;
                                                               out_nVert x 1 */
                                          int* out_nVert, /* & of int, number of
                                                             vertices */
                                          void* allocator); /* & of an allocator
                                                             */

#ifdef __cplusplus
} /*extern "C"*/
#endif

#endif /* CONVHULL_3D_INCLUDED */

/************
 * INTERNAL:
 ***********/

#ifdef CONVHULL_3D_ENABLE

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
#define CV_STRNCPY(a, b, c) strncpy_s(a, c + 1, b, c);
#define CV_STRCAT(a, b) strcat_s(a, sizeof(b), b);
#else
#define CV_STRNCPY(a, b, c) strncpy(a, b, c);
#define CV_STRCAT(a, b) strcat(a, b);
#endif
#ifdef CONVHULL_3D_USE_SINGLE_PRECISION
#define CH_FLT_MIN FLT_MIN
#define CH_FLT_MAX FLT_MAX
#define CH_NOISE_VAL 0.00001f
#define ch_pow powf
#define ch_sqrt sqrtf
#else
#define CH_FLT_MIN DBL_MIN
#define CH_FLT_MAX DBL_MAX
#define CH_NOISE_VAL 0.0000001
#define ch_pow pow
#define ch_sqrt sqrt
#endif
#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef ch_malloc
#define ch_malloc malloc
#endif
#ifndef ch_calloc
#define ch_calloc calloc
#endif
#ifndef ch_realloc
#define ch_realloc realloc
#endif
#ifndef ch_free
#define ch_free free
#endif
#ifndef ch_stateful_malloc
#define ch_stateful_malloc(allocator, size) ch_malloc(size)
#endif
#ifndef ch_stateful_calloc
#define ch_stateful_calloc(allocator, num, size) ch_calloc(num, size)
#endif
#ifndef ch_stateful_realloc
#define ch_stateful_realloc(allocator, ptr, size) ch_realloc(ptr, size)
#endif
#ifndef ch_stateful_free
#define ch_stateful_free(allocator, ptr) ch_free(ptr)
#endif
#ifndef ch_stateful_resize
#define ch_stateful_resize(allocator, ptr, size) \
  default_memory_resize(allocator, ptr, size)
#define CONVHULL_CREATE_DEFAULT_RESIZE 1
#endif

#define CH_MAX_NUM_FACES 50000
#define CONVHULL_3D_MAX_DIMENSIONS 3
#define CONVHULL_ND_MAX_DIMENSIONS 5

/* structs for qsort */
typedef struct float_w_idx {
  CH_FLOAT val;
  int idx;
} float_w_idx;

/* internal functions prototypes: */
static int cmp_asc_float(const void*, const void*);
static int cmp_desc_float(const void*, const void*);
static int cmp_asc_int(const void*, const void*);
static void sort_float(CH_FLOAT*, CH_FLOAT*, int*, int, int, void*);
static void sort_int(int*, int);
static ch_vec3 cross(ch_vec3*, ch_vec3*);
static CH_FLOAT det_4x4(CH_FLOAT*);
static void plane_3d(CH_FLOAT*, CH_FLOAT*, CH_FLOAT*);
static void ismember(int*, int*, int*, int, int);

/* internal functions definitions: */
#ifdef CONVHULL_CREATE_DEFAULT_RESIZE
static void* default_memory_resize(void* allocator, void* ptr, size_t size) {
  if (ptr) ch_stateful_free(allocator, ptr);
  return ch_stateful_malloc(allocator, size);
}
#endif

static int cmp_asc_float(const void* a, const void* b) {
  struct float_w_idx* a1 = (struct float_w_idx*)a;
  struct float_w_idx* a2 = (struct float_w_idx*)b;
  if ((*a1).val < (*a2).val)
    return -1;
  else if ((*a1).val > (*a2).val)
    return 1;
  else
    return 0;
}

static int cmp_desc_float(const void* a, const void* b) {
  struct float_w_idx* a1 = (struct float_w_idx*)a;
  struct float_w_idx* a2 = (struct float_w_idx*)b;
  if ((*a1).val > (*a2).val)
    return -1;
  else if ((*a1).val < (*a2).val)
    return 1;
  else
    return 0;
}

static int cmp_asc_int(const void* a, const void* b) {
  int* a1 = (int*)a;
  int* a2 = (int*)b;
  if ((*a1) < (*a2))
    return -1;
  else if ((*a1) > (*a2))
    return 1;
  else
    return 0;
}

static void sort_float(
    CH_FLOAT* in_vec,  /* vector[len] to be sorted */
    CH_FLOAT* out_vec, /* if NULL, then in_vec is sorted "in-place" */
    int* new_idices,   /* set to NULL if you don't need them */
    int len, /* number of elements in vectors, must be consistent with the input
                data */
    int descendFLAG, /* !1:ascending, 1:descending */
    void* allocator  /* (stateful) allocator */
) {
  int i;
  struct float_w_idx* data;

  data = (float_w_idx*)ch_stateful_malloc(allocator, len * sizeof(float_w_idx));
  for (i = 0; i < len; i++) {
    data[i].val = in_vec[i];
    data[i].idx = i;
  }
  if (descendFLAG)
    qsort(data, len, sizeof(data[0]), cmp_desc_float);
  else
    qsort(data, len, sizeof(data[0]), cmp_asc_float);
  for (i = 0; i < len; i++) {
    if (out_vec != NULL)
      out_vec[i] = data[i].val;
    else
      in_vec[i] = data[i].val; /* overwrite input vector */
    if (new_idices != NULL) new_idices[i] = data[i].idx;
  }
  ch_stateful_free(allocator, data);
}

static void sort_int(int* io_vec, /* vector[len] to be sorted */
                     int len      /* number of elements in vectors, must be
                                     consistent with the input data */
) {
  qsort(io_vec, len, sizeof(io_vec[0]), cmp_asc_int);
}

static ch_vec3 cross(ch_vec3* v1, ch_vec3* v2) {
  ch_vec3 cross;
  cross.x = v1->y * v2->z - v1->z * v2->y;
  cross.y = v1->z * v2->x - v1->x * v2->z;
  cross.z = v1->x * v2->y - v1->y * v2->x;
  return cross;
}

/* calculates the determinent of a 4x4 matrix */
static CH_FLOAT det_4x4(CH_FLOAT* m) {
  return m[3] * m[6] * m[9] * m[12] - m[2] * m[7] * m[9] * m[12] -
         m[3] * m[5] * m[10] * m[12] + m[1] * m[7] * m[10] * m[12] +
         m[2] * m[5] * m[11] * m[12] - m[1] * m[6] * m[11] * m[12] -
         m[3] * m[6] * m[8] * m[13] + m[2] * m[7] * m[8] * m[13] +
         m[3] * m[4] * m[10] * m[13] - m[0] * m[7] * m[10] * m[13] -
         m[2] * m[4] * m[11] * m[13] + m[0] * m[6] * m[11] * m[13] +
         m[3] * m[5] * m[8] * m[14] - m[1] * m[7] * m[8] * m[14] -
         m[3] * m[4] * m[9] * m[14] + m[0] * m[7] * m[9] * m[14] +
         m[1] * m[4] * m[11] * m[14] - m[0] * m[5] * m[11] * m[14] -
         m[2] * m[5] * m[8] * m[15] + m[1] * m[6] * m[8] * m[15] +
         m[2] * m[4] * m[9] * m[15] - m[0] * m[6] * m[9] * m[15] -
         m[1] * m[4] * m[10] * m[15] + m[0] * m[5] * m[10] * m[15];
}

/* Helper function for det_NxN()  */
static void createSubMatrix(CH_FLOAT* m, int N, int i, CH_FLOAT* sub_m) {
  int j, k;
  for (j = N, k = 0; j < N * N; j++) {
    if (j % N != i) { /* i is the index to remove */
      sub_m[k] = m[j];
      k++;
    }
  }
}

static CH_FLOAT det_NxN(CH_FLOAT* m, int d) {
  CH_FLOAT sum;
  CH_FLOAT sub_m[CONVHULL_ND_MAX_DIMENSIONS * CONVHULL_ND_MAX_DIMENSIONS];
  int sign;

  if (d == 0) return 1.0;
  sum = 0.0;
  sign = 1;
  for (int i = 0; i < d; i++) {
    createSubMatrix(m, d, i, sub_m);
    sum += sign * m[i] * det_NxN(sub_m, d - 1);
    sign *= -1;
  }
  return sum;
}

/* Calculates the coefficients of the equation of a PLANE in 3D.
 * Original Copyright (c) 2014, George Papazafeiropoulos
 * Distributed under the BSD (2-clause) license
 */
static void plane_3d(CH_FLOAT* p, CH_FLOAT* c, CH_FLOAT* d) {
  int i, j, k, l;
  int r[3];
  CH_FLOAT sign, det, norm_c;
  CH_FLOAT pdiff[2][3], pdiff_s[2][2];

  for (i = 0; i < 2; i++)
    for (j = 0; j < 3; j++) pdiff[i][j] = p[(i + 1) * 3 + j] - p[i * 3 + j];
  memset(c, 0, 3 * sizeof(CH_FLOAT));
  sign = 1.0;
  for (i = 0; i < 3; i++) r[i] = i;
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 2; j++) {
      for (k = 0, l = 0; k < 3; k++) {
        if (r[k] != i) {
          pdiff_s[j][l] = pdiff[j][k];
          l++;
        }
      }
    }
    det = pdiff_s[0][0] * pdiff_s[1][1] - pdiff_s[1][0] * pdiff_s[0][1];
    c[i] = sign * det;
    sign *= -1.0;
  }
  norm_c = (CH_FLOAT)0.0;
  for (i = 0; i < 3; i++) norm_c += (ch_pow(c[i], (CH_FLOAT)2.0));
  norm_c = ch_sqrt(norm_c);
  for (i = 0; i < 3; i++) c[i] /= norm_c;
  (*d) = (CH_FLOAT)0.0;
  for (i = 0; i < 3; i++) (*d) += -p[i] * c[i];
}

/* Calculates the coefficients of the equation of a PLANE in ND.
 * Original Copyright (c) 2014, George Papazafeiropoulos
 * Distributed under the BSD (2-clause) license
 */
static void plane_nd(const int Nd, CH_FLOAT* p, CH_FLOAT* c, CH_FLOAT* d) {
  int i, j, k, l;
  int r[CONVHULL_ND_MAX_DIMENSIONS];
  CH_FLOAT sign, det, norm_c;
  CH_FLOAT pdiff[CONVHULL_ND_MAX_DIMENSIONS - 1][CONVHULL_ND_MAX_DIMENSIONS],
      pdiff_s[(CONVHULL_ND_MAX_DIMENSIONS - 1) *
              (CONVHULL_ND_MAX_DIMENSIONS - 1)];

  if (Nd == 3) {
    plane_3d(p, c, d);
    return;
  }

  for (i = 0; i < Nd - 1; i++)
    for (j = 0; j < Nd; j++) pdiff[i][j] = p[(i + 1) * Nd + j] - p[i * Nd + j];
  memset(c, 0, Nd * sizeof(CH_FLOAT));
  sign = 1.0;
  for (i = 0; i < Nd; i++) r[i] = i;
  for (i = 0; i < Nd; i++) {
    for (j = 0; j < Nd - 1; j++) {
      for (k = 0, l = 0; k < Nd; k++) {
        if (r[k] != i) {
          pdiff_s[j * (Nd - 1) + l] = pdiff[j][k];
          l++;
        }
      }
    }
    /* Determinant 1 dimension lower */
    if (Nd == 3)
      det = pdiff_s[0 * (Nd - 1) + 0] * pdiff_s[1 * (Nd - 1) + 1] -
            pdiff_s[1 * (Nd - 1) + 0] * pdiff_s[0 * (Nd - 1) + 1];
    else if (Nd == 5)
      det = det_4x4((CH_FLOAT*)pdiff_s);
    else {
      det = det_NxN((CH_FLOAT*)pdiff_s, Nd - 1);
    }
    c[i] = sign * det;
    sign *= -1.0;
  }
  norm_c = (CH_FLOAT)0.0;
  for (i = 0; i < Nd; i++) norm_c += (ch_pow(c[i], (CH_FLOAT)2.0));
  norm_c = ch_sqrt(norm_c);
  for (i = 0; i < Nd; i++) c[i] /= norm_c;
  (*d) = (CH_FLOAT)0.0;
  for (i = 0; i < Nd; i++) (*d) += -p[i] * c[i];
}

static void ismember(int* pLeft,  /* left vector; nLeftElements x 1 */
                     int* pRight, /* right vector; nRightElements x 1 */
                     int* pOut,   /* 0, unless pRight elements are present in
                                     pLeft then 1; nLeftElements x 1 */
                     int nLeftElements, /* number of elements in pLeft */
                     int nRightElements /* number of elements in pRight */
) {
  int i, j;
  memset(pOut, 0, nLeftElements * sizeof(int));
  for (i = 0; i < nLeftElements; i++)
    for (j = 0; j < nRightElements; j++)
      if (pLeft[i] == pRight[j]) pOut[i] = 1;
}

static CH_FLOAT rnd(int x, int y) {
  // Reference(s):
  //
  // - Improvements to the canonical one-liner GLSL rand() for OpenGL ES 2.0
  //   http://byteblacksmith.com/improvements-to-the-canonical-one-liner-glsl-rand-for-opengl-es-2-0/
  //
  CH_FLOAT a = (CH_FLOAT)12.9898;
  CH_FLOAT b = (CH_FLOAT)78.233;
  CH_FLOAT c = (CH_FLOAT)43758.5453;
  CH_FLOAT dt = x * a + y * b;
#ifdef CONVHULL_3D_USE_SINGLE_PRECISION
  float sn = fmodf(dt, 3.14f);
  float intpart;
  return modff(sinf(sn) * c, &intpart);
#else
  double sn = fmod(dt, 3.14);
  double intpart;
  return modf(sin(sn) * c, &intpart);
#endif  // CONVHULL_3D_USE_SINGLE_PRECISION
}

/* A C version of the 3D quickhull matlab implementation from here:
 * https://www.mathworks.com/matlabcentral/fileexchange/48509-computational-geometry-toolbox?focused=3851550&tab=example
 * (*out_faces) is returned as NULL, if triangulation fails *
 * Original Copyright (c) 2014, George Papazafeiropoulos
 * Distributed under the BSD (2-clause) license
 * Reference: "The Quickhull Algorithm for Convex Hull, C. Bradford Barber,
 * David P. Dobkin and Hannu Huhdanpaa, Geometry Center Technical Report GCG53,
 * July 30, 1993"
 */
void convhull_3d_build(ch_vertex* const in_vertices, const int nVert,
                       int** out_faces, int* nOut_faces) {
  convhull_3d_build_alloc(in_vertices, nVert, out_faces, nOut_faces, NULL);
}

void convhull_3d_build_alloc(ch_vertex* const in_vertices, const int nVert,
                             int** out_faces, int* nOut_faces,
                             void* allocator) {
  int i, j, k, l, h;
  int nFaces, p, d;
  int *aVec, *faces;
  CH_FLOAT dfi, v, max_p, min_p;
  CH_FLOAT span[CONVHULL_3D_MAX_DIMENSIONS];
  CH_FLOAT cfi[CONVHULL_3D_MAX_DIMENSIONS];
  CH_FLOAT p_s[CONVHULL_3D_MAX_DIMENSIONS * CONVHULL_3D_MAX_DIMENSIONS];
  CH_FLOAT *points, *cf, *df;

  if (nVert <= 3 || in_vertices == NULL) {
    (*out_faces) = NULL;
    (*nOut_faces) = 0;
    return;
  }

  /* 3 dimensions. The code should theoretically work for >=2 dimensions, but
   * "plane_3d" and "det_4x4" are hardcoded for 3,
   * so would need to be rewritten */
  d = 3;

  /* Add noise to the points */
  points = (CH_FLOAT*)ch_stateful_malloc(allocator,
                                         nVert * (d + 1) * sizeof(CH_FLOAT));
  for (i = 0; i < nVert; i++) {
    for (j = 0; j < d; j++)
      points[i * (d + 1) + j] =
          in_vertices[i].v[j] +
          CH_NOISE_VAL * rnd(i, j); /* noise mitigates duplicates */
    points[i * (d + 1) + d] = 1.0f; /* add a last column of ones. Used only for
                                       determinant calculation */
  }

  /* Find the span */
  for (j = 0; j < d; j++) {
    max_p = (CH_FLOAT)-2.23e+13;
    min_p = (CH_FLOAT)2.23e+13;
    for (i = 0; i < nVert; i++) {
      max_p = MAX(max_p, points[i * (d + 1) + j]);
      min_p = MIN(min_p, points[i * (d + 1) + j]);
    }
    span[j] = max_p - min_p;
#ifndef CONVHULL_ALLOW_BUILD_IN_HIGHER_DIM
    /* If you hit this assertion error, then the input vertices do not span all
     * 3 dimensions. Therefore the convex hull could be built in less
     * dimensions. In these cases, consider reducing the dimensionality of the
     * points and calling convhull_nd_build() instead with d<3
     * You can turn this assert off using CONVHULL_ALLOW_BUILD_IN_HIGHER_DIM if
     * you still wish to build in a higher number of dimensions. */
    assert(span[j] > 0.000000001);
#endif
  }

  /* The initial convex hull is a simplex with (d+1) facets, where d is the
   * number of dimensions */
  nFaces = (d + 1);
  faces = (int*)ch_stateful_calloc(allocator, nFaces * d, sizeof(int));
  aVec = (int*)ch_stateful_malloc(allocator, nFaces * sizeof(int));
  for (i = 0; i < nFaces; i++) aVec[i] = i;

  /* Each column of cf contains the coefficients of a plane */
  cf = (CH_FLOAT*)ch_stateful_malloc(allocator, nFaces * d * sizeof(CH_FLOAT));
  df = (CH_FLOAT*)ch_stateful_malloc(allocator, nFaces * sizeof(CH_FLOAT));
  for (i = 0; i < nFaces; i++) {
    /* Set the indices of the points defining the face  */
    for (j = 0, k = 0; j < (d + 1); j++) {
      if (aVec[j] != i) {
        faces[i * d + k] = aVec[j];
        k++;
      }
    }

    /* Calculate and store the plane coefficients of the face */
    for (j = 0; j < d; j++)
      for (k = 0; k < d; k++)
        p_s[j * d + k] = points[(faces[i * d + j]) * (d + 1) + k];

    /* Calculate and store the plane coefficients of the face */
    plane_3d(p_s, cfi, &dfi);
    for (j = 0; j < d; j++) cf[i * d + j] = cfi[j];
    df[i] = dfi;
  }
  CH_FLOAT
  A[(CONVHULL_3D_MAX_DIMENSIONS + 1) * (CONVHULL_3D_MAX_DIMENSIONS + 1)];
  int fVec[CONVHULL_3D_MAX_DIMENSIONS + 1];
  int face_tmp[2];

  /* Check to make sure that faces are correctly oriented */
  int bVec[CONVHULL_3D_MAX_DIMENSIONS + 1];
  for (i = 0; i < d + 1; i++) bVec[i] = i;

  /* A contains the coordinates of the points forming a simplex */
  memset(A, 0, sizeof(A));
  for (k = 0; k < (d + 1); k++) {
    /* Get the point that is not on the current face (point p) */
    for (i = 0; i < d; i++) fVec[i] = faces[k * d + i];
    sort_int(fVec, d); /* sort ascending */
    p = k;
    for (i = 0; i < d; i++)
      for (j = 0; j < (d + 1); j++)
        A[i * (d + 1) + j] = points[(faces[k * d + i]) * (d + 1) + j];
    for (; i < (d + 1); i++)
      for (j = 0; j < (d + 1); j++)
        A[i * (d + 1) + j] = points[p * (d + 1) + j];

    /* det(A) determines the orientation of the face */
    v = det_4x4(A);

    /* Orient so that each point on the original simplex can't see the opposite
     * face */
    if (v < 0) {
      /* Reverse the order of the last two vertices to change the volume */
      for (j = 0; j < 2; j++) face_tmp[j] = faces[k * d + d - j - 1];
      for (j = 0; j < 2; j++) faces[k * d + d - j - 1] = face_tmp[1 - j];

      /* Modify the plane coefficients of the properly oriented faces */
      for (j = 0; j < d; j++) cf[k * d + j] = -cf[k * d + j];
      df[k] = -df[k];
      for (i = 0; i < d; i++)
        for (j = 0; j < (d + 1); j++)
          A[i * (d + 1) + j] = points[(faces[k * d + i]) * (d + 1) + j];
      for (; i < (d + 1); i++)
        for (j = 0; j < (d + 1); j++)
          A[i * (d + 1) + j] = points[p * (d + 1) + j];
    }
  }

  /* Coordinates of the center of the point set */
  CH_FLOAT meanp[CONVHULL_3D_MAX_DIMENSIONS];
  CH_FLOAT *absdist, *reldist, *desReldist;
  memset(meanp, 0, sizeof(meanp));
  for (i = d + 1; i < nVert; i++)
    for (j = 0; j < d; j++) meanp[j] += points[i * (d + 1) + j];
  for (j = 0; j < d; j++) meanp[j] = meanp[j] / (CH_FLOAT)(nVert - d - 1);

  /* Absolute distance of points from the center */
  absdist = (CH_FLOAT*)ch_stateful_malloc(
      allocator, (nVert - d - 1) * d * sizeof(CH_FLOAT));
  for (i = d + 1, k = 0; i < nVert; i++, k++)
    for (j = 0; j < d; j++)
      absdist[k * d + j] = (points[i * (d + 1) + j] - meanp[j]) / span[j];

  /* Relative distance of points from the center */
  reldist = (CH_FLOAT*)ch_stateful_calloc(allocator, (nVert - d - 1),
                                          sizeof(CH_FLOAT));
  desReldist = (CH_FLOAT*)ch_stateful_malloc(
      allocator, (nVert - d - 1) * sizeof(CH_FLOAT));
  for (i = 0; i < (nVert - d - 1); i++)
    for (j = 0; j < d; j++)
      reldist[i] += ch_pow(absdist[i * d + j], (CH_FLOAT)2.0);

  /* Sort from maximum to minimum relative distance */
  int num_pleft, cnt;
  int *ind, *pleft;
  ind = (int*)ch_stateful_malloc(allocator, (nVert - d - 1) * sizeof(int));
  pleft = (int*)ch_stateful_malloc(allocator, (nVert - d - 1) * sizeof(int));
  sort_float(reldist, desReldist, ind, (nVert - d - 1), 1, allocator);

  /* Initialize the vector of points left. The points with the larger relative
   distance from the center are scanned first. */
  num_pleft = (nVert - d - 1);
  for (i = 0; i < num_pleft; i++) pleft[i] = ind[i] + d + 1;

  /* Loop over all remaining points that are not deleted. Deletion of points
   occurs every #iter2del# iterations of this while loop */
  memset(A, 0, sizeof(A));

  /* cnt is equal to the points having been selected without deletion of
   nonvisible points (i.e. points inside the current convex hull) */
  cnt = 0;

  /* The main loop for the quickhull algorithm */
  CH_FLOAT detA;
  CH_FLOAT* points_cf;
  CH_FLOAT points_s[CONVHULL_3D_MAX_DIMENSIONS];
  int face_s[CONVHULL_3D_MAX_DIMENSIONS];
  int gVec[CONVHULL_3D_MAX_DIMENSIONS];
  int *visible_ind, *visible, *nonvisible_faces, *f0, *u, *horizon, *hVec, *pp,
      *hVec_mem_face;
  int num_visible_ind, num_nonvisible_faces, n_newfaces, n_realloc_faces, count,
      vis;
  int f0_sum, u_len, start, num_p, index, horizon_size1;
  int FUCKED;
  FUCKED = 0;
  /* These pointers need to be assigned NULL as they only use realloc/resize
   * (which act like malloc on a NULL pointer */
  visible = nonvisible_faces = f0 = u = horizon = hVec = pp = hVec_mem_face =
      NULL;
  nFaces = d + 1;
  visible_ind = (int*)ch_stateful_malloc(allocator, nFaces * sizeof(int));
  points_cf =
      (CH_FLOAT*)ch_stateful_malloc(allocator, nFaces * sizeof(CH_FLOAT));
  while ((num_pleft > 0)) {
    /* i is the first point of the points left */
    i = pleft[0];

    /* Delete the point selected */
    for (j = 0; j < num_pleft - 1; j++) pleft[j] = pleft[j + 1];
    num_pleft--;
    if (num_pleft == 0)
      ch_stateful_free(allocator, pleft);
    else
      pleft =
          (int*)ch_stateful_realloc(allocator, pleft, num_pleft * sizeof(int));

    /* Update point selection counter */
    cnt++;

    /* find visible faces */
    for (j = 0; j < d; j++) points_s[j] = points[i * (d + 1) + j];
    points_cf = (CH_FLOAT*)ch_stateful_realloc(allocator, points_cf,
                                               nFaces * sizeof(CH_FLOAT));
    visible_ind =
        (int*)ch_stateful_realloc(allocator, visible_ind, nFaces * sizeof(int));
#ifdef CONVHULL_3D_USE_CBLAS
#ifdef CONVHULL_3D_USE_SINGLE_PRECISION
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, nFaces, d, 1.0f,
                points_s, d, cf, d, 0.0f, points_cf, nFaces);
#else
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, nFaces, d, 1.0,
                points_s, d, cf, d, 0.0, points_cf, nFaces);
#endif
#else
    for (j = 0; j < nFaces; j++) {
      points_cf[j] = 0;
      for (k = 0; k < d; k++) points_cf[j] += points_s[k] * cf[j * d + k];
    }
#endif
    num_visible_ind = 0;
    for (j = 0; j < nFaces; j++) {
      if (points_cf[j] + df[j] > 0.0) {
        num_visible_ind++; /* will sum to 0 if none are visible */
        visible_ind[j] = 1;
      } else
        visible_ind[j] = 0;
    }
    num_nonvisible_faces = nFaces - num_visible_ind;

    /* proceed if there are any visible faces */
    if (num_visible_ind != 0) {
      /* Find visible face indices */
      visible = (int*)ch_stateful_resize(allocator, visible,
                                         num_visible_ind * sizeof(int));
      for (j = 0, k = 0; j < nFaces; j++) {
        if (visible_ind[j] == 1) {
          visible[k] = j;
          k++;
        }
      }

      /* Find nonvisible faces */
      nonvisible_faces = (int*)ch_stateful_resize(
          allocator, nonvisible_faces, num_nonvisible_faces * d * sizeof(int));
      f0 = (int*)ch_stateful_resize(allocator, f0,
                                    num_nonvisible_faces * d * sizeof(int));
      for (j = 0, k = 0; j < nFaces; j++) {
        if (visible_ind[j] == 0) {
          for (l = 0; l < d; l++)
            nonvisible_faces[k * d + l] = faces[j * d + l];
          k++;
        }
      }

      /* Create horizon (count is the number of the edges of the horizon) */
      count = 0;
      for (j = 0; j < num_visible_ind; j++) {
        /* visible face */
        vis = visible[j];
        for (k = 0; k < d; k++) face_s[k] = faces[vis * d + k];
        sort_int(face_s, d);
        ismember(nonvisible_faces, face_s, f0, num_nonvisible_faces * d, d);
        u_len = 0;

        /* u are the nonvisible faces connected to the face v, if any */
        for (k = 0; k < num_nonvisible_faces; k++) {
          f0_sum = 0;
          for (l = 0; l < d; l++) f0_sum += f0[k * d + l];
          if (f0_sum == d - 1) {
            u_len++;
            if (u_len == 1)
              u = (int*)ch_stateful_resize(allocator, u, u_len * sizeof(int));
            else
              u = (int*)ch_stateful_realloc(allocator, u, u_len * sizeof(int));
            u[u_len - 1] = k;
          }
        }
        for (k = 0; k < u_len; k++) {
          /* The boundary between the visible face v and the k(th) nonvisible
           * face connected to the face v forms part of the horizon */
          count++;
          if (count == 1)
            horizon = (int*)ch_stateful_resize(allocator, horizon,
                                               count * (d - 1) * sizeof(int));
          else
            horizon = (int*)ch_stateful_realloc(allocator, horizon,
                                                count * (d - 1) * sizeof(int));
          for (l = 0; l < d; l++) gVec[l] = nonvisible_faces[u[k] * d + l];
          for (l = 0, h = 0; l < d; l++) {
            if (f0[u[k] * d + l]) {
              horizon[(count - 1) * (d - 1) + h] = gVec[l];
              h++;
            }
          }
        }
      }
      horizon_size1 = count;
      for (j = 0, l = 0; j < nFaces; j++) {
        if (!visible_ind[j]) {
          /* Delete visible faces */
          for (k = 0; k < d; k++) faces[l * d + k] = faces[j * d + k];

          /* Delete the corresponding plane coefficients of the faces */
          for (k = 0; k < d; k++) cf[l * d + k] = cf[j * d + k];
          df[l] = df[j];
          l++;
        }
      }

      /* Update the number of faces */
      nFaces = nFaces - num_visible_ind;

      /* start is the first row of the new faces */
      start = nFaces;

      /* Add faces connecting horizon to the new point */
      n_newfaces = horizon_size1;
      n_realloc_faces = nFaces + n_newfaces;
      if (n_realloc_faces > CH_MAX_NUM_FACES)
        n_realloc_faces = CH_MAX_NUM_FACES + 1;
      faces = (int*)ch_stateful_realloc(allocator, faces,
                                        n_realloc_faces * d * sizeof(int));
      cf = (CH_FLOAT*)ch_stateful_realloc(
          allocator, cf, n_realloc_faces * d * sizeof(CH_FLOAT));
      df = (CH_FLOAT*)ch_stateful_realloc(allocator, df,
                                          n_realloc_faces * sizeof(CH_FLOAT));

      for (j = 0; j < n_newfaces; j++) {
        nFaces++;
        for (k = 0; k < d - 1; k++)
          faces[(nFaces - 1) * d + k] = horizon[j * (d - 1) + k];
        faces[(nFaces - 1) * d + (d - 1)] = i;

        /* Calculate and store appropriately the plane coefficients of the faces
         */
        for (k = 0; k < d; k++)
          for (l = 0; l < d; l++)
            p_s[k * d + l] =
                points[(faces[(nFaces - 1) * d + k]) * (d + 1) + l];
        plane_3d(p_s, cfi, &dfi);
        for (k = 0; k < d; k++) cf[(nFaces - 1) * d + k] = cfi[k];
        df[(nFaces - 1)] = dfi;
        if (nFaces > CH_MAX_NUM_FACES) {
          FUCKED = 1;
          nFaces = 0;
          break;
        }
      }

      /* Orient each new face properly */
      hVec = (int*)ch_stateful_resize(allocator, hVec, nFaces * sizeof(int));
      hVec_mem_face = (int*)ch_stateful_resize(allocator, hVec_mem_face,
                                               nFaces * sizeof(int));
      for (j = 0; j < nFaces; j++) hVec[j] = j;
      for (k = start; k < nFaces; k++) {
        for (j = 0; j < d; j++) face_s[j] = faces[k * d + j];
        sort_int(face_s, d);
        ismember(hVec, face_s, hVec_mem_face, nFaces, d);
        num_p = 0;
        for (j = 0; j < nFaces; j++)
          if (!hVec_mem_face[j]) num_p++;
        pp = (int*)ch_stateful_resize(allocator, pp, num_p * sizeof(int));
        for (j = 0, l = 0; j < nFaces; j++) {
          if (!hVec_mem_face[j]) {
            pp[l] = hVec[j];
            l++;
          }
        }
        index = 0;
        detA = 0.0;

        /* While new point is coplanar, choose another point */
        while (detA == 0.0) {
          for (j = 0; j < d; j++)
            for (l = 0; l < d + 1; l++)
              A[j * (d + 1) + l] = points[(faces[k * d + j]) * (d + 1) + l];
          for (; j < d + 1; j++)
            for (l = 0; l < d + 1; l++)
              A[j * (d + 1) + l] = points[pp[index] * (d + 1) + l];
          index++;
          detA = det_4x4(A);
        }

        /* Orient faces so that each point on the original simplex can't see the
         * opposite face */
        if (detA < 0.0) {
          /* If orientation is improper, reverse the order to change the volume
           * sign */
          for (j = 0; j < 2; j++) face_tmp[j] = faces[k * d + d - j - 1];
          for (j = 0; j < 2; j++) faces[k * d + d - j - 1] = face_tmp[1 - j];

          /* Modify the plane coefficients of the properly oriented faces */
          for (j = 0; j < d; j++) cf[k * d + j] = -cf[k * d + j];
          df[k] = -df[k];
          for (l = 0; l < d; l++)
            for (j = 0; j < d + 1; j++)
              A[l * (d + 1) + j] = points[(faces[k * d + l]) * (d + 1) + j];
          for (; l < d + 1; l++)
            for (j = 0; j < d + 1; j++)
              A[l * (d + 1) + j] = points[pp[index] * (d + 1) + j];
#ifndef NDEBUG
          /* Check */
          detA = det_4x4(A);
          /* If you hit this assertion error, then the face cannot be properly
           * orientated */
          assert(detA > 0.0);
#endif
        }
      }
    }
    if (FUCKED) {
      break;
    }
  }

  /* output */
  if (FUCKED) {
    (*out_faces) = NULL;
    (*nOut_faces) = 0;
  } else {
    (*out_faces) =
        (int*)ch_stateful_malloc(allocator, nFaces * d * sizeof(int));
    memcpy((*out_faces), faces, nFaces * d * sizeof(int));
    (*nOut_faces) = nFaces;
  }

  /* clean-up */
  ch_stateful_free(allocator, u);
  ch_stateful_free(allocator, pp);
  ch_stateful_free(allocator, horizon);
  ch_stateful_free(allocator, f0);
  ch_stateful_free(allocator, nonvisible_faces);
  ch_stateful_free(allocator, visible);
  ch_stateful_free(allocator, hVec);
  ch_stateful_free(allocator, hVec_mem_face);
  ch_stateful_free(allocator, visible_ind);
  ch_stateful_free(allocator, points_cf);
  ch_stateful_free(allocator, absdist);
  ch_stateful_free(allocator, reldist);
  ch_stateful_free(allocator, desReldist);
  ch_stateful_free(allocator, ind);
  ch_stateful_free(allocator, points);
  ch_stateful_free(allocator, faces);
  ch_stateful_free(allocator, aVec);
  ch_stateful_free(allocator, cf);
  ch_stateful_free(allocator, df);
}

void convhull_3d_export_obj(ch_vertex* const vertices, const int nVert,
                            int* const faces, const int nFaces,
                            const int keepOnlyUsedVerticesFLAG,
                            char* const obj_filename) {
  int i, j;
  char* path = (char*)calloc(strlen(obj_filename) + 1, sizeof(char));
  // char* path = new char[strlen(obj_filename) + 1]; // Allocate memory for
  // destination buffer
  strcpy(path, obj_filename);
  FILE* obj_file;
#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
  CV_STRCAT(path, ".obj");
  fopen_s(&obj_file, path, "wt");
#else
  errno = 0;
  obj_file = fopen(strcat(path, ".obj"), "wt");
#endif
  if (obj_file == NULL) {
    printf("Error %d \n", errno);
    printf("It's null");
  }
  fprintf(obj_file, "o\n");
  CH_FLOAT scale;
  ch_vec3 v1, v2, normal;

  /* export vertices */
  if (keepOnlyUsedVerticesFLAG) {
    for (i = 0; i < nFaces; i++)
      for (j = 0; j < 3; j++)
        fprintf(obj_file, "v %f %f %f\n", vertices[faces[i * 3 + j]].x,
                vertices[faces[i * 3 + j]].y, vertices[faces[i * 3 + j]].z);
  } else {
    for (i = 0; i < nVert; i++)
      fprintf(obj_file, "v %f %f %f\n", vertices[i].x, vertices[i].y,
              vertices[i].z);
  }

  /* export the face normals */
  for (i = 0; i < nFaces; i++) {
    /* calculate cross product between v1-v0 and v2-v0 */
    v1 = vertices[faces[i * 3 + 1]];
    v2 = vertices[faces[i * 3 + 2]];
    v1.x -= vertices[faces[i * 3]].x;
    v1.y -= vertices[faces[i * 3]].y;
    v1.z -= vertices[faces[i * 3]].z;
    v2.x -= vertices[faces[i * 3]].x;
    v2.y -= vertices[faces[i * 3]].y;
    v2.z -= vertices[faces[i * 3]].z;
    normal = cross(&v1, &v2);

    /* normalise to unit length */
    scale = ((CH_FLOAT)1.0) / (ch_sqrt(ch_pow(normal.x, (CH_FLOAT)2.0) +
                                       ch_pow(normal.y, (CH_FLOAT)2.0) +
                                       ch_pow(normal.z, (CH_FLOAT)2.0)) +
                               (CH_FLOAT)2.23e-9);
    normal.x *= scale;
    normal.y *= scale;
    normal.z *= scale;
    fprintf(obj_file, "vn %f %f %f\n", normal.x, normal.y, normal.z);
  }

  /* export the face indices */
  if (keepOnlyUsedVerticesFLAG) {
    for (i = 0; i < nFaces; i++) {
      /* vertices are in same order as the faces, and normals are in order */
      fprintf(obj_file, "f %u//%u %u//%u %u//%u\n", i * 3 + 1, i + 1,
              i * 3 + 1 + 1, i + 1, i * 3 + 2 + 1, i + 1);
    }
  } else {
    /* just normals are in order  */
    for (i = 0; i < nFaces; i++) {
      fprintf(obj_file, "f %u//%u %u//%u %u//%u\n", faces[i * 3] + 1, i + 1,
              faces[i * 3 + 1] + 1, i + 1, faces[i * 3 + 2] + 1, i + 1);
    }
  }
  fclose(obj_file);
}

void convhull_3d_export_m(ch_vertex* const vertices, const int nVert,
                          int* const faces, const int nFaces,
                          char* const m_filename) {
  int i;
  char path[256] = {"\0"};
  memcpy(path, m_filename, strlen(m_filename));
  FILE* m_file;
#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
  CV_STRCAT(path, ".m");
  fopen_s(&m_file, path, "wt");
#else
  m_file = fopen(strcat(path, ".m"), "wt");
#endif

  /* save face indices and vertices for verification in matlab: */
  fprintf(m_file, "vertices = [\n");
  for (i = 0; i < nVert; i++)
    fprintf(m_file, "%f, %f, %f;\n", vertices[i].x, vertices[i].y,
            vertices[i].z);
  fprintf(m_file, "];\n\n\n");
  fprintf(m_file, "faces = [\n");
  for (i = 0; i < nFaces; i++) {
    fprintf(m_file, " %u, %u, %u;\n", faces[3 * i + 0] + 1,
            faces[3 * i + 1] + 1, faces[3 * i + 2] + 1);
  }
  fprintf(m_file, "];\n\n\n");
  fclose(m_file);
}

void extract_vertices_from_obj_file(char* const obj_filename,
                                    ch_vertex** out_vertices, int* out_nVert) {
  extract_vertices_from_obj_file_alloc(obj_filename, out_vertices, out_nVert,
                                       NULL);
}

void extract_vertices_from_obj_file_alloc(char* const obj_filename,
                                          ch_vertex** out_vertices,
                                          int* out_nVert, void* allocator) {
  FILE* obj_file;
#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
  CV_STRCAT(obj_filename, ".obj");
  fopen_s(&obj_file, obj_filename, "r");
#else
  obj_file = fopen(strcat(obj_filename, ".obj"), "r");
#endif

  /* determine number of vertices */
  unsigned int nVert = 0;
  char line[256];
  while (fgets(line, sizeof(line), obj_file)) {
    char* vexists = strstr(line, "v ");
    if (vexists != NULL) nVert++;
  }
  (*out_nVert) = nVert;
  (*out_vertices) =
      (ch_vertex*)ch_stateful_malloc(allocator, nVert * sizeof(ch_vertex));

  /* extract the vertices */
  rewind(obj_file);
  int i = 0;
  int vertID, prev_char_isDigit, current_char_isDigit;
  char vert_char[256] = {0};
  while (fgets(line, sizeof(line), obj_file)) {
    char* vexists = strstr(line, "v ");
    if (vexists != NULL) {
      prev_char_isDigit = 0;
      vertID = -1;
      for (size_t j = 0; j < strlen(line) - 1; j++) {
        if (isdigit(line[j]) || line[j] == '.' || line[j] == '-' ||
            line[j] == '+' || line[j] == 'E' || line[j] == 'e') {
          vert_char[strlen(vert_char)] = line[j];
          current_char_isDigit = 1;
        } else
          current_char_isDigit = 0;
        if ((prev_char_isDigit && !current_char_isDigit) ||
            j == strlen(line) - 2) {
          vertID++;
          if (vertID > 4) {
            /* not a valid file */
            ch_stateful_free(allocator, (*out_vertices));
            (*out_vertices) = NULL;
            (*out_nVert) = 0;
            return;
          }
          (*out_vertices)[i].v[vertID] = (CH_FLOAT)atof(vert_char);
          memset(vert_char, 0, 256 * sizeof(char));
        }
        prev_char_isDigit = current_char_isDigit;
      }
      i++;
    }
  }
}

/**** NEW! ****/

/* A C version of the ND quickhull matlab implementation from here:
 * https://www.mathworks.com/matlabcentral/fileexchange/48509-computational-geometry-toolbox?focused=3851550&tab=example
 * (*out_faces) is returned as NULL, if triangulation fails *
 * Original Copyright (c) 2014, George Papazafeiropoulos
 * Distributed under the BSD (2-clause) license
 * Reference: "The Quickhull Algorithm for Convex Hull, C. Bradford Barber,
 * David P. Dobkin and Hannu Huhdanpaa, Geometry Center Technical Report GCG53,
 * July 30, 1993"
 */
void convhull_nd_build(CH_FLOAT* const in_vertices, const int nVert,
                       const int d, int** out_faces, CH_FLOAT** out_cf,
                       CH_FLOAT** out_df, int* nOut_faces) {
  convhull_nd_build_alloc(in_vertices, nVert, d, out_faces, out_cf, out_df,
                          nOut_faces, NULL);
}

void convhull_nd_build_alloc(CH_FLOAT* const in_vertices, const int nVert,
                             const int d, int** out_faces, CH_FLOAT** out_cf,
                             CH_FLOAT** out_df, int* nOut_faces,
                             void* allocator) {
  int i, j, k, l, h;
  int nFaces, p;
  int *aVec, *faces;
  CH_FLOAT dfi, v, max_p, min_p;
  CH_FLOAT span[CONVHULL_ND_MAX_DIMENSIONS];
  CH_FLOAT cfi[CONVHULL_ND_MAX_DIMENSIONS];
  CH_FLOAT p_s[CONVHULL_ND_MAX_DIMENSIONS * CONVHULL_ND_MAX_DIMENSIONS];
  CH_FLOAT *points, *cf, *df;

  assert(d <= CONVHULL_ND_MAX_DIMENSIONS);

  /* Solution not possible... */
  if (d > CONVHULL_ND_MAX_DIMENSIONS || nVert <= d || in_vertices == NULL) {
    (*out_faces) = NULL;
    (*nOut_faces) = 0;
    if (out_cf != NULL) (*out_cf) = NULL;
    if (out_df != NULL) (*out_df) = NULL;
    return;
  }

  /* Add noise to the points */
  points = (CH_FLOAT*)ch_stateful_malloc(allocator,
                                         nVert * (d + 1) * sizeof(CH_FLOAT));
  for (i = 0; i < nVert; i++) {
    for (j = 0; j < d; j++)
      points[i * (d + 1) + j] =
          in_vertices[i * d + j] + CH_NOISE_VAL * rnd(i, j);
    points[i * (d + 1) + d] = 1.0; /* add a last column of ones. Used only for
                                      determinant calculation */
  }

  /* Find the span */
  for (j = 0; j < d; j++) {
    max_p = (CH_FLOAT)-2.23e+13;
    min_p = (CH_FLOAT)2.23e+13;
    for (i = 0; i < nVert; i++) {
      max_p = MAX(max_p, points[i * (d + 1) + j]);
      min_p = MIN(min_p, points[i * (d + 1) + j]);
    }
    span[j] = max_p - min_p;
#ifndef CONVHULL_ALLOW_BUILD_IN_HIGHER_DIM
    /* If you hit this assertion error, then the input vertices do not span all
     * 'd' dimensions. Therefore the convex hull could be built in less
     * dimensions. In these cases, consider reducing the dimensionality of the
     * points and calling convhull_nd_build() with a smaller d
     * You can turn this assert off using CONVHULL_ALLOW_BUILD_IN_HIGHER_DIM if
     * you still wish to build in a higher number of dimensions. */
    assert(span[j] > 0.000000001);
#endif
  }

  /* The initial convex hull is a simplex with (d+1) facets, where d is the
   * number of dimensions */
  nFaces = (d + 1);
  faces = (int*)ch_stateful_calloc(allocator, nFaces * d, sizeof(int));
  aVec = (int*)ch_stateful_malloc(allocator, nFaces * sizeof(int));
  for (i = 0; i < nFaces; i++) aVec[i] = i;

  /* Each column of cf contains the coefficients of a plane */
  cf = (CH_FLOAT*)ch_stateful_malloc(allocator, nFaces * d * sizeof(CH_FLOAT));
  df = (CH_FLOAT*)ch_stateful_malloc(allocator, nFaces * sizeof(CH_FLOAT));
  for (i = 0; i < nFaces; i++) {
    /* Set the indices of the points defining the face  */
    for (j = 0, k = 0; j < (d + 1); j++) {
      if (aVec[j] != i) {
        faces[i * d + k] = aVec[j];
        k++;
      }
    }

    /* Calculate and store the plane coefficients of the face */
    for (j = 0; j < d; j++)
      for (k = 0; k < d; k++)
        p_s[j * d + k] = points[(faces[i * d + j]) * (d + 1) + k];

    /* Calculate and store the plane coefficients of the face */
    plane_nd(d, p_s, cfi, &dfi);
    for (j = 0; j < d; j++) cf[i * d + j] = cfi[j];
    df[i] = dfi;
  }
  CH_FLOAT
  A[(CONVHULL_ND_MAX_DIMENSIONS + 1) * (CONVHULL_ND_MAX_DIMENSIONS + 1)];
  int fVec[CONVHULL_ND_MAX_DIMENSIONS + 1];
  int face_tmp[2];

  /* Check to make sure that faces are correctly oriented */
  int bVec[CONVHULL_ND_MAX_DIMENSIONS + 1];
  for (i = 0; i < d + 1; i++) bVec[i] = i;

  /* A contains the coordinates of the points forming a simplex */
  memset(A, 0, sizeof(A));
  for (k = 0; k < (d + 1); k++) {
    /* Get the point that is not on the current face (point p) */
    for (i = 0; i < d; i++) fVec[i] = faces[k * d + i];
    sort_int(fVec, d); /* sort ascending */
    p = k;
    for (i = 0; i < d; i++)
      for (j = 0; j < (d + 1); j++)
        A[i * (d + 1) + j] = points[(faces[k * d + i]) * (d + 1) + j];
    for (; i < (d + 1); i++)
      for (j = 0; j < (d + 1); j++)
        A[i * (d + 1) + j] = points[p * (d + 1) + j];

    /* det(A) determines the orientation of the face */
    if (d == 3)
      v = det_4x4(A);
    else
      v = det_NxN(A, d + 1);

    /* Orient so that each point on the original simplex can't see the opposite
     * face */
    if (v < 0) {
      /* Reverse the order of the last two vertices to change the volume */
      for (j = 0; j < 2; j++) face_tmp[j] = faces[k * d + d - j - 1];
      for (j = 0; j < 2; j++) faces[k * d + d - j - 1] = face_tmp[1 - j];

      /* Modify the plane coefficients of the properly oriented faces */
      for (j = 0; j < d; j++) cf[k * d + j] = -cf[k * d + j];
      df[k] = -df[k];
      for (i = 0; i < d; i++)
        for (j = 0; j < (d + 1); j++)
          A[i * (d + 1) + j] = points[(faces[k * d + i]) * (d + 1) + j];
      for (; i < (d + 1); i++)
        for (j = 0; j < (d + 1); j++)
          A[i * (d + 1) + j] = points[p * (d + 1) + j];
    }
  }

  /* Coordinates of the center of the point set */
  CH_FLOAT meanp[CONVHULL_ND_MAX_DIMENSIONS];
  CH_FLOAT *reldist, *desReldist, *absdist;
  memset(meanp, 0, sizeof(meanp));
  for (i = d + 1; i < nVert; i++)
    for (j = 0; j < d; j++) meanp[j] += points[i * (d + 1) + j];
  for (j = 0; j < d; j++) meanp[j] = meanp[j] / (CH_FLOAT)(nVert - d - 1);

  /* Absolute distance of points from the center */
  absdist = (CH_FLOAT*)ch_stateful_malloc(
      allocator, (nVert - d - 1) * d * sizeof(CH_FLOAT));
  for (i = d + 1, k = 0; i < nVert; i++, k++)
    for (j = 0; j < d; j++)
      absdist[k * d + j] = (points[i * (d + 1) + j] - meanp[j]) / span[j];

  /* Relative distance of points from the center */
  reldist = (CH_FLOAT*)ch_stateful_calloc(allocator, (nVert - d - 1),
                                          sizeof(CH_FLOAT));
  desReldist = (CH_FLOAT*)ch_stateful_malloc(
      allocator, (nVert - d - 1) * sizeof(CH_FLOAT));
  for (i = 0; i < (nVert - d - 1); i++)
    for (j = 0; j < d; j++)
      reldist[i] += ch_pow(absdist[i * d + j], (CH_FLOAT)2.0);

  /* Sort from maximum to minimum relative distance */
  int num_pleft, cnt;
  int *ind, *pleft;
  ind = (int*)ch_stateful_malloc(allocator, (nVert - d - 1) * sizeof(int));
  pleft = (int*)ch_stateful_malloc(allocator, (nVert - d - 1) * sizeof(int));
  sort_float(reldist, desReldist, ind, (nVert - d - 1), 1, allocator);

  /* Initialize the vector of points left. The points with the larger relative
   distance from the center are scanned first. */
  num_pleft = (nVert - d - 1);
  for (i = 0; i < num_pleft; i++) pleft[i] = ind[i] + d + 1;

  /* Loop over all remaining points that are not deleted. Deletion of points
   occurs every #iter2del# iterations of this while loop */
  memset(A, 0, sizeof(A));

  /* cnt is equal to the points having been selected without deletion of
   nonvisible points (i.e. points inside the current convex hull) */
  cnt = 0;

  /* The main loop for the quickhull algorithm */
  CH_FLOAT detA;
  CH_FLOAT* points_cf;
  CH_FLOAT points_s[CONVHULL_ND_MAX_DIMENSIONS];
  int face_s[CONVHULL_ND_MAX_DIMENSIONS];
  int gVec[CONVHULL_ND_MAX_DIMENSIONS];
  int *visible_ind, *visible, *nonvisible_faces, *f0, *u, *horizon, *hVec, *pp,
      *hVec_mem_face;
  int num_visible_ind, num_nonvisible_faces, n_newfaces, n_realloc_faces, count,
      vis;
  int f0_sum, u_len, start, num_p, index, horizon_size1;
  int FUCKED;
  FUCKED = 0;
  /* These pointers need to be assigned NULL as they only use realloc/resize
   * (which act like malloc on a NULL pointer */
  visible = nonvisible_faces = f0 = u = horizon = hVec = pp = hVec_mem_face =
      NULL;
  nFaces = d + 1;
  visible_ind = (int*)ch_stateful_malloc(allocator, nFaces * sizeof(int));
  points_cf =
      (CH_FLOAT*)ch_stateful_malloc(allocator, nFaces * sizeof(CH_FLOAT));
  while ((num_pleft > 0)) {
    /* i is the first point of the points left */
    i = pleft[0];

    /* Delete the point selected */
    for (j = 0; j < num_pleft - 1; j++) pleft[j] = pleft[j + 1];
    num_pleft--;
    if (num_pleft == 0)
      ch_stateful_free(allocator, pleft);
    else
      pleft =
          (int*)ch_stateful_realloc(allocator, pleft, num_pleft * sizeof(int));

    /* Update point selection counter */
    cnt++;

    /* find visible faces */
    for (j = 0; j < d; j++) points_s[j] = points[i * (d + 1) + j];
    points_cf = (CH_FLOAT*)ch_stateful_realloc(allocator, points_cf,
                                               nFaces * sizeof(CH_FLOAT));
    visible_ind =
        (int*)ch_stateful_realloc(allocator, visible_ind, nFaces * sizeof(int));
#ifdef CONVHULL_3D_USE_CBLAS
#ifdef CONVHULL_3D_USE_SINGLE_PRECISION
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, nFaces, d, 1.0f,
                points_s, d, cf, d, 0.0f, points_cf, nFaces);
#else
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, nFaces, d, 1.0,
                points_s, d, cf, d, 0.0, points_cf, nFaces);
#endif
#else
    for (j = 0; j < nFaces; j++) {
      points_cf[j] = 0;
      for (k = 0; k < d; k++) points_cf[j] += points_s[k] * cf[j * d + k];
    }
#endif
    num_visible_ind = 0;
    for (j = 0; j < nFaces; j++) {
      if (points_cf[j] + df[j] > 0.0) {
        num_visible_ind++; /* will sum to 0 if none are visible */
        visible_ind[j] = 1;
      } else
        visible_ind[j] = 0;
    }
    num_nonvisible_faces = nFaces - num_visible_ind;

    /* proceed if there are any visible faces */
    if (num_visible_ind != 0) {
      /* Find visible face indices */
      visible = (int*)ch_stateful_resize(allocator, visible,
                                         num_visible_ind * sizeof(int));
      for (j = 0, k = 0; j < nFaces; j++) {
        if (visible_ind[j] == 1) {
          visible[k] = j;
          k++;
        }
      }

      /* Find nonvisible faces */
      nonvisible_faces = (int*)ch_stateful_resize(
          allocator, nonvisible_faces, num_nonvisible_faces * d * sizeof(int));
      f0 = (int*)ch_stateful_resize(allocator, f0,
                                    num_nonvisible_faces * d * sizeof(int));
      for (j = 0, k = 0; j < nFaces; j++) {
        if (visible_ind[j] == 0) {
          for (l = 0; l < d; l++)
            nonvisible_faces[k * d + l] = faces[j * d + l];
          k++;
        }
      }

      /* Create horizon (count is the number of the edges of the horizon) */
      count = 0;
      for (j = 0; j < num_visible_ind; j++) {
        /* visible face */
        vis = visible[j];
        for (k = 0; k < d; k++) face_s[k] = faces[vis * d + k];
        sort_int(face_s, d);
        ismember(nonvisible_faces, face_s, f0, num_nonvisible_faces * d, d);
        u_len = 0;

        /* u are the nonvisible faces connected to the face v, if any */
        for (k = 0; k < num_nonvisible_faces; k++) {
          f0_sum = 0;
          for (l = 0; l < d; l++) f0_sum += f0[k * d + l];
          if (f0_sum == d - 1) {
            u_len++;
            if (u_len == 1)
              u = (int*)ch_stateful_resize(allocator, u, u_len * sizeof(int));
            else
              u = (int*)ch_stateful_realloc(allocator, u, u_len * sizeof(int));
            u[u_len - 1] = k;
          }
        }
        for (k = 0; k < u_len; k++) {
          /* The boundary between the visible face v and the k(th) nonvisible
           * face connected to the face v forms part of the horizon */
          count++;
          if (count == 1)
            horizon = (int*)ch_stateful_resize(allocator, horizon,
                                               count * (d - 1) * sizeof(int));
          else
            horizon = (int*)ch_stateful_realloc(allocator, horizon,
                                                count * (d - 1) * sizeof(int));
          for (l = 0; l < d; l++) gVec[l] = nonvisible_faces[u[k] * d + l];
          for (l = 0, h = 0; l < d; l++) {
            if (f0[u[k] * d + l]) {
              horizon[(count - 1) * (d - 1) + h] = gVec[l];
              h++;
            }
          }
        }
      }
      horizon_size1 = count;
      for (j = 0, l = 0; j < nFaces; j++) {
        if (!visible_ind[j]) {
          /* Delete visible faces */
          for (k = 0; k < d; k++) faces[l * d + k] = faces[j * d + k];

          /* Delete the corresponding plane coefficients of the faces */
          for (k = 0; k < d; k++) cf[l * d + k] = cf[j * d + k];
          df[l] = df[j];
          l++;
        }
      }

      /* Update the number of faces */
      nFaces = nFaces - num_visible_ind;

      /* start is the first row of the new faces */
      start = nFaces;

      /* Add faces connecting horizon to the new point */
      n_newfaces = horizon_size1;
      n_realloc_faces = nFaces + n_newfaces;
      if (n_realloc_faces > CH_MAX_NUM_FACES)
        n_realloc_faces = CH_MAX_NUM_FACES + 1;
      faces = (int*)ch_stateful_realloc(
          allocator, faces, (nFaces + n_newfaces) * d * sizeof(int));
      cf = (CH_FLOAT*)ch_stateful_realloc(
          allocator, cf, (nFaces + n_newfaces) * d * sizeof(CH_FLOAT));
      df = (CH_FLOAT*)ch_stateful_realloc(
          allocator, df, (nFaces + n_newfaces) * sizeof(CH_FLOAT));

      for (j = 0; j < n_newfaces; j++) {
        nFaces++;
        for (k = 0; k < d - 1; k++)
          faces[(nFaces - 1) * d + k] = horizon[j * (d - 1) + k];
        faces[(nFaces - 1) * d + (d - 1)] = i;

        /* Calculate and store appropriately the plane coefficients of the faces
         */
        for (k = 0; k < d; k++)
          for (l = 0; l < d; l++)
            p_s[k * d + l] =
                points[(faces[(nFaces - 1) * d + k]) * (d + 1) + l];
        plane_nd(d, p_s, cfi, &dfi);
        for (k = 0; k < d; k++) cf[(nFaces - 1) * d + k] = cfi[k];
        df[(nFaces - 1)] = dfi;
        if (nFaces > CH_MAX_NUM_FACES) {
          FUCKED = 1;
          nFaces = 0;
          break;
        }
      }

      /* Orient each new face properly */
      hVec = (int*)ch_stateful_resize(allocator, hVec, nFaces * sizeof(int));
      hVec_mem_face = (int*)ch_stateful_resize(allocator, hVec_mem_face,
                                               nFaces * sizeof(int));
      for (j = 0; j < nFaces; j++) hVec[j] = j;
      for (k = start; k < nFaces; k++) {
        for (j = 0; j < d; j++) face_s[j] = faces[k * d + j];
        sort_int(face_s, d);
        ismember(hVec, face_s, hVec_mem_face, nFaces, d);
        num_p = 0;
        for (j = 0; j < nFaces; j++)
          if (!hVec_mem_face[j]) num_p++;
        pp = (int*)ch_stateful_resize(allocator, pp, num_p * sizeof(int));
        for (j = 0, l = 0; j < nFaces; j++) {
          if (!hVec_mem_face[j]) {
            pp[l] = hVec[j];
            l++;
          }
        }
        index = 0;
        detA = 0.0;

        /* While new point is coplanar, choose another point */
        while (detA == 0.0) {
          for (j = 0; j < d; j++)
            for (l = 0; l < d + 1; l++)
              A[j * (d + 1) + l] = points[(faces[k * d + j]) * (d + 1) + l];
          for (; j < d + 1; j++)
            for (l = 0; l < d + 1; l++)
              A[j * (d + 1) + l] = points[pp[index] * (d + 1) + l];
          index++;
          if (d == 3)
            detA = det_4x4(A);
          else
            detA = det_NxN((CH_FLOAT*)A, d + 1);
        }

        /* Orient faces so that each point on the original simplex can't see the
         * opposite face */
        if (detA < 0.0) {
          /* If orientation is improper, reverse the order to change the volume
           * sign */
          for (j = 0; j < 2; j++) face_tmp[j] = faces[k * d + d - j - 1];
          for (j = 0; j < 2; j++) faces[k * d + d - j - 1] = face_tmp[1 - j];

          /* Modify the plane coefficients of the properly oriented faces */
          for (j = 0; j < d; j++) cf[k * d + j] = -cf[k * d + j];
          df[k] = -df[k];
          for (l = 0; l < d; l++)
            for (j = 0; j < d + 1; j++)
              A[l * (d + 1) + j] = points[(faces[k * d + l]) * (d + 1) + j];
          for (; l < d + 1; l++)
            for (j = 0; j < d + 1; j++)
              A[l * (d + 1) + j] = points[pp[index] * (d + 1) + j];
#ifndef NDEBUG
          /* Check */
          if (d == 3)
            detA = det_4x4(A);
          else
            detA = det_NxN((CH_FLOAT*)A, d + 1);
          /* If you hit this assertion error, then the face cannot be properly
           * orientated and building the convex hull is likely impossible */
          assert(detA > 0.0);
#endif
        }
      }
    }
    if (FUCKED) {
      break;
    }
  }

  /* output */
  if (FUCKED) {
    (*out_faces) = NULL;
    if (out_cf != NULL) (*out_cf) = NULL;
    if (out_df != NULL) (*out_df) = NULL;
    (*nOut_faces) = 0;
  } else {
    (*out_faces) =
        (int*)ch_stateful_malloc(allocator, nFaces * d * sizeof(int));
    memcpy((*out_faces), faces, nFaces * d * sizeof(int));
    (*nOut_faces) = nFaces;
    if (out_cf != NULL) {
      (*out_cf) = (CH_FLOAT*)ch_stateful_malloc(allocator,
                                                nFaces * d * sizeof(CH_FLOAT));
      memcpy((*out_cf), cf, nFaces * d * sizeof(CH_FLOAT));
    }
    if (out_df != NULL) {
      (*out_df) =
          (CH_FLOAT*)ch_stateful_malloc(allocator, nFaces * sizeof(CH_FLOAT));
      memcpy((*out_df), df, nFaces * sizeof(CH_FLOAT));
    }
  }

  /* clean-up */
  ch_stateful_free(allocator, u);
  ch_stateful_free(allocator, pp);
  ch_stateful_free(allocator, horizon);
  ch_stateful_free(allocator, f0);
  ch_stateful_free(allocator, nonvisible_faces);
  ch_stateful_free(allocator, visible);
  ch_stateful_free(allocator, hVec);
  ch_stateful_free(allocator, hVec_mem_face);
  ch_stateful_free(allocator, visible_ind);
  ch_stateful_free(allocator, points_cf);
  ch_stateful_free(allocator, absdist);
  ch_stateful_free(allocator, reldist);
  ch_stateful_free(allocator, desReldist);
  ch_stateful_free(allocator, ind);
  ch_stateful_free(allocator, points);
  ch_stateful_free(allocator, faces);
  ch_stateful_free(allocator, aVec);
  ch_stateful_free(allocator, cf);
  ch_stateful_free(allocator, df);
}

void delaunay_nd_mesh(const float* points, const int nPoints, const int nd,
                      int** Mesh, int* nMesh) {
  delaunay_nd_mesh_alloc(points, nPoints, nd, Mesh, nMesh, NULL);
}

void delaunay_nd_mesh_alloc(const float* points, const int nPoints,
                            const int nd, int** Mesh, int* nMesh,
                            void* allocator) {
  int i, j, k, nHullFaces, maxW_idx, nVisible;
  int* hullfaces;
  CH_FLOAT w0, w_optimal, w_optimal2;
  CH_FLOAT *projpoints, *cf, *df, *p0, *p, *visible;

  /* Project the N-dimensional points onto a N+1-dimensional paraboloid */
  projpoints = (CH_FLOAT*)ch_stateful_malloc(
      allocator, nPoints * (nd + 1) * sizeof(CH_FLOAT));
  for (i = 0; i < nPoints; i++) {
    projpoints[i * (nd + 1) + nd] = 0.0;
    for (j = 0; j < nd; j++) {
      projpoints[i * (nd + 1) + j] =
          (CH_FLOAT)(points[i * nd + j] + 0.0000001 * rnd(i, j));
      projpoints[i * (nd + 1) + nd] +=
          (projpoints[i * (nd + 1) + j] *
           projpoints[i * (nd + 1) + j]); /* w vector */
    }
  }

  /* The N-dimensional delaunay triangulation requires first computing the
   * convex hull of this N+1-dimensional paraboloid */
  hullfaces = NULL;
  cf = df = NULL;
  convhull_nd_build(projpoints, nPoints, nd + 1, &hullfaces, &cf, &df,
                    &nHullFaces);

  /* Solution not possible... */
  if (nd > CONVHULL_ND_MAX_DIMENSIONS || !hullfaces || !nHullFaces) {
    (*Mesh) = NULL;
    (*nMesh) = 0;
    ch_stateful_free(allocator, projpoints);
    return;
  }

  /* Find the coordinates of the point with the maximum (N+1 dimension)
   * coordinate (i.e. the w vector) */
#ifdef CONVHULL_3D_USE_CBLAS
  if (sizeof(CH_FLOAT) == sizeof(double))
    maxW_idx = (int)cblas_idamax(nPoints, (double*)&projpoints[nd], nd + 1);
  else
    maxW_idx = (int)cblas_isamax(nPoints, (float*)&projpoints[nd], nd + 1);
#else
  CH_FLOAT maxVal;
  maxVal = (CH_FLOAT)-2.23e13;
  maxW_idx = -1;
  for (i = 0; i < nPoints; i++) {
    if (projpoints[i * (nd + 1) + nd] > maxVal) {
      maxVal = projpoints[i * (nd + 1) + nd];
      maxW_idx = i;
    }
  }
  assert(maxW_idx != -1);
#endif
  w0 = projpoints[maxW_idx * (nd + 1) + nd];
  p0 = (CH_FLOAT*)ch_stateful_malloc(allocator, nd * sizeof(CH_FLOAT));
  for (j = 0; j < nd; j++) p0[j] = projpoints[maxW_idx * (nd + 1) + j];

  /* Find the point where the plane tangent to the point (p0,w0) on the
   * paraboloid crosses the w axis. This is the point that can see the entire
   * lower hull. */
  w_optimal = 0.0;
  for (j = 0; j < nd; j++)
    w_optimal += ((CH_FLOAT)2.0 * ch_pow(p0[j], (CH_FLOAT)2.0));
  w_optimal = w0 - w_optimal;

  /* Subtract 1000 times the absolute value of w_optimal to ensure that the
   * point where the tangent plane
   * crosses the w axis will see all points on the lower hull. This avoids
   * numerical roundoff errors. */
  w_optimal2 = (CH_FLOAT)(w_optimal - 1000.0 * fabs(w_optimal));

  /* Set the point where the tangent plane crosses the w axis */
  p = (CH_FLOAT*)ch_stateful_calloc(allocator, (nd + 1), sizeof(CH_FLOAT));
  p[nd] = w_optimal2;

  /* Find all faces that are visible from this point */
  visible =
      (CH_FLOAT*)ch_stateful_malloc(allocator, nHullFaces * sizeof(CH_FLOAT));
#ifdef CONVHULL_3D_USE_CBLAS
  if (sizeof(CH_FLOAT) == sizeof(double)) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nHullFaces, 1,
                nd + 1, 1.0, (double*)cf, nd + 1, (double*)p, 1, 0.0,
                (double*)visible, 1);
  } else {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nHullFaces, 1,
                nd + 1, 1.0f, (float*)cf, nd + 1, (float*)p, 1, 0.0f,
                (float*)visible, 1);
  }
#else
  for (i = 0; i < nHullFaces; i++) {
    visible[i] = 0.0;
    for (j = 0; j < nd + 1; j++) visible[i] += cf[i * (nd + 1) + j] * p[j];
  }
#endif
  nVisible = 0;
  for (j = 0; j < nHullFaces; j++) {
    visible[j] += df[j];
    if (visible[j] > 0.0) nVisible++;
  }

  /* Output */
  (*nMesh) = nVisible;
  if (nVisible > 0) {
    (*Mesh) =
        (int*)ch_stateful_malloc(allocator, nVisible * (nd + 1) * sizeof(int));
    for (i = 0, j = 0; i < nHullFaces; i++) {
      if (visible[i] > 0.0) {
        for (k = 0; k < nd + 1; k++)
          (*Mesh)[j * (nd + 1) + k] = hullfaces[i * (nd + 1) + k];
        j++;
      }
    }
    assert(j == nVisible);
  }

  /* clean up */
  ch_stateful_free(allocator, projpoints);
  ch_stateful_free(allocator, hullfaces);
  ch_stateful_free(allocator, cf);
  ch_stateful_free(allocator, df);
  ch_stateful_free(allocator, p0);
  ch_stateful_free(allocator, p);
  ch_stateful_free(allocator, visible);
}

#endif /* CONVHULL_3D_ENABLE */