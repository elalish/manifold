// MIT License

// Copyright (c) 2019 wi-re
// Copyright 2023 The Manifold Authors.

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Modified from https://github.com/wi-re/tbtSVD, removing CUDA dependence and
// approximate inverse square roots.

#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>

namespace SVD {
// Constants used for calculation of givens quaternions
constexpr float _gamma = 5.828427124f;   // sqrt(8)+3;
constexpr float _cStar = 0.923879532f;   // cos(pi/8)
constexpr float _sStar = 0.3826834323f;  // sin(pi/8)
// Threshold value
constexpr float _SVD_EPSILON = 1e-6f;
// Iteration counts for Jacobi Eigen Analysis, influence precision
constexpr int JACOBI_STEPS = 12;

// Helper function used to swap X with Y and Y with  X if c == true
void condSwap(bool c, float& X, float& Y) {
  float Z = X;
  X = c ? Y : X;
  Y = c ? Z : Y;
}
// Helper function used to swap X with Y and Y with -X if c == true
void condNegSwap(bool c, float& X, float& Y) {
  float Z = -X;
  X = c ? Y : X;
  Y = c ? Z : Y;
}
// Helper class to contain a quaternion. Could be replaced with float4 (CUDA
// based type) but this might lead to unintended conversions when using the
// supplied matrices
struct quaternion {
  float x = 0.f, y = 0.f, z = 0.f, w = 1.f;
  float& operator[](int32_t arg) { return ((float*)this)[arg]; }
};
// A simple 3x3 Matrix class
struct Mat3x3 {
  float m_00 = 1.f, m_01 = 0.f, m_02 = 0.f;
  float m_10 = 0.f, m_11 = 1.f, m_12 = 0.f;
  float m_20 = 0.f, m_21 = 0.f, m_22 = 1.f;
  static Mat3x3 fromPtr(float* ptr, int32_t i, int32_t offset) {
    return Mat3x3{
        ptr[i + 0 * offset], ptr[i + 1 * offset], ptr[i + 2 * offset],
        ptr[i + 3 * offset], ptr[i + 4 * offset], ptr[i + 5 * offset],
        ptr[i + 6 * offset], ptr[i + 7 * offset], ptr[i + 8 * offset]};
  }
  auto det() const {
    return fmaf(m_00, fmaf(m_11, m_22, -m_21 * m_12),
                fmaf(-m_01, fmaf(m_10, m_22, -m_20 * m_12),
                     m_02 * fmaf(m_10, m_21, -m_20 * m_11)));
  }
  void toPtr(float* ptr, int32_t i, int32_t offset) const {
    ptr[i + 0 * offset] = m_00;
    ptr[i + 1 * offset] = m_01;
    ptr[i + 2 * offset] = m_02;
    ptr[i + 3 * offset] = m_10;
    ptr[i + 4 * offset] = m_11;
    ptr[i + 5 * offset] = m_12;
    ptr[i + 6 * offset] = m_20;
    ptr[i + 7 * offset] = m_21;
    ptr[i + 8 * offset] = m_22;
  }
  Mat3x3(float a11 = 1.f, float a12 = 0.f, float a13 = 0.f, float a21 = 0.f,
         float a22 = 1.f, float a23 = 0.f, float a31 = 0.f, float a32 = 0.f,
         float a33 = 1.f)
      : m_00(a11),
        m_01(a12),
        m_02(a13),
        m_10(a21),
        m_11(a22),
        m_12(a23),
        m_20(a31),
        m_21(a32),
        m_22(a33) {}
  Mat3x3(const quaternion& q) {
    m_00 = 1.f - 2.f * (fmaf(q.y, q.y, q.z * q.z));
    m_01 = 2 * fmaf(q.x, q.y, -q.w * q.z);
    m_02 = 2 * fmaf(q.x, q.z, q.w * q.y);
    m_10 = 2.f * fmaf(q.x, q.y, +q.w * q.z);
    m_11 = 1 - 2 * fmaf(q.x, q.x, q.z * q.z);
    m_12 = 2 * fmaf(q.y, q.z, -q.w * q.x);
    m_20 = 2.f * fmaf(q.x, q.z, -q.w * q.y);
    m_21 = 2 * fmaf(q.y, q.z, q.w * q.x);
    m_22 = 1 - 2 * fmaf(q.x, q.x, q.y * q.y);
  }
  Mat3x3 transpose() const {
    return Mat3x3{m_00, m_10, m_20, m_01, m_11, m_21, m_02, m_12, m_22};
  }
  Mat3x3 operator*(const float& o) const {
    return Mat3x3{
        m_00 * o, m_01 * o, m_02 * o, m_10 * o, m_11 * o,
        m_12 * o, m_20 * o, m_21 * o, m_22 * o,
    };
  }
  Mat3x3& operator*=(const float& o) {
    m_00 *= o;
    m_01 *= o;
    m_02 *= o;
    m_10 *= o;
    m_11 *= o;
    m_12 *= o;
    m_20 *= o;
    m_21 *= o;
    m_22 *= o;
    return *this;
  }
  Mat3x3 operator-(const Mat3x3& o) const {
    return Mat3x3{m_00 - o.m_00, m_01 - o.m_01, m_02 - o.m_02,
                  m_10 - o.m_10, m_11 - o.m_11, m_12 - o.m_12,
                  m_20 - o.m_20, m_21 - o.m_21, m_22 - o.m_22};
  }
  Mat3x3 operator*(const Mat3x3& o) const {
    return Mat3x3{fmaf(m_00, o.m_00, fmaf(m_01, o.m_10, m_02 * o.m_20)),
                  fmaf(m_00, o.m_01, fmaf(m_01, o.m_11, m_02 * o.m_21)),
                  fmaf(m_00, o.m_02, fmaf(m_01, o.m_12, m_02 * o.m_22)),
                  fmaf(m_10, o.m_00, fmaf(m_11, o.m_10, m_12 * o.m_20)),
                  fmaf(m_10, o.m_01, fmaf(m_11, o.m_11, m_12 * o.m_21)),
                  fmaf(m_10, o.m_02, fmaf(m_11, o.m_12, m_12 * o.m_22)),
                  fmaf(m_20, o.m_00, fmaf(m_21, o.m_10, m_22 * o.m_20)),
                  fmaf(m_20, o.m_01, fmaf(m_21, o.m_11, m_22 * o.m_21)),
                  fmaf(m_20, o.m_02, fmaf(m_21, o.m_12, m_22 * o.m_22))};
  }
};
// A simple symmetric 3x3 Matrix class (contains no storage for (0, 1) (0, 2)
// and (1, 2)
struct Symmetric3x3 {
  float m_00 = 1.f;
  float m_10 = 0.f, m_11 = 1.f;
  float m_20 = 0.f, m_21 = 0.f, m_22 = 1.f;

  Symmetric3x3(float a11 = 1.f, float a21 = 0.f, float a22 = 1.f,
               float a31 = 0.f, float a32 = 0.f, float a33 = 1.f)
      : m_00(a11), m_10(a21), m_11(a22), m_20(a31), m_21(a32), m_22(a33) {}
  Symmetric3x3(Mat3x3 o)
      : m_00(o.m_00),
        m_10(o.m_10),
        m_11(o.m_11),
        m_20(o.m_20),
        m_21(o.m_21),
        m_22(o.m_22) {}
};
// Helper struct to store 2 floats to avoid OUT parameters on functions
struct givens {
  float ch = _cStar;
  float sh = _sStar;
};
// Helper struct to store 2 Matrices to avoid OUT parameters on functions
struct QR {
  Mat3x3 Q;
  Mat3x3 R;
};
// Helper struct to store 3 Matrices to avoid OUT parameters on functions
struct SVDSet {
  Mat3x3 U, S, V;
};
// Calculates the squared norm of the vector [x y z] using a standard scalar
// product d = x * x + y *y + z * z
float dist2(float x, float y, float z) { return fmaf(x, x, fmaf(y, y, z * z)); }
// For an explanation of the math see
// http://pages.cs.wisc.edu/~sifakis/papers/SVD_TR1690.pdf Computing the
// Singular Value Decomposition of 3 x 3 matrices with minimal branching and
// elementary floating point operations See Algorithm 2 in reference. Given a
// matrix A this function returns the givens quaternion (x and w component, y
// and z are 0)
givens approximateGivensQuaternion(Symmetric3x3& A) {
  givens g{2.f * (A.m_00 - A.m_11), A.m_10};
  bool b = _gamma * g.sh * g.sh < g.ch * g.ch;
  float w = 1.f / sqrt(fmaf(g.ch, g.ch, g.sh * g.sh));
  if (w != w) b = 0;
  return givens{b ? w * g.ch : (float)_cStar, b ? w * g.sh : (float)_sStar};
}
// Function used to apply a givens rotation S. Calculates the weights and
// updates the quaternion to contain the cumulative rotation
void jacobiConjugation(const int32_t x, const int32_t y, const int32_t z,
                       Symmetric3x3& S, quaternion& q) {
  auto g = approximateGivensQuaternion(S);
  float scale = 1.f / fmaf(g.ch, g.ch, g.sh * g.sh);
  float a = fmaf(g.ch, g.ch, -g.sh * g.sh) * scale;
  float b = 2.f * g.sh * g.ch * scale;
  Symmetric3x3 _S = S;
  // perform conjugation S = Q'*S*Q
  S.m_00 = fmaf(a, fmaf(a, _S.m_00, b * _S.m_10),
                b * (fmaf(a, _S.m_10, b * _S.m_11)));
  S.m_10 = fmaf(a, fmaf(-b, _S.m_00, a * _S.m_10),
                b * (fmaf(-b, _S.m_10, a * _S.m_11)));
  S.m_11 = fmaf(-b, fmaf(-b, _S.m_00, a * _S.m_10),
                a * (fmaf(-b, _S.m_10, a * _S.m_11)));
  S.m_20 = fmaf(a, _S.m_20, b * _S.m_21);
  S.m_21 = fmaf(-b, _S.m_20, a * _S.m_21);
  S.m_22 = _S.m_22;
  // update cumulative rotation qV
  float tmp[3];
  tmp[0] = q[0] * g.sh;
  tmp[1] = q[1] * g.sh;
  tmp[2] = q[2] * g.sh;
  g.sh *= q[3];
  // (x,y,z) corresponds to ((0,1,2),(1,2,0),(2,0,1)) for (p,q) =
  // ((0,1),(1,2),(0,2))
  q[z] = fmaf(q[z], g.ch, g.sh);
  q[3] = fmaf(q[3], g.ch, -tmp[z]);  // w
  q[x] = fmaf(q[x], g.ch, tmp[y]);
  q[y] = fmaf(q[y], g.ch, -tmp[x]);
  // re-arrange matrix for next iteration
  _S.m_00 = S.m_11;
  _S.m_10 = S.m_21;
  _S.m_11 = S.m_22;
  _S.m_20 = S.m_10;
  _S.m_21 = S.m_20;
  _S.m_22 = S.m_00;
  S.m_00 = _S.m_00;
  S.m_10 = _S.m_10;
  S.m_11 = _S.m_11;
  S.m_20 = _S.m_20;
  S.m_21 = _S.m_21;
  S.m_22 = _S.m_22;
}
// Function used to contain the givens permutations and the loop of the jacobi
// steps controlled by JACOBI_STEPS Returns the quaternion q containing the
// cumulative result used to reconstruct S
quaternion jacobiEigenAnalysis(Symmetric3x3 S) {
  quaternion q;
  for (int32_t i = 0; i < JACOBI_STEPS; i++) {
    jacobiConjugation(0, 1, 2, S, q);
    jacobiConjugation(1, 2, 0, S, q);
    jacobiConjugation(2, 0, 1, S, q);
  }
  return q;
}
// Implementation of Algorithm 3
void sortSingularValues(Mat3x3& B, Mat3x3& V) {
  float rho1 = dist2(B.m_00, B.m_10, B.m_20);
  float rho2 = dist2(B.m_01, B.m_11, B.m_21);
  float rho3 = dist2(B.m_02, B.m_12, B.m_22);
  bool c;
  c = rho1 < rho2;
  condNegSwap(c, B.m_00, B.m_01);
  condNegSwap(c, V.m_00, V.m_01);
  condNegSwap(c, B.m_10, B.m_11);
  condNegSwap(c, V.m_10, V.m_11);
  condNegSwap(c, B.m_20, B.m_21);
  condNegSwap(c, V.m_20, V.m_21);
  condSwap(c, rho1, rho2);
  c = rho1 < rho3;
  condNegSwap(c, B.m_00, B.m_02);
  condNegSwap(c, V.m_00, V.m_02);
  condNegSwap(c, B.m_10, B.m_12);
  condNegSwap(c, V.m_10, V.m_12);
  condNegSwap(c, B.m_20, B.m_22);
  condNegSwap(c, V.m_20, V.m_22);
  condSwap(c, rho1, rho3);
  c = rho2 < rho3;
  condNegSwap(c, B.m_01, B.m_02);
  condNegSwap(c, V.m_01, V.m_02);
  condNegSwap(c, B.m_11, B.m_12);
  condNegSwap(c, V.m_11, V.m_12);
  condNegSwap(c, B.m_21, B.m_22);
  condNegSwap(c, V.m_21, V.m_22);
}
// Implementation of Algorithm 4
givens QRGivensQuaternion(float a1, float a2) {
  // a1 = pivot point on diagonal
  // a2 = lower triangular entry we want to annihilate
  float epsilon = (float)_SVD_EPSILON;
  float rho = sqrt(fmaf(a1, a1, +a2 * a2));
  givens g{fabsf(a1) + fmaxf(rho, epsilon), rho > epsilon ? a2 : 0};
  bool b = a1 < 0.f;
  condSwap(b, g.sh, g.ch);
  float w = 1.f / sqrt(fmaf(g.ch, g.ch, g.sh * g.sh));
  g.ch *= w;
  g.sh *= w;
  return g;
}
// Implements a QR decomposition of a Matrix, see Sec 4.2
QR QRDecomposition(Mat3x3& B) {
  Mat3x3 Q, R;
  // first givens rotation (ch,0,0,sh)
  auto g1 = QRGivensQuaternion(B.m_00, B.m_10);
  auto a = fmaf(-2.f, g1.sh * g1.sh, 1.f);
  auto b = 2.f * g1.ch * g1.sh;
  // apply B = Q' * B
  R.m_00 = fmaf(a, B.m_00, b * B.m_10);
  R.m_01 = fmaf(a, B.m_01, b * B.m_11);
  R.m_02 = fmaf(a, B.m_02, b * B.m_12);
  R.m_10 = fmaf(-b, B.m_00, a * B.m_10);
  R.m_11 = fmaf(-b, B.m_01, a * B.m_11);
  R.m_12 = fmaf(-b, B.m_02, a * B.m_12);
  R.m_20 = B.m_20;
  R.m_21 = B.m_21;
  R.m_22 = B.m_22;
  // second givens rotation (ch,0,-sh,0)
  auto g2 = QRGivensQuaternion(R.m_00, R.m_20);
  a = fmaf(-2.f, g2.sh * g2.sh, 1.f);
  b = 2.f * g2.ch * g2.sh;
  // apply B = Q' * B;
  B.m_00 = fmaf(a, R.m_00, b * R.m_20);
  B.m_01 = fmaf(a, R.m_01, b * R.m_21);
  B.m_02 = fmaf(a, R.m_02, b * R.m_22);
  B.m_10 = R.m_10;
  B.m_11 = R.m_11;
  B.m_12 = R.m_12;
  B.m_20 = fmaf(-b, R.m_00, a * R.m_20);
  B.m_21 = fmaf(-b, R.m_01, a * R.m_21);
  B.m_22 = fmaf(-b, R.m_02, a * R.m_22);
  // third givens rotation (ch,sh,0,0)
  auto g3 = QRGivensQuaternion(B.m_11, B.m_21);
  a = fmaf(-2.f, g3.sh * g3.sh, 1.f);
  b = 2.f * g3.ch * g3.sh;
  // R is now set to desired value
  R.m_00 = B.m_00;
  R.m_01 = B.m_01;
  R.m_02 = B.m_02;
  R.m_10 = fmaf(a, B.m_10, b * B.m_20);
  R.m_11 = fmaf(a, B.m_11, b * B.m_21);
  R.m_12 = fmaf(a, B.m_12, b * B.m_22);
  R.m_20 = fmaf(-b, B.m_10, a * B.m_20);
  R.m_21 = fmaf(-b, B.m_11, a * B.m_21);
  R.m_22 = fmaf(-b, B.m_12, a * B.m_22);
  // construct the cumulative rotation Q=Q1 * Q2 * Q3
  // the number of floating point operations for three quaternion
  // multiplications is more or less comparable to the explicit form of the
  // joined matrix. certainly more memory-efficient!
  auto sh12 = 2.f * fmaf(g1.sh, g1.sh, -0.5f);
  auto sh22 = 2.f * fmaf(g2.sh, g2.sh, -0.5f);
  auto sh32 = 2.f * fmaf(g3.sh, g3.sh, -0.5f);
  Q.m_00 = sh12 * sh22;
  Q.m_01 = fmaf(4.f * g2.ch * g3.ch, sh12 * g2.sh * g3.sh,
                2.f * g1.ch * g1.sh * sh32);
  Q.m_02 = fmaf(4.f * g1.ch * g3.ch, g1.sh * g3.sh,
                -2.f * g2.ch * sh12 * g2.sh * sh32);

  Q.m_10 = -2.f * g1.ch * g1.sh * sh22;
  Q.m_11 =
      fmaf(-8.f * g1.ch * g2.ch * g3.ch, g1.sh * g2.sh * g3.sh, sh12 * sh32);
  Q.m_12 = fmaf(
      -2.f * g3.ch, g3.sh,
      4.f * g1.sh * fmaf(g3.ch * g1.sh, g3.sh, g1.ch * g2.ch * g2.sh * sh32));

  Q.m_20 = 2.f * g2.ch * g2.sh;
  Q.m_21 = -2.f * g3.ch * sh22 * g3.sh;
  Q.m_22 = sh22 * sh32;
  return QR{Q, R};
}
// Wrapping function used to contain all of the required sub calls.
SVDSet svd(Mat3x3 A) {
  Mat3x3 V(jacobiEigenAnalysis(A.transpose() * A));
  auto B = A * V;
  sortSingularValues(B, V);
  QR qr = QRDecomposition(B);
  return SVDSet{qr.Q, qr.R, V};
}
// The largest singular value of A.
float spectralNorm(Mat3x3 A) {
  SVDSet usv = svd(A);
  return usv.S.m_00;
}
}  // namespace SVD