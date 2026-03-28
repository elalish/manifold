// Copyright 2021 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#ifdef MANIFOLD_DEBUG
#include <chrono>
#endif

#include "linalg.h"

namespace manifold {
/** @addtogroup Math
 * @ingroup Core
 * @brief Simple math operations.
 * */

/** @addtogroup LinAlg
 *  @{
 */
namespace la = linalg;
using vec2 = la::vec<double, 2>;
using vec3 = la::vec<double, 3>;
using vec4 = la::vec<double, 4>;
using bvec4 = la::vec<bool, 4>;
using mat2 = la::mat<double, 2, 2>;
using mat3x2 = la::mat<double, 3, 2>;
using mat4x2 = la::mat<double, 4, 2>;
using mat2x3 = la::mat<double, 2, 3>;
using mat3 = la::mat<double, 3, 3>;
using mat4x3 = la::mat<double, 4, 3>;
using mat3x4 = la::mat<double, 3, 4>;
using mat4 = la::mat<double, 4, 4>;
using ivec2 = la::vec<int, 2>;
using ivec3 = la::vec<int, 3>;
using ivec4 = la::vec<int, 4>;
using quat = la::vec<double, 4>;
/** @} */

/** @addtogroup Scalar
 * @ingroup Math
 *  @brief Simple scalar operations.
 *  @{
 */

constexpr double kPi = 3.14159265358979323846264338327950288;
constexpr double kTwoPi = 6.28318530717958647692528676655900576;
constexpr double kHalfPi = 1.57079632679489661923132169163975144;

/**
 * Convert degrees to radians.
 *
 * @param a Angle in degrees.
 */
constexpr double radians(double a) { return a * kPi / 180; }

/**
 * Convert radians to degrees.
 *
 * @param a Angle in radians.
 */
constexpr double degrees(double a) { return a * 180 / kPi; }

/**
 * Performs smooth Hermite interpolation between 0 and 1 when edge0 < x < edge1.
 *
 * @param edge0 Specifies the value of the lower edge of the Hermite function.
 * @param edge1 Specifies the value of the upper edge of the Hermite function.
 * @param a Specifies the source value for interpolation.
 */
constexpr double smoothstep(double edge0, double edge1, double a) {
  const double x = la::clamp((a - edge0) / (edge1 - edge0), 0, 1);
  return x * x * (3 - 2 * x);
}

/**
 * Deterministic trigonometric helpers.
 *
 * Adapted from FreeBSD msun implementations via musl libc sources:
 * - https://git.musl-libc.org/cgit/musl/plain/src/math/__sin.c
 * - https://git.musl-libc.org/cgit/musl/plain/src/math/__cos.c
 * - https://git.musl-libc.org/cgit/musl/plain/src/math/__tan.c
 * - https://git.musl-libc.org/cgit/musl/plain/src/math/__rem_pio2.c
 * - https://git.musl-libc.org/cgit/musl/plain/src/math/e_acos.c
 * - https://git.musl-libc.org/cgit/musl/plain/src/math/atan.c
 * - https://git.musl-libc.org/cgit/musl/plain/src/math/atan2.c
 * - https://git.musl-libc.org/cgit/musl/plain/src/math/tan.c
 *
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 * Developed at SunPro/SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this software is freely
 * granted, provided that this notice is preserved.
 */
namespace math {
inline uint64_t AsUint64(double x) {
  uint64_t u;
  std::memcpy(&u, &x, sizeof(u));
  return u;
}

inline double FromUint64(uint64_t u) {
  double x;
  std::memcpy(&x, &u, sizeof(x));
  return x;
}

inline uint32_t HighWord(double x) {
  return static_cast<uint32_t>(AsUint64(x) >> 32);
}

inline uint32_t LowWord(double x) {
  return static_cast<uint32_t>(AsUint64(x));
}

inline double WithLowWord(double x, uint32_t low) {
  uint64_t u = AsUint64(x);
  u = (u & 0xffffffff00000000ULL) | static_cast<uint64_t>(low);
  return FromUint64(u);
}

constexpr inline double SinKernel(double x, double y, int iy) {
  constexpr double S1 = -1.66666666666666324348e-01;
  constexpr double S2 = 8.33333333332248946124e-03;
  constexpr double S3 = -1.98412698298579493134e-04;
  constexpr double S4 = 2.75573137070700676789e-06;
  constexpr double S5 = -2.50507602534068634195e-08;
  constexpr double S6 = 1.58969099521155010221e-10;

  const double z = x * x;
  const double w = z * z;
  const double r = S2 + z * (S3 + z * S4) + z * w * (S5 + z * S6);
  const double v = z * x;
  if (iy == 0) return x + v * (S1 + z * r);
  return x - ((z * (0.5 * y - v * r) - y) - v * S1);
}

constexpr inline double CosKernel(double x, double y) {
  constexpr double C1 = 4.16666666666666019037e-02;
  constexpr double C2 = -1.38888888888741095749e-03;
  constexpr double C3 = 2.48015872894767294178e-05;
  constexpr double C4 = -2.75573143513906633035e-07;
  constexpr double C5 = 2.08757232129817482790e-09;
  constexpr double C6 = -1.13596475577881948265e-11;

  const double z = x * x;
  const double w = z * z;
  const double r =
      z * (C1 + z * (C2 + z * C3)) + w * w * (C4 + z * (C5 + z * C6));
  const double hz = 0.5 * z;
  const double w1 = 1.0 - hz;
  return w1 + (((1.0 - w1) - hz) + (z * r - x * y));
}

inline double TanKernel(double x, double y, int odd) {
  constexpr double T[] = {
      3.33333333333334091986e-01,  1.33333333333201242699e-01,
      5.39682539762260521377e-02,  2.18694882948595424599e-02,
      8.86323982359930005737e-03,  3.59207910759131235356e-03,
      1.45620945432529025516e-03,  5.88041240820264096874e-04,
      2.46463134818469906812e-04,  7.81794442939557092300e-05,
      7.14072491382608190305e-05,  -1.85586374855275456654e-05,
      2.59073051863633712884e-05,
  };
  constexpr double pio4 = 7.85398163397448278999e-01;
  constexpr double pio4lo = 3.06161699786838301793e-17;

  const uint32_t hx = HighWord(x);
  const bool big = (hx & 0x7fffffff) >= 0x3FE59428;  // |x| >= 0.6744
  bool sign = false;
  if (big) {
    sign = (hx >> 31) != 0;
    if (sign) {
      x = -x;
      y = -y;
    }
    x = (pio4 - x) + (pio4lo - y);
    y = 0.0;
  }

  const double z = x * x;
  const double w = z * z;
  const double r =
      T[1] + w * (T[3] + w * (T[5] + w * (T[7] + w * (T[9] + w * T[11]))));
  const double v =
      z * (T[2] + w * (T[4] + w * (T[6] + w * (T[8] + w * (T[10] + w * T[12])))));
  const double s = z * x;
  const double rr = y + z * (s * (r + v) + y) + s * T[0];
  const double ww = x + rr;
  if (big) {
    const double s2 = 1 - 2 * odd;
    const double vv = s2 - 2.0 * (x + (rr - ww * ww / (ww + s2)));
    return sign ? -vv : vv;
  }
  if (!odd) return ww;
  // Compute -1/(x+r) with reduced cancellation error.
  const double w0 = WithLowWord(ww, 0);
  const double vv = rr - (w0 - x);
  const double aa = -1.0 / ww;
  const double a0 = WithLowWord(aa, 0);
  return a0 + aa * (1.0 + a0 * w0 + a0 * vv);
}

inline double acos(double x) {
  constexpr double pio2_hi = 1.57079632679489655800e+00;
  constexpr double pio2_lo = 6.12323399573676603587e-17;
  constexpr double pS0 = 1.66666666666666657415e-01;
  constexpr double pS1 = -3.25565818622400915405e-01;
  constexpr double pS2 = 2.01212532134862925881e-01;
  constexpr double pS3 = -4.00555345006794114027e-02;
  constexpr double pS4 = 7.91534994289814532176e-04;
  constexpr double pS5 = 3.47933107596021167570e-05;
  constexpr double qS1 = -2.40339491173441421878e+00;
  constexpr double qS2 = 2.02094576023350569471e+00;
  constexpr double qS3 = -6.88283971605453293030e-01;
  constexpr double qS4 = 7.70381505559019352791e-02;
  auto R = [=](double z) {
    const double p =
        z * (pS0 + z * (pS1 + z * (pS2 + z * (pS3 + z * (pS4 + z * pS5)))));
    const double q = 1.0 + z * (qS1 + z * (qS2 + z * (qS3 + z * qS4)));
    return p / q;
  };
  double z, w, s, c, df;
  uint64_t xx;
  uint32_t hx, lx, ix;
  std::memcpy(&xx, &x, sizeof(xx));
  hx = xx >> 32;
  ix = hx & 0x7fffffff;
  if (ix >= 0x3ff00000) {
    lx = xx;
    if (((ix - 0x3ff00000) | lx) == 0) {
      if (hx >> 31) return 2 * pio2_hi + 0x1p-120f;
      return 0;
    }
    return 0 / (x - x);
  }
  if (ix < 0x3fe00000) {
    if (ix <= 0x3c600000) return pio2_hi + 0x1p-120f;
    return pio2_hi - (x - (pio2_lo - x * R(x * x)));
  }
  if (hx >> 31) {
    z = (1.0 + x) * 0.5;
    s = std::sqrt(z);
    w = R(z) * s - pio2_lo;
    return 2 * (pio2_hi - (s + w));
  }
  z = (1.0 - x) * 0.5;
  s = std::sqrt(z);
  std::memcpy(&xx, &s, sizeof(xx));
  xx &= 0xffffffff00000000;
  std::memcpy(&df, &xx, sizeof(xx));
  c = (z - df * df) / (s + df);
  w = R(z) * s + c;
  return 2 * (df + w);
}

inline int RemPio2(double x, double y[2]) {
  constexpr double toint = 1.5 / DBL_EPSILON;
  constexpr double pio4 = 0x1.921fb54442d18p-1;
  constexpr double invpio2 = 6.36619772367581382433e-01;
  constexpr double pio2_1 = 1.57079632673412561417e+00;
  constexpr double pio2_1t = 6.07710050650619224932e-11;
  constexpr double pio2_2 = 6.07710050630396597660e-11;
  constexpr double pio2_2t = 2.02226624879595063154e-21;
  constexpr double pio2_3 = 2.02226624871116645580e-21;
  constexpr double pio2_3t = 8.47842766036889956997e-32;

  uint64_t ux;
  std::memcpy(&ux, &x, sizeof(ux));
  const bool sign = (ux >> 63) != 0;
  const uint32_t ix = static_cast<uint32_t>((ux >> 32) & 0x7fffffff);

  if (ix <= 0x400f6a7a) {  // |x| ~<= 5pi/4
    if ((ix & 0xfffff) == 0x921fb) goto medium;
    if (ix <= 0x4002d97c) {  // |x| ~<= 3pi/4
      if (!sign) {
        const double z = x - pio2_1;
        y[0] = z - pio2_1t;
        y[1] = (z - y[0]) - pio2_1t;
        return 1;
      }
      const double z = x + pio2_1;
      y[0] = z + pio2_1t;
      y[1] = (z - y[0]) + pio2_1t;
      return -1;
    }
    if (!sign) {
      const double z = x - 2 * pio2_1;
      y[0] = z - 2 * pio2_1t;
      y[1] = (z - y[0]) - 2 * pio2_1t;
      return 2;
    }
    const double z = x + 2 * pio2_1;
    y[0] = z + 2 * pio2_1t;
    y[1] = (z - y[0]) + 2 * pio2_1t;
    return -2;
  }
  if (ix <= 0x401c463b) {  // |x| ~<= 9pi/4
    if (ix <= 0x4015fdbc) {  // |x| ~<= 7pi/4
      if (ix == 0x4012d97c) goto medium;
      if (!sign) {
        const double z = x - 3 * pio2_1;
        y[0] = z - 3 * pio2_1t;
        y[1] = (z - y[0]) - 3 * pio2_1t;
        return 3;
      }
      const double z = x + 3 * pio2_1;
      y[0] = z + 3 * pio2_1t;
      y[1] = (z - y[0]) + 3 * pio2_1t;
      return -3;
    }
    if (ix == 0x401921fb) goto medium;
    if (!sign) {
      const double z = x - 4 * pio2_1;
      y[0] = z - 4 * pio2_1t;
      y[1] = (z - y[0]) - 4 * pio2_1t;
      return 4;
    }
    const double z = x + 4 * pio2_1;
    y[0] = z + 4 * pio2_1t;
    y[1] = (z - y[0]) + 4 * pio2_1t;
    return -4;
  }

  if (ix < 0x413921fb) {  // |x| ~< 2^20*(pi/2), medium size
  medium:
    double fn = x * invpio2 + toint - toint;
    int n = static_cast<int32_t>(fn);
    double r = x - fn * pio2_1;
    double w = fn * pio2_1t;
    if (r - w < -pio4) {
      n--;
      fn--;
      r = x - fn * pio2_1;
      w = fn * pio2_1t;
    } else if (r - w > pio4) {
      n++;
      fn++;
      r = x - fn * pio2_1;
      w = fn * pio2_1t;
    }
    y[0] = r - w;
    uint64_t uy0;
    std::memcpy(&uy0, &y[0], sizeof(uy0));
    const int ey = static_cast<int>((uy0 >> 52) & 0x7ff);
    const int ex = static_cast<int>(ix >> 20);
    if (ex - ey > 16) {
      const double t = r;
      w = fn * pio2_2;
      r = t - w;
      w = fn * pio2_2t - ((t - r) - w);
      y[0] = r - w;
      std::memcpy(&uy0, &y[0], sizeof(uy0));
      const int ey2 = static_cast<int>((uy0 >> 52) & 0x7ff);
      if (ex - ey2 > 49) {
        const double t2 = r;
        w = fn * pio2_3;
        r = t2 - w;
        w = fn * pio2_3t - ((t2 - r) - w);
        y[0] = r - w;
      }
    }
    y[1] = (r - y[0]) - w;
    return n;
  }

  if (ix >= 0x7ff00000) {
    y[0] = y[1] = x - x;
    return 0;
  }

  int q;
  y[0] = std::remquo(x, kHalfPi, &q);
  y[1] = 0.0;
  return q;
}

inline double sin(double x) {
  uint64_t ux;
  std::memcpy(&ux, &x, sizeof(ux));
  const uint32_t ix = static_cast<uint32_t>((ux >> 32) & 0x7fffffff);
  if (ix <= 0x3fe921fb) {
    if (ix < 0x3e500000) return x;
    return SinKernel(x, 0.0, 0);
  }
  if (ix >= 0x7ff00000) return x - x;
  double y[2];
  const int n = RemPio2(x, y);
  switch (n & 3) {
    case 0:
      return SinKernel(y[0], y[1], 1);
    case 1:
      return CosKernel(y[0], y[1]);
    case 2:
      return -SinKernel(y[0], y[1], 1);
    default:
      return -CosKernel(y[0], y[1]);
  }
}

inline double cos(double x) {
  uint64_t ux;
  std::memcpy(&ux, &x, sizeof(ux));
  const uint32_t ix = static_cast<uint32_t>((ux >> 32) & 0x7fffffff);
  if (ix <= 0x3fe921fb) {
    if (ix < 0x3e46a09e) return 1.0;
    return CosKernel(x, 0.0);
  }
  if (ix >= 0x7ff00000) return x - x;
  double y[2];
  const int n = RemPio2(x, y);
  switch (n & 3) {
    case 0:
      return CosKernel(y[0], y[1]);
    case 1:
      return -SinKernel(y[0], y[1], 1);
    case 2:
      return -CosKernel(y[0], y[1]);
    default:
      return SinKernel(y[0], y[1], 1);
  }
}

inline double tan(double x) {
  const uint32_t ix = HighWord(x) & 0x7fffffff;
  if (ix <= 0x3fe921fb) {
    if (ix < 0x3e400000) return x;
    return TanKernel(x, 0.0, 0);
  }
  if (ix >= 0x7ff00000) return x - x;
  double y[2];
  const int n = RemPio2(x, y);
  return TanKernel(y[0], y[1], n & 1);
}

inline double asin(double x) {
  if (!la::isfinite(x) || x < -1.0 || x > 1.0) return NAN;
  if (x == 1.0) return kHalfPi;
  if (x == -1.0) return -kHalfPi;
  return kHalfPi - acos(x);
}

inline double atan(double x) {
  constexpr double atanhi[] = {
      4.63647609000806093515e-01, 7.85398163397448278999e-01,
      9.82793723247329054082e-01, 1.57079632679489655800e+00};
  constexpr double atanlo[] = {
      2.26987774529616870924e-17, 3.06161699786838301793e-17,
      1.39033110312309984516e-17, 6.12323399573676603587e-17};
  constexpr double aT[] = {
      3.33333333333329318027e-01,  -1.99999999998764832476e-01,
      1.42857142725034663711e-01,  -1.11111104054623557880e-01,
      9.09088713343650656196e-02,  -7.69187620504482999495e-02,
      6.66107313738753120669e-02,  -5.83357013379057348645e-02,
      4.97687799461593236017e-02,  -3.65315727442169155270e-02,
      1.62858201153657823623e-02};

  uint32_t ix = HighWord(x);
  const uint32_t sign = ix >> 31;
  ix &= 0x7fffffff;
  int id;

  if (ix >= 0x44100000) {  // |x| >= 2^66
    if (std::isnan(x)) return x;
    const double z = atanhi[3] + 0x1p-120f;
    return sign ? -z : z;
  }
  if (ix < 0x3fdc0000) {  // |x| < 0.4375
    if (ix < 0x3e400000) return x;  // |x| < 2^-27
    id = -1;
  } else {
    x = std::fabs(x);
    if (ix < 0x3ff30000) {  // |x| < 1.1875
      if (ix < 0x3fe60000) {  // 7/16 <= |x| < 11/16
        id = 0;
        x = (2.0 * x - 1.0) / (2.0 + x);
      } else {  // 11/16 <= |x| < 19/16
        id = 1;
        x = (x - 1.0) / (x + 1.0);
      }
    } else {
      if (ix < 0x40038000) {  // |x| < 2.4375
        id = 2;
        x = (x - 1.5) / (1.0 + 1.5 * x);
      } else {  // 2.4375 <= |x| < 2^66
        id = 3;
        x = -1.0 / x;
      }
    }
  }

  const double z = x * x;
  const double w = z * z;
  const double s1 =
      z * (aT[0] + w * (aT[2] + w * (aT[4] + w * (aT[6] + w * (aT[8] + w * aT[10])))));
  const double s2 = w * (aT[1] + w * (aT[3] + w * (aT[5] + w * (aT[7] + w * aT[9]))));
  if (id < 0) return x - x * (s1 + s2);
  const double zz = atanhi[id] - (x * (s1 + s2) - atanlo[id] - x);
  return sign ? -zz : zz;
}

inline double atan2(double y, double x) {
  constexpr double pi = 3.1415926535897931160E+00;
  constexpr double pi_lo = 1.2246467991473531772E-16;

  if (std::isnan(x) || std::isnan(y)) return x + y;
  uint32_t ix = HighWord(x);
  uint32_t iy = HighWord(y);
  const uint32_t lx = LowWord(x);
  const uint32_t ly = LowWord(y);

  if (((ix - 0x3ff00000) | lx) == 0) return atan(y);  // x = 1.0

  const uint32_t m = ((iy >> 31) & 1) | ((ix >> 30) & 2);
  ix &= 0x7fffffff;
  iy &= 0x7fffffff;

  if ((iy | ly) == 0) {  // y = 0
    switch (m) {
      case 0:
      case 1:
        return y;
      case 2:
        return pi;
      default:
        return -pi;
    }
  }
  if ((ix | lx) == 0) return (m & 1) ? -pi / 2 : pi / 2;  // x = 0

  if (ix == 0x7ff00000) {  // x is INF
    if (iy == 0x7ff00000) {
      switch (m) {
        case 0:
          return pi / 4;
        case 1:
          return -pi / 4;
        case 2:
          return 3 * pi / 4;
        default:
          return -3 * pi / 4;
      }
    }
    switch (m) {
      case 0:
        return 0.0;
      case 1:
        return -0.0;
      case 2:
        return pi;
      default:
        return -pi;
    }
  }

  if (ix + (64 << 20) < iy || iy == 0x7ff00000) {
    return (m & 1) ? -pi / 2 : pi / 2;  // |y/x| > 2^64
  }

  double z;
  if ((m & 2) && iy + (64 << 20) < ix) {
    z = 0;  // |y/x| < 2^-64 and x < 0
  } else {
    z = atan(std::fabs(y / x));
  }

  switch (m) {
    case 0:
      return z;
    case 1:
      return -z;
    case 2:
      return pi - (z - pi_lo);
    default:
      return (z - pi_lo) - pi;
  }
}
}  // namespace math

/**
 * Sine function where multiples of 90 degrees come out exact.
 *
 * @param x Angle in degrees.
 */
inline double sind(double x) {
  if (!la::isfinite(x)) return NAN;
  if (x < 0.0) return -sind(-x);
  int quo;
  x = std::remquo(std::fabs(x), 90.0, &quo);
  const double xr = radians(x);
  switch (quo % 4) {
    case 0:
      return math::sin(xr);
    case 1:
      return math::cos(xr);
    case 2:
      return -math::sin(xr);
    case 3:
      return -math::cos(xr);
  }
  return 0.0;
}

/**
 * Cosine function where multiples of 90 degrees come out exact.
 *
 * @param x Angle in degrees.
 */
inline double cosd(double x) { return sind(x + 90.0); }
/** @} */

/** @addtogroup Structs
 * @ingroup Core
 * @brief Miscellaneous data structures for interfacing with this library.
 *  @{
 */

/**
 * @brief Single polygon contour, wound CCW. First and last point are implicitly
 * connected. Should ensure all input is
 * [&epsilon;-valid](https://github.com/elalish/manifold/wiki/Manifold-Library#definition-of-%CE%B5-valid).
 */
using SimplePolygon = std::vector<vec2>;

/**
 * @brief Set of polygons with holes. Order of contours is arbitrary. Can
 * contain any depth of nested holes and any number of separate polygons. Should
 * ensure all input is
 * [&epsilon;-valid](https://github.com/elalish/manifold/wiki/Manifold-Library#definition-of-%CE%B5-valid).
 */
using Polygons = std::vector<SimplePolygon>;

/**
 * @brief Defines which edges to sharpen and how much for the Manifold.Smooth()
 * constructor.
 */
struct Smoothness {
  /// The halfedge index = 3 * tri + i, referring to Mesh.triVerts[tri][i].
  size_t halfedge;
  /// A value between 0 and 1, where 0 is sharp and 1 is the default and the
  /// curvature is interpolated between these values. The two paired halfedges
  /// can have different values while maintaining C-1 continuity (except for 0).
  double smoothness;
};

/**
 * @brief Axis-aligned 3D box, primarily for bounding.
 */
struct Box {
  vec3 min = vec3(std::numeric_limits<double>::infinity());
  vec3 max = vec3(-std::numeric_limits<double>::infinity());

  /**
   * Default constructor is an infinite box that contains all space.
   */
  constexpr Box() {}

  /**
   * Creates a box that contains the two given points.
   */
  constexpr Box(const vec3 p1, const vec3 p2) {
    min = la::min(p1, p2);
    max = la::max(p1, p2);
  }

  /**
   * Returns the dimensions of the Box.
   */
  constexpr vec3 Size() const { return max - min; }

  /**
   * Returns the center point of the Box.
   */
  constexpr vec3 Center() const { return 0.5 * (max + min); }

  /**
   * Returns the absolute-largest coordinate value of any contained
   * point.
   */
  constexpr double Scale() const {
    vec3 absMax = la::max(la::abs(min), la::abs(max));
    return la::max(absMax.x, la::max(absMax.y, absMax.z));
  }

  /**
   * Does this box contain (includes equal) the given point?
   */
  constexpr bool Contains(const vec3& p) const {
    return la::all(la::gequal(p, min)) && la::all(la::gequal(max, p));
  }

  /**
   * Does this box contain (includes equal) the given box?
   */
  constexpr bool Contains(const Box& box) const {
    return la::all(la::gequal(box.min, min)) &&
           la::all(la::gequal(max, box.max));
  }

  /**
   * Expand this box to include the given point.
   */
  void Union(const vec3 p) {
    min = la::min(min, p);
    max = la::max(max, p);
  }

  /**
   * Expand this box to include the given box.
   */
  constexpr Box Union(const Box& box) const {
    Box out;
    out.min = la::min(min, box.min);
    out.max = la::max(max, box.max);
    return out;
  }

  /**
   * Transform the given box by the given axis-aligned affine transform.
   *
   * Ensure the transform passed in is axis-aligned (rotations are all
   * multiples of 90 degrees), or else the resulting bounding box will no longer
   * bound properly.
   */
  constexpr Box Transform(const mat3x4& transform) const {
    Box out;
    vec3 minT = transform * vec4(min, 1.0);
    vec3 maxT = transform * vec4(max, 1.0);
    out.min = la::min(minT, maxT);
    out.max = la::max(minT, maxT);
    return out;
  }

  /**
   * Shift this box by the given vector.
   */
  constexpr Box operator+(vec3 shift) const {
    Box out;
    out.min = min + shift;
    out.max = max + shift;
    return out;
  }

  /**
   * Shift this box in-place by the given vector.
   */
  Box& operator+=(vec3 shift) {
    min += shift;
    max += shift;
    return *this;
  }

  /**
   * Scale this box by the given vector.
   */
  constexpr Box operator*(vec3 scale) const {
    Box out;
    out.min = min * scale;
    out.max = max * scale;
    return out;
  }

  /**
   * Scale this box in-place by the given vector.
   */
  Box& operator*=(vec3 scale) {
    min *= scale;
    max *= scale;
    return *this;
  }

  /**
   * Does this box overlap the one given (including equality)?
   */
  constexpr bool DoesOverlap(const Box& box) const {
    return min.x <= box.max.x && min.y <= box.max.y && min.z <= box.max.z &&
           max.x >= box.min.x && max.y >= box.min.y && max.z >= box.min.z;
  }

  /**
   * Does the given point project within the XY extent of this box
   * (including equality)?
   */
  constexpr bool DoesOverlap(vec3 p) const {  // projected in z
    return p.x <= max.x && p.x >= min.x && p.y <= max.y && p.y >= min.y;
  }

  /**
   * Does this box have finite bounds?
   */
  constexpr bool IsFinite() const {
    return la::all(la::isfinite(min)) && la::all(la::isfinite(max));
  }
};

/**
 * @brief Axis-aligned 2D box, primarily for bounding.
 */
struct Rect {
  vec2 min = vec2(std::numeric_limits<double>::infinity());
  vec2 max = vec2(-std::numeric_limits<double>::infinity());

  /**
   * Default constructor is an empty rectangle..
   */
  constexpr Rect() {}

  /**
   * Create a rectangle that contains the two given points.
   */
  constexpr Rect(const vec2 a, const vec2 b) {
    min = la::min(a, b);
    max = la::max(a, b);
  }

  /** @name Information
   *  Details of the rectangle
   */
  ///@{

  /**
   * Return the dimensions of the rectangle.
   */
  constexpr vec2 Size() const { return max - min; }

  /**
   * Return the area of the rectangle.
   */
  constexpr double Area() const {
    auto sz = Size();
    return sz.x * sz.y;
  }

  /**
   * Returns the absolute-largest coordinate value of any contained
   * point.
   */
  constexpr double Scale() const {
    vec2 absMax = la::max(la::abs(min), la::abs(max));
    return la::max(absMax.x, absMax.y);
  }

  /**
   * Returns the center point of the rectangle.
   */
  constexpr vec2 Center() const { return 0.5 * (max + min); }

  /**
   * Does this rectangle contain (includes on border) the given point?
   */
  constexpr bool Contains(const vec2& p) const {
    return la::all(la::gequal(p, min)) && la::all(la::gequal(max, p));
  }

  /**
   * Does this rectangle contain (includes equal) the given rectangle?
   */
  constexpr bool Contains(const Rect& rect) const {
    return la::all(la::gequal(rect.min, min)) &&
           la::all(la::gequal(max, rect.max));
  }

  /**
   * Does this rectangle overlap the one given (including equality)?
   */
  constexpr bool DoesOverlap(const Rect& rect) const {
    return min.x <= rect.max.x && min.y <= rect.max.y && max.x >= rect.min.x &&
           max.y >= rect.min.y;
  }

  /**
   * Is the rectangle empty (containing no space)?
   */
  constexpr bool IsEmpty() const { return max.y <= min.y || max.x <= min.x; };

  /**
   * Does this recangle have finite bounds?
   */
  constexpr bool IsFinite() const {
    return la::all(la::isfinite(min)) && la::all(la::isfinite(max));
  }

  ///@}

  /** @name Modification
   */
  ///@{

  /**
   * Expand this rectangle (in place) to include the given point.
   */
  void Union(const vec2 p) {
    min = la::min(min, p);
    max = la::max(max, p);
  }

  /**
   * Expand this rectangle to include the given Rect.
   */
  constexpr Rect Union(const Rect& rect) const {
    Rect out;
    out.min = la::min(min, rect.min);
    out.max = la::max(max, rect.max);
    return out;
  }

  /**
   * Shift this rectangle by the given vector.
   */
  constexpr Rect operator+(const vec2 shift) const {
    Rect out;
    out.min = min + shift;
    out.max = max + shift;
    return out;
  }

  /**
   * Shift this rectangle in-place by the given vector.
   */
  Rect& operator+=(const vec2 shift) {
    min += shift;
    max += shift;
    return *this;
  }

  /**
   * Scale this rectangle by the given vector.
   */
  constexpr Rect operator*(const vec2 scale) const {
    Rect out;
    out.min = min * scale;
    out.max = max * scale;
    return out;
  }

  /**
   * Scale this rectangle in-place by the given vector.
   */
  Rect& operator*=(const vec2 scale) {
    min *= scale;
    max *= scale;
    return *this;
  }

  /**
   * Transform the rectangle by the given axis-aligned affine transform.
   *
   * Ensure the transform passed in is axis-aligned (rotations are all
   * multiples of 90 degrees), or else the resulting rectangle will no longer
   * bound properly.
   */
  constexpr Rect Transform(const mat2x3& m) const {
    Rect rect;
    rect.min = m * vec3(min, 1);
    rect.max = m * vec3(max, 1);
    return rect;
  }
  ///@}
};

/**
 * @brief Boolean operation type: Add (Union), Subtract (Difference), and
 * Intersect.
 */
enum class OpType : char { Add, Subtract, Intersect };

constexpr int DEFAULT_SEGMENTS = 0;
constexpr double DEFAULT_ANGLE = 10.0;
constexpr double DEFAULT_LENGTH = 1.0;
/**
 * @brief These static properties control how circular shapes are quantized by
 * default on construction.
 *
 * If circularSegments is specified, it takes
 * precedence. If it is zero, then instead the minimum is used of the segments
 * calculated based on edge length and angle, rounded up to the nearest
 * multiple of four. To get numbers not divisible by four, circularSegments
 * must be specified.
 */
class Quality {
 private:
 public:
  static void SetMinCircularAngle(double angle);
  static void SetMinCircularEdgeLength(double length);
  static void SetCircularSegments(int number);
  static int GetCircularSegments(double radius);
  static void ResetToDefaults();
};
/** @} */

/** @addtogroup Debug
 * @ingroup Optional
 * @{
 */

/**
 * @brief Global parameters that control debugging output. Only has an
 * effect when compiled with the MANIFOLD_DEBUG flag.
 */
struct ExecutionParams {
  /// Perform extra sanity checks and assertions on the intermediate data
  /// structures.
  bool intermediateChecks = false;
  /// Perform 3D mesh self-intersection test on intermediate boolean results to
  /// test for ϵ-validity. For debug purposes only.
  bool selfIntersectionChecks = false;
  /// If processOverlaps is false, a geometric check will be performed to assert
  /// all triangles are CCW.
  bool processOverlaps = true;
  /// Suppresses printed errors regarding CW triangles. Has no effect if
  /// processOverlaps is true.
  bool suppressErrors = false;
  /// Deprecated! This value no longer has any effect, as cleanup now only
  /// occurs on intersected triangles.
  bool cleanupTriangles = true;
  /// Verbose level:
  /// - 0 for no verbose output
  /// - 1 for Boolean debug dumps on failures and invalid intermediate meshes.
  /// - 2 for Boolean timing and size statistics, plus triangulator action.
  int verbose = 0;
};
/** @} */

#ifdef MANIFOLD_DEBUG

inline std::ostream& operator<<(std::ostream& stream, const Box& box) {
  return stream << "min: " << box.min << ", "
                << "max: " << box.max;
}

inline std::ostream& operator<<(std::ostream& stream, const Rect& box) {
  return stream << "min: " << box.min << ", "
                << "max: " << box.max;
}

inline std::ostream& operator<<(std::ostream& stream, const Smoothness& s) {
  return stream << "halfedge: " << s.halfedge << ", "
                << "smoothness: " << s.smoothness;
}

/**
 * Print the contents of this vector to standard output. Only exists if compiled
 * with MANIFOLD_DEBUG flag.
 */
template <typename T>
void Dump(const std::vector<T>& vec) {
  std::cout << "Vec = " << std::endl;
  for (size_t i = 0; i < vec.size(); ++i) {
    std::cout << i << ", " << vec[i] << ", " << std::endl;
  }
  std::cout << std::endl;
}

template <typename T>
void Diff(const std::vector<T>& a, const std::vector<T>& b) {
  std::cout << "Diff = " << std::endl;
  if (a.size() != b.size()) {
    std::cout << "a and b must have the same length, aborting Diff"
              << std::endl;
    return;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i])
      std::cout << i << ": " << a[i] << ", " << b[i] << std::endl;
  }
  std::cout << std::endl;
}

struct Timer {
  std::chrono::high_resolution_clock::time_point start, end;

  void Start() { start = std::chrono::high_resolution_clock::now(); }

  void Stop() { end = std::chrono::high_resolution_clock::now(); }

  float Elapsed() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
        .count();
  }
  void Print(std::string message) {
    std::cout << "----------- " << std::round(Elapsed()) << " ms for "
              << message << std::endl;
  }
};
#endif
}  // namespace manifold
