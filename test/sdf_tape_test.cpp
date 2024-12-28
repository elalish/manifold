// Copyright 2024 The Manifold Authors.
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

#include <chrono>
#include <iostream>

#include "../src/sdf/tape.h"
#include "../src/sdf/value.h"
#include "test.h"

using namespace manifold;
using namespace manifold::sdf;

int recursive_interval(EvalContext<Interval<double>>& ctx, vec3 start,
                       vec3 delta, const double edgeLength,
                       const double level) {
  if (delta.x < edgeLength && delta.y < edgeLength && delta.z < edgeLength)
    return 1;  // we should do one evaluation...
  ctx.buffer[0] = {start.x, start.x + delta.x};
  ctx.buffer[1] = {start.y, start.y + delta.y};
  ctx.buffer[2] = {start.z, start.z + delta.z};
  auto result = ctx.eval() - level;

  int count = 1;  // we did one evaluation
  if (result.lower >= 0.0 || result.upper <= 0.0) return 1;
  // in case it is not a cube, we may want to avoid dividing too much in some
  // axis
  vec3 new_delta = {delta.x < edgeLength ? delta.x : (delta.x / 2),
                    delta.y < edgeLength ? delta.y : (delta.y / 2),
                    delta.z < edgeLength ? delta.z : (delta.z / 2)};
  // can easily be converted into a worklist
  for (int a = 0; a <= (delta.x < edgeLength ? 0 : 1); a++)
    for (int b = 0; b <= (delta.y < edgeLength ? 0 : 1); b++)
      for (int c = 0; c <= (delta.z < edgeLength ? 0 : 1); c++) {
        auto new_start = start;
        if (a == 1) new_start.x += new_delta.x;
        if (b == 1) new_start.y += new_delta.y;
        if (c == 1) new_start.z += new_delta.z;
        count +=
            recursive_interval(ctx, new_start, new_delta, edgeLength, level);
      }

  return count;
}

TEST(TAPE, Gyroid) {
  const double n = 20;
  const double period = kTwoPi;
  Value constantKPi4 = Value::Constant(kPi / 4);
  auto x = Value::X() - constantKPi4;
  auto y = Value::Y() - constantKPi4;
  auto z = Value::Z() - constantKPi4;

  auto result = x.cos() * y.sin() + y.cos() * z.sin() + z.cos() * x.sin();
  auto tape = result.genTape();

  // verify by comparing with grid evaluation results
  auto gyroid = [](vec3 p) {
    p -= kPi / 4;
    return std::cos(p.x) * std::sin(p.y) + std::cos(p.y) * std::sin(p.z) +
           std::cos(p.z) * std::sin(p.x);
  };

  std::vector<double> buffer(tape.second, 0.0);
  EvalContext<double> ctxSimple{tape.first,
                                VecView(buffer.data(), buffer.size())};
  for (double x = -period; x < period; x += period / n) {
    for (double y = -period; y < period; y += period / n) {
      for (double z = -period; z < period; z += period / n) {
        ctxSimple.buffer[0] = x;
        ctxSimple.buffer[1] = y;
        ctxSimple.buffer[2] = z;
        ASSERT_NEAR(ctxSimple.eval(), gyroid({x, y, z}), 1e-6);
      }
    }
  }

  std::vector<Interval<double>> intervalBuffer(tape.second,
                                               Interval<double>::constant(0.0));
  EvalContext<Interval<double>> ctx{
      tape.first, VecView(intervalBuffer.data(), intervalBuffer.size())};
  auto start = std::chrono::high_resolution_clock::now();
  std::cout << recursive_interval(ctx, vec3(-period), vec3(period * 2),
                                  period / n, -0.4)
            << std::endl;
  auto end = std::chrono::high_resolution_clock::now();
  auto time = static_cast<int>(
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count());
  printf("interval evaluation: %dus\n", time);
}

TEST(TAPE, Blobs) {
  std::vector<vec4> balls = {{0, 0, 0, 2},     //
                             {1, 2, 3, 2},     //
                             {-2, 2, -2, 1},   //
                             {-2, -3, -2, 2},  //
                             {-3, -1, -3, 1},  //
                             {2, -3, -2, 2},   //
                             {-2, 3, 2, 2},    //
                             {-2, -3, 2, 2},   //
                             {1, -1, 1, -2},   //
                             {-4, -3, -2, 1}};
  auto lengthFn = [](Value x, Value y, Value z) {
    return (x * x + y * y + z * z).sqrt();
  };
  auto smoothstepFn = [](Value edge0, Value edge1, Value a) {
    auto x = ((a - edge0) / (edge1 - edge0))
                 .min(Value::Constant(1))
                 .max(Value::Constant(0));
    return x * x * (Value::Constant(3) - Value::Constant(2) * x);
  };
  Value d = Value::Constant(0);
  for (const auto& ball : balls) {
    auto tmp = smoothstepFn(Value::Constant(-1), Value::Constant(1),
                            Value::Constant(ball.w).abs() -
                                lengthFn(Value::Constant(ball.x) - Value::X(),
                                         Value::Constant(ball.y) - Value::Y(),
                                         Value::Constant(ball.z) - Value::Z()));
    if (ball.w > 0)
      d = d + tmp;
    else
      d = d - tmp;
  }
  auto start = std::chrono::high_resolution_clock::now();
  auto tape = d.genTape();
  auto end = std::chrono::high_resolution_clock::now();
  auto time = static_cast<int>(
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count());
  printf("codegen time: %dus, %ld\n", time, tape.first.size());

  auto blobs = [&balls](vec3 p) {
    double d = 0;
    for (const auto& ball : balls) {
      d += (ball.w > 0 ? 1 : -1) *
           smoothstep(-1, 1, std::abs(ball.w) - la::length(vec3(ball) - p));
    }
    return d;
  };
  std::vector<double> buffer(tape.second, 0.0);
  EvalContext<double> ctxSimple{tape.first,
                                VecView(buffer.data(), buffer.size())};
  for (double x = -5; x < 5; x += 0.05) {
    for (double y = -5; y < 5; y += 0.05) {
      for (double z = -5; z < 5; z += 0.05) {
        ctxSimple.buffer[0] = x;
        ctxSimple.buffer[1] = y;
        ctxSimple.buffer[2] = z;
        ASSERT_NEAR(ctxSimple.eval(), blobs({x, y, z}), 1e-12);
      }
    }
  }

  std::vector<Interval<double>> intervalBuffer(tape.second,
                                               Interval<double>::constant(0.0));
  EvalContext<Interval<double>> ctx{
      tape.first, VecView(intervalBuffer.data(), intervalBuffer.size())};
  start = std::chrono::high_resolution_clock::now();
  std::cout << recursive_interval(ctx, vec3(-5), vec3(10), 0.05, 0.5)
            << std::endl;
  end = std::chrono::high_resolution_clock::now();
  time = static_cast<int>(
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count());
  printf("interval evaluation: %dus\n", time);
}

TEST(TAPE, Blobs2) {
  auto lengthFn = [](Value x, Value y, Value z) {
    return (x * x + y * y + z * z).sqrt();
  };
  auto smoothstepFn = [](Value edge0, Value edge1, Value a) {
    auto x = ((a - edge0) / (edge1 - edge0))
                 .min(Value::Constant(1))
                 .max(Value::Constant(0));
    return x * x * (Value::Constant(3) - Value::Constant(2) * x);
  };
  Value d = Value::Constant(0);
  for (int i = 0; i < 1000; i++) {
    auto f = double(i + 1);
    auto tmp = smoothstepFn(
        Value::Constant(-1), Value::Constant(1),
        Value::Constant(f).abs() - lengthFn(Value::Constant(f) - Value::X(),
                                            Value::Constant(f) - Value::Y(),
                                            Value::Constant(f) - Value::Z()));
    d = d + tmp;
  }
  auto tape = d.genTape();
}
