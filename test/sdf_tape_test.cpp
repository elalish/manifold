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
  auto z = Value::Y() - constantKPi4;

  auto result = x.cos() * y.sin() + y.cos() * z.sin() + z.cos() * x.sin();
  auto tape = result.genTape();
  std::vector<Interval<double>> intervalBuffer;
  for (auto d : tape.second) intervalBuffer.push_back(Interval(d));
  EvalContext<Interval<double>> ctx{
      tape.first, VecView(intervalBuffer.data(), intervalBuffer.size())};
  std::cout << recursive_interval(ctx, vec3(-period), vec3(period * 2),
                                  period / n, -0.4)
            << std::endl;
  std::cout << recursive_interval(ctx, vec3(-period), vec3(period * 2),
                                  period / n, 0.4)
            << std::endl;
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
  auto tape = d.genTape();

  std::vector<Interval<double>> intervalBuffer;
  for (auto d : tape.second) intervalBuffer.push_back(Interval(d));
  EvalContext<Interval<double>> ctx{
      tape.first, VecView(intervalBuffer.data(), intervalBuffer.size())};
  std::cout << recursive_interval(ctx, vec3(-5), vec3(10), 0.05, 0.5)
            << std::endl;
}
