// Copyright 2026 The Manifold Authors.
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

#ifndef MANIFOLD_NO_IOSTREAM

#include <algorithm>
#include <sstream>
#include <streambuf>
#include <string>

#include "gtest/gtest.h"
#include "manifold/manifold.h"

using namespace manifold;

namespace {
class LimitedOutput : public std::streambuf {
 public:
  explicit LimitedOutput(std::streamsize capacity, bool failSync = false)
      : remaining_(capacity), failSync_(failSync) {}

 private:
  std::streamsize xsputn(const char*, std::streamsize count) override {
    const auto written = std::min(remaining_, count);
    remaining_ -= written;
    return written;
  }

  int_type overflow(int_type c) override {
    if (traits_type::eq_int_type(c, traits_type::eof())) {
      return traits_type::not_eof(c);
    }
    if (remaining_ == 0) return traits_type::eof();
    --remaining_;
    return c;
  }

  int sync() override { return failSync_ ? -1 : 0; }
  std::streamsize remaining_;
  bool failSync_;
};

}  // namespace

TEST(OBJStream, ReportShortWrites) {
  const Manifold cube = Manifold::Cube();
  for (const int capacity : {0, 64, 512}) {
    for (const bool member : {false, true}) {
      LimitedOutput buffer(capacity);
      std::ostream output(&buffer);
      ASSERT_TRUE(output.good());
      const bool written =
          member ? cube.WriteOBJ(output) : WriteOBJ(output, cube.GetMeshGL64());
      EXPECT_FALSE(written);
      EXPECT_TRUE(output.bad());
    }
  }
}

TEST(OBJStream, ReportFlushErrors) {
  const Manifold cube = Manifold::Cube();
  for (const bool member : {false, true}) {
    LimitedOutput buffer(10000, true);
    std::ostream output(&buffer);
    const bool written =
        member ? cube.WriteOBJ(output) : WriteOBJ(output, cube.GetMeshGL64());
    EXPECT_FALSE(written);
    EXPECT_TRUE(output.bad());
  }
}

TEST(OBJStream, SuccessfulWrite) {
  const Manifold cube = Manifold::Cube();
  for (const bool member : {false, true}) {
    std::ostringstream output;
    ASSERT_TRUE(member ? cube.WriteOBJ(output)
                       : WriteOBJ(output, cube.GetMeshGL64()));
    EXPECT_TRUE(output.good());
    std::istringstream input(output.str());
    const MeshGL64 mesh = ReadOBJ(input);
    EXPECT_EQ(mesh.NumVert(), cube.NumVert());
    EXPECT_EQ(mesh.NumTri(), cube.NumTri());
    EXPECT_DOUBLE_EQ(Manifold(mesh).Volume(), cube.Volume());
  }
}

TEST(OBJStream, EmptyMeshWrite) {
  for (const bool member : {false, true}) {
    std::ostringstream output;
    ASSERT_TRUE(member ? Manifold().WriteOBJ(output)
                       : WriteOBJ(output, MeshGL64{}));
    EXPECT_TRUE(output.good());
    EXPECT_NE(output.str().find("# ======== end mesh ======="),
              std::string::npos);
    std::istringstream input(output.str());
    EXPECT_TRUE(ReadOBJ(input).vertProperties.empty());
  }
}

TEST(OBJStream, InitiallyFailedOutput) {
  const Manifold cube = Manifold::Cube();
  for (const auto state :
       {std::ios::failbit, std::ios::badbit, std::ios::eofbit}) {
    for (const bool member : {false, true}) {
      std::ostringstream output;
      output.setstate(state);
      EXPECT_FALSE(member ? cube.WriteOBJ(output)
                          : WriteOBJ(output, cube.GetMeshGL64()));
      EXPECT_EQ(output.rdstate(), state);
      EXPECT_TRUE(output.str().empty());
    }
  }
}

TEST(OBJStream, PreserveOutputExceptionMasks) {
  for (const bool member : {false, true}) {
    for (const bool failSync : {false, true}) {
      LimitedOutput outputBuffer(failSync ? 10000 : 0, failSync);
      std::ostream output(&outputBuffer);
      output.exceptions(std::ios::badbit);
      if (member) {
        EXPECT_THROW(Manifold::Cube().WriteOBJ(output), std::ios_base::failure);
      } else {
        EXPECT_THROW(WriteOBJ(output, Manifold::Cube().GetMeshGL64()),
                     std::ios_base::failure);
      }
      EXPECT_EQ(output.exceptions(), std::ios::badbit);
    }
  }
}

#endif  // MANIFOLD_NO_IOSTREAM
