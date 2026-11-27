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
#include <stdexcept>
#include <streambuf>
#include <string>

#include "gtest/gtest.h"
#include "manifold/manifold.h"

using namespace manifold;

namespace {
class FailingInput : public std::streambuf {
 public:
  explicit FailingInput(std::string prefix) : prefix_(std::move(prefix)) {
    setg(prefix_.data(), prefix_.data(), prefix_.data() + prefix_.size());
  }

 private:
  int_type underflow() override { throw std::runtime_error("read failed"); }
  std::string prefix_;
};

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

std::string CubeOBJ() {
  std::ostringstream output;
  Manifold::Cube().WriteOBJ(output);
  return output.str();
}
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

TEST(OBJStream, ReadErrorDoesNotReturnPartialMesh) {
  for (const std::string& prefix :
       {std::string(), std::string("v 0 0 0\n"), CubeOBJ()}) {
    FailingInput freeBuffer(prefix);
    std::istream freeInput(&freeBuffer);
    const MeshGL64 mesh = ReadOBJ(freeInput);
    EXPECT_TRUE(freeInput.bad());
    EXPECT_FALSE(freeInput.eof());
    EXPECT_TRUE(mesh.vertProperties.empty());
    EXPECT_TRUE(mesh.triVerts.empty());

    FailingInput memberBuffer(prefix);
    std::istream memberInput(&memberBuffer);
    EXPECT_EQ(Manifold::ReadOBJ(memberInput).Status(),
              Manifold::Error::InvalidConstruction);
    EXPECT_TRUE(memberInput.bad());
    EXPECT_FALSE(memberInput.eof());
  }
}

TEST(OBJStream, SuccessfulWriteAndNormalEOF) {
  const Manifold cube = Manifold::Cube();
  for (const bool member : {false, true}) {
    std::ostringstream output;
    const bool written =
        member ? cube.WriteOBJ(output) : WriteOBJ(output, cube.GetMeshGL64());
    ASSERT_TRUE(written);
    const std::string text = output.str();
    for (const bool newline : {false, true}) {
      const std::string inputText =
          newline ? text : text.substr(0, text.size() - 1);
      std::istringstream freeInput(inputText);
      const MeshGL64 mesh = ReadOBJ(freeInput);
      EXPECT_TRUE(freeInput.eof());
      EXPECT_FALSE(freeInput.bad());
      EXPECT_EQ(mesh.NumVert(), cube.NumVert());
      EXPECT_EQ(mesh.NumTri(), cube.NumTri());
      std::istringstream memberInput(inputText);
      const Manifold restored = Manifold::ReadOBJ(memberInput);
      EXPECT_EQ(restored.Status(), Manifold::Error::NoError);
      EXPECT_DOUBLE_EQ(restored.Volume(), 1);
    }
  }
}

TEST(OBJStream, LastVertexWithoutNewline) {
  for (const std::string text : {"v 1 2 3", "v 1 2 3\n", "v 1 2 3\r\n"}) {
    std::istringstream input(text);
    const MeshGL64 mesh = ReadOBJ(input);
    EXPECT_EQ(mesh.vertProperties, (std::vector<double>{1, 2, 3}));
    EXPECT_FALSE(input.bad());
  }
}

TEST(OBJStream, InitiallyFailedStreams) {
  const Manifold cube = Manifold::Cube();
  std::istringstream input(CubeOBJ());
  input.setstate(std::ios::failbit);
  EXPECT_TRUE(ReadOBJ(input).vertProperties.empty());
  EXPECT_EQ(Manifold::ReadOBJ(input).Status(),
            Manifold::Error::InvalidConstruction);
  std::ostringstream output;
  output.setstate(std::ios::failbit);
  EXPECT_FALSE(WriteOBJ(output, cube.GetMeshGL64()));
  EXPECT_FALSE(cube.WriteOBJ(output));
}

TEST(OBJStream, PreserveExceptionMasks) {
  for (const bool member : {false, true}) {
    FailingInput inputBuffer("v 0 0 0\n");
    std::istream input(&inputBuffer);
    input.exceptions(std::ios::badbit);
    if (member) {
      EXPECT_THROW(Manifold::ReadOBJ(input), std::runtime_error);
    } else {
      EXPECT_THROW(ReadOBJ(input), std::runtime_error);
    }
    LimitedOutput outputBuffer(0);
    std::ostream output(&outputBuffer);
    output.exceptions(std::ios::badbit);
    if (member) {
      EXPECT_THROW(Manifold::Cube().WriteOBJ(output), std::ios_base::failure);
    } else {
      EXPECT_THROW(WriteOBJ(output, Manifold::Cube().GetMeshGL64()),
                   std::ios_base::failure);
    }
  }
}

#endif  // MANIFOLD_NO_IOSTREAM
