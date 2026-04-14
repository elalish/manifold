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

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "impl.h"

namespace manifold::debug {

inline bool ParseBoolEnv(const char* value) {
  if (value == nullptr) return false;
  return std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
         std::strcmp(value, "TRUE") == 0 || std::strcmp(value, "on") == 0 ||
         std::strcmp(value, "ON") == 0;
}

inline bool BooleanDumpEnabled() {
  // Reuse existing verbosity plumbing from tests/debug flows.
  if (ManifoldParams().verbose >= 1) return true;
  return ParseBoolEnv(std::getenv("MANIFOLD_BOOLEAN_DEBUG_DUMP"));
}

inline std::filesystem::path BooleanDumpDir() {
  const char* value = std::getenv("MANIFOLD_BOOLEAN_DEBUG_DUMP_DIR");
  if (value == nullptr || *value == '\0') return ".";
  return value;
}

inline void EnsureDumpDir(const std::filesystem::path& dir) {
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
}

inline const char* OpName(const OpType op) {
  switch (op) {
    case OpType::Add:
      return "add";
    case OpType::Subtract:
      return "subtract";
    case OpType::Intersect:
      return "intersect";
  }
  return "unknown";
}

inline std::string DumpPrefix(const char* kind, const char* stage, const OpType op,
                              std::atomic<uint64_t>& counter) {
  const uint64_t id = counter.fetch_add(1, std::memory_order_relaxed);
  std::ostringstream prefix;
  prefix << kind << "_" << std::setw(6) << std::setfill('0') << id << "_"
         << stage << "_" << OpName(op);
  return prefix.str();
}

template <typename T, typename WriteValue>
inline void DumpVector(const std::filesystem::path& path, const Vec<T>& data,
                       WriteValue writeValue) {
  std::ofstream out(path);
  if (!out.good()) return;
  for (const T& value : data) {
    writeValue(out, value);
    out << "\n";
  }
}

inline void DumpIntVector(const std::filesystem::path& path, const Vec<int>& data) {
  DumpVector(path, data, [](std::ofstream& out, int value) { out << value; });
}

inline void DumpPairVector(const std::filesystem::path& path,
                           const Vec<std::array<int, 2>>& data) {
  DumpVector(path, data, [](std::ofstream& out, const std::array<int, 2>& value) {
    out << value[0] << " " << value[1];
  });
}

inline void DumpVec3HexVector(const std::filesystem::path& path,
                              const Vec<vec3>& data) {
  DumpVector(path, data, [](std::ofstream& out, const vec3& value) {
    char x[128];
    char y[128];
    char z[128];
    std::snprintf(x, sizeof(x), "%.13a", value.x);
    std::snprintf(y, sizeof(y), "%.13a", value.y);
    std::snprintf(z, sizeof(z), "%.13a", value.z);
    out << x << " " << y << " " << z;
  });
}

inline void DumpImplObj(const std::filesystem::path& path,
                        const Manifold::Impl& impl) {
  std::ofstream out(path);
  if (out.good()) out << impl;
}

}  // namespace manifold::debug
