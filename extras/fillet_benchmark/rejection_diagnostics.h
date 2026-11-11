#pragma once

#include <string>

#include "manifold/cross_section.h"

namespace manifold::fillet_benchmark {
struct RejectionDiagnostics {
  std::string reason = "input_rejected";
  bool geometryValid = false;
  bool anglesValid = false;
  bool areaValid = false;
  int sharpTurns = 0;
  double maxTurnDegrees = 0;
  double rawArea = 0;
  double inputArea = 0;
  Polygons raw;
};
inline thread_local RejectionDiagnostics rejectionDiagnostics;
}  // namespace manifold::fillet_benchmark
