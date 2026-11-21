#!/usr/bin/env python3
"""Instrument a build-local source copy; keep production fillet2d untouched."""
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
start = source.index("bool IsValidFilletResult(")
end = source.index("\n#ifdef MANIFOLD_DEBUG\nvoid SavePolygons", start)
original = source[start:end]
# Fail explicitly if the validation changes; instrumentation must mirror it.
expected = '''bool IsValidFilletResult(const std::vector<CrossSection>& result,
                         double originalArea, int circularSegments,
                         bool invert) {
  if (result.empty()) return false;
  for (const CrossSection& crossSection : result) {
    if (!HasValidFilletGeometry(crossSection)) return false;
    if (!HasFilletAngles(crossSection, circularSegments, invert)) return false;
  }

  // Decomposed pieces are disjoint, so their areas add up.
  double area = 0;
  for (const CrossSection& crossSection : result) area += crossSection.Area();
  return invert ? area > originalArea : area < originalArea;
}
'''
assert original == expected, "Validator changed; update diagnostic instrumentation"
replacement = r'''bool IsValidFilletResult(const std::vector<CrossSection>& result,
                         double originalArea, int circularSegments,
                         bool invert) {
  auto& d = manifold::fillet_benchmark::rejectionDiagnostics;
  d.inputArea = originalArea;
  if (result.empty()) {
    d.reason = "trace_empty";
    return false;
  }
  d.geometryValid = true;
  d.anglesValid = true;
  const double limit = 1.1 * 2.0 * M_PI / circularSegments;
  for (const CrossSection& piece : result) {
    d.geometryValid &= HasValidFilletGeometry(piece);
    d.anglesValid &= HasFilletAngles(piece, circularSegments, invert);
    d.rawArea += piece.Area();
    for (const auto& loop : piece.ToPolygons()) {
      d.raw.push_back(loop);
      for (size_t i = 0; i < loop.size(); ++i) {
        const vec2 e1 = loop[(i + 1) % loop.size()] - loop[i];
        const vec2 e2 = loop[(i + 2) % loop.size()] - loop[(i + 1) % loop.size()];
        const double turn = ToRad(e2) - ToRad(e1);
        const double angle = NormalizeAngle(invert ? -turn : turn);
        if (angle <= M_PI) {
          d.maxTurnDegrees = std::max(d.maxTurnDegrees, angle * 180.0 / M_PI);
          if (!(angle < limit)) ++d.sharpTurns;
        }
      }
    }
  }
  d.areaValid = invert ? d.rawArea > d.inputArea : d.rawArea < d.inputArea;
  // Keep the original decision order and strict comparisons unchanged.
  for (const CrossSection& piece : result) {
    if (!HasValidFilletGeometry(piece)) {
      d.reason = "invalid_geometry";
      return false;
    }
    if (!HasFilletAngles(piece, circularSegments, invert)) {
      d.reason = "sharp_angle";
      return false;
    }
  }
  d.reason = d.areaValid ? "accepted" :
      (d.rawArea == d.inputArea ? "area_unchanged" : "area_wrong_direction");
  return d.areaValid;
}
'''
source = source[:start] + replacement + source[end:]
signature = '''int circularSegments) const {
  try {
    return FilletImpl(ToPolygons(), Area(), radius, circularSegments);'''
assert source.count(signature) == 1
source = source.replace(signature, '''int circularSegments) const {
  manifold::fillet_benchmark::rejectionDiagnostics = {};
  try {
    return FilletImpl(ToPolygons(), Area(), radius, circularSegments);''')
catch = '''  } catch (const std::exception&) {
    return {};
  }
}'''
assert source.count(catch) == 1
source = source.replace(catch, '''  } catch (const std::exception&) {
    manifold::fillet_benchmark::rejectionDiagnostics.reason = "exception";
    return {};
  }
}''')
Path(sys.argv[2]).write_text('#include "rejection_diagnostics.h"\n' + source)
