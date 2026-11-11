#define FILLET_REJECTION_DIAGNOSTICS
#include "main.cpp"
#include "rejection_diagnostics.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: fillet_rejections REPO OUTPUT.csv\n";
    return 1;
  }
  try {
    auto cases = Cases(argv[1], true);
    std::ofstream out(argv[2]);
    std::ofstream raw(std::string(argv[2]) + ".raw.txt");
    if (!out || !raw) throw std::runtime_error("Cannot open diagnostic output");
    out << std::setprecision(17);
    raw << std::setprecision(17);
    out << "case,input_vertices,radius,reason,returned_empty,raw_vertices,raw_contours,geometry_valid,angles_valid,area_valid,sharp_turns,max_turn_degrees,angle_limit_degrees,input_area,raw_area,clipper_empty,clipper_contours\n";
    size_t done = 0;
    for (const auto& c : cases) {
      const auto result = Run(c, 0);
      const auto& d = manifold::fillet_benchmark::rejectionDiagnostics;
      const auto gt = Run(c, 2);
      if ((d.reason == "accepted") == result.empty())
        throw std::runtime_error("Diagnostic decision disagrees with returned result");
      out << c.name << ',' << Vertices(c.polygons) << ',' << c.radius << ','
          << d.reason << ',' << result.empty() << ',' << Vertices(d.raw) << ','
          << d.raw.size() << ',' << d.geometryValid << ',' << d.anglesValid << ','
          << d.areaValid << ',' << d.sharpTurns << ',' << d.maxTurnDegrees << ','
          << 1.1 * 360.0 / segments << ',' << c.input.Area() << ',' << d.rawArea
          << ',' << gt.empty() << ',' << gt.size() << '\n';
      if (!d.anglesValid && !d.raw.empty()) {
        // Corpus-compatible blocks containing rejected pre-validation polygons.
        raw << c.name << " -1 -1 " << d.raw.size() << ' ' << std::abs(c.radius) << '\n';
        for (const auto& loop : d.raw) {
          raw << loop.size() << '\n';
          for (const auto& p : loop) raw << p.x << ' ' << p.y << '\n';
        }
      }
      out.flush();
      if (++done % 100 == 0) std::cerr << "Completed " << done << '/' << cases.size() << '\n';
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
