// Inspect the unchanged validator's decision on a saved corpus polygon.
#include <iomanip>
#include <iostream>
#include "manifold/cross_section.h"
#include "polygon_corpus.h"
#include "rejection_diagnostics.h"
using namespace manifold;
int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: fillet_rejection_probe CORPUS CASE [SEGMENTS]\n";
    return 2;
  }
  const int segments = argc > 3 ? std::stoi(argv[3]) : 20;
  for (const auto& c : ReadPolygonCorpus(argv[1])) {
    if (c.name != argv[2]) continue;
    CrossSection input(c.polys);
    std::cout << std::setprecision(17)
      << "factor,returned_pieces,reason,input_area,raw_area,sharp_turns,max_turn_degrees\n";
    for (double f : {-.5,-1.,-2.,-5.,-10.,-20.,-50.,-100.,.1,.3,.5,.7,.9}) {
      const auto result = input.Fillet(f*c.maxCircleRadius, segments);
      const auto& d = fillet_benchmark::rejectionDiagnostics;
      std::cout << f << ',' << result.size() << ',' << d.reason << ','
        << d.inputArea << ',' << d.rawArea << ',' << d.sharpTurns << ','
        << d.maxTurnDegrees << '\n';
    }
    return 0;
  }
  return 2;
}
