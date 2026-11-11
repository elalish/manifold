#define FILLET_REJECTION_DIAGNOSTICS
#include "main.cpp"
#include "boolean2_diagnostics.h"
#include <cstring>
#include <tbb/global_control.h>

// Exact coordinate/order comparison: parallel output must equal one-thread
// output, not just agree within an area tolerance.
bool SamePolygons(const Polygons& a, const Polygons& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].size() != b[i].size()) return false;
    for (size_t j = 0; j < a[i].size(); ++j)
      if (a[i][j].x != b[i][j].x || a[i][j].y != b[i][j].y) return false;
  }
  return true;
}

uint64_t PolygonHash(const Polygons& polygons) {
  uint64_t hash = 14695981039346656037ull;
  auto add = [&](uint64_t bits) { hash = (hash ^ bits) * 1099511628211ull; };
  add(polygons.size());
  for (const auto& loop : polygons) {
    add(loop.size());
    for (const auto& p : loop) for (double coordinate : {p.x, p.y}) {
      uint64_t bits;
      std::memcpy(&bits, &coordinate, sizeof(bits));
      add(bits);
    }
  }
  return hash;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: offset_parallel REPO OUTPUT.csv [--large-only] [--verify-only]\n";
    return 1;
  }
  try {
    bool largeOnly = false, verifyOnly = false;
    for (int i = 3; i < argc; ++i) {
      largeOnly |= std::string(argv[i]) == "--large-only";
      verifyOnly |= std::string(argv[i]) == "--verify-only";
    }
    auto cases = Cases(argv[1], true);
    for (int count : {128, 4096}) {
      Polygons p;
      for (int i = 0; i < count; ++i) {
        const double x = (i % 64) * 20, y = (i / 64) * 20;
        p.push_back({{x,y},{x+10,y},{x+10,y+10},{x,y+10}});
      }
      Add(cases, "disjoint/" + std::to_string(count), p, 1, false);
    }
    std::ofstream out(argv[2]);
    if (!out) throw std::runtime_error("Cannot open output");
    out << std::setprecision(12);
    out << "case,input_vertices,input_contours,threads,median_us,min_us,repetitions,exact_match,output_vertices,cleanup_us,merge_us,remap_us,bvh_us,broad_us,narrow_us,sweep_us,output_hash\n";
    size_t done = 0;
    for (const auto& c : cases) {
      if (largeOnly && Vertices(c.polygons) < 10000) continue;
      Polygons reference;
      for (int threads : {1, 2, 4, 8}) {
        tbb::global_control control(tbb::global_control::max_allowed_parallelism, threads);
        auto& phases = GlobalPhases();
        phases.Reset();
        SetTimingEnabled(true);
        auto warm = Run(c, 1);
        SetTimingEnabled(false);
        if (threads == 1) reference = warm;
        bool exact = SamePolygons(warm, reference);
        std::vector<double> times;
        double total = 0;
        if (verifyOnly) times.push_back(NAN);
        else do {
          const auto start = Clock::now();
          const auto p = Run(c, 1);
          const double us = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
          times.push_back(us);
          total += us;
          exact &= SamePolygons(p, reference);
          sink += Vertices(p);
        } while (times.size() < 5 || (times.size() < 21 && total < 20000));
        std::sort(times.begin(), times.end());
        out << c.name << ',' << Vertices(c.polygons) << ',' << c.polygons.size()
            << ',' << threads << ',' << times[times.size()/2] << ',' << times.front()
            << ',' << (verifyOnly ? 0 : times.size()) << ',' << exact << ',' << Vertices(warm)
            << ',' << phases.totalNs.load()/1000.0 << ',' << phases.mergeNs.load()/1000.0
            << ',' << phases.remapNs.load()/1000.0 << ',' << phases.bvhBuildNs.load()/1000.0
            << ',' << phases.broadPairWorkNs.load()/1000.0 << ',' << phases.narrowPhaseNs.load()/1000.0
            << ',' << phases.filterWindingNs.load()/1000.0 << ',' << PolygonHash(warm) << '\n';
        out.flush();
        if (!exact) throw std::runtime_error("Parallel output mismatch: " + c.name);
      }
      if (++done % 100 == 0) std::cerr << "Completed " << done << " cases\n";
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
