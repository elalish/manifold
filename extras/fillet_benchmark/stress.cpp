// Deterministic similarity-transform stress checks for CrossSection::Fillet.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include "manifold/cross_section.h"
#include "polygon_corpus.h"
using namespace manifold;
constexpr double pi = 3.14159265358979323846;
const double factors[] = {-0.5, -1, -2, -5, -10, -20, -50, -100,
                          .1, .3, .5, .7, .9};
struct Result { bool empty; double area; int invalid = 0, sharp = 0; };
void Dump(const Polygons& polygons, double radius, const std::string& path) {
  std::ofstream out(path);
  out << std::setprecision(17) << "REPRO 0 -1 " << polygons.size()
      << ' ' << radius << '\n';
  for (const auto& loop : polygons) {
    out << loop.size() << '\n';
    for (auto p : loop) out << p.x << ' ' << p.y << '\n';
  }
}
Result Run(const CrossSection& input, double radius, int segments) {
  const auto pieces = input.Fillet(radius, segments);
  Result out{pieces.empty(), 0};
  for (const auto& piece : pieces) {
    out.area += piece.Area();
    for (const auto& loop : piece.ToPolygons()) {
      if (loop.size() < 3) ++out.invalid;
      for (size_t i = 0; i < loop.size(); ++i) {
        const auto p = loop[i], q = loop[(i+1)%loop.size()];
        const auto r = loop[(i+2)%loop.size()];
        const auto a = q-p, b = r-q;
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || length2(a) == 0)
          ++out.invalid;
        double turn = std::atan2(a.x*b.y-a.y*b.x, dot(a,b));
        if (radius > 0 && turn >= 2*pi/segments*1.1 && turn <= pi)
          ++out.sharp;
      }
    }
  }
  if (!std::isfinite(out.area)) ++out.invalid;
  return out;
}
int main(int argc, char** argv) {
  if (argc < 3) return 2;
  auto corpus = ReadPolygonCorpus(argv[1]);
  if (std::string(argv[2]) == "--list") {
    for (const auto& c : corpus) {
      size_t n = 0; for (const auto& p : c.polys) n += p.size();
      if (c.maxCircleRadius > 0) std::cout << c.name << " " << n << "\n";
    }
    return corpus.empty() ? 2 : 0;
  }
  // file case first-seed seed-count transforms segments [first-failure-dump]
  if (argc < 8) return 2;
  const auto it = std::find_if(corpus.begin(), corpus.end(), [&](const auto& c) {
    return c.name == argv[2]; });
  if (it == corpus.end() || !(it->maxCircleRadius > 0)) return 2;
  const int first = std::stoi(argv[3]), seeds = std::stoi(argv[4]);
  const int count = std::stoi(argv[5]), segments = std::stoi(argv[6]);
  // argv[7] is the output prefix, kept mandatory for isolated parallel jobs.
  const std::string prefix = argv[7];
  const bool strict = argc > 8 && std::string(argv[8]) == "--strict";
  const auto start = std::chrono::steady_clock::now();
  CrossSection base(it->polys);
  Result reference[13];
  for (int j = 0; j < 13; ++j)
    reference[j] = Run(base, factors[j]*it->maxCircleRadius, segments);
  long calls=0, empty=0, flips=0, invalid=0, sharp=0, direction=0, drift=0;
  double maxDrift=0;
  int details=0;
  std::ofstream events(prefix+".events.csv");
  events << "seed,transform,factor,kind,relative_area_drift\n";
  std::ofstream strictEvents;
  if (strict) {
    strictEvents.open(prefix+".strict.csv");
    strictEvents << "seed,transform,factor,kind\n";
  }
  for (int seed=first; seed<first+seeds; ++seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0,1);
    for (int k=0; k<count; ++k) {
      const double angle=u(rng)*2*pi, scale=std::exp((u(rng)-.5)*4);
      const double tx=(u(rng)-.5)*100, ty=(u(rng)-.5)*100;
      Polygons q=it->polys;
      for (auto& loop:q) for (auto& p:loop) {
        const auto v=p;
        p={scale*(std::cos(angle)*v.x-std::sin(angle)*v.y)+tx,
           scale*(std::sin(angle)*v.x+std::cos(angle)*v.y)+ty};
      }
      CrossSection cs(q);
      // Optional transform index after --strict, for exact input replay.
      if (argc > 9 && k == std::stoi(argv[9]))
        Dump(q, it->maxCircleRadius*scale, prefix+".replay.txt");
      const double inputArea=cs.Area();
      double previousArea=inputArea;
      for (int j=0;j<13;++j) {
        const auto result=Run(cs, factors[j]*it->maxCircleRadius*scale, segments);
        ++calls; empty+=result.empty;
        if (strict && (result.empty || (j<8 &&
            result.area < previousArea*(1-1e-9)))) {
          strictEvents << seed << ',' << k << ',' << factors[j] << ','
                       << (result.empty?"empty":"negative_area_decrease") << '\n';
          Dump(q, it->maxCircleRadius*scale,
               prefix+".failure-s"+std::to_string(seed)+"-t"+
               std::to_string(k)+".txt");
        }
        if (j<8 && !result.empty) previousArea=result.area;
        const bool flip=result.empty!=reference[j].empty;
        const bool wrong=!result.empty &&
          (factors[j]>0 ? result.area>=inputArea : result.area<=inputArea);
        double delta=0;
        if (!result.empty && !reference[j].empty) {
          delta=std::abs(result.area/(scale*scale)-reference[j].area)/
                std::max(base.Area(),1e-100);
          maxDrift=std::max(maxDrift,delta);
        }
        flips+=flip; invalid+=result.invalid>0; sharp+=result.sharp>0;
        direction+=wrong; drift+=delta>1e-4;
        if (flip || result.invalid || result.sharp || wrong || delta>1e-4) {
          if (details++<100) events << seed << ',' << k << ',' << factors[j]
            << ',' << (flip?"empty_flip":result.invalid?"invalid":
                        result.sharp?"sharp":wrong?"area_direction":"area_drift")
            << ',' << std::setprecision(17) << delta << '\n';
          if (details==1) {
            Dump(q, it->maxCircleRadius*scale, prefix+".polygon.txt");
          }
        }
      }
    }
  }
  double seconds=std::chrono::duration<double>(
      std::chrono::steady_clock::now()-start).count();
  std::cout << calls << ',' << empty << ',' << flips << ',' << invalid << ','
    << sharp << ',' << direction << ',' << drift << ',' << std::setprecision(17)
    << maxDrift << ',' << seconds << '\n';
}
