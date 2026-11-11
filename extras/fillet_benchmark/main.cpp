#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "clipper2/clipper.h"
#include "clipper2/clipper.version.h"
#include "manifold/cross_section.h"
#include "polygon_corpus.h"

using namespace manifold;
namespace C = Clipper2Lib;
using Clock = std::chrono::steady_clock;
constexpr double pi = 3.14159265358979323846;
constexpr int segments = 32;
volatile size_t sink = 0;

struct Case {
  std::string name;
  CrossSection input;
  Polygons polygons;
  double radius;
};

C::PathsD Paths(const Polygons& polygons, vec2 origin = {0, 0}, double scale = 1) {
  C::PathsD out;
  for (const auto& p : polygons) {
    out.emplace_back();
    for (const auto& v : p)
      out.back().emplace_back((v.x - origin.x) * scale, (v.y - origin.y) * scale);
  }
  return out;
}

// Adaptive integer scale: about 1e9 units per input span, with room for offsets.
// Conversion is part of the Clipper timing; input construction is not timed.
Polygons ClipperFillet(const Case& c) {
  const auto bounds = c.input.Bounds();
  const vec2 origin = bounds.min;
  const double scale = 1e9 / std::max({bounds.Size().x, bounds.Size().y,
                                     std::abs(c.radius)});
  int error = 0;
  auto paths = C::ScalePaths<int64_t, double>(Paths(c.polygons, origin), scale, error);
  if (error) throw std::runtime_error("Clipper input scaling");
  const double r = c.radius * scale;
  const double arcTolerance = std::abs(r) * (1 - std::cos(pi / segments));
  paths = C::InflatePaths(paths, -r, C::JoinType::Round, C::EndType::Polygon,
                          2.0, arcTolerance);
  paths = C::InflatePaths(paths, r, C::JoinType::Round, C::EndType::Polygon,
                          2.0, arcTolerance);
  Polygons result;
  for (const auto& path : paths) {
    result.emplace_back();
    for (auto p : path)
      result.back().emplace_back(p.x / scale + origin.x, p.y / scale + origin.y);
  }
  return result;
}

Polygons Run(const Case& c, int method) {
  if (method == 2) return ClipperFillet(c);
  if (method == 1)
    return c.input.Offset(-c.radius, JoinType::Round, 2, segments)
        .Offset(c.radius, JoinType::Round, 2, segments).ToPolygons();
  Polygons result;
  for (const auto& piece : c.input.Fillet(c.radius, segments)) {
    auto p = piece.ToPolygons();
    result.insert(result.end(), p.begin(), p.end());
  }
  return result;
}

size_t Vertices(const Polygons& p) {
  size_t n = 0;
  for (const auto& loop : p) n += loop.size();
  return n;
}

C::Paths64 ValidationPaths(const Polygons& p, vec2 origin, double scale) {
  int error = 0;
  auto result = C::ScalePaths<int64_t, double>(Paths(p, origin, scale), 1e12, error);
  if (error) throw std::runtime_error("Validation scaling");
  return result;
}

double Perimeter(const C::Paths64& paths) {
  double result = 0;
  for (const auto& p : paths) for (size_t i = 0; i < p.size(); ++i) {
    const auto a = p[i], b = p[(i + 1) % p.size()];
    result += std::hypot(double(a.x - b.x), double(a.y - b.y));
  }
  return result;
}

bool Valid(const Polygons& p) {
  for (const auto& loop : p) {
    if (loop.size() < 3) return false;
    for (size_t i = 0; i < loop.size(); ++i) {
      const auto a = loop[i], b = loop[(i + 1) % loop.size()];
      if (!std::isfinite(a.x) || !std::isfinite(a.y) || (a.x == b.x && a.y == b.y))
        return false;
    }
  }
  return true;
}

int SharpTurns(const Polygons& p, double radius) {
  int count = 0;
  for (const auto& loop : p) {
    for (size_t i = 0; i < loop.size(); ++i) {
      const auto a = loop[(i + loop.size() - 1) % loop.size()];
      const auto b = loop[i], c = loop[(i + 1) % loop.size()];
      const auto u = b - a, v = c - b;
      double turn = std::atan2(u.x * v.y - u.y * v.x, u.x * v.x + u.y * v.y);
      if (radius < 0) turn = -turn;
      if (turn > 1.1 * 2 * pi / segments) ++count;
    }
  }
  return count;
}

void Add(std::vector<Case>& cases, const std::string& name,
         const Polygons& polygons, double maxRadius, bool sweep = true) {
  CrossSection input(polygons);
  if (input.IsEmpty() || !std::isfinite(input.Area()) || maxRadius <= 0) return;
  const auto p = input.ToPolygons();
  for (double fraction : {0.1, 0.5, 0.9, 1.01, -0.1, -0.5, -0.9}) {
    if (!sweep && fraction != 0.5 && fraction != -0.5) continue;
    cases.push_back({name + "/" + std::to_string(fraction), input, p,
                     maxRadius * fraction});
  }
}

std::vector<Case> Cases(const std::string& root, bool large) {
  std::vector<Case> cases;
  for (const auto& file : {"fillet", "polygon_corpus"}) {
    auto corpus = ReadPolygonCorpus(root + "/test/polygons/" + file + ".txt");
    if (corpus.empty()) throw std::runtime_error("Missing corpus: " + root);
    for (const auto& e : corpus)
      Add(cases, std::string(file) + "/" + e.name, e.polys, e.maxCircleRadius);
  }
  for (double scale : {1e-6, 1.0, 1e6}) {
    for (double shift : {0.0, 1e8}) {
      if (scale == 1e-6 && shift != 0) continue;
      for (const auto& shape : std::vector<std::pair<std::string, Polygons>>{
        {"square", {{{0,0},{10,0},{10,10},{0,10}}}},
        {"L", {{{0,0},{6,0},{6,2},{2,2},{2,6},{0,6}}}},
        {"neck", {{{0,0},{4,0},{4,1.9},{6,1.9},{6,0},{10,0},
                    {10,4},{6,4},{6,2.1},{4,2.1},{4,4},{0,4}}}},
        {"hole", {{{0,0},{10,0},{10,10},{0,10}},
                    {{4,4},{4,6},{6,6},{6,4}}}},
        {"tiny_edge", {{{0,0},{5,0},{5+1e-9,1e-10},{10,0},{10,10},{0,10}}}}
      }) {
        auto p = shape.second;
        for (auto& loop : p) for (auto& v : loop) v = v * scale + vec2(shift);
        Add(cases, "stress/" + shape.first + "/s=" + std::to_string(scale) +
            "/t=" + std::to_string(shift), p, scale);
      }
    }
  }
  std::mt19937 rng(20261109);
  std::uniform_real_distribution<double> radial(3, 10);
  for (int n : {16, 64, 256}) {
    for (int seed = 0; seed < 8; ++seed) {
      SimplePolygon p;
      for (int i = 0; i < n; ++i) {
        double a = 2 * pi * i / n, r = radial(rng);
        p.emplace_back(r * std::cos(a), r * std::sin(a));
      }
      Add(cases, "random/n=" + std::to_string(n) + "/" + std::to_string(seed),
          {p}, 1.0, false);
    }
  }
  if (large) {
    for (const auto& file : {"sponge", "zebra", "zebra3"}) {
      auto corpus = ReadPolygonCorpus(root + "/test/polygons/" + file + ".txt");
      if (corpus.empty()) throw std::runtime_error("Missing large corpus");
      for (const auto& e : corpus)
        Add(cases, std::string(file) + "/" + e.name, e.polys, e.maxCircleRadius, false);
    }
  }
  return cases;
}

#ifndef FILLET_REJECTION_DIAGNOSTICS
int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: fillet_benchmark REPO OUTPUT.csv [--large] [--clipper-gt] [case-substring]\n";
    return 1;
  }
  try {
    bool large = false;
    int referenceMethod = 0;
    std::string filter;
    for (int i = 3; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--large") large = true;
      else if (arg == "--clipper-gt") referenceMethod = 2;
      else filter = arg;
    }
    const auto cases = Cases(argv[1], large);
    std::ofstream out(argv[2]);
    if (!out) throw std::runtime_error("Cannot open output");
    out << std::setprecision(12);
    out << "case,method,input_vertices,radius,median_us,min_us,repetitions,empty,vertices,contours,valid,sharp_turns,area,reference_empty,reference_contours,xor_over_input,containment_over_input,regularization_error,oracle_error_bound,error,reference_method\n";
    const std::array<std::string, 3> names{"fillet2d", "boolean2_offset", "clipper2_offset"};
    std::cerr << "Clipper2 " << CLIPPER2_VERSION << ", segments=" << segments
              << ", cases=" << cases.size() << ", reference=" << names[referenceMethod] << "\n";
    size_t done = 0;
    for (const auto& c : cases) {
      if (!filter.empty() && c.name.find(filter) == std::string::npos) continue;
      std::array<Polygons, 3> results;
      std::array<std::string, 3> errors;
      std::array<std::vector<double>, 3> times;
      // Rotate method order per case; one untimed warmup, at least 5 samples,
      // up to 21 samples or 20ms per method. No validation in timed region.
      for (int method = 0; method < 3; ++method) {
        try { results[method] = Run(c, method); }
        catch (const std::exception& e) { errors[method] = e.what(); }
      }
      for (int k = 0; k < 3; ++k) {
        const int method = (k + done) % 3;
        if (!errors[method].empty()) continue;
        double total = 0;
        try {
          do {
            auto start = Clock::now();
            auto p = Run(c, method);
            auto end = Clock::now();
            sink += Vertices(p);
            const double us = std::chrono::duration<double, std::micro>(end - start).count();
            times[method].push_back(us);
            total += us;
          } while (times[method].size() < 5 ||
                   (times[method].size() < 21 && total < 20000));
        } catch (const std::exception& e) { errors[method] = e.what(); }
      }
      const auto bounds = c.input.Bounds();
      const double scale = 1 / std::max(bounds.Size().x, bounds.Size().y);
      const auto input = ValidationPaths(c.polygons, bounds.min, scale);
      const auto reference = ValidationPaths(results[referenceMethod], bounds.min, scale);
      const double inputArea = std::abs(C::Area(input));
      for (int method = 0; method < 3; ++method) {
        auto& t = times[method];
        std::sort(t.begin(), t.end());
        const auto& p = results[method];
        double difference = NAN, containment = NAN, regularization = NAN, area = NAN;
        double oracleError = NAN;
        const bool valid = Valid(p) && errors[method].empty();
        if (valid) {
          const auto paths = ValidationPaths(p, bounds.min, scale);
          area = C::Area(paths) / (1e24 * scale * scale);
          // Conservative grid-displacement estimate, not a certified bound on
          // the clipping algorithm. Mark ill-conditioned area ratios in reports.
          oracleError = 4 * (Perimeter(input) + Perimeter(reference) + Perimeter(paths)) / inputArea;
          difference = std::abs(C::Area(C::Xor(paths, reference, C::FillRule::NonZero))) / inputArea;
          const auto escaped = c.radius > 0
              ? C::Difference(paths, input, C::FillRule::NonZero)
              : C::Difference(input, paths, C::FillRule::NonZero);
          containment = std::abs(C::Area(escaped)) / inputArea;
          regularization = std::abs(C::Area(C::Union(paths, C::FillRule::NonZero))
                                      - C::Area(paths)) / inputArea;
        }
        out << c.name << ',' << names[method] << ',' << Vertices(c.polygons) << ','
            << c.radius << ',' << (t.empty() ? NAN : t[t.size()/2]) << ','
            << (t.empty() ? NAN : t.front()) << ',' << t.size() << ',' << p.empty()
            << ',' << Vertices(p) << ',' << p.size() << ',' << valid << ','
            << SharpTurns(p, c.radius) << ',' << area << ',' << results[referenceMethod].empty()
            << ',' << results[referenceMethod].size() << ',' << difference << ',' << containment
            << ',' << regularization << ',' << oracleError << ','
            << (errors[method].empty() ? "" : "exception") << ','
            << names[referenceMethod] << '\n';
      }
      out.flush();
      if (++done % 25 == 0) std::cerr << "Completed " << done << "/" << cases.size() << '\n';
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
#endif
