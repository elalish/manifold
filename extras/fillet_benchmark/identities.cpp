// Identity checks for CrossSection::Fillet, after the way the 2D booleans are
// tested: not "is there a result", but relations a correct result must
// satisfy, on inputs where exact ties are kept exact.
//
//   identities <corpus file>... [--generated N] [--segments S] [--only NAME]
//
// For every shape and radius:
//  - contain:  a positive radius never adds material and a negative one never
//              removes any: area(out - in) = 0, resp. area(in - out) = 0.
//  - dual:     closing the shape is opening its complement, and the other way
//              round: Fillet(P, r) against Box - Fillet(Box - P, -r), compared
//              away from the box's own corners. The two go through opposite
//              signs of the same code.
//  - exact:    the eight axis symmetries and scaling by a power of two change
//              no bit of any decision's inputs beyond sign and exponent, so
//              the result must be the same shape: same emptiness, same area.
//  - order:    starting each loop at another vertex and listing the loops in
//              reverse is the same shape.
//  - nudge:    at the radii the shape itself makes critical (half a
//              vertex-vertex or vertex-edge distance), and one unit in the
//              last place to either side, the result must still satisfy all of
//              the above. Whether the ball fits at the tie is free; the result
//              on either side of it is not.
//
// Differences are reported relative to the input area, in decades, with the
// worst cases written out for replay. Arc tessellation is identical on both
// sides of every comparison here, so there is no discretization allowance.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "manifold/cross_section.h"
#include "polygon_corpus.h"

using namespace manifold;

namespace {

struct Case {
  std::string name;
  Polygons polys;
  std::vector<double> radii;
};

double Area(const std::vector<CrossSection>& pieces) {
  double a = 0;
  for (const auto& p : pieces) a += p.Area();
  return a;
}

CrossSection Join(const std::vector<CrossSection>& pieces) {
  return CrossSection::BatchBoolean(pieces, OpType::Add);
}

// One of the eight axis symmetries, then a scale by a power of two. Exact.
Polygons Transform(const Polygons& in, int symmetry, double scale) {
  Polygons out = in;
  const bool swap = symmetry & 4;
  const double sx = symmetry & 1 ? -1 : 1, sy = symmetry & 2 ? -1 : 1;
  const bool flips = (sx * sy < 0) != swap;
  for (auto& loop : out) {
    for (auto& p : loop) {
      const vec2 v = swap ? vec2(p.y, p.x) : p;
      p = vec2(sx * v.x * scale, sy * v.y * scale);
    }
    if (flips) std::reverse(loop.begin(), loop.end());
  }
  return out;
}

Polygons Reorder(const Polygons& in) {
  Polygons out = in;
  for (auto& loop : out)
    std::rotate(loop.begin(), loop.begin() + loop.size() / 2, loop.end());
  std::reverse(out.begin(), out.end());
  return out;
}

void Flag(const std::string& line);

struct Tally {
  // Decades of difference / input area: <=1e-12, <=1e-9, <=1e-6, <=1e-3, more.
  long buckets[5] = {0, 0, 0, 0, 0};
  long emptyMismatch = 0;
  long count = 0;
  struct Worst {
    double value;
    std::string what;
  };
  std::vector<Worst> worst;

  void Add(double relative, const std::string& what) {
    ++count;
    const int b = !(relative > 1e-12)  ? 0
                  : !(relative > 1e-9) ? 1
                  : !(relative > 1e-6) ? 2
                  : !(relative > 1e-3) ? 3
                                       : 4;
    ++buckets[b];
    if (b >= 2) {
      Keep(relative, what);
      Flag(what + " diff " + std::to_string(relative));
    }
  }
  void Mismatch(const std::string& what) {
    ++count;
    ++emptyMismatch;
    Keep(std::numeric_limits<double>::infinity(), what);
    Flag(what + " one-side-empty");
  }
  void Keep(double value, const std::string& what) {
    worst.push_back({value, what});
    std::sort(worst.begin(), worst.end(),
              [](const Worst& a, const Worst& b) { return a.value > b.value; });
    if (worst.size() > 12) worst.pop_back();
  }
};

std::map<std::string, Tally> tallies;
// Every flagged comparison, for --log.
std::vector<std::string> flagged;
// Not a multiple of anything common: an arc of a right angle or a half-turn
// then never lands exactly on a whole number of segments, where rounding
// would choose the count.
int segments = 37;

void Flag(const std::string& line) { flagged.push_back(line); }
std::string dumpDir;
long dumped = 0;

std::string Describe(const std::string& name, double radius,
                     const std::string& extra) {
  std::ostringstream s;
  s << name << " r=" << std::setprecision(17) << radius << " " << extra;
  return s.str();
}

void Dump(const Case& c, double radius, const std::string& check) {
  if (dumpDir.empty() || dumped >= 200) return;
  std::ofstream out(dumpDir + "/" + std::to_string(dumped++) + "-" + check +
                    "-" + c.name + ".txt");
  out << std::setprecision(17) << "REPRO 0 -1 " << c.polys.size() << ' '
      << radius << '\n';
  for (const auto& loop : c.polys) {
    out << loop.size() << '\n';
    for (auto p : loop) out << p.x << ' ' << p.y << '\n';
  }
}

// Fillet, with "no result" told apart from "no change" by the caller.
std::vector<CrossSection> Run(const CrossSection& cs, double radius) {
  return cs.Fillet(radius, segments);
}

// Generated shapes are tallied by family (grid, turned); corpus shapes
// together.
// Grid shapes are split once more: at a half-integer radius every gap that
// wide is an exact tie and the ball may fit or not; one unit in the last
// place to either side nothing is tied, and every number involved is exact.
std::string Key(const Case& c, const std::string& check, double radius) {
  const size_t digit = c.name.find_first_of("0123456789");
  const std::string family = c.name.substr(0, digit);
  if (family == "turned") return check + " turned";
  if (family != "grid") return check;
  const double r4 = std::abs(radius) * 4;
  return check + (r4 == std::floor(r4) ? " grid tie" : " grid off");
}

void Check(const Case& c, double radius) {
  const CrossSection input(c.polys);
  if (input.IsEmpty()) return;
  const double inArea = input.Area();
  if (!(inArea > 0)) return;
  const auto pieces = Run(input, radius);
  const bool empty = pieces.empty();
  const CrossSection out = empty ? CrossSection() : Join(pieces);

  // contain
  if (!empty) {
    const double escaped =
        radius > 0 ? (out - input).Area() : (input - out).Area();
    tallies[Key(c, "contain", radius)].Add(escaped / inArea,
                                           Describe(c.name, radius, ""));
    if (escaped / inArea > 1e-9) Dump(c, radius, "contain");
  }

  // dual
  {
    const Rect bounds = input.Bounds();
    const double r = std::abs(radius);
    const vec2 size = bounds.max - bounds.min;
    const vec2 outerMin = bounds.min - vec2(8 * r + size.x, 8 * r + size.y);
    const vec2 outerSize = size * 3.0 + vec2(16 * r, 16 * r);
    const CrossSection box =
        CrossSection::Square(outerSize).Translate(outerMin);
    const CrossSection window = CrossSection::Square(size + vec2(4 * r, 4 * r))
                                    .Translate(bounds.min - vec2(2 * r, 2 * r));
    const CrossSection complement = box - input;
    const auto dualPieces = Run(complement, -radius);
    // "No result" is also what Fillet returns where nothing changes and,
    // for a positive radius, where nothing is left. Each side is read as
    // whichever of those the other side agrees with best.
    std::vector<CrossSection> ours, theirs;
    if (!empty) {
      ours.push_back(out);
    } else {
      ours.push_back(input);
      if (radius > 0) ours.push_back(CrossSection());
    }
    if (!dualPieces.empty()) {
      theirs.push_back(box - Join(dualPieces));
    } else {
      theirs.push_back(input);
      if (radius < 0) theirs.push_back(box);
    }
    double diff = std::numeric_limits<double>::infinity();
    for (const auto& x : ours)
      for (const auto& y : theirs)
        diff = std::min(
            diff,
            ((x - y) + (y - x)).Boolean(window, OpType::Intersect).Area());
    const std::string note =
        empty != dualPieces.empty() ? "one side without result" : "";
    tallies[Key(c, "dual", radius)].Add(diff / inArea,
                                        Describe(c.name, radius, note));
    if (diff / inArea > 1e-6) Dump(c, radius, "dual");
  }

  // exact, order
  const double outArea = empty ? 0 : out.Area();
  auto compare = [&](const std::string& check, const Polygons& polys,
                     double scale, const std::string& label) {
    const CrossSection moved(polys);
    const auto other = Run(moved, radius * scale);
    if (other.empty() != empty) {
      tallies[Key(c, check, radius)].Mismatch(Describe(c.name, radius, label));
      Dump(c, radius, check);
      return;
    }
    if (empty) {
      tallies[Key(c, check, radius)].Add(0, "");
      return;
    }
    const double diff = std::abs(Area(other) / (scale * scale) - outArea);
    tallies[Key(c, check, radius)].Add(diff / inArea,
                                       Describe(c.name, radius, label));
    if (diff / inArea > 1e-9) Dump(c, radius, check);
  };
  for (int symmetry = 1; symmetry < 8; ++symmetry)
    compare("exact", Transform(c.polys, symmetry, 1), 1,
            "symmetry " + std::to_string(symmetry));
  compare("exact", Transform(c.polys, 0, 0.125), 0.125, "scale 1/8");
  compare("exact", Transform(c.polys, 0, 32), 32, "scale 32");
  compare("order", Reorder(c.polys), 1, "reordered");
}

// Radii the shape itself makes critical: half the distance between two
// vertices, or between a vertex and an edge's line, where the foot is on the
// edge. A handful of the smallest distinct ones.
std::vector<double> CriticalRadii(const Polygons& polys, size_t limit) {
  std::vector<vec2> points;
  std::vector<std::pair<vec2, vec2>> edges;
  for (const auto& loop : polys)
    for (size_t i = 0; i < loop.size(); ++i) {
      points.push_back(loop[i]);
      edges.push_back({loop[i], loop[(i + 1) % loop.size()]});
    }
  if (points.size() > 400) return {};
  std::vector<double> radii;
  for (size_t i = 0; i < points.size(); ++i) {
    for (size_t j = i + 1; j < points.size(); ++j) {
      const double d = length(points[i] - points[j]);
      if (d > 0) radii.push_back(d / 2);
    }
    for (const auto& e : edges) {
      const vec2 d = e.second - e.first;
      const double t = dot(points[i] - e.first, d) / dot(d, d);
      if (!(t > 0 && t < 1)) continue;
      const double h = std::abs(la::cross(d, points[i] - e.first)) / length(d);
      if (h > 0) radii.push_back(h / 2);
    }
  }
  // Not the distances between vertices that all but coincide: those are
  // below what the boolean that prepares the input tells apart, so the
  // input itself is not the same shape under a symmetry.
  double extent = 0;
  for (const vec2 p : points)
    for (const vec2 q : {points.front(), points.back()})
      extent = std::max(extent, length(p - q));
  radii.erase(std::remove_if(radii.begin(), radii.end(),
                             [&](double r) { return r < 1e-6 * extent; }),
              radii.end());
  std::sort(radii.begin(), radii.end());
  radii.erase(std::unique(radii.begin(), radii.end()), radii.end());
  // Spread the picks over the range rather than taking only the smallest.
  std::vector<double> picked;
  for (size_t k = 0; k < limit && !radii.empty(); ++k)
    picked.push_back(
        radii[k * (radii.size() - 1) / std::max<size_t>(1, limit - 1)]);
  picked.erase(std::unique(picked.begin(), picked.end()), picked.end());
  return picked;
}

void AddNudged(std::vector<double>& radii, double r) {
  for (const double sign : {1.0, -1.0}) {
    radii.push_back(sign * std::nextafter(r, 0.0));
    radii.push_back(sign * r);
    radii.push_back(sign * std::nextafter(r, 2 * r));
  }
}

// Unions and differences of rectangles on an integer grid: every edge is
// axis-aligned and every gap an integer, so half-integer radii are exact
// ties everywhere at once. Raw bits only, so the shapes are the same on every
// platform.
Case GridCase(uint64_t seed) {
  std::mt19937_64 rng(seed);
  auto pick = [&](int n) { return static_cast<int>(rng() % n); };
  CrossSection shape;
  const int adds = 3 + pick(5);
  for (int i = 0; i < adds; ++i) {
    const vec2 at(pick(10), pick(10));
    const vec2 size(1 + pick(5), 1 + pick(5));
    shape = shape + CrossSection::Square(size).Translate(at);
  }
  const int cuts = pick(4);
  for (int i = 0; i < cuts; ++i) {
    const vec2 at(pick(12), pick(12));
    const vec2 size(1 + pick(3), 1 + pick(3));
    shape = shape - CrossSection::Square(size).Translate(at);
  }
  Case c{"grid" + std::to_string(seed), shape.ToPolygons(), {}};
  for (const double r : {0.25, 0.5, 1.0, 1.5, 2.0}) AddNudged(c.radii, r);
  return c;
}

// The same, turned by an angle that is not a multiple of a right angle: the
// ties are then only near-ties, each rounded its own way.
Case TurnedCase(uint64_t seed) {
  Case c = GridCase(seed);
  std::mt19937_64 rng(seed ^ 0x9e3779b97f4a7c15ull);
  const double angle =
      (rng() >> 11) * (1.0 / 9007199254740992.0) * 6.283185307179586;
  const double co = std::cos(angle), si = std::sin(angle);
  for (auto& loop : c.polys)
    for (auto& p : loop) p = vec2(co * p.x - si * p.y, si * p.x + co * p.y);
  c.name = "turned" + std::to_string(seed);
  return c;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> files;
  int generated = 0;
  std::string only, logFile;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--generated") {
      generated = std::stoi(argv[++i]);
    } else if (arg == "--segments") {
      segments = std::stoi(argv[++i]);
    } else if (arg == "--only") {
      only = argv[++i];
    } else if (arg == "--dump") {
      dumpDir = argv[++i];
    } else if (arg == "--log") {
      logFile = argv[++i];
    } else {
      files.push_back(arg);
    }
  }
  std::vector<Case> cases;
  for (const auto& file : files)
    for (const auto& entry : ReadPolygonCorpus(file)) {
      if (!(entry.maxCircleRadius > 0)) continue;
      size_t vertices = 0;
      for (const auto& loop : entry.polys) vertices += loop.size();
      if (vertices > 5000) continue;
      Case c{entry.name, entry.polys, {}};
      for (const double f : {0.1, 0.3, 0.7, 0.9, -0.3, -3.0, -30.0})
        c.radii.push_back(f * entry.maxCircleRadius);
      for (const double f : {0.5, 1.0, 2.0})
        AddNudged(c.radii, f * entry.maxCircleRadius);
      for (const double r : CriticalRadii(entry.polys, 6))
        AddNudged(c.radii, r);
      cases.push_back(c);
    }
  for (int seed = 1; seed <= generated; ++seed) {
    cases.push_back(GridCase(seed));
    cases.push_back(TurnedCase(seed));
  }
  long calls = 0;
  for (const Case& c : cases) {
    if (!only.empty() && c.name != only) continue;
    for (const double radius : c.radii) {
      Check(c, radius);
      ++calls;
    }
  }
  std::cout << cases.size() << " shapes, " << calls << " shape-radius pairs, "
            << segments << " segments\n";
  std::cout << "check              comparisons  <=1e-12  <=1e-9  <=1e-6  "
               "<=1e-3  larger  "
               "one-side-empty\n";
  for (const auto& [name, t] : tallies) {
    std::cout << std::left << std::setw(19) << name << std::right
              << std::setw(11) << t.count;
    for (const long b : t.buckets) std::cout << std::setw(9) << b;
    std::cout << std::setw(10) << t.emptyMismatch << '\n';
  }
  if (!logFile.empty()) {
    std::ofstream log(logFile);
    for (const auto& line : flagged) log << line << '\n';
  }
  for (const auto& [name, t] : tallies) {
    if (t.worst.empty()) continue;
    std::cout << "worst " << name << ":\n";
    for (const auto& w : t.worst)
      std::cout << "  " << std::setprecision(3) << w.value << "  " << w.what
                << '\n';
  }
}
