// Identity checks for CrossSection::Fillet, after the way the 2D booleans are
// tested: not "is there a result", but relations a correct result must
// satisfy, on inputs built to be degenerate.
//
//   identities <corpus file>... [--generated N] [--segments S] [--only NAME]
//              [--log FILE] [--dump DIR] [--quick]
//
// A fillet is not continuous in its radius: where a gap is exactly as wide as
// the ball, a whole channel is there or not, and either answer is right. So
// results are not compared for equality. What always holds is that the result
// shrinks as a positive radius grows and grows with a negative one, so every
// result must lie between the results a little to either side of its radius:
//
//   Fillet(P, r (1 + d))  inside  Fillet(P, r')  inside  Fillet(P, r (1 - d))
//
// for r > 0 and r' within e of r, e much smaller than d; the other way round
// for r < 0. The two outer results are the bracket. Any consistent way of
// deciding a tie lies within it; a missing result, or one that mixes the two
// answers into something outside it, does not.
//
// For every shape and radius r, the bracket is taken once, and then at
// r (1 - e), r and r (1 + e):
//  - plain:    the result itself is within the bracket, and present if both
//              sides of the bracket are.
//  - contain:  a positive radius never adds material and a negative one never
//              removes any.
//  - exact:    so is the result for each of the eight axis symmetries and for
//              the shape scaled by a power of two, mapped back. These change
//              no bit of any number beyond sign and exponent.
//  - order:    and for the loops listed in another order from another vertex.
//  - dual:     and for the complement: Box - Fillet(Box - P, -r), away from
//              the box's own corners. Closing a shape is opening its
//              complement, through the opposite sign of the same code.
//
// e is 1e-9 and d is 1e-6, relative: like the booleans' own tests, nothing is
// asked below what the input can resolve. Radii are the corpus's multiples of
// the largest inscribed radius plus those the shape itself makes critical
// (half a vertex-vertex or vertex-edge distance), and for the generated grid
// shapes the half-integers, which are exact ties everywhere at once.
//
// What escapes the bracket is reported relative to the input area, in decades.
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

constexpr double kNear = 1e-9;
constexpr double kBracket = 1e-6;

struct Case {
  std::string name;
  Polygons polys;
  std::vector<double> radii;
};

CrossSection Join(const std::vector<CrossSection>& pieces) {
  return CrossSection::BatchBoolean(pieces, OpType::Add);
}

// One of the eight axis symmetries, then a scale by a power of two. Exact.
struct Map {
  int symmetry = 0;
  double scale = 1;

  bool Flips() const {
    const bool swap = symmetry & 4;
    return (((symmetry & 1) != 0) != ((symmetry & 2) != 0)) != swap;
  }
  Polygons Apply(const Polygons& in, bool inverse) const {
    Polygons out = in;
    const bool swap = symmetry & 4;
    const double sx = symmetry & 1 ? -1 : 1, sy = symmetry & 2 ? -1 : 1;
    for (auto& loop : out) {
      for (auto& p : loop) {
        if (!inverse) {
          const vec2 v = swap ? vec2(p.y, p.x) : p;
          p = vec2(sx * v.x * scale, sy * v.y * scale);
        } else {
          const vec2 v(sx * p.x / scale, sy * p.y / scale);
          p = swap ? vec2(v.y, v.x) : v;
        }
      }
      if (Flips()) std::reverse(loop.begin(), loop.end());
    }
    return out;
  }
};

Polygons Reorder(const Polygons& in) {
  Polygons out = in;
  for (auto& loop : out)
    std::rotate(loop.begin(), loop.begin() + loop.size() / 2, loop.end());
  std::reverse(out.begin(), out.end());
  return out;
}

struct Tally {
  // Decades of escaped area / input area: <=1e-9, <=1e-6, <=1e-4, <=1e-2,
  // more.
  long buckets[5] = {0, 0, 0, 0, 0};
  long missing = 0;
  long count = 0;
};

std::map<std::string, Tally> tallies;
std::vector<std::string> flagged;
// Many, so that two polygons inscribed in the same arc from different
// starting angles differ by far less than anything flagged here; and not a
// multiple of anything common, so that an arc of a right angle or a
// half-turn never lands exactly on a whole number of segments, where
// rounding would choose the count.
int segments = 367;
std::string dumpDir;
long dumped = 0;
long inconclusive = 0;
// Only the plain and contain checks.
bool quick = false;

std::string Describe(const Case& c, double radius, const std::string& extra) {
  std::ostringstream s;
  s << c.name << " r=" << std::setprecision(17) << radius << " " << extra;
  return s.str();
}

void Dump(const Case& c, double radius, const std::string& check) {
  if (dumpDir.empty() || dumped >= 400) return;
  std::ofstream out(dumpDir + "/" + std::to_string(dumped++) + "-" + check +
                    "-" + c.name + ".txt");
  out << std::setprecision(17) << "REPRO 0 -1 " << c.polys.size() << ' '
      << radius << '\n';
  for (const auto& loop : c.polys) {
    out << loop.size() << '\n';
    for (auto p : loop) out << p.x << ' ' << p.y << '\n';
  }
}

// Tallied by where the shape comes from.
std::string Key(const Case& c, const std::string& check) {
  const size_t digit = c.name.find_first_of("0123456789");
  const std::string family = c.name.substr(0, digit);
  if (family == "turned") return check + " turned";
  if (family == "grid") return check + " grid";
  return check + " corpus";
}

void Record(const Case& c, const std::string& check, double radius,
            double relative, const std::string& label) {
  Tally& t = tallies[Key(c, check)];
  ++t.count;
  const int b = !(relative > 1e-9)   ? 0
                : !(relative > 1e-6) ? 1
                : !(relative > 1e-4) ? 2
                : !(relative > 1e-2) ? 3
                                     : 4;
  ++t.buckets[b];
  if (b >= 2) {
    flagged.push_back(check + " " + Describe(c, radius, label) + " escaped " +
                      std::to_string(relative));
    Dump(c, radius, check);
  }
}

void RecordMissing(const Case& c, const std::string& check, double radius,
                   const std::string& label) {
  Tally& t = tallies[Key(c, check)];
  ++t.count;
  ++t.missing;
  flagged.push_back(check + " " + Describe(c, radius, label) + " no result");
  Dump(c, radius, check);
}

void Check(const Case& c, double radius) {
  const CrossSection input(c.polys);
  if (input.IsEmpty()) return;
  const double inArea = input.Area();
  if (!(inArea > 0)) return;
  const double sign = radius > 0 ? 1 : -1;
  const double r = std::abs(radius);

  // The bracket. A positive radius shrinks the shape more as it grows; a
  // negative one grows it more. "No result" is also what Fillet returns
  // where nothing changes or nothing is left; with that on either side
  // there is nothing to hold the results in between to.
  const auto less = input.Fillet(sign * r * (1 - kBracket), segments);
  const auto more = input.Fillet(sign * r * (1 + kBracket), segments);
  if (less.empty() || more.empty()) {
    ++inconclusive;
    return;
  }
  const CrossSection inner = Join(radius > 0 ? more : less);
  const CrossSection outer = Join(radius > 0 ? less : more);

  const Rect bounds = input.Bounds();
  const vec2 size = bounds.max - bounds.min;
  const CrossSection box =
      CrossSection::Square(size * 3.0 + vec2(16 * r, 16 * r))
          .Translate(bounds.min - size - vec2(8 * r, 8 * r));
  const CrossSection window = CrossSection::Square(size + vec2(4 * r, 4 * r))
                                  .Translate(bounds.min - vec2(2 * r, 2 * r));
  const CrossSection complement = box - input;
  const CrossSection innerInWindow = inner.Boolean(window, OpType::Intersect);

  auto within = [&](const std::string& check, double at, bool present,
                    const CrossSection& result, const std::string& label,
                    bool clipped) {
    if (!present) {
      RecordMissing(c, check, at, label);
      return;
    }
    const double escaped = ((clipped ? innerInWindow : inner) - result).Area() +
                           (result - outer).Area();
    Record(c, check, at, escaped / inArea, label);
  };

  std::vector<Map> maps;
  for (int symmetry = 1; symmetry < 8; ++symmetry)
    maps.push_back({symmetry, 1});
  maps.push_back({0, 0.125});
  maps.push_back({0, 32});

  for (const double nudge : {1 - kNear, 1.0, 1 + kNear}) {
    const double at = radius * nudge;
    const auto plain = input.Fillet(at, segments);
    const CrossSection out = plain.empty() ? CrossSection() : Join(plain);
    within("plain", at, !plain.empty(), out, "", false);
    if (!plain.empty()) {
      const double escaped =
          radius > 0 ? (out - input).Area() : (input - out).Area();
      Record(c, "contain", at, escaped / inArea, "");
    }

    if (quick) continue;
    for (const Map& map : maps) {
      const CrossSection moved(map.Apply(c.polys, false));
      const auto pieces = moved.Fillet(at * map.scale, segments);
      Polygons back;
      for (const auto& piece : pieces) {
        const Polygons loops = map.Apply(piece.ToPolygons(), true);
        back.insert(back.end(), loops.begin(), loops.end());
      }
      within("exact", at, !pieces.empty(), CrossSection(back),
             map.scale == 1 ? "symmetry " + std::to_string(map.symmetry)
                            : "scale " + std::to_string(map.scale),
             false);
    }

    const auto reordered = CrossSection(Reorder(c.polys)).Fillet(at, segments);
    within("order", at, !reordered.empty(),
           reordered.empty() ? CrossSection() : Join(reordered), "", false);

    const auto dual = complement.Fillet(-at, segments);
    within("dual", at, !dual.empty(),
           dual.empty() ? CrossSection()
                        : (box - Join(dual)).Boolean(window, OpType::Intersect),
           "", true);
  }
}

// Radii the shape itself makes critical: half the distance between two
// vertices, or between a vertex and an edge's line where the foot is on the
// edge. A handful, spread over their range.
std::vector<double> CriticalRadii(const Polygons& polys, size_t limit) {
  std::vector<vec2> points;
  std::vector<std::pair<vec2, vec2>> edges;
  for (const auto& loop : polys)
    for (size_t i = 0; i < loop.size(); ++i) {
      points.push_back(loop[i]);
      edges.push_back({loop[i], loop[(i + 1) % loop.size()]});
    }
  if (points.size() > 400 || points.empty()) return {};
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
  // below what the boolean that prepares the input tells apart.
  double extent = 0;
  for (const vec2 p : points)
    for (const vec2 q : {points.front(), points.back()})
      extent = std::max(extent, length(p - q));
  radii.erase(std::remove_if(radii.begin(), radii.end(),
                             [&](double r) { return r < 1e-6 * extent; }),
              radii.end());
  std::sort(radii.begin(), radii.end());
  radii.erase(std::unique(radii.begin(), radii.end()), radii.end());
  std::vector<double> picked;
  for (size_t k = 0; k < limit && !radii.empty(); ++k)
    picked.push_back(
        radii[k * (radii.size() - 1) / std::max<size_t>(1, limit - 1)]);
  picked.erase(std::unique(picked.begin(), picked.end()), picked.end());
  return picked;
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
  for (const double r : {0.25, 0.5, 1.0, 1.5, 2.0}) {
    c.radii.push_back(r);
    c.radii.push_back(-r);
  }
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
    } else if (arg == "--quick") {
      quick = true;
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
      for (const double f :
           {0.1, 0.3, 0.5, 0.7, 0.9, 1.0, -0.3, -0.5, -1.0, -2.0, -3.0, -30.0})
        c.radii.push_back(f * entry.maxCircleRadius);
      for (const double r : CriticalRadii(entry.polys, 6)) {
        c.radii.push_back(r);
        c.radii.push_back(-r);
      }
      cases.push_back(c);
    }
  for (int seed = 1; seed <= generated; ++seed) {
    cases.push_back(GridCase(seed));
    cases.push_back(TurnedCase(seed));
  }
  long pairs = 0;
  for (const Case& c : cases) {
    if (!only.empty() && c.name != only) continue;
    for (const double radius : c.radii) {
      Check(c, radius);
      ++pairs;
    }
  }
  std::cout << cases.size() << " shapes, " << pairs << " shape-radius pairs ("
            << inconclusive << " without a bracket), " << segments
            << " segments\n";
  std::cout << "check             comparisons   <=1e-9   <=1e-6   <=1e-4   "
               "<=1e-2   larger  no-result\n";
  for (const auto& [name, t] : tallies) {
    std::cout << std::left << std::setw(18) << name << std::right
              << std::setw(11) << t.count;
    for (const long b : t.buckets) std::cout << std::setw(9) << b;
    std::cout << std::setw(11) << t.missing << '\n';
  }
  if (!logFile.empty()) {
    std::ofstream log(logFile);
    for (const auto& line : flagged) log << line << '\n';
  }
}
