/**
 * OpenSCAD runtime for Manifold.js compiled output.
 * Loaded via import from compiled files. Initializes manifold-3d and provides
 * OpenSCAD built-ins and helpers. Exports all symbols for use by compiled code.
 */
import Module from "manifold-3d";

const wasm = await Module();
wasm.setup();
const { Manifold, CrossSection } = wasm;

// Type checks (OpenSCAD built-ins)
function is_undef_fn(x) { return x === undefined || x === null; }
function is_bool_fn(x) { return typeof x === "boolean"; }
function is_num_fn(x) { return typeof x === "number" && !Number.isNaN(x); }
function is_string_fn(x) { return typeof x === "string"; }
function is_list_fn(x) { return Array.isArray(x); }
function is_function_fn(x) { return typeof x === "function"; }

// Trig (OpenSCAD uses degrees!)
function sin_fn(x) { return Math.sin(x * Math.PI / 180); }
function cos_fn(x) { return Math.cos(x * Math.PI / 180); }
function tan_fn(x) { return Math.tan(x * Math.PI / 180); }
function asin_fn(x) { return Math.asin(x) * 180 / Math.PI; }
function acos_fn(x) { return Math.acos(x) * 180 / Math.PI; }
function atan_fn(x) { return Math.atan(x) * 180 / Math.PI; }
function atan2_fn(y, x) { return Math.atan2(y, x) * 180 / Math.PI; }

// Math (OpenSCAD built-ins)
var abs_fn = Math.abs;
var sign_fn = Math.sign;
var floor_fn = Math.floor;
var ceil_fn = Math.ceil;
var round_fn = Math.round;
var sqrt_fn = Math.sqrt;
var exp_fn = Math.exp;
function ln_fn(x) { return Math.log(x); }
function log_fn(x) { return Math.log(x); }
function min_fn(...a) { return a.length === 1 && Array.isArray(a[0]) ? Math.min(...a[0]) : Math.min(...a); }
function max_fn(...a) { return a.length === 1 && Array.isArray(a[0]) ? Math.max(...a[0]) : Math.max(...a); }
function norm_fn(v) { return Math.sqrt(v.reduce((s, x) => s + x * x, 0)); }
function cross_fn(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }

// String & list (OpenSCAD built-ins)
function len_fn(x) {
  if (typeof x === "string" || Array.isArray(x)) return x.length;
  // OpenSCAD returns undef and emits a warning for non-string/non-list inputs.
  console.warn("WARNING: len() parameter could not be converted");
  return undefined;
}
function str_fn(...a) { return a.map(x => x === undefined ? "undef" : String(x)).join(""); }
function chr_fn(n) { return Array.isArray(n) ? n.map(c => String.fromCharCode(c)).join("") : String.fromCharCode(n); }
function ord_fn(s) { return s == null || s.length === 0 ? undefined : s.charCodeAt(0); }
function concat_fn(...a) { return [].concat(...a); }
function search_fn(needle, haystack, num_returns, idx_col) {
  if (is_string_fn(needle) && is_string_fn(haystack)) {
    var result = [];
    for (var ch of needle) {
      var indices = [];
      for (var i = 0; i < haystack.length; i++) { if (haystack[i] === ch) indices.push(i); }
      result.push(num_returns === 0 ? indices : indices.slice(0, num_returns || 1));
    }
    return num_returns === 1 || num_returns === undefined ? result.map(r => r.length > 0 ? r[0] : []) : result;
  }
  if (is_list_fn(haystack) && is_list_fn(needle)) {
    return needle.map(function(n) {
      var indices = [];
      for (var i = 0; i < haystack.length; i++) {
        var item = idx_col !== undefined ? haystack[i][idx_col] : haystack[i];
        if (__eq(item, n)) indices.push(i);
      }
      return num_returns === 0 ? indices : (indices.length > 0 ? indices[0] : []);
    });
  }
  return [];
}
function lookup_fn(key, table) {
  if (key <= table[0][0]) return table[0][1];
  if (key >= table[table.length-1][0]) return table[table.length-1][1];
  for (var i = 0; i < table.length - 1; i++) {
    if (table[i][0] <= key && key <= table[i+1][0]) {
      var t = (key - table[i][0]) / (table[i+1][0] - table[i][0]);
      return table[i][1] + t * (table[i+1][1] - table[i][1]);
    }
  }
  return undefined;
}

// Control
function openscad_assert_fn(cond, msg) { if (!cond) { console.trace("Assertion failed:", msg); throw new Error(msg || "Assertion failed"); } }
function __eq(a, b) {
  if (a === b) return true;
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    for (var i = 0; i < a.length; i++) { if (!__eq(a[i], b[i])) return false; }
    return true;
  }
  return false;
}
function __add(a, b) {
  if (Array.isArray(a)) {
    if (Array.isArray(b)) return a.map((x, i) => __add(x, b[i]));
    return a.map(x => __add(x, b));
  }
  if (Array.isArray(b)) return b.map(x => __add(a, x));
  return a + b;
}
function __sub(a, b) {
  if (Array.isArray(a)) {
    if (Array.isArray(b)) return a.map((x, i) => __sub(x, b[i]));
    return a.map(x => __sub(x, b));
  }
  if (Array.isArray(b)) return b.map(x => __sub(a, x));
  return a - b;
}
function __mul(a, b) {
  if (Array.isArray(a)) {
    if (Array.isArray(b)) {
      if (a.length > 0 && Array.isArray(a[0])) {
        if (b.length > 0 && Array.isArray(b[0])) {
          var res = [];
          for (var i = 0; i < a.length; i++) {
            res[i] = [];
            for (var j = 0; j < b[0].length; j++) {
              var sum = 0;
              for (var k = 0; k < a[0].length; k++) sum += a[i][k] * b[k][j];
              res[i].push(sum);
            }
          }
          return res;
        } else {
          return a.map(row => __mul(row, b));
        }
      } else {
        if (b.length > 0 && Array.isArray(b[0])) {
          var res2 = [];
          for (var j2 = 0; j2 < b[0].length; j2++) {
            var sum2 = 0;
            for (var k2 = 0; k2 < a.length; k2++) sum2 += a[k2] * b[k2][j2];
            res2.push(sum2);
          }
          return res2;
        } else {
          var sum3 = 0;
          for (var i3 = 0; i3 < Math.min(a.length, b.length); i3++) sum3 += a[i3] * b[i3];
          return sum3;
        }
      }
    }
    return a.map(x => __mul(x, b));
  }
  if (Array.isArray(b)) return b.map(x => __mul(a, x));
  return a * b;
}
function __div(a, b) {
  if (Array.isArray(a)) {
    if (Array.isArray(b)) return a.map((x, i) => __div(x, b[i]));
    return a.map(x => __div(x, b));
  }
  if (Array.isArray(b)) return b.map(x => __div(a, x));
  return a / b;
}
function __mod(a, b) {
  if (Array.isArray(a)) {
    if (Array.isArray(b)) return a.map((x, i) => __mod(x, b[i]));
    return a.map(x => __mod(x, b));
  }
  if (Array.isArray(b)) return b.map(x => __mod(a, x));
  return a % b;
}
function __neg(a) {
  if (Array.isArray(a)) return a.map(__neg);
  return -a;
}
function __pos(a) {
  if (Array.isArray(a)) return a.map(__pos);
  return +a;
}

// OpenSCAD version
function version_fn() { return [2019, 5, 0]; }
function version_num_fn() { return 20190500; }

// Constants
var PI = Math.PI;
var INF = Infinity;
var NAN = NaN;
var undef = undefined;
var _EPSILON = 1e-9;

// Children stack for module calls
var __children_stack = [];
const __color_prop_layout = new WeakMap();
function __with_children(fn, count, call) {
  __children_stack.push({ fn: fn, count: count });
  try {
    return call();
  } finally {
    __children_stack.pop();
  }
}

function __is_finite_matrix4(m) {
  return Array.isArray(m) &&
    m.length === 4 &&
    m.every((row) => Array.isArray(row) &&
      row.length === 4 &&
      row.every((v) => typeof v === "number" && Number.isFinite(v)));
}

// Manifold expects a flat 4x4 matrix in column-major order.
function __to_manifold_mat4(m) {
  if (!__is_finite_matrix4(m)) return undefined;
  const out = new Array(16);
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      out[col * 4 + row] = m[row][col];
    }
  }
  return out;
}

// Guard transform() against invalid matrices produced by complex attachment math.
function __safe_transform(shape, m) {
  const mm = __to_manifold_mat4(m);
  if (!mm) return shape;
  try {
    return shape.transform(mm);
  } catch {
    return shape;
  }
}

function __identity4() {
  return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
}

function __safe_attach_transform(...args) {
  try {
    const m = _attach_transform_fn(...args);
    return __is_finite_matrix4(m) ? m : __identity4();
  } catch {
    return __identity4();
  }
}

// 2D helpers used by offset()/projection() fallbacks.
function __safe_offset2d(shape, delta, joinType = "Round", miterLimit = 2, circularSegments = 0) {
  try {
    if (shape && typeof shape.offset === "function") {
      return shape.offset(delta, joinType, miterLimit, circularSegments);
    }
  } catch {}
  return shape;
}

function __safe_project3d(shape) {
  try {
    if (shape && typeof shape.project === "function") return shape.project();
  } catch {}
  return CrossSection.square(0);
}

// Common OpenSCAD/CSS color names mapped to linearized [0, 1] RGB.
const __named_colors = {
  aqua: [0, 1, 1],
  beige: [0.9608, 0.9608, 0.8627],
  black: [0, 0, 0],
  blue: [0, 0, 1],
  brown: [0.6471, 0.1647, 0.1647],
  coral: [1, 0.498, 0.3137],
  crimson: [0.8627, 0.0784, 0.2353],
  cyan: [0, 1, 1],
  fuchsia: [1, 0, 1],
  gold: [1, 0.8431, 0],
  gray: [0.502, 0.502, 0.502],
  green: [0, 0.502, 0],
  grey: [0.502, 0.502, 0.502],
  indigo: [0.2941, 0, 0.5098],
  khaki: [0.9412, 0.902, 0.549],
  lavender: [0.902, 0.902, 0.9804],
  lime: [0, 1, 0],
  magenta: [1, 0, 1],
  maroon: [0.502, 0, 0],
  navy: [0, 0, 0.502],
  olive: [0.502, 0.502, 0],
  orange: [1, 0.6471, 0],
  pink: [1, 0.7529, 0.7961],
  purple: [0.502, 0, 0.502],
  red: [1, 0, 0],
  salmon: [0.9804, 0.502, 0.4471],
  silver: [0.7529, 0.7529, 0.7529],
  tan: [0.8235, 0.7059, 0.549],
  teal: [0, 0.502, 0.502],
  transparent: [0, 0, 0, 0],
  violet: [0.9333, 0.5098, 0.9333],
  white: [1, 1, 1],
  yellow: [1, 1, 0],
};

function __clamp01(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return 0;
  if (n <= 0) return 0;
  if (n >= 1) return 1;
  return n;
}

function __parse_hex_color(s) {
  if (!s.startsWith("#")) return undefined;
  const h = s.slice(1);
  if (h.length === 3 || h.length === 4) {
    const r = parseInt(h[0] + h[0], 16);
    const g = parseInt(h[1] + h[1], 16);
    const b = parseInt(h[2] + h[2], 16);
    const a = h.length === 4 ? parseInt(h[3] + h[3], 16) : 255;
    if ([r, g, b, a].some((x) => Number.isNaN(x))) return undefined;
    return [r / 255, g / 255, b / 255, a / 255];
  }
  if (h.length === 6 || h.length === 8) {
    const r = parseInt(h.slice(0, 2), 16);
    const g = parseInt(h.slice(2, 4), 16);
    const b = parseInt(h.slice(4, 6), 16);
    const a = h.length === 8 ? parseInt(h.slice(6, 8), 16) : 255;
    if ([r, g, b, a].some((x) => Number.isNaN(x))) return undefined;
    return [r / 255, g / 255, b / 255, a / 255];
  }
  return undefined;
}

function __parse_color_value(c) {
  if (Array.isArray(c)) {
    if (c.length < 3) return undefined;
    let r = Number(c[0]);
    let g = Number(c[1]);
    let b = Number(c[2]);
    let a = c.length >= 4 ? Number(c[3]) : 1;
    if (![r, g, b, a].every(Number.isFinite)) return undefined;
    const maxChan = Math.max(Math.abs(r), Math.abs(g), Math.abs(b), Math.abs(a));
    // Support accidental [0..255] style vectors.
    if (maxChan > 1) {
      r /= 255;
      g /= 255;
      b /= 255;
      a /= 255;
    }
    return [r, g, b, a];
  }

  if (typeof c === "string") {
    const key = c.trim().toLowerCase();
    if (key === "" || key === "default") return undefined;
    const named = __named_colors[key];
    if (named) {
      return named.length === 4 ? named : [named[0], named[1], named[2], 1];
    }
    return __parse_hex_color(key);
  }

  return undefined;
}

// Apply OpenSCAD color() by appending custom RGBA + marker channels.
function __apply_color(shape, c, alpha) {
  if (!shape || typeof shape.setProperties !== "function" || typeof shape.numProp !== "function") {
    return shape;
  }

  let base = __parse_color_value(c);
  if (!base) {
    // Allow color(alpha = x) form by assuming white base if alpha is provided.
    if (alpha === undefined || alpha === null) return shape;
    base = [1, 1, 1, 1];
  }

  let outAlpha = base[3];
  if (alpha !== undefined && alpha !== null && Number.isFinite(Number(alpha))) {
    outAlpha = Number(alpha);
  }

  const rgba = [
    __clamp01(base[0]),
    __clamp01(base[1]),
    __clamp01(base[2]),
    __clamp01(outAlpha),
  ];

  const oldNumProp = Math.max(0, Number(shape.numProp()) || 0);
  const trackedLayout = __color_prop_layout.get(shape);
  let colorOffset = Math.max(3, oldNumProp);
  let markerOffset = colorOffset + 4;
  let newNumProp = markerOffset + 1;
  if (trackedLayout && Number.isInteger(trackedLayout.colorOffset) && Number.isInteger(trackedLayout.markerOffset)) {
    const trackedColorOffset = trackedLayout.colorOffset;
    const trackedMarkerOffset = trackedLayout.markerOffset;
    if (trackedColorOffset >= 0 && trackedMarkerOffset === trackedColorOffset + 4 && trackedMarkerOffset < oldNumProp) {
      colorOffset = trackedColorOffset;
      markerOffset = trackedMarkerOffset;
      newNumProp = oldNumProp;
    }
  }

  try {
    const painted = shape.setProperties(newNumProp, (newProp, position, oldProp) => {
      for (let i = 0; i < newNumProp; i++) {
        if (i < oldProp.length) {
          newProp[i] = oldProp[i];
        } else if (i < 3) {
          newProp[i] = position[i];
        } else {
          newProp[i] = 0;
        }
      }
      newProp[colorOffset] = rgba[0];
      newProp[colorOffset + 1] = rgba[1];
      newProp[colorOffset + 2] = rgba[2];
      newProp[colorOffset + 3] = rgba[3];
      // Marker channel lets the viewer distinguish custom RGBA from manifold built-in properties.
      newProp[markerOffset] = 1;
    });
    __color_prop_layout.set(painted, { colorOffset, markerOffset });
    return painted;
  } catch {
    return shape;
  }
}

// OpenSCAD iteration can target lists, strings, and occasionally scalars.
function __flat_map_iter(v, fn) {
  if (v === undefined || v === null) return [];
  if (Array.isArray(v)) return v.flatMap(fn);
  if (typeof v === "string") return Array.from(v).flatMap(fn);
  return [v].flatMap(fn);
}

// Range expansion: convert OpenSCAD range [start:step:end] to an actual array
function __range(start, step, end) {
  var result = [];
  if (step > 0) { for (var i = start; i <= end; i += step) result.push(i); }
  else if (step < 0) { for (var i = start; i >= end; i += step) result.push(i); }
  return result;
}

// Detect CrossSection (2D) vs Manifold (3D) for dispatch
function __is2D(x) {
  return x != null && typeof x.offset === "function" && typeof x.toPolygons === "function";
}

function __isEmpty(x) {
  if (!x) return true;
  if (typeof x.isEmpty === 'function' && x.isEmpty()) return true;
  if (typeof x.numTri === 'function' && x.numTri() === 0) return true;
  if (typeof x.numVert === 'function' && x.numVert() === 0) return true;
  return false;
}

// Boolean ops: use CrossSection for 2D, Manifold for 3D (no thin extrusion of 2D)
function __union2d3d(items) {
  const valid = items.filter(x => !__isEmpty(x));
  if (valid.length === 0) return Manifold.union([]);
  return __is2D(valid[0]) ? CrossSection.union(valid) : Manifold.union(valid);
}
function __difference2d3d(first, rest) {
  if (__isEmpty(first)) return first;
  const validRest = rest.filter(x => !__isEmpty(x));
  if (validRest.length === 0) return first;
  if (__is2D(first)) return CrossSection.difference([first, ...validRest]);
  return validRest.length === 1 ? first.subtract(validRest[0]) : first.subtract(Manifold.union(validRest));
}
function __intersection2d3d(items) {
  if (items.length === 0) return Manifold.union([]);
  const valid = items.filter(x => !__isEmpty(x));
  if (valid.length < items.length) {
    const firstValid2D = valid.find(__is2D);
    return firstValid2D ? CrossSection.union([]) : Manifold.union([]);
  }
  return __is2D(valid[0]) ? CrossSection.intersection(valid) : Manifold.intersection(valid);
}
function __hull2d3d(items) {
  const valid = items.filter(x => !__isEmpty(x));
  if (valid.length === 0) return Manifold.union([]);
  return __is2D(valid[0]) ? CrossSection.hull(valid) : Manifold.hull(valid);
}

function __mesh_points3d(manifold, maxPoints = 192) {
  if (!manifold || typeof manifold.getMesh !== "function") return [];
  const mesh = manifold.getMesh();
  const numProp = (mesh && typeof mesh.numProp === "number" && mesh.numProp >= 3) ? mesh.numProp : 3;
  const vertProps = mesh && mesh.vertProperties;
  if (!vertProps || vertProps.length < 3) return [];

  const count = Math.floor(vertProps.length / numProp);
  if (count <= 0) return [];

  const step = Math.max(1, Math.ceil(count / maxPoints));
  const points = [];
  const seen = new Set();

  for (let i = 0; i < count; i += step) {
    const base = i * numProp;
    const x = vertProps[base];
    const y = vertProps[base + 1];
    const z = vertProps[base + 2];
    const key = `${x},${y},${z}`;
    if (seen.has(key)) continue;
    seen.add(key);
    points.push([x, y, z]);
  }

  // Include the final vertex to reduce directional sampling bias.
  const tail = (count - 1) * numProp;
  const tx = vertProps[tail];
  const ty = vertProps[tail + 1];
  const tz = vertProps[tail + 2];
  const tailKey = `${tx},${ty},${tz}`;
  if (!seen.has(tailKey)) points.push([tx, ty, tz]);

  return points;
}

function __is_likely_convex3d(manifold) {
  if (!manifold || typeof manifold.hull !== "function" || typeof manifold.volume !== "function") return false;
  if (typeof manifold.isEmpty === "function" && manifold.isEmpty()) return true;
  try {
    const volume = manifold.volume();
    const hullVolume = manifold.hull().volume();
    if (!Number.isFinite(volume) || !Number.isFinite(hullVolume)) return false;
    const eps = Math.max(1e-6, Math.abs(hullVolume) * 1e-4);
    return Math.abs(hullVolume - volume) <= eps;
  } catch (_err) {
    return false;
  }
}

function __minkowski_convex_pair3d(a, b) {
  const pointsA = __mesh_points3d(a);
  const pointsB = __mesh_points3d(b);
  if (pointsA.length === 0 || pointsB.length === 0) return Manifold.union([]);

  const sums = [];
  for (let i = 0; i < pointsA.length; i++) {
    const pa = pointsA[i];
    for (let j = 0; j < pointsB.length; j++) {
      const pb = pointsB[j];
      sums.push([pa[0] + pb[0], pa[1] + pb[1], pa[2] + pb[2]]);
    }
  }
  return Manifold.hull(sums);
}

function __minkowski_convex_chain3d(items) {
  let acc = items[0];
  for (let i = 1; i < items.length; i++) {
    acc = __minkowski_convex_pair3d(acc, items[i]);
  }
  return acc;
}

// Minkowski sum: native API when available; otherwise use a convex approximation.
function __minkowski2d3d(items) {
  const valid = items.filter(x => !__isEmpty(x));
  if (valid.length === 0) return Manifold.union([]);
  if (valid.length === 1) return valid[0];
  if (__is2D(valid[0])) return CrossSection.hull(valid); // 2D: no minkowski in CrossSection

  // Some manifold-3d builds do not expose any minkowski API.
  const first = valid[0];
  const hasInstanceMinkowski = first && typeof first.minkowskiSum === "function";
  const hasStaticMinkowski = typeof Manifold.minkowskiSum === "function";
  if (!hasInstanceMinkowski && !hasStaticMinkowski) {
    if (valid.every(__is_likely_convex3d)) {
      try {
        return __minkowski_convex_chain3d(valid);
      } catch (_err) {
        // Fall through to last-resort behavior.
      }
    }
    return Manifold.hull(valid);
  }

  let acc = valid[0];
  for (let i = 1; i < valid.length; i++) {
    acc = hasInstanceMinkowski ? acc.minkowskiSum(valid[i]) : Manifold.minkowskiSum(acc, valid[i]);
  }
  return acc;
}

// Export all runtime symbols for compiled code
export {
  Manifold, CrossSection, wasm,
  is_undef_fn, is_bool_fn, is_num_fn, is_string_fn, is_list_fn, is_function_fn,
  sin_fn, cos_fn, tan_fn, asin_fn, acos_fn, atan_fn, atan2_fn,
  abs_fn, sign_fn, floor_fn, ceil_fn, round_fn, sqrt_fn, exp_fn, ln_fn, log_fn,
  min_fn, max_fn, norm_fn, cross_fn,
  len_fn, str_fn, chr_fn, ord_fn, concat_fn, search_fn, lookup_fn,
  openscad_assert_fn, __eq, __add, __sub, __mul, __div, __mod, __neg, __pos,
  version_fn, version_num_fn,
  PI, INF, NAN, undef, _EPSILON,
  __children_stack, __with_children,
  __is_finite_matrix4, __to_manifold_mat4, __safe_transform, __identity4,
  __safe_attach_transform, __safe_offset2d, __safe_project3d,
  __apply_color,
  __flat_map_iter, __range, __is2D, __union2d3d, __difference2d3d, __intersection2d3d, __hull2d3d, __minkowski2d3d,
  __extrude, __revolve
};

function __extrude(child, height, ...args) {
  if (!__is2D(child) || __isEmpty(child)) return Manifold.union([]);
  return Manifold.extrude(child, height, ...args);
}

function __revolve(child, circularSegments, degrees) {
  if (!__is2D(child) || __isEmpty(child)) return Manifold.union([]);
  return Manifold.revolve(child, circularSegments, degrees);
}
