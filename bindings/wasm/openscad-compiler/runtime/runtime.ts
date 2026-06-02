/**
 * OpenSCAD runtime for Manifold.js compiled output.
 * Loaded via import from compiled files. Initializes manifold-3d and provides
 * OpenSCAD built-ins and helpers. Exports all symbols for use by compiled code.
 */
import Module from "manifold-3d";
import * as opentype from "opentype.js";
import { createCanvas, loadImage } from "canvas";

const opentypeParse = opentype.parse || (opentype as any).default?.parse;

declare const document: any;
declare const OffscreenCanvas: any;
declare const _attach_transform_fn: any;

const wasm = await Module();
wasm.setup();
const { Manifold, CrossSection } = wasm;

// Type checks (OpenSCAD built-ins)
function is_undef_fn(x: any) { return x === undefined || x === null; }
function is_bool_fn(x: any) { return typeof x === "boolean"; }
function is_num_fn(x: any) { return typeof x === "number" && !Number.isNaN(x); }
function is_string_fn(x: any) { return typeof x === "string"; }
function is_list_fn(x: any) { return Array.isArray(x); }
function is_function_fn(x: any) { return typeof x === "function"; }

// Trig (OpenSCAD uses degrees!)
function sin_fn(x: any) { return Math.sin(x * Math.PI / 180); }
function cos_fn(x: any) { return Math.cos(x * Math.PI / 180); }
function tan_fn(x: any) { return Math.tan(x * Math.PI / 180); }
function asin_fn(x: any) { return Math.asin(x) * 180 / Math.PI; }
function acos_fn(x: any) { return Math.acos(x) * 180 / Math.PI; }
function atan_fn(x: any) { return Math.atan(x) * 180 / Math.PI; }
function atan2_fn(y: any, x: any) { return Math.atan2(y, x) * 180 / Math.PI; }

// Math (OpenSCAD built-ins)
var abs_fn = Math.abs;
var sign_fn = Math.sign;
var floor_fn = Math.floor;
var ceil_fn = Math.ceil;
var round_fn = Math.round;
var sqrt_fn = Math.sqrt;
var exp_fn = Math.exp;
function ln_fn(x: any) { return Math.log(x); }
function log_fn(x: any) { return Math.log(x); }
function min_fn(...a: any[]) { return a.length === 1 && Array.isArray(a[0]) ? Math.min(...a[0]) : Math.min(...a); }
function max_fn(...a: any[]) { return a.length === 1 && Array.isArray(a[0]) ? Math.max(...a[0]) : Math.max(...a); }
function norm_fn(v: any) { return Math.sqrt(v.reduce((s: any, x: any) => s + x * x, 0)); }
function cross_fn(a: any, b: any) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }

// String & list (OpenSCAD built-ins)
function len_fn(x: any) {
  if (typeof x === "string" || Array.isArray(x)) return x.length;
  // OpenSCAD returns undef and emits a warning for non-string/non-list inputs.
  console.warn("WARNING: len() parameter could not be converted");
  return undefined;
}
function str_fn(...a: any[]) { return a.map(x => x === undefined ? "undef" : String(x)).join(""); }
function chr_fn(n: any) { return Array.isArray(n) ? n.map(c => String.fromCharCode(c)).join("") : String.fromCharCode(n); }
function ord_fn(s: any) { return s == null || s.length === 0 ? undefined : s.charCodeAt(0); }
function concat_fn(...a: any[]) { return [].concat(...a); }
function search_fn(needle: any, haystack: any, num_returns: any, idx_col: any) {
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
  if (is_string_fn(needle) && is_list_fn(haystack)) {
    return [...needle].map(function(n) {
      var indices = [];

      for (var i = 0; i < haystack.length; i++) {
        var item = idx_col !== undefined ? haystack[i][idx_col] : haystack[i];

        if (__eq(item, n)) indices.push(i);
      }

      return num_returns === 0 ? indices : (indices.length > 0 ? indices[0] : undefined);
    });
  }
  return undefined;
}
function lookup_fn(key: any, table: any) {
  if (key <= table[0][0]) return table[0][1];
  if (key >= table[table.length - 1][0]) return table[table.length - 1][1];
  for (var i = 0; i < table.length - 1; i++) {
    if (table[i][0] <= key && key <= table[i + 1][0]) {
      var t = (key - table[i][0]) / (table[i + 1][0] - table[i][0]);
      return table[i][1] + t * (table[i+1][1] - table[i][1]);
    }
  }
  return undefined;
}

// Control
function openscad_assert_fn(cond: any, msg: any) { if (!cond) { console.trace("Assertion failed:", msg); throw new Error(msg || "Assertion failed"); } }
function __eq(a: any, b: any) {
  if (a === b) return true;
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    for (var i = 0; i < a.length; i++) { if (!__eq(a[i], b[i])) return false; }
    return true;
  }
  return false;
}
function __add(a: any, b: any): any {
  if (Array.isArray(a)) {
    if (Array.isArray(b)) return a.map((x, i) => __add(x, b[i]));
    return a.map(x => __add(x, b));
  }
  if (Array.isArray(b)) return b.map((x: any): any => __add(a, x));
  return a + b;
}
function __sub(a: any, b: any): any {
  if (Array.isArray(a)) {
    if (Array.isArray(b)) return a.map((x, i): any => __sub(x, b[i]));
    return a.map((x: any): any => __sub(x, b));
  }
  if (Array.isArray(b)) return b.map((x: any): any => __sub(a, x));
  return a - b;
}
function __mul(a: any, b: any) {
  if (Array.isArray(a)) {
    if (Array.isArray(b)) {
      if (a.length > 0 && Array.isArray(a[0])) {
        if (b.length > 0 && Array.isArray(b[0])) {
          var res: any[] = [];
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
          return a.map((row: any): any => __mul(row, b));
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
    return a.map((x: any): any => __mul(x, b));
  }
  if (Array.isArray(b)) return b.map((x: any): any => __mul(a, x));
  return a * b;
}
function __div(a: any, b: any) {
  if (Array.isArray(a)) {
    if (Array.isArray(b)) return a.map((x, i): any => __div(x, b[i]));
    return a.map((x: any): any => __div(x, b));
  }
  if (Array.isArray(b)) return b.map((x: any): any => __div(a, x));
  return a / b;
}
function __mod(a: any, b: any): any {
  if (Array.isArray(a)) {
    if (Array.isArray(b)) return a.map((x, i) => __mod(x, b[i]));
    return a.map(x => __mod(x, b));
  }
  if (Array.isArray(b)) return b.map(x => __mod(a, x));
  return a % b;
}
function __neg(a: any): any {
  if (Array.isArray(a)) return a.map(__neg);
  return -a;
}
function __pos(a: any): any {
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
var __children_stack: any[] = [];
const __color_prop_layout = new WeakMap();
function __with_children(fn: any, count: any, call: any, name?: string) {
  __children_stack.push({ fn: fn, count: count, name: name });
  try {
    return call();
  } finally {
    __children_stack.pop();
  }
}

function parent_module_fn(d: any = 1) {
  const depth = Number(d);
  if (!Number.isInteger(depth) || depth < 0) return "";
  const idx = __children_stack.length - 1 - depth;
  if (idx < 0 || idx >= __children_stack.length) return "";
  return __children_stack[idx].name || "";
}

function __is_finite_matrix4(m: any) {
  return Array.isArray(m) &&
    m.length === 4 &&
    m.every((row: any) => Array.isArray(row) &&
      row.length === 4 &&
      row.every((v: any) => typeof v === "number" && Number.isFinite(v)));
}

// Manifold expects a flat 4x4 matrix in column-major order.
function __to_manifold_mat4(m: any) {
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
function __safe_transform(shape: any, m: any) {
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

function __safe_attach_transform(...args: any[]) {
  try {
    const m = _attach_transform_fn(...args);
    return __is_finite_matrix4(m) ? m : __identity4();
  } catch {
    return __identity4();
  }
}

// 2D helpers used by offset()/projection() fallbacks.
function __safe_offset2d(shape: any, delta: any, joinType = "Round", miterLimit = 2, circularSegments = 0, fa = 12, fs = 2) {
  try {
    if (shape && typeof shape.offset === "function") {
      if (circularSegments <= 0) {
        __sync_quality(fa, fs);
      }
      return shape.offset(delta, joinType, miterLimit, circularSegments);
    }
  } catch {}
  return shape;
}

function __safe_project3d(shape: any, cut = false) {
  try {
    if (shape) {
      if (cut && typeof shape.slice === "function") {
        return shape.slice(0);
      }
      if (typeof shape.project === "function") {
        return shape.project();
      }
    }
  } catch {}
  return CrossSection.square(0);
}

// Common OpenSCAD/CSS color names mapped to linearized [0, 1] RGB.
const __named_colors: Record<string, number[]> = {
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

function __clamp01(v: any) {
  const n = Number(v);
  if (!Number.isFinite(n)) return 0;
  if (n <= 0) return 0;
  if (n >= 1) return 1;
  return n;
}

function __parse_hex_color(s: any) {
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

function __parse_color_value(c: any) {
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
function __apply_color(shape: any, c: any, alpha: any) {
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
    const painted = shape.setProperties(newNumProp, (newProp: any, position: any, oldProp: any) => {
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
function __flat_map_iter(v: any, fn: any) {
  if (v === undefined || v === null) return [];
  if (Array.isArray(v)) return v.flatMap((item, i) => fn(item, i));
  if (typeof v === "string") return Array.from(v).flatMap(fn);
  return [v].flatMap(fn);
}

// Range expansion: convert OpenSCAD range [start:step:end] to an actual array
function __range(start: any, step: any, end: any) {
  var result = [];
  if (step > 0) { for (var i = start; i <= end; i += step) result.push(i); }
  else if (step < 0) { for (var i = start; i >= end; i += step) result.push(i); }
  return result;
}

// Detect CrossSection (2D) vs Manifold (3D) for dispatch
function __is2D(x: any) {
  return x != null && typeof x.offset === "function" && typeof x.toPolygons === "function";
}

function __isEmpty(x: any) {
  if (!x) return true;
  if (typeof x.isEmpty === 'function' && x.isEmpty()) return true;
  if (typeof x.numTri === 'function' && x.numTri() === 0) return true;
  if (typeof x.numVert === 'function' && x.numVert() === 0) return true;
  return false;
}

// Boolean ops: use CrossSection for 2D, Manifold for 3D (no thin extrusion of 2D)
function __union2d3d(items: any[]) {
  const valid = items.filter(x => !__isEmpty(x));
  if (valid.length === 0) return Manifold.union([]);
  return __is2D(valid[0]) ? CrossSection.union(valid) : Manifold.union(valid);
}
function __difference2d3d(first: any, rest: any[]) {
  if (__isEmpty(first)) return first;
  const validRest = rest.filter(x => !__isEmpty(x));
  if (validRest.length === 0) return first;
  if (__is2D(first)) return CrossSection.difference([first, ...validRest]);
  return validRest.length === 1 ? first.subtract(validRest[0]) : first.subtract(Manifold.union(validRest));
}
function __intersection2d3d(items: any[]) {
  if (items.length === 0) return Manifold.union([]);
  const valid = items.filter(x => !__isEmpty(x));
  if (valid.length < items.length) {
    const firstValid2D = valid.find(__is2D);
    return firstValid2D ? CrossSection.union([]) : Manifold.union([]);
  }
  return __is2D(valid[0]) ? CrossSection.intersection(valid) : Manifold.intersection(valid);
}
function __hull2d3d(items: any[]) {
  const valid = items.filter(x => !__isEmpty(x));
  if (valid.length === 0) return Manifold.union([]);
  return __is2D(valid[0]) ? CrossSection.hull(valid) : Manifold.hull(valid);
}

function __mesh_points3d(manifold: any, maxPoints = 192) {
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

function __is_likely_convex3d(manifold: any) {
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

function __minkowski_convex_pair3d(a: any, b: any) {
  const pointsA = __mesh_points3d(a);
  const pointsB = __mesh_points3d(b);
  if (pointsA.length === 0 || pointsB.length === 0) return Manifold.union([]);

  const sums = [];
  for (let i = 0; i < pointsA.length; i++) {
    const pa = pointsA[i]!;
    for (let j = 0; j < pointsB.length; j++) {
      const pb = pointsB[j]!;
      sums.push([pa[0] + pb[0], pa[1] + pb[1], pa[2] + pb[2]]);
    }
  }
  return Manifold.hull(sums as any);
}

function __minkowski_convex_chain3d(items: any[]) {
  let acc = items[0];
  for (let i = 1; i < items.length; i++) {
    acc = __minkowski_convex_pair3d(acc, items[i]);
  }
  return acc;
}

function __minkowski2d3d(items: any[]) {
  const valid = items.filter(x => !__isEmpty(x));
  if (valid.length === 0)
    return Manifold.union([]);
  if (valid.length === 1)
    return valid[0];

  // Check ALL items upfront
  for (const item of valid) {
    if (__is2D(item))
      throw new Error("2D minkowski not implemented");
    if (typeof item.minkowskiSum !== "function")
      throw new Error("Your manifold-3d build does not expose minkowskiSum");
  }

  let acc = valid[0];
  for (let i = 1; i < valid.length; i++) {
    acc = acc.minkowskiSum(valid[i]);
  }
  return acc;
}

function __normalizeScale(scale: number | number[] | undefined): [number, number] | undefined {
  if (scale === undefined) return undefined;
  const [sx, sy] = Array.isArray(scale)
    ? [scale[0], scale[1]]
    : [scale, scale];
  return [Math.max(0, sx as number), Math.max(0, sy as number)];
}

function __getDiagonalSlices(maxDeltaSqr: number, height: number, fn: number, fs: number): number {
  if (fn > 0) return fn;
  return Math.ceil(Math.sqrt(maxDeltaSqr + height * height) / fs);
}

function __getHelixSlices(rSqr: number, height: number, twist: number, fn: number, fa: number, fs: number, fe: number): number {
  const minSlices = Math.max(Math.ceil(twist / 120), 1);

  if (fn > 0) {
    return Math.max(Math.ceil((twist / 360) * fn), minSlices);
  }

  if (fe > 0) {
    const r = Math.sqrt(rSqr);
    const twistRad = (twist * Math.PI) / 180;
    const circumference = r * twistRad;
    return Math.max(Math.ceil(circumference / fe), minSlices);
  }

  const twistRad = (twist * Math.PI) / 180;

  const helixLength = Math.sqrt(rSqr * twistRad * twistRad + height * height);

  const faSlices = Math.ceil(twist / fa);
  const fsSlices = Math.ceil(helixLength / fs);

  return Math.max(Math.min(faSlices, fsSlices), minSlices);
}

function __getConicalHelixSlices(rSqr: number, height: number, twist: number, scale: number, fn: number, fa: number, fs: number): number {
  const minSlices = Math.max(Math.ceil(twist / 120), 1);

  if (fn > 0) {
    return Math.max(Math.ceil((twist / 360) * fn), minSlices);
  }

  const r = Math.sqrt(rSqr);
  const twistRad = (twist * Math.PI) / 180;

  const rEnd = r * scale;
  const rMid = (r + rEnd) / 2;
  const dh = height / (twist / 360);
  const helixLength = Math.sqrt(
    rMid * rMid * twistRad * twistRad + height * height
  );

  const faSlices = Math.ceil(twist / fa);
  const fsSlices = Math.ceil(helixLength / fs);

  return Math.max(Math.min(faSlices, fsSlices), minSlices);
}

function __computeExtrudeDivisions(shape: any, height: number, options: { twist?: number; scale?: number | number[]; fn?: number; fa?: number; fs?: number; fe?: number; slices?: number; }): number {
  if (options.slices !== undefined) {
    return Math.max(1, options.slices);
  }

  const twist = Math.abs(options.twist ?? 0);
  const fn = options.fn ?? 0;
  const fa = options.fa ?? 12;
  const fs = options.fs ?? 2;
  const fe = options.fe ?? 0;

  const normScale = __normalizeScale(options.scale);
  const sx = normScale?.[0] ?? 1;
  const sy = normScale?.[1] ?? 1;

  const isNonUniformScale = sx !== sy;
  const isUniformScale = sx === sy && sx !== 1;

  let rSqr = 1;
  try {
    const box = shape.bounds();
    const dx = box.max[0] - box.min[0];
    const dy = box.max[1] - box.min[1];
    rSqr = Math.pow(Math.max(dx, dy) / 2, 2);
  } catch {
    rSqr = 100;
  }

  const maxDeltaSqr = rSqr * Math.pow(Math.max(Math.abs(sx - 1), Math.abs(sy - 1)), 2);

  if (twist === 0) {
    if (isNonUniformScale) {
      return __getDiagonalSlices(maxDeltaSqr, height, fn, fs);
    }
    return 1;
  }

  if (isNonUniformScale) {
    const diag = __getDiagonalSlices(maxDeltaSqr, height, fn, fs);
    const helix = __getHelixSlices(rSqr, height, twist, fn, fa, fs, fe);
    return Math.max(diag, helix);
  }

  if (isUniformScale) {
    return __getConicalHelixSlices(rSqr, height, twist, sx, fn, fa, fs);
  }

  return __getHelixSlices(rSqr, height, twist, fn, fa, fs, fe);
}

function __extrude(shape: any, height = 1, options: {twist?: number; scale?: number | number[]; center?: boolean; fn?: number; fa?: number; fs?: number; fe?: number; slices?: number;} = {}) {
  if (__isEmpty(shape)) {
    return Manifold.union([]);
  }

  if (!__is2D(shape)) {
    return shape;
  }

  const normScale = __normalizeScale(options.scale);

  const nDivisions = __computeExtrudeDivisions(shape, height, { ...options, scale: normScale });

  return shape.extrude(
    height,
    Math.max(0, nDivisions - 1),
    options.twist !== undefined ? Math.abs(options.twist) : undefined,
    normScale,
    options.center
  );
}

function __revolve(shape: any, fn = 0, fa = 12, fs = 2, angle = 360) {
  if (__isEmpty(shape)) {
    return Manifold.union([]);
  }
  if (__is2D(shape)) {
    if (fn <= 0) {
      __sync_quality(fa, fs);
    }
    return shape.revolve(fn, angle);
  }
  return shape;
}

function __sampleQuadratic(x0: number, y0: number, x1: number, y1: number, x2: number, y2: number, steps: number): [number, number][] {
  const pts: [number, number][] = [];
  for (let i = 1; i <= steps; i++) {
    const t = i / steps, mt = 1 - t;
    pts.push([
      mt*mt*x0 + 2*mt*t*x1 + t*t*x2,
      mt*mt*y0 + 2*mt*t*y1 + t*t*y2,
    ]);
  }
  return pts;
}

function __sampleCubic(x0: number, y0: number, x1: number, y1: number, x2: number, y2: number, x3: number, y3: number, steps: number): [number, number][] {
  const pts: [number, number][] = [];
  for (let i = 1; i <= steps; i++) {
    const t = i / steps, mt = 1 - t;
    pts.push([
      mt*mt*mt*x0 + 3*mt*mt*t*x1 + 3*mt*t*t*x2 + t*t*t*x3,
      mt*mt*mt*y0 + 3*mt*mt*t*y1 + 3*mt*t*t*y2 + t*t*t*y3,
    ]);
  }
  return pts;
}

function __pathToContours(commands: any[], fn: number): [number, number][][] {
  const steps = Math.max(2, fn > 0 ? Math.round(fn / 4) : 4);
  const contours: [number, number][][] = [];
  let current: [number, number][] | null = null;
 
  for (const cmd of commands) {
    switch (cmd.type) {
      case "M": // Move to — starts a new contour
        if (current && current.length >= 3) contours.push(current);
        current = [[cmd.x, cmd.y]];
        break;
 
      case "L": // Line to
        current?.push([cmd.x, cmd.y]);
        break;
 
      case "Q": { // Quadratic bezier
        if (!current) break;
        const prev = current[current.length - 1]!;
        const pts = __sampleQuadratic(prev[0], prev[1], cmd.x1, cmd.y1, cmd.x, cmd.y, steps);
        for (const [px, py] of pts) current.push([px, py]);
        break;
      }
 
      case "C": { // Cubic bezier
        if (!current) break;
        const prev = current[current.length - 1]!;
        const pts = __sampleCubic(prev[0], prev[1], cmd.x1, cmd.y1, cmd.x2, cmd.y2, cmd.x, cmd.y, steps);
        for (const [px, py] of pts) current.push([px, py]);
        break;
      }
 
      case "Z": // Close contour
        if (current && current.length >= 3) contours.push(current);
        current = null;
        break;
    }
  }
  if (current && current.length >= 3) contours.push(current);
  return contours.map(c => c.map(([x, y]): [number, number] => [x, -y]));
}

const __opentypeFontCache = new Map<string, any>();

function __getOpentypeFont(base64DataUrl: string): any | undefined {
  const cached = __opentypeFontCache.get(base64DataUrl);
  if (cached) return cached;

  try {
    const base64 = base64DataUrl.replace(/^data:[^;]+;base64,/, "");
    const binaryStr = atob(base64);
    const bytes = new Uint8Array(binaryStr.length);
    for (let i = 0; i < binaryStr.length; i++) {
      bytes[i] = binaryStr.charCodeAt(i);
    }
    const font = opentypeParse(bytes.buffer as ArrayBuffer);
    __opentypeFontCache.set(base64DataUrl, font);
    return font;
  } catch (e) {
    console.log("err: ", e);
    return undefined;
  }
}

function __opentypeGlyphContours(ch: string, font: any, size: number, fn: number): { contours: [number, number][][]; width: number } | undefined {
  const fontSize = size * 100 / 72;

  if (ch === " ") {
    const spaceGlyph = font.charToGlyph(" ");

    const scale = fontSize / font.unitsPerEm;

    const spaceWidth =
      (spaceGlyph?.advanceWidth ?? font.unitsPerEm * 0.25) * scale;

    return {
      contours: [],
      width: spaceWidth
    };
  }

  const glyph = font.charToGlyph(ch);

  if (!glyph || glyph.index === 0) {
    return undefined;
  }

  const scale = fontSize / font.unitsPerEm;

  const advance =
    (glyph.advanceWidth ?? 0) * scale;

  const glyphPath = glyph.getPath(
    0,
    0,
    fontSize
  );

  const commands = glyphPath.commands;

  if (!commands || commands.length === 0) {
    return {
      contours: [],
      width: advance
    };
  }

  const contours = __pathToContours(commands, fn);

  return {
    contours,
    width: advance
  };
}

function __opentypeTextContours(chars: string[], size: number, spacing: number, direction: string, fontBase64: string, fn: number): [number, number][][] | undefined {
  const font = __getOpentypeFont(fontBase64);
  
  if (!font) return undefined;

  const contours: [number, number][][] = [];
  const isVertical = direction === "ttb" || direction === "btt";

  if (isVertical) {
    const ySign = direction === "ttb" ? -1 : 1;
    let cursorY = 0;
    for (const ch of chars) {
      const glyph = __opentypeGlyphContours(ch, font, size, fn);
      if (!glyph) return undefined;
      const xOffset = -glyph.width / 2;
      for (const contour of glyph.contours) {
        contours.push(contour.map(([x, y]): [number, number] => [x + xOffset, cursorY + y * ySign]));
      }
      cursorY += size * spacing * ySign;
    }
  } else {
    let cursorX = 0;
    for (const ch of chars) {
      const glyph = __opentypeGlyphContours(ch, font, size, fn);
      if (!glyph) return undefined;
      for (const contour of glyph.contours) {
        contours.push(contour.map(([x, y]): [number, number] => [x + cursorX, y]));
      }
      cursorX += glyph.width * spacing;
    }
  }

  return contours;
}

const __canvasGlyphCache = new Map<string, { contours: [number, number][][], width: number }>();

function __fontToCss(font: string, px: number): string {
  const spec = String(font || "Liberation Sans");
  const family = (spec.split(":")[0] || "Liberation Sans").replace(/"/g, "");
  const styleSpec = spec.toLowerCase();
  const weight = styleSpec.includes("bold") ? "700" : "400";
  const style = styleSpec.includes("italic") ? "italic" : "normal";
  return `${style} ${weight} ${px}px "${family}", Arial, sans-serif`;
}

function __canvasForText(): any {
  if (typeof document !== "undefined" && typeof document.createElement === "function") {
    return document.createElement("canvas");
  }
  if (typeof OffscreenCanvas !== "undefined") {
    return new OffscreenCanvas(1, 1);
  }
  return undefined;
}

function __canvasGlyphContours(ch: string, font: string, size: number): { contours: [number, number][][], width: number } | undefined {
  if (ch === " ") return { contours: [], width: size * 0.35 };

  const px = 128;
  const cacheKey = `${font}|${ch}|${px}`;
  const cached = __canvasGlyphCache.get(cacheKey);
  if (cached) return cached;

  const canvas = __canvasForText();
  const ctx = canvas?.getContext("2d", { willReadFrequently: true } as any) as any;
  if (!canvas || !ctx) return undefined;

  ctx.font = __fontToCss(font, px);
  const metrics = ctx.measureText(ch);
  const ascent = Math.ceil(metrics.actualBoundingBoxAscent || px * 0.8);
  const descent = Math.ceil(metrics.actualBoundingBoxDescent || px * 0.25);
  const leftBearing = Math.ceil(metrics.actualBoundingBoxLeft || 0);
  const rightBearing = Math.ceil(metrics.actualBoundingBoxRight || metrics.width || px * 0.6);
  const pad = 8;
  const widthPx = Math.max(1, Math.ceil(leftBearing + rightBearing + pad * 2));
  const heightPx = Math.max(1, ascent + descent + pad * 2);

  canvas.width = widthPx;
  canvas.height = heightPx;
  ctx.clearRect(0, 0, widthPx, heightPx);
  ctx.font = __fontToCss(font, px);
  ctx.fillStyle = "#fff";
  ctx.textBaseline = "alphabetic";
  ctx.fillText(ch, pad + leftBearing, pad + ascent);

  const image = ctx.getImageData(0, 0, widthPx, heightPx);
  let minX = widthPx, minY = heightPx, maxX = -1, maxY = -1;
  for (let y = 0; y < heightPx; y++) {
    for (let x = 0; x < widthPx; x++) {
      if (image.data[(y * widthPx + x) * 4 + 3] > 32) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  const scale = (size * 100 / 72) / px;
  const result = { contours: [] as [number, number][][], width: Math.max(1, metrics.width) * scale };
  if (maxX < minX || maxY < minY) {
    __canvasGlyphCache.set(cacheKey, result);
    return result;
  }

  for (let y = minY; y <= maxY; y++) {
    let x = minX;
    while (x <= maxX) {
      while (x <= maxX && image.data[(y * widthPx + x) * 4 + 3] <= 32) x++;
      const start = x;
      while (x <= maxX && image.data[(y * widthPx + x) * 4 + 3] > 32) x++;
      if (start < x) {
        const x0 = (start - minX) * scale;
        const x1 = (x - minX) * scale;
        const y0 = (maxY - y) * scale;
        const y1 = (maxY - y + 1) * scale;
        result.contours.push([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);
      }
    }
  }

  __canvasGlyphCache.set(cacheKey, result);
  return result;
}

function __canvasTextContours(chars: string[], size: number, spacing: number, direction: string, font: string): [number, number][][] | undefined {
  if (typeof document === "undefined" && typeof OffscreenCanvas === "undefined") return undefined;

  const contours: [number, number][][] = [];
  const isVertical = direction === "ttb" || direction === "btt";
  if (isVertical) {
    const ySign = direction === "ttb" ? -1 : 1;
    let cursorY = 0;
    for (const ch of chars) {
      const glyph = __canvasGlyphContours(ch, font, size);
      if (!glyph) return undefined;
      const xOffset = -glyph.width / 2;
      for (const contour of glyph.contours) {
        contours.push(contour.map(([x, y]): [number, number] => [x + xOffset, cursorY + y * ySign]));
      }
      cursorY += size * spacing * ySign;
    }
  } else {
    let cursorX = 0;
    for (const ch of chars) {
      const glyph = __canvasGlyphContours(ch, font, size);
      if (!glyph) return undefined;
      for (const contour of glyph.contours) {
        contours.push(contour.map(([x, y]): [number, number] => [x + cursorX, y]));
      }
      cursorX += glyph.width * spacing;
    }
  }

  return contours;
}

function __text(text: string, size: number = 10, font: string, halign: string = "left", valign: string = "baseline", spacing: number = 1, direction: string = "ltr", fn: number = 0, fontBase64Data: string | undefined = undefined): any {
  if (!text || text.length === 0) return CrossSection.square([0.001, 0.001], false);

  void fn;

  const dir = (direction || "ltr").toLowerCase();
  const chars = dir === "rtl" ? Array.from(text).reverse() : Array.from(text);

  let contours: [number, number][][] | undefined;

  contours = __canvasTextContours(chars, size, spacing, dir, font);

  if (!fontBase64Data) console.log("fontBase64Data is undefined");
  else console.log("fontBase64Data is defined");
  console.log("canvas contours:", contours?.length);

  if (!contours && fontBase64Data) {
    contours = __opentypeTextContours(chars, size, spacing, dir, fontBase64Data, fn);
  }
  console.log("opentype contours:", contours?.length);
  if (!contours || contours.length === 0) {
    console.log("returning simple square");
    return CrossSection.square([0.001, 0.001], false);
  }

  let left = Infinity, right = -Infinity, top = -Infinity, bottom = Infinity;
  for (const c of contours) {
    for (const [x, y] of c) {
      if (x < left) left = x;
      if (x > right) right = x;
      if (y > top) top = y;
      if (y < bottom) bottom = y;
    }
  }

  let dx = 0;
  if      (halign === "center") dx = -(left + right) / 2;
  else if (halign === "right")  dx = -right;
  else                          dx = -left;

  let dy = 0;
  if (valign === "center") dy = -(top + bottom) / 2;
  else if (valign === "top")    dy = -top;
  else if (valign === "bottom") dy = -bottom;

  const shifted = contours.map(c => c.map(([x, y]): [number, number] => [x + dx, y + dy]));
  return CrossSection.ofPolygons(shifted, 'EvenOdd');
}

function __sync_quality(fa: any, fs: any) {
  if (typeof wasm.setMinCircularAngle === "function") {
    if (typeof fa === "number" && fa > 0) {
      wasm.setMinCircularAngle(fa);
    }
  }
  if (typeof wasm.setMinCircularEdgeLength === "function") {
    if (typeof fs === "number" && fs > 0) {
      wasm.setMinCircularEdgeLength(fs);
    }
  }
}

function __rotate(shape: any, a: any, v?: any) {
  if (!shape) return shape;

  if (__is2D(shape)) {
    if (v !== undefined) {
      const vz = Array.isArray(v) ? v[2] : (v && typeof v === 'object' ? (v.z || v[2]) : 0);
      const vzNum = Number(vz) || 0;
      if (Math.abs(vzNum) > 1e-9) {
        return shape.rotate(Number(a) * Math.sign(vzNum));
      }
      return shape;
    }
    if (Array.isArray(a)) {
      return shape.rotate(Number(a[2]) || 0);
    }
    return shape.rotate(Number(a) || 0);
  }

  // 3D shape
  if (v !== undefined) {
    const theta = (Number(a) || 0) * Math.PI / 180;
    const cosT = Math.cos(theta);
    const sinT = Math.sin(theta);
    const oneMinusCosT = 1 - cosT;

    let vx = 0, vy = 0, vz = 0;
    if (Array.isArray(v)) {
      vx = Number(v[0]) || 0;
      vy = Number(v[1]) || 0;
      vz = Number(v[2]) || 0;
    } else if (v && typeof v === 'object') {
      vx = Number(v.x || v[0]) || 0;
      vy = Number(v.y || v[1]) || 0;
      vz = Number(v.z || v[2]) || 0;
    } else {
      vx = Number(v) || 0;
    }

    const len = Math.sqrt(vx * vx + vy * vy + vz * vz);
    if (len < 1e-9) return shape;

    const ux = vx / len;
    const uy = vy / len;
    const uz = vz / len;

    const R = [
      [oneMinusCosT * ux * ux + cosT, oneMinusCosT * ux * uy - sinT * uz, oneMinusCosT * ux * uz + sinT * uy, 0],
      [oneMinusCosT * ux * uy + sinT * uz, oneMinusCosT * uy * uy + cosT, oneMinusCosT * uy * uz - sinT * ux, 0],
      [oneMinusCosT * ux * uz - sinT * uy, oneMinusCosT * uy * uz + sinT * ux, oneMinusCosT * uz * uz + cosT, 0],
      [0, 0, 0, 1]
    ];

    return __safe_transform(shape, R);
  } else {
    if (Array.isArray(a)) {
      return shape.rotate(a);
    }
    return shape.rotate([0, 0, Number(a) || 0]);
  }
}

function __translate(shape: any, v: any) {
  if (!shape) return shape;
  if (__is2D(shape)) {
    let x = 0, y = 0;
    if (Array.isArray(v)) {
      x = Number(v[0]) || 0;
      y = Number(v[1]) || 0;
    } else if (v && typeof v === "object") {
      x = Number(v.x || v[0]) || 0;
      y = Number(v.y || v[1]) || 0;
    } else {
      x = Number(v) || 0;
    }
    return shape.translate([x, y]);
  } else {
    let x = 0, y = 0, z = 0;
    if (Array.isArray(v)) {
      x = Number(v[0]) || 0;
      y = Number(v[1]) || 0;
      z = Number(v[2]) || 0;
    } else if (v && typeof v === "object") {
      x = Number(v.x || v[0]) || 0;
      y = Number(v.y || v[1]) || 0;
      z = Number(v.z || v[2]) || 0;
    } else {
      x = Number(v) || 0;
    }
    return shape.translate([x, y, z]);
  }
}

function __scale(shape: any, v: any) {
  if (!shape) return shape;
  if (__is2D(shape)) {
    let x = 1, y = 1;
    if (Array.isArray(v)) {
      x = v[0] !== undefined && v[0] !== null ? Number(v[0]) : 1;
      y = v[1] !== undefined && v[1] !== null ? Number(v[1]) : 1;
    } else if (v && typeof v === "object") {
      x = (v.x !== undefined ? v.x : v[0]) !== undefined ? Number(v.x !== undefined ? v.x : v[0]) : 1;
      y = (v.y !== undefined ? v.y : v[1]) !== undefined ? Number(v.y !== undefined ? v.y : v[1]) : 1;
    } else if (typeof v === "number" && !Number.isNaN(v)) {
      x = y = v;
    }
    return shape.scale([x, y]);
  } else {
    let x = 1, y = 1, z = 1;
    if (Array.isArray(v)) {
      x = v[0] !== undefined && v[0] !== null ? Number(v[0]) : 1;
      y = v[1] !== undefined && v[1] !== null ? Number(v[1]) : 1;
      z = v[2] !== undefined && v[2] !== null ? Number(v[2]) : 1;
    } else if (v && typeof v === "object") {
      x = (v.x !== undefined ? v.x : v[0]) !== undefined ? Number(v.x !== undefined ? v.x : v[0]) : 1;
      y = (v.y !== undefined ? v.y : v[1]) !== undefined ? Number(v.y !== undefined ? v.y : v[1]) : 1;
      z = (v.z !== undefined ? v.z : v[2]) !== undefined ? Number(v.z !== undefined ? v.z : v[2]) : 1;
    } else if (typeof v === "number" && !Number.isNaN(v)) {
      x = y = z = v;
    }
    return shape.scale([x, y, z]);
  }
}

function __mirror(shape: any, v: any) {
  if (!shape) return shape;
  if (__is2D(shape)) {
    let x = 0, y = 0;
    if (Array.isArray(v)) {
      x = Number(v[0]) || 0;
      y = Number(v[1]) || 0;
    } else if (v && typeof v === "object") {
      x = Number(v.x || v[0]) || 0;
      y = Number(v.y || v[1]) || 0;
    } else {
      x = Number(v) || 0;
    }
    return shape.mirror([x, y]);
  } else {
    let x = 0, y = 0, z = 0;
    if (Array.isArray(v)) {
      x = Number(v[0]) || 0;
      y = Number(v[1]) || 0;
      z = Number(v[2]) || 0;
    } else if (v && typeof v === "object") {
      x = Number(v.x || v[0]) || 0;
      y = Number(v.y || v[1]) || 0;
      z = Number(v.z || v[2]) || 0;
    } else {
      x = Number(v) || 0;
    }
    return shape.mirror([x, y, z]);
  }
}

function __sphere(radius: number, fn = 0, fa = 12, fs = 2) {
  let N: number;
  if (fn > 0) {
    N = Math.max(3, Math.ceil(fn));
  } else {
    const N_fa = 360 / fa;
    const N_fs = (2 * Math.PI * radius) / fs;
    N = Math.ceil(Math.max(Math.min(N_fa, N_fs), 5));
  }

  const R = Math.floor((N + 1) / 2);

  const verts: number[] = [];
  const tris: number[] = [];

  // Generate rings
  const rings: number[][] = [];
  for (let i = 0; i < R; i++) {
    const phi = (Math.PI * (i + 0.5)) / R;
    const ring_r = radius * Math.sin(phi);
    const z = radius * Math.cos(phi);
    const ring: number[] = [];
    for (let j = 0; j < N; j++) {
      const theta = (2 * Math.PI * j) / N;
      ring.push(verts.length / 3);
      verts.push(ring_r * Math.cos(theta), ring_r * Math.sin(theta), z);
    }
    rings.push(ring);
  }

  // Top cap: flat triangulation of first ring
  const top = rings[0]!;
  for (let j = 1; j < N - 1; j++) {
    tris.push(top[0]!, top[j]!, top[j + 1]!);
  }

  // Middle bands
  for (let r = 0; r < R - 1; r++) {
    const lo = rings[r]!;
    const hi = rings[r + 1]!;
    for (let j = 0; j < N; j++) {
      const jn = (j + 1) % N;
      tris.push(lo[j]!, hi[j]!, hi[jn]!);
      tris.push(lo[j]!, hi[jn]!, lo[jn]!);
    }
  }

  // Bottom cap: flat triangulation of last ring
  const bot = rings[R - 1]!;
  for (let j = 1; j < N - 1; j++) {
    tris.push(bot[0]!, bot[j + 1]!, bot[j]!);
  }

  const mesh: any = {
    vertProperties: new Float32Array(verts),
    triVerts: new Uint32Array(tris),
    numProp: 3,
  };

  const sphere = new Manifold(mesh);
  
  return sphere;
}

function __cylinder(height: number, radiusLow: number, radiusHigh = -1.0, fn = 0, center = false, fa = 12, fs = 2) {
  let segs = fn;
  if (segs <= 0) {
    const r = Math.max(radiusLow, radiusHigh < 0 ? radiusLow : radiusHigh);
    const N_fa = 360 / fa;
    const N_fs = (2 * Math.PI * r) / fs;
    segs = Math.ceil(Math.max(Math.min(N_fa, N_fs), 5));
  }
  return Manifold.cylinder(height, radiusLow, radiusHigh, segs, center);
}

function __circle(radius: number, circularSegments = 0, fa = 12, fs = 2) {
  if (circularSegments <= 0) {
    __sync_quality(fa, fs);
  }
  return CrossSection.circle(radius, circularSegments);
}

function __getSignedArea(contour: [number, number][]): number {
  let area = 0;
  const n = contour.length;
  for (let i = 0; i < n; i++) {
    const p1 = contour[i]!;
    const p2 = contour[(i + 1) % n]!;
    area += p1[0] * p2[1] - p2[0] * p1[1];
  }
  return area / 2;
}

function __forceWinding(contour: [number, number][], ccw: boolean): [number, number][] {
  const area = __getSignedArea(contour);
  if (ccw && area < 0) {
    contour.reverse();
  } else if (!ccw && area > 0) {
    contour.reverse();
  }
  return contour;
}

function __polygon(points: any, paths?: any) {
  if (!points || !Array.isArray(points) || points.length === 0) {
    return CrossSection.ofPolygons([]);
  }

  // If paths is not specified
  if (paths === undefined || paths === null) {
    const ccwPoints = __forceWinding([...points], true);
    return CrossSection.ofPolygons([ccwPoints]);
  }

  // If paths is a 1D array of indices (e.g. [0, 1, 2, 3])
  if (Array.isArray(paths) && paths.length > 0 && !Array.isArray(paths[0])) {
    const contour = paths.map((idx: any) => points[Number(idx) || 0]).filter(Boolean);
    const ccwContour = __forceWinding(contour, true);
    return CrossSection.ofPolygons([ccwContour]);
  }

  // If paths is a 2D array of indices (e.g. [[0, 1, 2], [3, 4, 5]])
  if (Array.isArray(paths)) {
    const contours = paths.map((path: any, pathIdx: number) => {
      let contour: any[] = [];
      if (Array.isArray(path)) {
        contour = path.map((idx: any) => points[Number(idx) || 0]).filter(Boolean);
      } else if (typeof path === "number") {
        contour = [points[path]];
      }
      if (contour.length > 0) {
        // First path is outer boundary (CCW), subsequent paths are holes (CW)
        return __forceWinding(contour, pathIdx === 0);
      }
      return [];
    }).filter((c: any) => c.length > 0);
    return CrossSection.ofPolygons(contours);
  }

  // Fallback
  const ccwPoints = __forceWinding([...points], true);
  return CrossSection.ofPolygons([ccwPoints]);
}

function __parse_color_for_scope(c: any, alpha: any): any {
  const base = __parse_color_value(c);
  if (!base) return undefined;
  const a = (alpha !== undefined && alpha !== null && Number.isFinite(Number(alpha)))
    ? Number(alpha) : base[3];
  return [base[0], base[1], base[2], a];
}

async function __surface(dataUrl: string, opts: { center?: boolean; fn?: number; fa?: number; fs?: number } = {}) {
  const { center = false } = opts;
  const { width, height, data } = await decodeImageToPixels(dataUrl);

  // Build height values (0–100) from grayscale (OpenSCAD: black (0) → 0, white (255) → 100)
  const Z = (x: number, y: number): number => {
    const row = y;
    const i = (row * width + x) * 4;          // RGBA offset
    const gray = 0.2126 * data![i]! + 0.7152 * data![i + 1]! + 0.0722 * data![i + 2]!;
    return (gray / 255) * 100;
  };

  const ox = center ? -(width  - 1) / 2 : 0;
  const oy = center ? -(height - 1) / 2 : 0;

  // Build a watertight mesh
  const numTop = width * height;
  const numQuads = (width - 1) * (height - 1);
  const vertProps: number[] = [];
  const tris: number[] = [];

  const topIdx = (x: number, y: number) => y * width  + x;
  const centerIdx = (x: number, y: number) => numTop + y * (width - 1) + x;
  const botIdx = (x: number, y: number) => numTop + numQuads + y * width + x;

  // top surface grid vertices
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      vertProps.push(x + ox, y + oy, Z(x, y));
    }
  }
  // top surface center vertices (one per quad, with average height of 4 corners)
  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width - 1; x++) {
      const zCenter = (Z(x, y) + Z(x + 1, y) + Z(x, y + 1) + Z(x + 1, y + 1)) / 4;
      vertProps.push(x + 0.5 + ox, y + 0.5 + oy, zCenter);
    }
  }
  // bottom surface vertices (same XY, z=0)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      vertProps.push(x + ox, y + oy, 0);
    }
  }

  // top surface faces (CCW when viewed from +Z, 4 triangles per quad)
  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width - 1; x++) {
      const a = topIdx(x, y);
      const b = topIdx(x + 1, y);
      const c = topIdx(x, y + 1);
      const d = topIdx(x + 1, y + 1);
      const center = centerIdx(x, y);
      tris.push(a, b, center);
      tris.push(b, d, center);
      tris.push(d, c, center);
      tris.push(c, a, center);
    }
  }

  // bottom surface faces (CW from +Z = CCW from -Z, outward normal points down)
  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width - 1; x++) {
      const a = botIdx(x, y);
      const b = botIdx(x + 1, y);
      const c = botIdx(x, y + 1);
      const d = botIdx(x + 1, y + 1);
      tris.push(a, c, b);
      tris.push(b, c, d);
    }
  }

  // side walls (4 edges of the grid)
  for (let x = 0; x < width - 1; x++) {
    tris.push(topIdx(x, 0), botIdx(x, 0), topIdx(x + 1, 0));
    tris.push(topIdx(x + 1, 0), botIdx(x, 0), botIdx(x + 1, 0));
  }
  // Back edge (y = height-1, normal points in +Y)
  const yb = height - 1;
  for (let x = 0; x < width - 1; x++) {
    tris.push(topIdx(x, yb), topIdx(x + 1, yb), botIdx(x, yb));
    tris.push(topIdx(x + 1, yb), botIdx(x + 1, yb), botIdx(x, yb));
  }
  // Left edge (x = 0, normal points in -X)
  for (let y = 0; y < height - 1; y++) {
    tris.push(topIdx(0, y), topIdx(0, y + 1), botIdx(0, y));
    tris.push(topIdx(0, y + 1), botIdx(0, y + 1), botIdx(0, y));
  }
  // Right edge (x = width-1, normal points in +X)
  const xr = width - 1;
  for (let y = 0; y < height - 1; y++) {
    tris.push(topIdx(xr, y), botIdx(xr, y), topIdx(xr, y + 1));
    tris.push(topIdx(xr, y + 1), botIdx(xr, y), botIdx(xr, y + 1));
  }

  return new Manifold(new wasm.Mesh({
    vertProperties: new Float32Array(vertProps),
    triVerts: new Uint32Array(tris),
    numProp: 3,
  }));
}

async function decodeImageToPixels(dataUrl: string): Promise<{ width: number; height: number; data: Uint8ClampedArray; }> {
  if (typeof OffscreenCanvas !== "undefined") {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const canvas = new OffscreenCanvas(img.width, img.height);
        const ctx = canvas.getContext("2d")!;
        ctx.drawImage(img, 0, 0);
        const { data, width, height } = ctx.getImageData(0, 0, img.width, img.height);
        resolve({ width, height, data });
      };
      img.onerror = () => reject(new Error("__surface: failed to decode image"));
      img.src = dataUrl;
    });
  } else {
    const img = await loadImage(dataUrl);
    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    const { data, width, height } = ctx.getImageData(0, 0, img.width, img.height);
    return { width, height, data: data as unknown as Uint8ClampedArray };
  }
}

// Export all runtime symbols for compiled code
export {
  Manifold, CrossSection, wasm,
  __sphere, __cylinder, __circle, __rotate, __polygon, __translate, __scale, __mirror,
  is_undef_fn, is_bool_fn, is_num_fn, is_string_fn, is_list_fn, is_function_fn,
  sin_fn, cos_fn, tan_fn, asin_fn, acos_fn, atan_fn, atan2_fn,
  abs_fn, sign_fn, floor_fn, ceil_fn, round_fn, sqrt_fn, exp_fn, ln_fn, log_fn,
  min_fn, max_fn, norm_fn, cross_fn,
  len_fn, str_fn, chr_fn, ord_fn, concat_fn, search_fn, lookup_fn,
  openscad_assert_fn, __eq, __add, __sub, __mul, __div, __mod, __neg, __pos,
  version_fn, version_num_fn,
  PI, INF, NAN, undef, _EPSILON,
  __children_stack, __with_children, parent_module_fn,
  __is_finite_matrix4, __to_manifold_mat4, __safe_transform, __identity4,
  __safe_attach_transform, __safe_offset2d, __safe_project3d,
  __apply_color,
  __flat_map_iter, __range, __is2D, __union2d3d, __difference2d3d, __intersection2d3d, __hull2d3d, __minkowski2d3d,
  __extrude, __revolve,
  __text, __parse_color_for_scope, __surface
};
