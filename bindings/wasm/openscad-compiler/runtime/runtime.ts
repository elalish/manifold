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

const wasm = await Module();
wasm.setup();
const { Manifold, CrossSection } = wasm;

function is_undef_fn(x: any) { return arguments.length === 1 ? (x === undefined || x === null) : undefined; }
function is_bool_fn(x: any) { return arguments.length === 1 ? (typeof x === "boolean") : undefined; }
function is_num_fn(x: any) { return arguments.length === 1 ? (typeof x === "number" && !Number.isNaN(x)) : undefined; }
function is_string_fn(x: any) { return arguments.length === 1 ? (typeof x === "string") : undefined; }
function is_list_fn(x: any) { return arguments.length === 1 ? Array.isArray(x) : undefined; }
function is_function_fn(x: any) { return arguments.length === 1 ? (typeof x === "function") : undefined; }
function is_object_fn(x: any) { return arguments.length === 1 ? (x !== null && typeof x === "object" && !Array.isArray(x)) : undefined; }

// Trig (OpenSCAD uses degrees)
function sin_fn(x: any) { return Math.sin(x * Math.PI / 180); }
function cos_fn(x: any) { return Math.cos(x * Math.PI / 180); }
function tan_fn(x: any) { return Math.tan(x * Math.PI / 180); }
function asin_fn(x: any) { return Math.asin(x) * 180 / Math.PI; }
function acos_fn(x: any) { return Math.acos(x) * 180 / Math.PI; }
function atan_fn(x: any) { return Math.atan(x) * 180 / Math.PI; }
function atan2_fn(y: any, x: any) { return Math.atan2(y, x) * 180 / Math.PI; }

// Math (OpenSCAD built-ins)
let abs_fn = Math.abs;
let sign_fn = Math.sign;
let floor_fn = Math.floor;
let ceil_fn = Math.ceil;
let round_fn = Math.round;
let sqrt_fn = Math.sqrt;
let exp_fn = Math.exp;
function ln_fn(x: any) { return Math.log(x); }
function log_fn(x: any) { return Math.log(x); }
function min_fn(...a: any[]) { return a.length === 1 && Array.isArray(a[0]) ? Math.min(...a[0]) : Math.min(...a); }
function max_fn(...a: any[]) { return a.length === 1 && Array.isArray(a[0]) ? Math.max(...a[0]) : Math.max(...a); }
function norm_fn(v: any) { return Math.sqrt(v.reduce((s: any, x: any) => s + x * x, 0)); }
function cross_fn(a: any, b: any) {
  if (!Array.isArray(a) || !Array.isArray(b)) return undefined;
  // 2D vector yeilds the scalar z-component and 3D vector yeilds the vector
  if (a.length === 2 && b.length === 2) return a[0] * b[1] - a[1] * b[0];
  if (a.length === 3 && b.length === 3) {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  }
  console.warn("WARNING: Invalid arguments to cross()");
  return undefined;
}

// String & list (OpenSCAD built-ins)
function len_fn(x: any) {
  if (typeof x === "string" || Array.isArray(x)) return x.length;
  // returns undef and emits a warning for non-string/non-list inputs.
  console.warn("WARNING: len() parameter could not be converted");
  return undefined;
}

function __ostr(x: any): string {
  if (x === undefined || x === null) return "undef";
  if (typeof x === "boolean") return x ? "true" : "false";
  if (typeof x === "string") return x;
  if (Array.isArray(x)) return "[" + x.map(__ostrInner).join(", ") + "]";
  return String(x);
}
function __ostrInner(x: any): string {
  return typeof x === "string" ? `"${x}"` : __ostr(x);
}
function str_fn(...a: any[]) { return a.map(__ostr).join(""); }
function chr_fn(n: any) { return Array.isArray(n) ? n.map(c => String.fromCharCode(c)).join("") : String.fromCharCode(n); }
function ord_fn(s: any) { return s == null || s.length === 0 ? undefined : s.charCodeAt(0); }
function concat_fn(...a: any[]) { return [].concat(...a); }

function __search_match(needle: any, entry: any, idx_col: any) {
  const col = idx_col !== undefined ? idx_col : 0;
  if (col === 0 && __eq(entry, needle)) return true;
  return Array.isArray(entry) && col < entry.length && __eq(entry[col], needle);
}

function search_fn(needle: any, haystack: any, num_returns: any, idx_col: any) {
  if (!is_list_fn(needle) && !is_string_fn(needle)) {
    const indices: any[] = [];
    for (let i = 0; i < haystack.length; i++) {
      if (__search_match(needle, haystack[i], idx_col)) indices.push(i);
    }
    if (num_returns === 0) return indices;
    return indices.length > 0 ? [indices[0]] : [];
  }
  if (is_string_fn(needle) && is_string_fn(haystack)) {
    let result: any[] = [];
    for (let ch of needle) {
      let indices = [];
      for (let i = 0; i < haystack.length; i++) { if (haystack[i] === ch) indices.push(i); }
      if (num_returns === 1 || num_returns === undefined) {
        // omits characters with no match from the result
        if (indices.length > 0) result.push(indices[0]);
      } else {
        result.push(num_returns === 0 ? indices : indices.slice(0, num_returns));
      }
    }
    return result;
  }
  if (is_list_fn(haystack) && is_list_fn(needle)) {
    return needle.map(function(n: any) {
      let indices = [];
      for (let i = 0; i < haystack.length; i++) {
        if (__search_match(n, haystack[i], idx_col)) indices.push(i);
      }
      return num_returns === 0 ? indices : (indices.length > 0 ? indices[0] : []);
    });
  }
  if (is_string_fn(needle) && is_list_fn(haystack)) {
    return [...needle].map(function(n) {
      let indices = [];
      for (let i = 0; i < haystack.length; i++) {
        let item = idx_col !== undefined ? haystack[i][idx_col] : haystack[i];
        if (__eq(item, n)) indices.push(i);
      }
      if (num_returns === 0) return indices;
      if (indices.length === 0) return [];
      return num_returns === 1 || num_returns === undefined
        ? indices[0]
        : indices.slice(0, num_returns);
    });
  }
  return undefined;
}
function lookup_fn(key: any, table: any) {
  if (key <= table[0][0]) return table[0][1];
  if (key >= table[table.length - 1][0]) return table[table.length - 1][1];
  for (let i = 0; i < table.length - 1; i++) {
    if (table[i][0] <= key && key <= table[i + 1][0]) {
      let t = (key - table[i][0]) / (table[i + 1][0] - table[i][0]);
      return table[i][1] + t * (table[i+1][1] - table[i][1]);
    }
  }
  return undefined;
}

function __truthy(x: any) {
  if (x === undefined || x === null || x === false) return false;
  // OpenSCAD treats every non-zero number as true, including nan and inf.
  if (typeof x === "number") return x !== 0;
  if (typeof x === "string" || Array.isArray(x)) return x.length > 0;
  return true;
}

// Control
function openscad_assert_fn(cond: any, msg: any) {
  if (!__truthy(cond)) {
    console.trace("Assertion failed:", msg);
    throw new Error(msg || "Assertion failed");
  }
}

function __eq(a: any, b: any) {
  if (a === b) return true;
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) { if (!__eq(a[i], b[i])) return false; }
    return true;
  }
  return false;
}

function __cmpCat(x: any): string {
  const t = typeof x;
  if (t === "number") return "n";
  if (t === "boolean") return "b";
  if (t === "string") return "s";
  if (Array.isArray(x)) return (x as any).__isRange ? "r" : "v";
  return "u";
}
function __veccmp(a: any[], b: any[]): number {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    if (a[i] < b[i]) return -1;
    if (a[i] > b[i]) return 1;
  }
  return Math.sign(a.length - b.length);
}
function __lt(a: any, b: any): any {
  const ca = __cmpCat(a); if (ca !== __cmpCat(b) || ca === "u") return undefined;
  return (ca === "v" || ca === "r") ? __veccmp(a, b) < 0 : a < b;
}
function __gt(a: any, b: any): any {
  const ca = __cmpCat(a); if (ca !== __cmpCat(b) || ca === "u") return undefined;
  return (ca === "v" || ca === "r") ? __veccmp(a, b) > 0 : a > b;
}
function __le(a: any, b: any): any {
  const ca = __cmpCat(a); if (ca !== __cmpCat(b) || ca === "u") return undefined;
  return (ca === "v" || ca === "r") ? __veccmp(a, b) <= 0 : a <= b;
}
function __ge(a: any, b: any): any {
  const ca = __cmpCat(a); if (ca !== __cmpCat(b) || ca === "u") return undefined;
  return (ca === "v" || ca === "r") ? __veccmp(a, b) >= 0 : a >= b;
}
function __isNum(x: any): boolean { return typeof x === "number"; }
function __isVec(x: any): boolean { return Array.isArray(x) && !(x as any).__isRange; }

function __add(a: any, b: any): any {
  if (__isNum(a) && __isNum(b)) return a + b;
  if (__isVec(a) && __isVec(b)) {
    let n = Math.min(a.length, b.length), r: any[] = [];
    for (let i = 0; i < n; i++) r.push(__add(a[i], b[i]));
    return r;
  }
  return undefined;
}
function __sub(a: any, b: any): any {
  if (__isNum(a) && __isNum(b)) return a - b;
  if (__isVec(a) && __isVec(b)) {
    let n = Math.min(a.length, b.length), r: any[] = [];
    for (let i = 0; i < n; i++) r.push(__sub(a[i], b[i]));
    return r;
  }
  return undefined;
}
function __mul(a: any, b: any): any {
  if (__isNum(a) && __isNum(b)) return a * b;
  if (__isNum(a) && __isVec(b)) return b.map((x: any): any => __mul(a, x)); // scalar * vector
  if (__isVec(a) && __isNum(b)) return a.map((x: any): any => __mul(x, b)); // vector * scalar
  if (__isVec(a) && __isVec(b)) {
    if (a.length > 0 && Array.isArray(a[0])) {
      if (b.length > 0 && Array.isArray(b[0])) {
        let res: any[] = [];                                               // matrix * matrix
        for (let i = 0; i < a.length; i++) {
          res[i] = [];
          for (let j = 0; j < b[0].length; j++) {
            let sum = 0;
            for (let k = 0; k < a[0].length; k++) sum += a[i][k] * b[k][j];
            res[i].push(sum);
          }
        }
        return res;
      }
      return a.map((row: any): any => __mul(row, b));                      // matrix * vector
    }
    if (b.length > 0 && Array.isArray(b[0])) {
      let res2: any[] = [];                                                // vector * matrix
      for (let j2 = 0; j2 < b[0].length; j2++) {
        let sum2 = 0;
        for (let k2 = 0; k2 < a.length; k2++) sum2 += a[k2] * b[k2][j2];
        res2.push(sum2);
      }
      return res2;
    }
    let sum3 = 0;                                                          // vector . vector
    for (let i3 = 0; i3 < Math.min(a.length, b.length); i3++) sum3 += a[i3] * b[i3];
    return sum3;
  }
  return undefined;
}
function __div(a: any, b: any): any {
  if (__isNum(a) && __isNum(b)) return a / b;
  if (__isVec(a) && __isNum(b)) return a.map((x: any): any => __div(x, b)); // vector / scalar
  if (__isNum(a) && __isVec(b)) return b.map((x: any): any => __div(a, x)); // scalar / vector
  if (__isVec(a) && __isVec(b)) {
    let n = Math.min(a.length, b.length), r: any[] = [];
    for (let i = 0; i < n; i++) r.push(__div(a[i], b[i]));
    return r;
  }
  return undefined;
}
function __mod(a: any, b: any): any {
  if (__isNum(a) && __isNum(b)) return a % b;
  return undefined;
}

// Bitwise operators
function __toI64(x: any): bigint | undefined {
  if (typeof x !== "number" || !isFinite(x)) return undefined;
  return BigInt.asIntN(64, BigInt(Math.trunc(x)));
}
function __band(a: any, b: any): any {
  const ia = __toI64(a), ib = __toI64(b);
  if (ia === undefined || ib === undefined) return undefined;
  return Number(BigInt.asIntN(64, ia & ib));
}
function __bor(a: any, b: any): any {
  const ia = __toI64(a), ib = __toI64(b);
  if (ia === undefined || ib === undefined) return undefined;
  return Number(BigInt.asIntN(64, ia | ib));
}
function __bnot(a: any): any {
  const ia = __toI64(a);
  if (ia === undefined) return undefined;
  return Number(BigInt.asIntN(64, ~ia));
}
function __shl(a: any, b: any): any {
  const ia = __toI64(a), ib = __toI64(b);
  if (ia === undefined || ib === undefined) return undefined;
  if (ib < 0n || ib >= 64n) return undefined; // shifts of 64+ bits are undef
  return Number(BigInt.asIntN(64, ia << ib));
}
function __shr(a: any, b: any): any {
  const ia = __toI64(a), ib = __toI64(b);
  if (ia === undefined || ib === undefined) return undefined;
  if (ib < 0n || ib >= 64n) return undefined;
  return Number(BigInt.asIntN(64, ia >> ib));
}
function __neg(a: any): any {
  if (__isNum(a)) return -a;
  if (__isVec(a)) return a.map(__neg);
  return undefined;
}
function __pos(a: any): any {
  if (__isNum(a)) return +a;
  if (__isVec(a)) return a.map(__pos);
  return undefined;
}


function __index(obj: any, idx: any): any {
  if (typeof idx !== "number" || obj == null) return undefined;
  if (typeof obj === "string" || Array.isArray(obj)) return obj[idx];
  return undefined;
}

// OpenSCAD version
function version_fn() { return [2019, 5, 0]; }
function version_num_fn() { return 20190500; }

// Constants
let PI = Math.PI;
let INF = Infinity;
let NAN = NaN;
let undef = undefined;
let _EPSILON = 1e-9;

// Special-variable context
const __ctx: Record<string, any> = {
  $fn: 0, $fa: 12, $fs: 2,
  $vpr: [55, 0, 25], $vpt: [0, 0, 0], $vpd: 140, $vpf: 22.5,
  $t: 0, $preview: false, $parent_modules: 0,
  $color: undefined, $idx: undefined,
};

function __withSpecials(overrides: Record<string, any>, body: () => any) {
  const saved: Record<string, any> = {};
  for (const k in overrides) { saved[k] = __ctx[k]; __ctx[k] = overrides[k]; }
  try { return body(); }
  finally { Object.assign(__ctx, saved); }
}

// Children stack for module calls
let __children_stack: any[] = [];
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

function __to_manifold_mat3(m: any) {
  if (!__is_finite_matrix4(m)) return undefined;
  return [m[0][0], m[1][0], 0, m[0][1], m[1][1], 0, m[0][3], m[1][3], 1];
}

function __safe_transform(shape: any, m: any) {
  const mm = __is2D(shape) ? __to_manifold_mat3(m) : __to_manifold_mat4(m);
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

// 2D helpers used by offset()/projection() fallbacks
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

function __rangeCount(start: any, step: any, end: any): number {
  if (step === 0 || Number.isNaN(start) || Number.isNaN(step) || Number.isNaN(end)) return 0;
  let n = (end - start) / step;
  if (!Number.isFinite(n) || n < 0) return 0;
  return Math.floor(n + 1e-10) + 1;
}

function __range(start: any, step: any, end: any) {
  let result: any[] = [];
  let n = __rangeCount(start, step, end);
  for (let i = 0; i < n; i++) result.push(start + i * step);
  Object.defineProperty(result, "__isRange", {
    value: true, enumerable: false, writable: true, configurable: true,
  });
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

const __TOL_FACTOR = Math.pow(2, -24);
function __maxBBoxDim3d(items: any[]): number {
  let lo = [Infinity, Infinity, Infinity];
  let hi = [-Infinity, -Infinity, -Infinity];
  for (const it of items) {
    if (!it || typeof it.boundingBox !== "function") continue;
    const bb = it.boundingBox();
    if (!bb || !bb.min || !bb.max) continue;
    for (let i = 0; i < 3; i++) {
      if (bb.min[i] < lo[i]!) lo[i] = bb.min[i];
      if (bb.max[i] > hi[i]!) hi[i] = bb.max[i];
    }
  }
  let maxDim = 0;
  for (let i = 0; i < 3; i++) {
    const d = hi[i]! - lo[i]!;
    if (Number.isFinite(d) && d > maxDim) maxDim = d;
  }
  return maxDim;
}
function __withTol3d(items: any[]): any[] {
  const maxDim = __maxBBoxDim3d(items);
  if (!(maxDim > 0)) return items;
  const tol = maxDim * __TOL_FACTOR;
  return items.map(it => {
    if (it && typeof it.setTolerance === "function" && typeof it.tolerance === "function") {
      return tol > it.tolerance() ? it.setTolerance(tol) : it;
    }
    return it;
  });
}

// OpenSCAD cannot mix 2D and 3D in a boolean op - it keeps the dimension of the first child and ignores (with a warning) any children of the other dimension.
function __sameDim(items: any[], ref2D: boolean): any[] {
  return items.filter(x => __is2D(x) === ref2D);
}

// Boolean ops: use CrossSection for 2D, Manifold for 3D
function __union2d3d(items: any[]) {
  const valid = items.filter(x => !__isEmpty(x));
  if (valid.length === 0) return Manifold.union([]);
  const is2D = __is2D(valid[0]);
  const same = __sameDim(valid, is2D);
  return is2D ? CrossSection.union(same) : Manifold.union(__withTol3d(same));
}
function __difference2d3d(first: any, rest: any[]) {
  if (__isEmpty(first)) return first;
  const is2D = __is2D(first);
  const validRest = __sameDim(rest.filter(x => !__isEmpty(x)), is2D);
  if (validRest.length === 0) return first;
  if (is2D) return CrossSection.difference([first, ...validRest]);
  const [tf, ...tr] = __withTol3d([first, ...validRest]);
  return tr.length === 1 ? tf.subtract(tr[0]) : tf.subtract(Manifold.union(tr));
}
function __intersection2d3d(items: any[]) {
  if (items.length === 0) return Manifold.union([]);
  const valid = items.filter(x => !__isEmpty(x));
  if (valid.length < items.length) {
    const firstValid2D = valid.find(__is2D);
    return firstValid2D ? CrossSection.union([]) : Manifold.union([]);
  }
  const is2D = __is2D(valid[0]);
  const same = __sameDim(valid, is2D);
  // Intersecting across dimensions (e.g. a 3D solid with a 2D shape) has no common volume, so OpenSCAD yields an empty result.
  if (same.length < valid.length) return is2D ? CrossSection.union([]) : Manifold.union([]);
  return is2D ? CrossSection.intersection(same) : Manifold.intersection(__withTol3d(same));
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

// Returns `x` when it is a finite number, otherwise `dflt`
function __finiteOr(x: any, dflt: number): number {
  return (typeof x === "number" && Number.isFinite(x)) ? x : dflt;
}

function __normalizeScale(scale: number | number[] | undefined): [number, number] | undefined {
  if (scale === undefined || scale === null) return undefined;
  // A range (e.g. [1:3]) is not a valid scale - OpenSCAD falls back to no scaling
  if (Array.isArray(scale) && (scale as any).__isRange) return undefined;
  // Each component defaults to 1 (no scaling) when it isn't a finite number
  const num = (v: any) => __finiteOr(v, 1);
  const [sx, sy] = Array.isArray(scale)
    ? [num(scale[0]), num(scale[1])]
    : [num(scale), num(scale)];
  return [Math.max(0, sx), Math.max(0, sy)];
}

function __getHelixLength(rMax: number, height: number, twist: number, scaleX: number, scaleY: number): number {
  const steps = 100;
  let len = 0;
  const twistRad = (twist * Math.PI) / 180;
  const dt = 1 / steps;
  for (let i = 0; i < steps; i++) {
    const t = (i + 0.5) * dt;
    const sx = 1 + (scaleX - 1) * t;
    const sy = 1 + (scaleY - 1) * t;
    const sin = Math.sin(t * twistRad);
    const cos = Math.cos(t * twistRad);
    
    const vx = -rMax * sx * twistRad * sin;
    const vy = rMax * sy * twistRad * cos;
    const vz = height;
    
    const speed = Math.sqrt(vx * vx + vy * vy + vz * vz);
    len += speed * dt;
  }
  return len;
}

function __computeExtrudeDivisions(shape: any, height: number, options: { twist?: number; scale?: number | number[] | undefined; fn?: number; fa?: number; fs?: number; fe?: number; slices?: number; }): number {
  // Explicit, finite slices count - an invalid value (undef, inf, nan, string, bool, range) is ignored and the count is auto-computed
  if (typeof options.slices === "number" && Number.isFinite(options.slices)) {
    return Math.max(1, options.slices);
  }

  const twist = Math.abs(__finiteOr(options.twist, 0));
  const fn = options.fn ?? 0;
  const fa = options.fa ?? 12;
  const fs = options.fs ?? 2;
  const fe = options.fe ?? 0;

  const normScale = __normalizeScale(options.scale);
  const sx = normScale?.[0] ?? 1;
  const sy = normScale?.[1] ?? 1;

  let rMax = 0;
  try {
    const polys = shape.toPolygons();
    for (const poly of polys) {
      for (const p of poly) {
        const dist = Math.sqrt(p[0] * p[0] + p[1] * p[1]);
        if (dist > rMax) rMax = dist;
      }
    }
  } catch {
    rMax = 10;
  }

  if (twist === 0) {
    if (sx !== sy) {
      const maxDeltaSqr = rMax * rMax * Math.max(Math.abs(sx - 1), Math.abs(sy - 1)) ** 2;
      const diagSlices = Math.ceil(Math.sqrt(maxDeltaSqr + height * height) / fs);
      return fn > 0 ? fn : diagSlices;
    }
    return 1;
  }

  const minSlices = Math.max(Math.ceil(twist / 120), 1);
  if (fn > 0) {
    return Math.max(Math.ceil((twist / 360) * fn), minSlices);
  }

  if (fe > 0) {
    const twistRad = (twist * Math.PI) / 180;
    const circumference = rMax * twistRad;
    return Math.max(Math.ceil(circumference / fe), minSlices);
  }

  const helixLen = __getHelixLength(rMax, height, twist, sx, sy);
  const fsSlices = Math.ceil(helixLen / fs);
  const faSlices = Math.ceil(twist / fa);

  return Math.max(Math.min(faSlices, fsSlices), minSlices);
}

// Interpolate points along v0->v1, excluding the endpoint v1
function __addSegmentedEdge(out: [number, number][], v0: [number, number], v1: [number, number], segs: number) {
  for (let j = 0; j < segs; j++) {
    const t = j / segs;
    out.push([(1 - t) * v0[0] + t * v1[0], (1 - t) * v0[1] + t * v1[1]]);
  }
}

function __maxEdgeLen(v0: [number, number], v1: [number, number], twist: number, sx: number, sy: number, slices: number): number {
  if (sx === sy) {
    return Math.hypot(v1[0] - v0[0], v1[1] - v0[1]) * Math.max(sx, 1);
  }
  let maxLen = 0;
  for (let j = 0; j <= slices; j++) {
    const t = j / slices;
    const scx = 1 + (sx - 1) * t, scy = 1 + (sy - 1) * t;
    const ang = -twist * t * Math.PI / 180;
    const ca = Math.cos(ang), sa = Math.sin(ang);
    const x0 = (v0[0] * ca - v0[1] * sa) * scx, y0 = (v0[0] * sa + v0[1] * ca) * scy;
    const x1 = (v1[0] * ca - v1[1] * sa) * scx, y1 = (v1[0] * sa + v1[1] * ca) * scy;
    const len = Math.hypot(x1 - x0, y1 - y0);
    if (len > maxLen) maxLen = len;
  }
  return maxLen;
}

// Split each edge into ceil(maxEdgeLen / fs) segments.
function __splitOutlineByFs(o: [number, number][], twist: number, sx: number, sy: number, fs: number, slices: number): [number, number][] {
  const n = o.length;
  const out: [number, number][] = [];
  for (let i = 1; i <= n; i++) {
    const v0 = o[i - 1]!, v1 = o[i % n]!;
    const segs = Math.max(1, Math.ceil(__maxEdgeLen(v0, v1, twist, sx, sy, slices) / fs));
    __addSegmentedEdge(out, v0, v1, segs);
  }
  return out;
}

function __splitOutlineByFn(o: [number, number][], twist: number, sx: number, sy: number, target: number, slices: number): [number, number][] {
  const n = o.length;
  const maxLen: number[] = [];
  for (let i = 1; i <= n; i++) {
    maxLen.push(__maxEdgeLen(o[i - 1]!, o[i % n]!, twist, sx, sy, slices));
  }
  const segCount = new Array(n).fill(1);
  const metric = (k: number) => maxLen[k]! / (segCount[k] + 0.5);
  let segTotal = n;
  while (segTotal < target) {
    let top = 0;
    for (let k = 1; k < n; k++) if (metric(k) > metric(top)) top = k;
    const topMetric = metric(top);
    const group: number[] = [];
    for (let k = 0; k < n; k++) {
      const mk = metric(k);
      if (Math.min(mk, topMetric) / Math.max(mk, topMetric) >= 0.999) group.push(k);
    }
    if (segTotal + group.length > target) break;
    for (const g of group) { segCount[g]++; segTotal++; }
  }
  const out: [number, number][] = [];
  for (let i = 1; i <= n; i++) __addSegmentedEdge(out, o[i - 1]!, o[i % n]!, segCount[i - 1]);
  return out;
}

// Refine a single outline for a non-linear (twisted/non-uniformly-scaled) extrude
function __splitOutline(o: [number, number][], twist: number, sx: number, sy: number, slices: number,fn: number, fa: number, fs: number, segments: number): [number, number][] {
  if (segments > 0 || fn > 0.0) {
    const minVerts = segments > 0 ? segments : Math.max(fn, 3);
    return o.length >= minVerts ? o : __splitOutlineByFn(o, twist, sx, sy, minVerts, slices);
  }
  const faSegs = Math.ceil(360.0 / fa);
  if (o.length >= faSegs) return o;
  const fsOutline = __splitOutlineByFs(o, twist, sx, sy, fs, slices);
  return fsOutline.length >= faSegs
    ? __splitOutlineByFn(o, twist, sx, sy, faSegs, slices)
    : fsOutline;
}

function __extrudeTwisted(shape: any, height: number, twistDeg: number, slices: number, scaleVec: [number, number] | undefined, center: boolean | undefined, fn: number, fa: number, fs: number): any {
  const rawPolys: [number, number][][] = shape.toPolygons();
  if (!rawPolys.length) return Manifold.union([]);

  const sx = scaleVec ? scaleVec[0] : 1;
  const sy = scaleVec ? scaleVec[1] : 1;

  // Refine each outline exactly as OpenSCAD does, so the cross-section vertex count matches OpenSCAD's twisted mesh.
  const polys = rawPolys.map(c => __splitOutline(c, twistDeg, sx, sy, slices, fn, fa, fs, 0));

  // Flatten all contours, keeping per-contour boundary order for the walls and a position -> flat-index map for recovering the cap triangulation
  const flat: [number, number][] = [];
  const contours: { off: number; len: number }[] = [];
  const keyMap = new Map<string, number>();
  const key = (x: number, y: number) => `${Math.round(x * 1000)},${Math.round(y * 1000)}`;
  for (const c of polys) {
    const off = flat.length;
    for (const p of c) {
      keyMap.set(key(p[0], p[1]), flat.length);
      flat.push([p[0], p[1]]);
    }
    contours.push({ off, len: c.length });
  }

  // Recover cap triangles (as flat indices) from the bottom face of a plain, untwisted extrude of the refined outline.
  let capTris: [number, number, number][] | null = [];
  try {
    const capShape = CrossSection.ofPolygons(polys);
    const pm = capShape.extrude(1).getMesh();
    const np = pm.numProp, vp = pm.vertProperties, tv = pm.triVerts;
    for (let i = 0; i < tv.length && capTris; i += 3) {
      const ia = tv[i]!, ib = tv[i + 1]!, ic = tv[i + 2]!;
      if (vp[ia * np + 2]! < 1e-4 && vp[ib * np + 2]! < 1e-4 && vp[ic * np + 2]! < 1e-4) {
        const a = keyMap.get(key(vp[ia * np]!, vp[ia * np + 1]!));
        const b = keyMap.get(key(vp[ib * np]!, vp[ib * np + 1]!));
        const cc = keyMap.get(key(vp[ic * np]!, vp[ic * np + 1]!));
        if (a === undefined || b === undefined || cc === undefined) { capTris = null; break; }
        capTris.push([a, b, cc]);
      }
    }
  } catch { capTris = null; }
  if (!capTris || !capTris.length) {
    // Fall back to Manifold's own twist extrude if the cap mapping failed.
    return shape.extrude(height, Math.max(0, slices - 1), Math.abs(twistDeg), scaleVec, center);
  }

  const S = Math.max(1, Math.round(slices));
  const topZero = scaleVec !== undefined && sx === 0 && sy === 0;
  const z0 = center ? -height / 2 : 0;
  const M = flat.length;

  // Rotate by -twist*t (positive twist -> clockwise), then apply the interpolated scale, matching OpenSCAD's Scaling * Rotation order.
  const verts = new Float32Array((S + 1) * M * 3);
  for (let i = 0; i <= S; i++) {
    const t = i / S;
    const ang = -twistDeg * t * Math.PI / 180;
    const ca = Math.cos(ang), sa = Math.sin(ang);
    const isx = 1 + (sx - 1) * t;
    const isy = 1 + (sy - 1) * t;
    const z = z0 + height * t;
    for (let k = 0; k < M; k++) {
      const x = flat[k]![0], y = flat[k]![1];
      const base = (i * M + k) * 3;
      verts[base] = (x * ca - y * sa) * isx;
      verts[base + 1] = (x * sa + y * ca) * isy;
      verts[base + 2] = z;
    }
  }

  const tris: number[] = [];
  const idx = (i: number, k: number) => i * M + k;
  const dist2 = (p: number, q: number) => {
    const pb = p * 3, qb = q * 3;
    const dx = verts[pb]! - verts[qb]!;
    const dy = verts[pb + 1]! - verts[qb + 1]!;
    const dz = verts[pb + 2]! - verts[qb + 2]!;
    return dx * dx + dy * dy + dz * dz;
  };

  // Bottom cap (slice 0) keeps the plain-extrude winding (faces -z); top cap (slice S) is reversed to face +z. Skip the top cap if it collapses to a point.
  for (const [a, b, c] of capTris) tris.push(idx(0, a), idx(0, b), idx(0, c));
  if (!topZero) {
    for (const [a, b, c] of capTris) tris.push(idx(S, a), idx(S, c), idx(S, b));
  }

  // Walls: triangulate each quad p1->c1->c2->p2 along its shorter diagonal
  for (const { off, len } of contours) {
    for (let e = 0; e < len; e++) {
      const a = off + e;
      const b = off + ((e + 1) % len);
      for (let i = 0; i < S; i++) {
        const p1 = idx(i, a), c1 = idx(i, b);
        const c2 = idx(i + 1, b), p2 = idx(i + 1, a);
        if (topZero && i === S - 1) {
          // Top collapsed to a point: fan to a single shared apex vertex (all slice-S vertices coincide at the origin, so pick one canonical index)
          tris.push(p1, c1, idx(S, 0));
        } else if (dist2(p1, c2) <= dist2(c1, p2)) {
          tris.push(p1, c1, c2, p1, c2, p2);
        } else {
          tris.push(p1, c1, p2, c1, c2, p2);
        }
      }
    }
  }

  return new Manifold({ vertProperties: verts, triVerts: new Uint32Array(tris), numProp: 3 } as any);
}

function __extrude(shape: any, height = 100, options: {twist?: number; scale?: number | number[] | undefined; center?: boolean; fn?: number; fa?: number; fs?: number; fe?: number; slices?: number;} = {}) {
  if (__isEmpty(shape)) {
    return Manifold.union([]);
  }

  if (!__is2D(shape)) {
    return shape;
  }

  // An invalid height defaults to OpenSCAD's linear_extrude default of 100
  height = __finiteOr(height, 100);
  // An invalid twist means "no twist"
  const twist = __finiteOr(options.twist, 0);
  const normScale = __normalizeScale(options.scale);

  const nDivisions = __computeExtrudeDivisions(shape, height, { ...options, scale: normScale });

  if (twist !== 0) {
    return __extrudeTwisted(
      shape, height, twist, nDivisions, normScale, options.center,
      options.fn ?? 0, options.fa ?? 12, options.fs ?? 2,
    );
  }

  return shape.extrude(
    height,
    Math.max(0, nDivisions - 1),
    undefined,
    normScale,
    options.center
  );
}

function __revolve(shape: any, fn = 0, fa = 12, fs = 2, angle = 360) {
  if (__isEmpty(shape)) {
    return Manifold.union([]);
  }
  if (__is2D(shape)) {
    let num_sections: number;
    const absAngle = Math.abs(angle);

    if (fn > 0) {
      num_sections = Math.max(1, Math.ceil(Math.max(fn, 3) * absAngle / 360));
    } else {
      const bounds = shape.bounds();
      const r = Math.max(Math.abs(bounds.max[0]), Math.abs(bounds.min[0]));
      const N_fa = 360 / fa;
      const N_fs = 2 * Math.PI * r / fs;
      num_sections = Math.max(1, Math.ceil(Math.max(Math.min(N_fa, N_fs), 5) * absAngle / 360));
    }

    const revolved = shape.revolve(num_sections, absAngle);
    return angle < 0 ? revolved.mirror([0, 1, 0]) : revolved;
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

function __fontSpecToFilename(fontSpec: string): string {
  const cleaned = fontSpec.replace(/"/g, "").trim();
  const parts = cleaned.split(":");
  const family = (parts[0] || "Liberation Sans").trim().replace(/\s+/g, "");

  let style = "Regular";
  for (let i = 1; i < parts.length; i++) {
    const part = parts[i]!.trim();
    const match = part.match(/^style\s*=\s*(.+)$/i);
    if (match) {
      style = match[1]!.trim().replace(/\s+/g, "");
      break;
    }
  }

  return `${family}-${style}`;
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

    const spaceWidth = (spaceGlyph?.advanceWidth ?? font.unitsPerEm * 0.25) * scale;

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

function __text(text: string, size: number = 10, font: string, halign: string = "left", valign: string = "baseline", spacing: number = 1, direction: string = "ltr", fn: number = 0, fontBase64Data: string | Record<string, string> | undefined = undefined): any {
  if (!text || text.length === 0) return CrossSection.square([0.001, 0.001], false);

  void fn;

  const dir = (direction || "ltr").toLowerCase();
  const chars = dir === "rtl" ? Array.from(text).reverse() : Array.from(text);

  let contours: [number, number][][] | undefined;

  contours = __canvasTextContours(chars, size, spacing, dir, font);

  if (!contours && fontBase64Data) {
    let base64: string | undefined;
    if (typeof fontBase64Data === "string") {
      base64 = fontBase64Data;
    } else if (typeof fontBase64Data === "object" && fontBase64Data !== null) {
      const filename = __fontSpecToFilename(font);
      base64 = fontBase64Data[filename];
      if (!base64) {
        const keys = Object.keys(fontBase64Data);
        if (keys.length > 0) {
          base64 = fontBase64Data[keys[0]!];
        }
      }
    }
    if (base64) {
      contours = __opentypeTextContours(chars, size, spacing, dir, base64, fn);
    }
  }
  if (!contours || contours.length === 0) {
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
  else                          dx = 0;

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
  // A non-finite (or non-positive) size produces no geometry instead of crashing
  if (!Number.isFinite(radius) || radius <= 0) {
    return Manifold.union([]);
  }
  let N: number;
  if (fn > 0) {
    // OpenSCAD takes the explicit-$fn path whenever $fn > 0, clamping to a minimum of 3 - a non-finite $fn is clamped to that minimum
    N = Number.isFinite(fn) ? Math.max(3, Math.ceil(fn)) : 3;
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

function __radius(dSpec: any, rSpec: any, dGen: any, rGen: any, dflt: any) {
  const def = (x: any) => x !== undefined && x !== null && !(typeof x === "number" && Number.isNaN(x));
  if (def(dSpec)) return dSpec / 2;
  if (def(rSpec)) return rSpec;
  if (def(dGen)) return dGen / 2;
  if (def(rGen)) return rGen;
  return dflt;
}

function __cylinder(height: number, radiusLow: number, radiusHigh = -1.0, fn = 0, center = false, fa = 12, fs = 2) {
  // Non-finite dimensions produce no geometry
  if (!Number.isFinite(height) || !Number.isFinite(radiusLow) ||
      (radiusHigh >= 0 && !Number.isFinite(radiusHigh))) {
    return Manifold.union([]);
  }
  let segs = fn;
  if (segs > 0) {
    // OpenSCAD clamps a non-finite $fn to the minimum of 3 fragments.
    if (!Number.isFinite(segs)) segs = 3;
  } else {
    const r = Math.max(radiusLow, radiusHigh < 0 ? radiusLow : radiusHigh);
    const N_fa = 360 / fa;
    const N_fs = (2 * Math.PI * r) / fs;
    segs = Math.ceil(Math.max(Math.min(N_fa, N_fs), 5));
  }
  return Manifold.cylinder(height, radiusLow, radiusHigh, segs, center);
}

function __circle(radius: number, fn = 0, fa = 12, fs = 2) {
  // Match OpenSCAD: a non-finite (or non-positive) radius produces no geometry.
  if (!Number.isFinite(radius) || radius <= 0) {
    return CrossSection.square(0);
  }
  let N: number;
  if (fn > 0) {
    // OpenSCAD clamps a non-finite $fn to the minimum of 3 fragments.
    N = Number.isFinite(fn) ? Math.max(3, Math.ceil(fn)) : 3;
  } else {
    const N_fa = 360 / fa;
    const N_fs = (2 * Math.PI * radius) / fs;
    N = Math.ceil(Math.max(Math.min(N_fa, N_fs), 5));
  }
  return CrossSection.circle(radius, N);
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

  // A point with a non-finite coordinate yields no geometry
  if (points.some((p: any) => Array.isArray(p) && p.some((c: any) => !Number.isFinite(Number(c))))) {
    return CrossSection.square(0);
  }

  if (paths === undefined || paths === null) {
    const ccwPoints = __forceWinding([...points], true);
    return CrossSection.ofPolygons([ccwPoints]);
  }

  if (Array.isArray(paths) && paths.length > 0 && !Array.isArray(paths[0])) {
    const contour = paths.map((idx: any) => points[Number(idx) || 0]).filter(Boolean);
    const ccwContour = __forceWinding(contour, true);
    return CrossSection.ofPolygons([ccwContour]);
  }

  if (Array.isArray(paths)) {
    const contours = paths.map((path: any) => {
      let contour: any[] = [];
      if (Array.isArray(path)) {
        contour = path.map((idx: any) => points[Number(idx) || 0]).filter(Boolean);
      } else if (typeof path === "number") {
        contour = [points[path]];
      }
      return contour;
    }).filter((c: any) => c.length > 0);
    // Use EvenOdd fill rule - OpenSCAD polygon() does not assume outer/hole winding
    return CrossSection.ofPolygons(contours, 'EvenOdd');
  }

  const ccwPoints = __forceWinding([...points], true);
  return CrossSection.ofPolygons([ccwPoints]);
}

function __polyhedron(points: any, faces: any) {
  // A point with a non-finite coordinate yields no geometry
  if (Array.isArray(points) &&
      points.some((p: any) => Array.isArray(p) && p.some((c: any) => !Number.isFinite(Number(c))))) {
    return Manifold.union([]);
  }
  const verts: number[] = [];
  if (Array.isArray(points)) {
    for (const p of points) {
      verts.push(Number(p?.[0]) || 0, Number(p?.[1]) || 0, Number(p?.[2]) || 0);
    }
  }
  const tris: number[] = [];
  if (Array.isArray(faces)) {
    for (const face of faces) {
      if (!Array.isArray(face) || face.length < 3) continue;
      for (let i = 1; i + 1 < face.length; i++) {
        tris.push(Number(face[i + 1]), Number(face[i]), Number(face[0]));
      }
    }
  }
  const mesh = new wasm.Mesh({
    numProp: 3,
    vertProperties: new Float32Array(verts),
    triVerts: new Uint32Array(tris),
  });
  // OpenSCAD accepts "polyhedron soup" where coincident vertices are duplicated per-face (so edges aren't shared)
  // Manifold requires shared edges, so weld coincident vertices along open edges before constructing the solid
  mesh.merge();
  return new Manifold(mesh);
}

function __parse_color_for_scope(c: any, alpha: any): any {
  const base = __parse_color_value(c);
  if (!base) return undefined;
  const a = (alpha !== undefined && alpha !== null && Number.isFinite(Number(alpha)))
    ? Number(alpha) : base[3];
  return [base[0], base[1], base[2], a];
}


async function gridFromImage(dataUrl: string): Promise<{ width: number; height: number; Z: (x: number, y: number) => number }> {
  const { width, height, data } = await decodeImageToPixels(dataUrl);
  const Z = (x: number, y: number): number => {
    const i = (y * width + x) * 4;
    const gray = 0.2126 * data![i]! + 0.7152 * data![i + 1]! + 0.0722 * data![i + 2]!;
    return (gray / 255) * 100;
  };
  return { width, height, Z };
}

function gridFromText(text: string): { width: number; height: number; Z: (x: number, y: number) => number } {
  const rows: number[][] = [];
  for (const line of text.split(/\r?\n/)) {
    const t = line.trim();
    // skip blanks, '#' (OpenSCAD/Octave) and '%' (Matlab) comment lines
    if (t === "" || t.startsWith("#") || t.startsWith("%")) continue;
    const vals = t.split(/\s+/).map(Number).filter(v => Number.isFinite(v));
    if (vals.length) rows.push(vals);
  }
  if (rows.length === 0) throw new Error("__surface: empty data file");
  const height = rows.length;
  const width = Math.max(...rows.map(r => r.length));
  // OpenSCAD reads row 0 as the first data row; row index -> Y, column index -> X. Value used directly as Z (no normalization).
  const Z = (x: number, y: number): number => rows[y]?.[x] ?? 0;
  return { width, height, Z };
}

async function __surface(source: string, opts: { center?: boolean; kind?: "image" | "text"; fn?: number; fa?: number; fs?: number } = {}
) {
  const { center = false, kind = "image" } = opts;
  const grid = kind === "text" ? gridFromText(source) : await gridFromImage(source);
  return buildSurfaceMesh(grid, center);
}

function buildSurfaceMesh({ width, height, Z }: { width: number; height: number; Z: (x: number, y: number) => number }, center: boolean) {
  const ox = center ? -(width - 1) / 2 : 0;
  const oy = center ? -(height - 1) / 2 : 0;

  let zmin = Infinity;
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++)
      if (Z(x, y) < zmin) zmin = Z(x, y);
  const zFloor = zmin - 1;

  const numTop = width * height;
  const numQuads = (width - 1) * (height - 1);
  const vertProps: number[] = [];
  const tris: number[] = [];

  const topIdx = (x: number, y: number) => y * width + x;
  const centerIdx = (x: number, y: number) => numTop + y * (width - 1) + x;
  const botIdx = (x: number, y: number) => numTop + numQuads + y * width + x;

  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++)
      vertProps.push(x + ox, y + oy, Z(x, y));

  for (let y = 0; y < height - 1; y++)
    for (let x = 0; x < width - 1; x++) {
      const zc = (Z(x, y) + Z(x + 1, y) + Z(x, y + 1) + Z(x + 1, y + 1)) / 4;
      vertProps.push(x + 0.5 + ox, y + 0.5 + oy, zc);
    }

  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++)
      vertProps.push(x + ox, y + oy, zFloor);

  for (let y = 0; y < height - 1; y++)
    for (let x = 0; x < width - 1; x++) {
      const a = topIdx(x, y), b = topIdx(x + 1, y);
      const c = topIdx(x, y + 1), d = topIdx(x + 1, y + 1);
      const ctr = centerIdx(x, y);
      tris.push(a, b, ctr, b, d, ctr, d, c, ctr, c, a, ctr);
    }

  for (let y = 0; y < height - 1; y++)
    for (let x = 0; x < width - 1; x++) {
      const a = botIdx(x, y), b = botIdx(x + 1, y);
      const c = botIdx(x, y + 1), d = botIdx(x + 1, y + 1);
      tris.push(a, c, b, b, c, d);
    }

  for (let x = 0; x < width - 1; x++)
    tris.push(topIdx(x, 0), botIdx(x, 0), topIdx(x + 1, 0),
              topIdx(x + 1, 0), botIdx(x, 0), botIdx(x + 1, 0));
  const yb = height - 1;
  for (let x = 0; x < width - 1; x++)
    tris.push(topIdx(x, yb), topIdx(x + 1, yb), botIdx(x, yb),
              topIdx(x + 1, yb), botIdx(x + 1, yb), botIdx(x, yb));
  for (let y = 0; y < height - 1; y++)
    tris.push(topIdx(0, y), topIdx(0, y + 1), botIdx(0, y),
              topIdx(0, y + 1), botIdx(0, y + 1), botIdx(0, y));
  const xr = width - 1;
  for (let y = 0; y < height - 1; y++)
    tris.push(topIdx(xr, y), botIdx(xr, y), topIdx(xr, y + 1),
              topIdx(xr, y + 1), botIdx(xr, y), botIdx(xr, y + 1));

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

function pow_fn(base: any, exp: any) { return Math.pow(base, exp); }

// Export all runtime symbols for compiled code
export {
  Manifold, CrossSection, wasm,
  __sphere, __cylinder, __circle, __radius, __rotate, __polygon, __polyhedron, __translate, __scale, __mirror,
  is_undef_fn, is_bool_fn, is_num_fn, is_string_fn, is_list_fn, is_function_fn, is_object_fn,
  sin_fn, cos_fn, tan_fn, asin_fn, acos_fn, atan_fn, atan2_fn,
  abs_fn, sign_fn, floor_fn, ceil_fn, round_fn, sqrt_fn, exp_fn, ln_fn, log_fn, pow_fn,
  min_fn, max_fn, norm_fn, cross_fn,
  len_fn, str_fn, chr_fn, ord_fn, concat_fn, search_fn, lookup_fn,
  openscad_assert_fn, __truthy, __eq, __lt, __gt, __le, __ge, __add, __sub, __mul, __div, __mod, __band, __bor, __shl, __shr, __bnot, __neg, __pos, __index,
  version_fn, version_num_fn,
  PI, INF, NAN, undef, _EPSILON,
  __ctx, __withSpecials,
  __children_stack, __with_children, parent_module_fn,
  __is_finite_matrix4, __to_manifold_mat4, __safe_transform, __identity4,
  __safe_offset2d, __safe_project3d,
  __apply_color,
  __flat_map_iter, __range, __rangeCount, __is2D, __union2d3d, __difference2d3d, __intersection2d3d, __hull2d3d, __minkowski2d3d,
  __extrude, __revolve,
  __text, __parse_color_for_scope, __surface
};
