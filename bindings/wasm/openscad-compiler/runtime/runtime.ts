/**
 * OpenSCAD runtime for Manifold.js compiled output.
 * Loaded via import from compiled files. Initializes manifold-3d and provides
 * OpenSCAD built-ins and helpers. Exports all symbols for use by compiled code.
 */
import {createCanvas, loadImage} from 'canvas';
import Module from 'manifold-3d';
import * as opentype from 'opentype.js';

const opentypeParse = opentype.parse || (opentype as any).default?.parse;

declare const document: any;
declare const OffscreenCanvas: any;

const wasm = await Module();
wasm.setup();
const {Manifold, CrossSection} = wasm;

function is_undef_fn(x: any) {
  return arguments.length === 1 ? (x === undefined || x === null) : undefined;
}
function is_bool_fn(x: any) {
  return arguments.length === 1 ? (typeof x === 'boolean') : undefined;
}
function is_num_fn(x: any) {
  return arguments.length === 1 ? (typeof x === 'number' && !Number.isNaN(x)) :
                                  undefined;
}
function is_string_fn(x: any) {
  return arguments.length === 1 ? (typeof x === 'string') : undefined;
}
function is_list_fn(x: any) {
  return arguments.length === 1 ? Array.isArray(x) : undefined;
}
function is_function_fn(x: any) {
  return arguments.length === 1 ? (typeof x === 'function') : undefined;
}

// unknown/undefined function or disabled experimental builting function calls
function __unknown_fn(_name: string, ..._args: any[]): undefined {
  return undefined;
}

// Trig from OpenSCAD's degree_trig
const M_DEG2RAD = 0.017453292519943295769;  // PI/180
const M_RAD2DEG = 57.2957795130823208767;   // 180/PI
const M_SQRT3 = 1.73205080756887719318;     // sqrt(3)
const M_SQRT3_4 = 0.86602540378443859659;   // sqrt(3)/2
const M_SQRT1_3 = 0.57735026918962573106;   // sqrt(3)/3
const M_SQRT1_2 = 0.70710678118654752440;   // sqrt(1/2)
const TRIG_HUGE_VAL = (1 << 26) * 360.0 * (1 << 26);

function deg2rad(x: number) {
  return x * M_DEG2RAD;
}
function rad2deg(x: number) {
  return x * M_RAD2DEG;
}
function cround(x: number) {
  return x < 0 ? -Math.round(-x) : Math.round(x);
}

function sin_fn(x: any) {
  if (x < 360.0 && x >= 0.0) {
    // already in range
  } else if (x < TRIG_HUGE_VAL && x > -TRIG_HUGE_VAL) {
    x -= 360.0 * Math.floor(x / 360.0);
  } else {
    return NaN;
  }
  const oppose = x >= 180.0;
  if (oppose) x -= 180.0;
  if (x > 90.0) x = 180.0 - x;
  if (x < 45.0) {
    if (x === 30.0)
      x = 0.5;
    else
      x = Math.sin(deg2rad(x));
  } else if (x === 45.0) {
    x = M_SQRT1_2;
  } else if (x === 60.0) {
    x = M_SQRT3_4;
  } else {
    x = Math.cos(deg2rad(90.0 - x));
  }
  return oppose ? -x : x;
}

function cos_fn(x: any) {
  if (x < 360.0 && x >= 0.0) {
    // already in range
  } else if (x < TRIG_HUGE_VAL && x > -TRIG_HUGE_VAL) {
    x -= 360.0 * Math.floor(x / 360.0);
  } else {
    return NaN;
  }
  let oppose = x >= 180.0;
  if (oppose) x -= 180.0;
  if (x > 90.0) {
    x = 180.0 - x;
    oppose = !oppose;
  }
  if (x > 45.0) {
    if (x === 60.0)
      x = 0.5;
    else
      x = Math.sin(deg2rad(90.0 - x));
  } else if (x === 45.0) {
    x = M_SQRT1_2;
  } else if (x === 30.0) {
    x = M_SQRT3_4;
  } else {
    x = Math.cos(deg2rad(x));
  }
  return oppose ? -x : x;
}

function tan_fn(x: any) {
  const cycles = Math.floor(x / 180.0);
  if (x < 180.0 && x >= 0.0) {
    // already in range
  } else if (x < TRIG_HUGE_VAL && x > -TRIG_HUGE_VAL) {
    x -= 180.0 * cycles;
  } else {
    return NaN;
  }
  const oppose = x > 90.0;
  if (oppose) x = 180.0 - x;
  if (x === 0.0) {
    x = (cycles % 2) === 0 ? 0.0 : -0.0;
  } else if (x === 30.0) {
    x = M_SQRT1_3;
  } else if (x === 45.0) {
    x = 1.0;
  } else if (x === 60.0) {
    x = M_SQRT3;
  } else if (x === 90.0) {
    x = (cycles % 2) === 0 ? Infinity : -Infinity;
  } else {
    x = Math.tan(deg2rad(x));
  }
  return oppose ? -x : x;
}

function asin_fn(x: any) {
  const degs = rad2deg(Math.asin(x));
  const whole = cround(degs);
  if (sin_fn(whole) === x) return whole;
  return degs;
}

function acos_fn(x: any) {
  const degs = rad2deg(Math.acos(x));
  const whole = cround(degs);
  if (cos_fn(whole) === x) return whole;
  return degs;
}

function atan_fn(x: any) {
  const degs = rad2deg(Math.atan(x));
  const whole = cround(degs);
  if (tan_fn(whole) === x) return whole;
  return degs;
}

function atan2_fn(y: any, x: any) {
  const degs = rad2deg(Math.atan2(y, x));
  const whole = cround(degs);
  if (Math.abs(degs - whole) < 3.0e-14) return whole;
  return degs;
}

// Math (OpenSCAD built-ins)
let abs_fn = Math.abs;
let sign_fn = Math.sign;
let floor_fn = Math.floor;
let ceil_fn = Math.ceil;
let round_fn = Math.round;
let sqrt_fn = Math.sqrt;
let exp_fn = Math.exp;
function ln_fn(x: any) {
  return Math.log(x);
}
function log_fn(x: any) {
  return Math.log(x);
}

function __minmax(reduce: (a: number, b: number) => number, a: any[]) {
  const vals = a.length === 1 && Array.isArray(a[0]) ? a[0] : a;
  if (vals.length === 0) return undefined;
  let acc: number|undefined = undefined;
  for (const v of vals) {
    if (typeof v !== 'number') return undefined;
    acc = acc === undefined ? v : reduce(acc, v);
  }
  return acc;
}

function min_fn(...a: any[]) {
  return __minmax(Math.min, a);
}
function max_fn(...a: any[]) {
  return __minmax(Math.max, a);
}

function norm_fn(...args: any[]) {
  if (args.length !== 1 || !__isVec(args[0])) return undefined;
  let sum = 0;
  for (const x of args[0]) {
    if (typeof x !== 'number') return undefined;
    sum += x * x;
  }
  return Math.sqrt(sum);
}

function cross_fn(a: any, b: any) {
  if (!Array.isArray(a) || !Array.isArray(b)) return undefined;
  if ((a.length !== 2 || b.length !== 2) &&
      (a.length !== 3 || b.length !== 3)) {
    console.warn('WARNING: Invalid arguments to cross()');
    return undefined;
  }
  // rejects vectors whose elements aren't finite numbers (NaN, INF, strings,
  // nested vectors) - returns undef
  if (![...a, ...b].every((x) => typeof x === 'number' && Number.isFinite(x))) {
    console.warn('WARNING: Invalid value in parameter vector for cross()');
    return undefined;
  }
  // 2D vector yields the scalar z-component and 3D vector yields the vector
  if (a.length === 2) return a[0] * b[1] - a[1] * b[0];
  return [
    a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
  ];
}

// String & list (OpenSCAD built-ins)
function len_fn(x: any) {
  // returns undef when called with the wrong number of arguments
  if (arguments.length !== 1) return undefined;
  if (typeof x === 'string') return Array.from(x).length;
  if (Array.isArray(x)) return x.length;

  // returns undef and emits a warning for non-string/non-list inputs
  console.warn('WARNING: len() parameter could not be converted');
  return undefined;
}

function __ostr(x: any): string {
  if (x === undefined || x === null) return 'undef';
  if (typeof x === 'boolean') return x ? 'true' : 'false';
  if (typeof x === 'string') return x;
  if (Array.isArray(x)) return '[' + x.map(__ostrInner).join(', ') + ']';
  if (typeof x === 'function') return x.__oscadSrc ?? String(x);
  return String(x);
}

// prints `function(args) body` like OpenSCAD
function __fnlit(fn: any, src: string) {
  try {
    fn.__oscadSrc = src;
  } catch {
  }
  return fn;
}

// Trampolines tail calls made via function values by wrapping them in `__TC`
// thunks that `__call` iterates instead of recursing, keeping self-recursion
// through function values off the JS call stack
class __TC {
  fn: any;
  args: any[];
  constructor(fn: any, args: any[]) {
    this.fn = fn;
    this.args = args;
  }
}
function __tc(fn: any, ...args: any[]): any {
  return new __TC(fn, args);
}
function __call(fn: any, ...args: any[]): any {
  // Calling a non-function evaluates to undef in OpenSCAD
  if (typeof fn !== 'function') return undefined;
  let r: any = fn(...args);
  while (r instanceof __TC) {
    if (typeof r.fn !== 'function') return undefined;
    r = r.fn(...r.args);
  }
  return r;
}
function __ostrInner(x: any): string {
  return typeof x === 'string' ? `"${x}"` : __ostr(x);
}
function str_fn(...a: any[]) {
  return a.map(__ostr).join('');
}

// OpenSCAD echo formatting
function __echoEscape(s: string): string {
  let r = '';
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (ch === '\\')
      r += '\\\\';
    else if (ch === '"')
      r += '\\"';
    else if (ch === '\t')
      r += '\\t';
    else if (ch === '\n')
      r += '\\n';
    else if (ch === '\r')
      r += '\\r';
    else
      r += ch;
  }
  return r;
}

function __oecho(x: any): string {
  if (typeof x === 'string') return '"' + __echoEscape(x) + '"';
  if (Array.isArray(x)) return '[' + x.map(__oecho).join(', ') + ']';
  return __ostr(x);
}

function __echo(...parts: string[]) {
  console.log(parts.join(', '));
}

function __chrOne(c: any): string {
  const n = Math.trunc(c);
  if (!Number.isFinite(n) || n < 1 || n > 0x10FFFF ||
      (n >= 0xD800 && n <= 0xDFFF))
    return '';
  return String.fromCodePoint(n);
}


function __chrCollect(x: any, out: string[]) {
  if (Array.isArray(x)) {
    for (const e of x) __chrCollect(e, out);
    return;
  }
  if (typeof x === 'number') {
    const c = __chrOne(x);
    if (c) out.push(c);
  }
}

// OpenSCAD chr()
function chr_fn(...args: any[]) {
  const out: string[] = [];
  for (const a of args) __chrCollect(a, out);
  return out.join('');
}

// OpenSCAD ord()
function ord_fn(...args: any[]) {
  if (args.length !== 1) return undefined;
  const s = args[0];
  if (typeof s !== 'string' || s.length === 0) return undefined;
  return s.codePointAt(0);
}

// OpenSCAD's concat
function concat_fn(...a: any[]) {
  const out: any[] = [];
  for (const x of a) {
    if (__isVec(x))
      out.push(...x);
    else
      out.push(x);
  }
  return out;
}

// OpenSCAD rands()
class __MT19937 {
  private mt = new Uint32Array(624);
  private idx = 624;
  constructor(seed: number) {
    this.seed(seed);
  }
  seed(s: number) {
    this.mt[0] = s >>> 0;
    for (let i = 1; i < 624; i++) {
      const p = this.mt[i - 1]! ^ (this.mt[i - 1]! >>> 30);
      const lo = (p & 0xffff) * 1812433253;
      const hi = ((p >>> 16) * 1812433253) & 0xffff;
      this.mt[i] = (((hi << 16) >>> 0) + lo + i) >>> 0;
    }
    this.idx = 624;
  }
  next(): number {
    if (this.idx >= 624) {
      for (let i = 0; i < 624; i++) {
        const y = ((this.mt[i]! & 0x80000000) >>> 0) |
            (this.mt[(i + 1) % 624]! & 0x7fffffff);
        let n = this.mt[(i + 397) % 624]! ^ (y >>> 1);
        if (y & 1) n ^= 0x9908b0df;
        this.mt[i] = n >>> 0;
      }
      this.idx = 0;
    }
    let y = this.mt[this.idx++]!;
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }
}

const __rng = new __MT19937((Date.now() ^ (Math.random() * 0x100000000)) >>> 0);

function __hashFloatingPoint(v: number): number {
  const PyHASH_BITS = 31n;
  const MOD = (1n << 31n) - 1n;
  if (!Number.isFinite(v)) {
    if (v === Infinity) return 314159;
    if (v === -Infinity) return -314159;
    return 0;  // NaN
  }
  if (v === 0) return 0;
  let e = Math.floor(Math.log2(Math.abs(v))) + 1;
  let m = v / 2 ** e;
  while (Math.abs(m) >= 1) {
    m /= 2;
    e++;
  }
  while (Math.abs(m) < 0.5) {
    m *= 2;
    e--;
  }
  let sign = 1n;
  if (m < 0) {
    sign = -1n;
    m = -m;
  }
  let x = 0n;
  let ee = BigInt(e);
  while (m !== 0) {
    x = ((x << 28n) & MOD) | (x >> (PyHASH_BITS - 28n));
    m *= 268435456.0;  // 2^28
    ee -= 28n;
    const y = BigInt(Math.trunc(m));
    m -= Number(y);
    x += y;
    if (x >= MOD) x -= MOD;
  }
  const eMod = ee >= 0n ? ee % PyHASH_BITS :
                          PyHASH_BITS - 1n - ((-1n - ee) % PyHASH_BITS);
  x = ((x << eMod) & MOD) | (x >> (PyHASH_BITS - eMod));
  x = x * sign;
  return Number(BigInt.asIntN(32, x));
}

function __generateCanonical(): number {
  const R = 2 ** 32;
  const u0 = __rng.next();
  const u1 = __rng.next();
  let ret = (u0 + u1 * R) / (R * R);
  if (ret >= 1) ret = 1 - Number.EPSILON / 2;
  return ret;
}

function rands_fn(...args: any[]) {
  // Requires 3 or 4 numeric arguments; any other shape warns and yields undef.
  if (args.length < 3 || args.length > 4) return undefined;
  for (const a of args) {
    if (typeof a !== 'number') return undefined;
  }
  const DBL_MAX = Number.MAX_VALUE;
  let min = args[0];
  if (!Number.isFinite(min)) min = -DBL_MAX / 2;
  let max = args[1];
  if (!Number.isFinite(max)) max = DBL_MAX / 2;
  if (max < min) {
    const t = min;
    min = max;
    max = t;
  }
  let numresultsd = Math.abs(args[2]);
  if (!Number.isFinite(numresultsd)) numresultsd = 1;
  const numresults = Math.trunc(numresultsd);
  if (args.length > 3) {
    __rng.seed(__hashFloatingPoint(args[3]) >>> 0);
  }
  const out: number[] = [];
  if (min >= max) {
    for (let i = 0; i < numresults; i++) out.push(min);
  } else {
    for (let i = 0; i < numresults; i++)
      out.push(__generateCanonical() * (max - min) + min);
  }
  return out;
}

function __search_match(needle: any, entry: any, idx_col: any) {
  const col = idx_col !== undefined ? idx_col : 0;
  if (col === 0 && __eq(entry, needle)) return true;
  return Array.isArray(entry) && col < entry.length && __eq(entry[col], needle);
}

function search_fn(needle: any, haystack: any, num_returns: any, idx_col: any) {
  if (needle === undefined || haystack === undefined) {
    console.warn(
        `WARNING: search() needs to be called with at least 2 arguments`);
    return undefined;
  }
  if (!is_list_fn(needle) && !is_string_fn(needle)) {
    const indices: any[] = [];
    for (let i = 0; i < haystack.length; i++) {
      if (__search_match(needle, haystack[i], idx_col)) indices.push(i);
    }
    if (num_returns === 0) return indices;
    return indices.slice(0, num_returns === undefined ? 1 : num_returns);
  }
  if (is_string_fn(needle) && is_string_fn(haystack)) {
    let result: any[] = [];
    const hs = [...haystack];
    for (let ch of needle) {
      let indices = [];
      for (let i = 0; i < hs.length; i++) {
        if (hs[i] === ch) indices.push(i);
      }
      if (num_returns === 1 || num_returns === undefined) {
        if (indices.length > 0) result.push(indices[0]);
      } else {
        result.push(
            num_returns === 0 ? indices : indices.slice(0, num_returns));
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
      return num_returns === 0 ? indices :
                                 (indices.length > 0 ? indices[0] : []);
    });
  }
  if (is_string_fn(needle) && is_list_fn(haystack)) {
    const col = idx_col !== undefined ? idx_col : 0;
    const result: any[] = [];
    for (const n of needle) {
      let indices = [];
      for (let i = 0; i < haystack.length; i++) {
        const entry = haystack[i];
        if (!Array.isArray(entry) || col >= entry.length) {
          console.warn(`WARNING: Invalid entry in search vector at index ${
              i}, required number of values in the entry: ${
              col + 1}. Invalid entry: ${entry}`);
          return [];
        }
        if (__eq(entry[col], n)) indices.push(i);
      }
      if (num_returns === 0)
        result.push(indices);
      else if (indices.length === 0)
        result.push([]);
      else
        result.push(
            num_returns === 1 || num_returns === undefined ?
                indices[0] :
                indices.slice(0, num_returns));
    }
    return result;
  }
  return undefined;
}

function lookup_fn(key: any, table: any, ...rest: any[]) {
  // OpenSCAD's lookup() takes exactly 2 parameters and a numeric key. Any other
  // shape is warned about yields undef
  if (rest.length > 0) {
    console.warn(
        `WARNING: lookup() number of parameters does not match: expected 2, found ${
            2 + rest.length}`);
    return undefined;
  }
  if (table === undefined) {
    console.warn(
        `WARNING: lookup() number of parameters does not match: expected 2, found 1`);
    return undefined;
  }
  if (typeof key !== 'number') {
    console.warn(
        `WARNING: lookup() parameter could not be converted: argument 0: expected number, found ${
            typeof key === 'string' ? `string ("${key}")` : typeof key}`);
    return undefined;
  }
  if (!Array.isArray(table)) return undefined;
  let lowSet = false, highSet = false;
  let low_p = 0, low_v: any, high_p = 0, high_v: any;
  for (const entry of table) {
    if (!Array.isArray(entry) || entry.length < 2) continue;
    const this_p = entry[0], this_v = entry[1];
    if (typeof this_p !== 'number') continue;
    if (this_p <= key && (!lowSet || this_p > low_p)) {
      low_p = this_p;
      low_v = this_v;
      lowSet = true;
    }
    if (this_p >= key && (!highSet || this_p < high_p)) {
      high_p = this_p;
      high_v = this_v;
      highSet = true;
    }
  }
  if (!lowSet && !highSet) return undefined;
  if (!lowSet) return high_v;
  if (!highSet) return low_v;
  if (high_p === low_p) return low_v;
  return low_v + (high_v - low_v) * (key - low_p) / (high_p - low_p);
}

function __truthy(x: any) {
  if (x === undefined || x === null || x === false) return false;
  if (typeof x === 'number') return x !== 0;
  if (typeof x === 'string' || Array.isArray(x)) return x.length > 0;
  return true;
}

// Control
function openscad_assert_fn(cond: any, msg: any) {
  if (!__truthy(cond)) {
    console.trace('Assertion failed:', msg);
    throw new Error(msg || 'Assertion failed');
  }
}

function __eq(a: any, b: any) {
  if (a === b) return true;
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (!__eq(a[i], b[i])) return false;
    }
    return true;
  }
  return false;
}

function __cmpCat(x: any): string {
  const t = typeof x;
  if (t === 'number') return 'n';
  if (t === 'boolean') return 'b';
  if (t === 'string') return 's';
  if (Array.isArray(x)) return (x as any).__isRange ? 'r' : 'v';
  return 'u';
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
  const ca = __cmpCat(a);
  if (ca !== __cmpCat(b) || ca === 'u') return undefined;
  return (ca === 'v' || ca === 'r') ? __veccmp(a, b) < 0 : a < b;
}
function __gt(a: any, b: any): any {
  const ca = __cmpCat(a);
  if (ca !== __cmpCat(b) || ca === 'u') return undefined;
  return (ca === 'v' || ca === 'r') ? __veccmp(a, b) > 0 : a > b;
}
function __le(a: any, b: any): any {
  const ca = __cmpCat(a);
  if (ca !== __cmpCat(b) || ca === 'u') return undefined;
  return (ca === 'v' || ca === 'r') ? __veccmp(a, b) <= 0 : a <= b;
}
function __ge(a: any, b: any): any {
  const ca = __cmpCat(a);
  if (ca !== __cmpCat(b) || ca === 'u') return undefined;
  return (ca === 'v' || ca === 'r') ? __veccmp(a, b) >= 0 : a >= b;
}
function __isNum(x: any): boolean {
  return typeof x === 'number';
}
function __isVec(x: any): boolean {
  return Array.isArray(x) && !(x as any).__isRange;
}

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
  if (__isNum(a) && __isVec(b))
    return b.map((x: any): any => __mul(a, x));  // scalar * vector
  if (__isVec(a) && __isNum(b))
    return a.map((x: any): any => __mul(x, b));  // vector * scalar
  if (__isVec(a) && __isVec(b)) {
    const aMat = a.length > 0 && __isVec(a[0]);
    const bMat = b.length > 0 && __isVec(b[0]);
    if (aMat && bMat) {  // matrix * matrix
      const aCols = a[0].length, bRows = b.length;
      if (aCols !== bRows) {
        console.warn(
            `WARNING: matrix*matrix requires left operand column count to match right operand row count (${
                aCols} != ${bRows})`);
        return undefined;
      }
      for (let i = 0; i < a.length; i++) {
        const rowLen = __isVec(a[i]) ? a[i].length : 0;
        if (rowLen !== bRows) {
          console.warn(
              `WARNING: matrix*matrix left operand row length does not match right operand row count (${
                  rowLen} != ${bRows}) at row ${i}`);
          return undefined;
        }
      }
      const res: any[] = [];
      for (let i = 0; i < a.length; i++) {
        res[i] = [];
        for (let j = 0; j < b[0].length; j++) {
          let sum = 0;
          for (let k = 0; k < aCols; k++) sum += a[i][k] * b[k][j];
          res[i].push(sum);
        }
      }
      return res;
    }
    if (aMat) {  // matrix * vector
      const aCols = a[0].length;
      if (aCols !== b.length) {
        console.warn(
            `WARNING: matrix*vector requires matrix column count to match vector length (${
                aCols} != ${b.length})`);
        return undefined;
      }
      const res: any[] = [];
      for (let i = 0; i < a.length; i++) {
        const row = a[i];
        const rowLen = __isVec(row) ? row.length : 0;
        if (rowLen !== b.length) {
          console.warn(
              `WARNING: matrix*vector left operand row length does not match vector length (${
                  rowLen} != ${b.length}) at row ${i}`);
          return undefined;
        }
        let sum = 0;
        for (let k = 0; k < b.length; k++) sum += row[k] * b[k];
        res.push(sum);
      }
      return res;
    }
    if (bMat) {  // vector * matrix
      if (a.length !== b.length) {
        console.warn(
            `WARNING: vector*matrix requires vector length to match matrix row count (${
                a.length} != ${b.length})`);
        return undefined;
      }
      const res: any[] = [];
      for (let j = 0; j < b[0].length; j++) {
        let sum = 0;
        for (let k = 0; k < a.length; k++) sum += a[k] * b[k][j];
        res.push(sum);
      }
      return res;
    }
    if (a.length !== b.length) {  // vector . vector
      console.warn(`WARNING: vector*vector requires matching lengths (${
          a.length} != ${b.length})`);
      return undefined;
    }
    let sum3 = 0;
    for (let i3 = 0; i3 < a.length; i3++) sum3 += a[i3] * b[i3];
    return sum3;
  }
  return undefined;
}
function __div(a: any, b: any): any {
  if (__isNum(a) && __isNum(b)) return a / b;
  if (__isVec(a) && __isNum(b))
    return a.map((x: any): any => __div(x, b));  // vector / scalar
  if (__isNum(a) && __isVec(b))
    return b.map((x: any): any => __div(a, x));  // scalar / vector
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
function __toI64(x: any): bigint|undefined {
  if (typeof x !== 'number' || !isFinite(x)) return undefined;
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
  if (ib < 0n || ib >= 64n) return undefined;  // shifts of 64+ bits are undef
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
  if (typeof idx !== 'number' || obj == null) return undefined;
  if (typeof obj === 'string') {
    const cps = Array.from(obj);
    const i = Math.floor(idx);
    return i >= 0 && i < cps.length ? cps[i] : undefined;
  }
  if (Array.isArray(obj)) return obj[Math.floor(idx)];
  return undefined;
}

// OpenSCAD version
function version_fn() {
  return [2019, 5, 0];
}
function version_num_fn() {
  return 20190500;
}

// Constants
let PI = Math.PI;
let INF = Infinity;
let NAN = NaN;
let undef = undefined;
let _EPSILON = 1e-9;

// Special-variable context
const __ctx: Record<string, any> = {
  $fn: 0,
  $fa: 12,
  $fs: 2,
  $vpr: [55, 0, 25],
  $vpt: [0, 0, 0],
  $vpd: 140,
  $vpf: 22.5,
  $t: 0,
  $preview: false,
  $parent_modules: 0,
  $color: undefined,
  $idx: undefined,
};

function __withSpecials(overrides: Record<string, any>, body: () => any) {
  const saved: Record<string, any> = {};
  for (const k in overrides) {
    saved[k] = __ctx[k];
    __ctx[k] = overrides[k];
  }
  try {
    return body();
  } finally {
    Object.assign(__ctx, saved);
  }
}

// Children stack for module calls
let __children_stack: any[] = [];
const __color_prop_layout = new WeakMap();
function __with_children(fn: any, count: any, call: any, name?: string) {
  __children_stack.push({fn: fn, count: count, name: name});
  try {
    return call();
  } finally {
    __children_stack.pop();
  }
}

// Resolve a children() selector into the ordered list of child fns to invoke
function __pick_children(childFns: any[], i: any): any[] {
  if (i === undefined) return childFns;
  if (Array.isArray(i)) {
    const out: any[] = [];
    for (const v of i) {
      if (Number.isInteger(v) && v >= 0 && v < childFns.length)
        out.push(childFns[v]);
    }
    return out;
  }
  if (Number.isInteger(i) && i >= 0 && i < childFns.length)
    return [childFns[i]];
  return [];
}

function parent_module_fn(d: any = 1) {
  const depth = Number(d);
  if (!Number.isInteger(depth) || depth < 0) return '';
  const idx = __children_stack.length - 1 - depth;
  if (idx < 0 || idx >= __children_stack.length) return '';
  return __children_stack[idx].name || '';
}

function __is_finite_matrix4(m: any) {
  return Array.isArray(m) && m.length === 4 &&
      m.every(
          (row: any) => Array.isArray(row) && row.length === 4 &&
              row.every(
                  (v: any) => typeof v === 'number' && Number.isFinite(v)));
}

// Pad smaller matrices to 4×4 with identity, then normalize by the bottom-right
// value before applying the affine transform
function __normalize_matrix4(m: any) {
  if (!Array.isArray(m) || m.length < 3) return undefined;
  const out = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  for (let row = 0; row < 3; row++) {
    const r = m[row];
    if (!Array.isArray(r) || r.length < 3) return undefined;
    for (let col = 0; col < Math.min(r.length, 4); col++) {
      const v = r[col];
      if (typeof v !== 'number' || !Number.isFinite(v)) return undefined;
      out[row]![col] = v;
    }
  }
  const w = Array.isArray(m[3]) ? m[3][3] : undefined;
  if (typeof w === 'number' && Number.isFinite(w) && w !== 0 && w !== 1) {
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 4; col++) {
        out[row]![col]! /= w;
      }
    }
  }
  return out;
}

// Manifold expects a flat 4x4 matrix in column-major order.
function __to_manifold_mat4(m: any) {
  const n = __normalize_matrix4(m);
  if (!n) return undefined;
  const out = new Array(16);
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      out[col * 4 + row] = n[row]![col];
    }
  }
  return out;
}

function __to_manifold_mat3(m: any) {
  const n = __normalize_matrix4(m);
  if (!n) return undefined;
  return [n[0]![0], n[1]![0], 0, n[0]![1], n[1]![1], 0, n[0]![3], n[1]![3], 1];
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
function __safe_offset2d(
    shape: any, delta: any, joinType = 'Round', miterLimit = 2,
    circularSegments = 0, fa = 12, fs = 2) {
  try {
    if (shape && typeof shape.offset === 'function') {
      if (circularSegments <= 0) {
        __sync_quality(fa, fs);
      }
      return shape.offset(delta, joinType, miterLimit, circularSegments);
    }
  } catch {
  }
  return shape;
}

function __safe_project3d(shape: any, cut = false) {
  try {
    if (shape) {
      if (cut && typeof shape.slice === 'function') {
        return shape.slice(0);
      }
      if (typeof shape.project === 'function') {
        return shape.project();
      }
    }
  } catch {
  }
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
  if (!s.startsWith('#')) return undefined;
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
    const maxChan =
        Math.max(Math.abs(r), Math.abs(g), Math.abs(b), Math.abs(a));
    // Support accidental [0..255] style vectors.
    if (maxChan > 1) {
      r /= 255;
      g /= 255;
      b /= 255;
      a /= 255;
    }
    return [r, g, b, a];
  }

  if (typeof c === 'string') {
    const key = c.trim().toLowerCase();
    if (key === '' || key === 'default') return undefined;
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
  if (!shape || typeof shape.setProperties !== 'function' ||
      typeof shape.numProp !== 'function') {
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
  if (trackedLayout && Number.isInteger(trackedLayout.colorOffset) &&
      Number.isInteger(trackedLayout.markerOffset)) {
    const trackedColorOffset = trackedLayout.colorOffset;
    const trackedMarkerOffset = trackedLayout.markerOffset;
    if (trackedColorOffset >= 0 &&
        trackedMarkerOffset === trackedColorOffset + 4 &&
        trackedMarkerOffset < oldNumProp) {
      colorOffset = trackedColorOffset;
      markerOffset = trackedMarkerOffset;
      newNumProp = oldNumProp;
    }
  }

  try {
    const painted = shape.setProperties(
        newNumProp, (newProp: any, position: any, oldProp: any) => {
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
          // Marker channel lets the viewer distinguish custom RGBA from
          // manifold built-in properties.
          newProp[markerOffset] = 1;
        });
    __color_prop_layout.set(painted, {colorOffset, markerOffset});
    return painted;
  } catch {
    return shape;
  }
}

function __each(v: any) {
  if (v === undefined || v === null) return [];
  if (Array.isArray(v)) return v;
  if (typeof v === 'string') return Array.from(v);
  return [v];
}

function __flat_map_iter(v: any, fn: any) {
  if (v === undefined || v === null) return [];
  if (Array.isArray(v)) return v.flatMap((item, i) => fn(item, i));
  if (typeof v === 'string') return Array.from(v).flatMap(fn);
  return [v].flatMap(fn);
}

const __UINT32_MAX = 4294967295;
const __rc_f64 = new Float64Array(1);
const __rc_u64 = new BigUint64Array(__rc_f64.buffer);
function __nextUp(x: number): number {
  __rc_f64[0] = x;
  __rc_u64[0] = __rc_u64[0]! + 1n;
  return __rc_f64[0]!;
}

// max uint32_t if step is 0 or range is infinite
function __rangeNumValues(start: number, step: number, end: number): number {
  if (Number.isNaN(start) || Number.isNaN(step) || Number.isNaN(end)) return 0;
  if (step < 0) {
    if (start < end) return 0;
  } else {
    if (start > end) return 0;
  }
  if (start === end || !Number.isFinite(step)) return 1;
  if (!Number.isFinite(start) || !Number.isFinite(end) || step === 0)
    return __UINT32_MAX;
  const numSteps = Math.floor(__nextUp((end - start) / step));
  return numSteps >= __UINT32_MAX ? __UINT32_MAX : numSteps + 1;
}

function __rangeCount(start: any, step: any, end: any): number {
  if (typeof start !== 'number' || typeof step !== 'number' ||
      typeof end !== 'number')
    return 0;
  const n = __rangeNumValues(start, step, end);
  if (n >= 1000000) {
    console.warn(
        `WARNING: Bad range parameter in for statement: too many elements (${
            n}).`);
    return 0;
  }
  // Starts at the end when step is 0, so [0:0:0] yields nothing
  if (step === 0) return 0;
  return n;
}

function __range(start: any, step: any, end: any) {
  let result: any[] = [];
  let n = __rangeCount(start, step, end);
  // Yield begin_val itself first, so an infinite step still yields start
  for (let i = 0; i < n; i++) result.push(i === 0 ? start : start + i * step);
  Object.defineProperty(result, '__isRange', {
    value: true,
    enumerable: false,
    writable: true,
    configurable: true,
  });
  return result;
}

// Detect CrossSection (2D) vs Manifold (3D) for dispatch
function __is2D(x: any) {
  return x != null && typeof x.offset === 'function' &&
      typeof x.toPolygons === 'function';
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
    if (!it || typeof it.boundingBox !== 'function') continue;
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
    if (it && typeof it.setTolerance === 'function' &&
        typeof it.tolerance === 'function') {
      return tol > it.tolerance() ? it.setTolerance(tol) : it;
    }
    return it;
  });
}

// OpenSCAD cannot mix 2D and 3D in a boolean op - it keeps the dimension of the
// first child and ignores (with a warning) any children of the other dimension.
function __sameDim(items: any[], ref2D: boolean): any[] {
  return items.filter(x => __is2D(x) === ref2D);
}

// Root modifier (!): the first geometry evaluated under a `!` statement becomes
// the design root - OpenSCAD renders only that subtree and ignores the rest of
// the design
let __root_item: any = null;
function __rootMod(g: any): any {
  if (__root_item === null && g !== null && g !== undefined) __root_item = g;
  return g;
}
function __applyRoot(items: any[], isBackground = false): any[] {
  if (__root_item === null) return items;
  return isBackground ? [] : [__root_item];
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
  // Intersecting across dimensions (e.g. a 3D solid with a 2D shape) has no
  // common volume, so OpenSCAD yields an empty result.
  if (same.length < valid.length)
    return is2D ? CrossSection.union([]) : Manifold.union([]);
  return is2D ? CrossSection.intersection(same) :
                Manifold.intersection(__withTol3d(same));
}
function __hull2d3d(items: any[]) {
  const valid = items.filter(x => !__isEmpty(x));
  if (valid.length === 0) return Manifold.union([]);
  return __is2D(valid[0]) ? CrossSection.hull(valid) : Manifold.hull(valid);
}

function __mesh_points3d(manifold: any, maxPoints = 192) {
  if (!manifold || typeof manifold.getMesh !== 'function') return [];
  const mesh = manifold.getMesh();
  const numProp =
      (mesh && typeof mesh.numProp === 'number' && mesh.numProp >= 3) ?
      mesh.numProp :
      3;
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
  if (!manifold || typeof manifold.hull !== 'function' ||
      typeof manifold.volume !== 'function')
    return false;
  if (typeof manifold.isEmpty === 'function' && manifold.isEmpty()) return true;
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
  if (valid.length === 0) return Manifold.union([]);
  if (valid.length === 1) return valid[0];

  // Check ALL items upfront
  for (const item of valid) {
    if (__is2D(item)) throw new Error('2D minkowski not implemented');
    if (typeof item.minkowskiSum !== 'function')
      throw new Error('Your manifold-3d build does not expose minkowskiSum');
  }

  let acc = valid[0];
  for (let i = 1; i < valid.length; i++) {
    acc = acc.minkowskiSum(valid[i]);
  }
  return acc;
}

// Returns `x` when it is a finite number, otherwise `dflt`
function __finiteOr(x: any, dflt: number): number {
  return (typeof x === 'number' && Number.isFinite(x)) ? x : dflt;
}

function __normalizeScale(scale: number|number[]|undefined): [number, number]|
    undefined {
  if (scale === undefined || scale === null) return undefined;
  // OpenSCAD accepts only a finite scalar or a 2-element finite vector for
  // scaling; anything else is treated as no scaling
  let sx: number, sy: number;
  if (Array.isArray(scale)) {
    if ((scale as any).__isRange || scale.length !== 2) return undefined;
    if (!Number.isFinite(scale[0]) || !Number.isFinite(scale[1]))
      return undefined;
    sx = scale[0] as number;
    sy = scale[1] as number;
  } else {
    if (typeof scale !== 'number' || !Number.isFinite(scale)) return undefined;
    sx = sy = scale;
  }
  return [Math.max(0, sx), Math.max(0, sy)];
}

// Arc length of an Archimedes spiral r = a*theta
function __archimedesLength(a: number, theta: number): number {
  return 0.5 * a * (theta * Math.sqrt(1 + theta * theta) + Math.asinh(theta));
}

// Slice calculation logic for curves, helices, and conical helices
function __computeExtrudeDivisions(shape: any, height: number, options: {
  twist?: number;
  scale?: number | number[] | undefined;
  fn?: number;
  fa?: number;
  fs?: number;
  fe?: number;
  slices?: number;
}): number {
  if (typeof options.slices === 'number' && Number.isFinite(options.slices)) {
    return Math.max(1, options.slices);
  }

  const twist = Math.abs(__finiteOr(options.twist, 0));
  const fn = options.fn ?? 0;
  const fa = options.fa ?? 12;
  const fs = options.fs ?? 2;
  const fe = (typeof options.fe === 'number' && Number.isFinite(options.fe) &&
              options.fe > 0) ?
      options.fe :
      0;
  const GRID_FINE = 0.00000095367431640625;

  const normScale = __normalizeScale(options.scale);
  const sx = normScale?.[0] ?? 1;
  const sy = normScale?.[1] ?? 1;

  // Tracks the maximum squared distance from the axis and the largest squared
  // displacement caused by scaling
  let rSqr = 0;
  let maxDeltaSqr = 0;
  try {
    const polys = shape.toPolygons();
    for (const poly of polys) {
      for (const p of poly) {
        const dSqr0 = p[0] * p[0] + p[1] * p[1];
        if (dSqr0 > rSqr) rSqr = dSqr0;
        const dx = p[0] - p[0] * sx;
        const dy = p[1] - p[1] * sy;
        const dSqr = dx * dx + dy * dy;
        if (dSqr > maxDeltaSqr) maxDeltaSqr = dSqr;
      }
    }
  } catch {
    rSqr = 100;
    maxDeltaSqr = (10 * Math.max(Math.abs(sx - 1), Math.abs(sy - 1))) ** 2;
  }

  const diagonalSlices = () => {
    if (Math.sqrt(maxDeltaSqr) < GRID_FINE) return 1;
    if (fn > 0) return Math.max(1, Math.trunc(fn));
    return Math.max(
        1, Math.ceil(Math.sqrt(maxDeltaSqr + height * height) / fs));
  };

  if (twist === 0) {
    return sx !== sy ? diagonalSlices() : 1;
  }

  const minSlices = Math.max(Math.ceil(twist / 120), 1);
  const twistRad = (twist * Math.PI) / 180;

  const helixSlices = () => {
    const r = Math.sqrt(rSqr);
    if (r < GRID_FINE) return minSlices;
    if (fn > 0) return Math.max(Math.ceil((twist / 360) * fn), minSlices);
    if (fe > 0) {
      // helix_slices_given_fe: max sagitta of a chord across the helix arc <=
      // fe
      if (fe >= r) return minSlices;
      const theta = 2 * (Math.PI - Math.acos(fe / r - 1));
      return Math.max(Math.ceil(twistRad / theta), minSlices);
    }
    const faSlices = Math.ceil(twist / fa);
    const helixLen = Math.sqrt(rSqr * twistRad * twistRad + height * height);
    const fsSlices = Math.ceil(helixLen / fs);
    return Math.max(Math.min(faSlices, fsSlices), minSlices);
  };

  if (sx === 1 && sy === 1) {
    return helixSlices();
  }

  if (sx !== sy) {
    // Twist with non-uniform scale: the larger of the diagonal and helix counts
    return Math.max(diagonalSlices(), helixSlices());
  }

  // Twist with uniform scale
  const r = Math.sqrt(rSqr);
  if (r < GRID_FINE) return minSlices;
  if (fn > 0) return Math.max(Math.ceil(twist * fn / 360), minSlices);
  const angleEnd = sx > 1 ? twistRad * sx / (sx - 1) : twistRad / (1 - sx);
  const angleStart = angleEnd - twistRad;
  const a = r / angleEnd;
  const spiralLength =
      __archimedesLength(a, angleEnd) - __archimedesLength(a, angleStart);
  const totalLength = Math.sqrt(spiralLength * spiralLength + height * height);
  const fsSlices = Math.ceil(totalLength / fs);
  const faSlices = Math.ceil(twist / fa);
  return Math.max(Math.min(faSlices, fsSlices), minSlices);
}

// Interpolate points along v0->v1, excluding the endpoint v1
function __addSegmentedEdge(
    out: [number, number][], v0: [number, number], v1: [number, number],
    segs: number) {
  for (let j = 0; j < segs; j++) {
    const t = j / segs;
    out.push([(1 - t) * v0[0] + t * v1[0], (1 - t) * v0[1] + t * v1[1]]);
  }
}

function __maxEdgeLen(
    v0: [number, number], v1: [number, number], twist: number, sx: number,
    sy: number, slices: number): number {
  if (sx === sy) {
    return Math.hypot(v1[0] - v0[0], v1[1] - v0[1]) * Math.max(sx, 1);
  }
  let maxLen = 0;
  for (let j = 0; j <= slices; j++) {
    const t = j / slices;
    const scx = 1 + (sx - 1) * t, scy = 1 + (sy - 1) * t;
    const ang = -twist * t * Math.PI / 180;
    const ca = Math.cos(ang), sa = Math.sin(ang);
    const x0 = (v0[0] * ca - v0[1] * sa) * scx,
          y0 = (v0[0] * sa + v0[1] * ca) * scy;
    const x1 = (v1[0] * ca - v1[1] * sa) * scx,
          y1 = (v1[0] * sa + v1[1] * ca) * scy;
    const len = Math.hypot(x1 - x0, y1 - y0);
    if (len > maxLen) maxLen = len;
  }
  return maxLen;
}

// Split each edge into ceil(maxEdgeLen / fs) segments.
function __splitOutlineByFs(
    o: [number, number][], twist: number, sx: number, sy: number, fs: number,
    slices: number): [number, number][] {
  const n = o.length;
  const out: [number, number][] = [];
  for (let i = 1; i <= n; i++) {
    const v0 = o[i - 1]!, v1 = o[i % n]!;
    const segs = Math.max(
        1, Math.ceil(__maxEdgeLen(v0, v1, twist, sx, sy, slices) / fs));
    __addSegmentedEdge(out, v0, v1, segs);
  }
  return out;
}

function __splitOutlineByFn(
    o: [number, number][], twist: number, sx: number, sy: number,
    target: number, slices: number): [number, number][] {
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
    for (let k = 1; k < n; k++)
      if (metric(k) > metric(top)) top = k;
    const topMetric = metric(top);
    const group: number[] = [];
    for (let k = 0; k < n; k++) {
      const mk = metric(k);
      if (Math.min(mk, topMetric) / Math.max(mk, topMetric) >= 0.999)
        group.push(k);
    }
    if (segTotal + group.length > target) break;
    for (const g of group) {
      segCount[g]++;
      segTotal++;
    }
  }
  const out: [number, number][] = [];
  for (let i = 1; i <= n; i++)
    __addSegmentedEdge(out, o[i - 1]!, o[i % n]!, segCount[i - 1]);
  return out;
}

// Refine a single outline for a non-linear (twisted/non-uniformly-scaled)
// extrude
function __splitOutline(
    o: [number, number][], twist: number, sx: number, sy: number,
    slices: number, fn: number, fa: number, fs: number,
    segments: number): [number, number][] {
  if (segments > 0 || fn > 0.0) {
    const minVerts = segments > 0 ? segments : Math.max(fn, 3);
    return o.length >= minVerts ?
        o :
        __splitOutlineByFn(o, twist, sx, sy, minVerts, slices);
  }
  const faSegs = Math.ceil(360.0 / fa);
  if (o.length >= faSegs) return o;
  const fsOutline = __splitOutlineByFs(o, twist, sx, sy, fs, slices);
  return fsOutline.length >= faSegs ?
      __splitOutlineByFn(o, twist, sx, sy, faSegs, slices) :
      fsOutline;
}

// OpenSCAD removes duplicate and collinear outline vertices before extrusion,
// so do the same to match its mesh output
function __dropCollinear(poly: [number, number][]): [number, number][] {
  const n = poly.length;
  if (n < 4) return poly;
  const out: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const p = poly[(i + n - 1) % n]!;
    const c = poly[i]!;
    const q = poly[(i + 1) % n]!;
    const ax = c[0] - p[0], ay = c[1] - p[1];
    const bx = q[0] - c[0], by = q[1] - c[1];
    const cross = ax * by - ay * bx;
    const scale = Math.hypot(ax, ay) * Math.hypot(bx, by);
    if (Math.abs(cross) > 1e-12 * Math.max(scale, 1e-30)) out.push(c);
  }
  return out.length >= 3 ? out : poly;
}

function __extrudeTwisted(
    shape: any, height: number, twistDeg: number, slices: number,
    scaleVec: [number, number]|undefined, center: boolean|undefined, fn: number,
    fa: number, fs: number, segments?: number): any {
  const rawPolys: [number, number][][] =
      shape.toPolygons().map(__dropCollinear);
  if (!rawPolys.length) return Manifold.union([]);

  const sx = scaleVec ? scaleVec[0] : 1;
  const sy = scaleVec ? scaleVec[1] : 1;

  // Refine outlines the same way as OpenSCAD to match its twisted mesh;
  // `segments=0` disables refinement
  const polys = segments === 0 ?
      rawPolys :
      rawPolys.map(
          c => __splitOutline(
              c, twistDeg, sx, sy, slices, fn, fa, fs, segments ?? 0));

  // Flatten all contours, keeping per-contour boundary order for the walls and
  // a position -> flat-index map for recovering the cap triangulation
  const flat: [number, number][] = [];
  const contours: {off: number; len: number}[] = [];
  const keyMap = new Map<string, number>();
  const key = (x: number, y: number) =>
      `${Math.round(x * 1000)},${Math.round(y * 1000)}`;
  for (const c of polys) {
    const off = flat.length;
    for (const p of c) {
      keyMap.set(key(p[0], p[1]), flat.length);
      flat.push([p[0], p[1]]);
    }
    contours.push({off, len: c.length});
  }

  // Recover cap triangles (as flat indices) from the bottom face of a plain,
  // untwisted extrude of the refined outline.
  let capTris: [number, number, number][]|null = [];
  try {
    const capShape = CrossSection.ofPolygons(polys);
    const pm = capShape.extrude(1).getMesh();
    const np = pm.numProp, vp = pm.vertProperties, tv = pm.triVerts;
    for (let i = 0; i < tv.length && capTris; i += 3) {
      const ia = tv[i]!, ib = tv[i + 1]!, ic = tv[i + 2]!;
      if (vp[ia * np + 2]! < 1e-4 && vp[ib * np + 2]! < 1e-4 &&
          vp[ic * np + 2]! < 1e-4) {
        const a = keyMap.get(key(vp[ia * np]!, vp[ia * np + 1]!));
        const b = keyMap.get(key(vp[ib * np]!, vp[ib * np + 1]!));
        const cc = keyMap.get(key(vp[ic * np]!, vp[ic * np + 1]!));
        if (a === undefined || b === undefined || cc === undefined) {
          capTris = null;
          break;
        }
        capTris.push([a, b, cc]);
      }
    }
  } catch {
    capTris = null;
  }
  if (!capTris || !capTris.length) {
    // Fall back to Manifold's own twist extrude if the cap mapping failed.
    return shape.extrude(
        height, Math.max(0, slices - 1), Math.abs(twistDeg), scaleVec, center);
  }

  const S = Math.max(1, Math.round(slices));
  const topZero = scaleVec !== undefined && sx === 0 && sy === 0;
  const z0 = center ? -height / 2 : 0;
  const M = flat.length;

  // Rotate by -twist*t (positive twist -> clockwise), then apply the
  // interpolated scale, matching OpenSCAD's Scaling * Rotation order.
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
  // OpenSCAD's add_slice_indices compares the XY-projected diagonal lengths
  // only
  const dist2d = (p: number, q: number) => {
    const pb = p * 3, qb = q * 3;
    const dx = verts[pb]! - verts[qb]!;
    const dy = verts[pb + 1]! - verts[qb + 1]!;
    return Math.hypot(dx, dy);
  };
  // OpenSCAD sgn_vdiff: lengths within 5 orders of magnitude count as equal
  const sgnVdiff = (l1: number, l2: number) => {
    const scale = l1 + l2;
    const diff = 2 * Math.abs(l1 - l2) * 1e5;
    return diff > scale ? (l1 < l2 ? -1 : 1) : 0;
  };
  // back_twist: rotation at slice top <= rotation at slice bottom
  const backTwist = twistDeg <= 0;

  // Bottom cap (slice 0) keeps the plain-extrude winding (faces -z); top cap
  // (slice S) is reversed to face +z. Skip the top cap if it collapses to a
  // point.
  for (const [a, b, c] of capTris) tris.push(idx(0, a), idx(0, b), idx(0, c));
  if (!topZero) {
    for (const [a, b, c] of capTris) tris.push(idx(S, a), idx(S, c), idx(S, b));
  }

  // Triangulate each wall quad using the shorter diagonal; if both are equal,
  // choose the diagonal the same way OpenSCAD does based on twist direction and
  // outline orientation
  for (const {off, len} of contours) {
    // Outline orientation via signed area: CCW outer outlines are positive, CW
    // holes negative
    let area2 = 0;
    for (let e = 0; e < len; e++) {
      const v0 = flat[off + e]!;
      const v1 = flat[off + ((e + 1) % len)]!;
      area2 += v0[0] * v1[1] - v1[0] * v0[1];
    }
    const flip = (area2 < 0) !== backTwist;

    for (let e = 0; e < len; e++) {
      const a = off + e;
      const b = off + ((e + 1) % len);
      for (let i = 0; i < S; i++) {
        const p1 = idx(i, a), c1 = idx(i, b);
        const c2 = idx(i + 1, b), p2 = idx(i + 1, a);
        if (topZero && i === S - 1) {
          // Top collapsed to a point: fan to a single shared apex vertex (all
          // slice-S vertices coincide at the origin, so pick one canonical
          // index)
          tris.push(p1, c1, idx(S, 0));
          continue;
        }
        const diffSign = sgnVdiff(dist2d(p1, c2), dist2d(c1, p2));
        const splitFirst = diffSign === -1 || (diffSign === 0 && !flip);
        // A zero component in the slice-top scale flips the split to avoid
        // 0-thickness ears
        const t1 = (i + 1) / S;
        const anyZero = 1 + (sx - 1) * t1 === 0 || 1 + (sy - 1) * t1 === 0;
        if (splitFirst !== anyZero) {
          tris.push(p1, c1, c2, p1, c2, p2);
        } else {
          tris.push(p1, c1, p2, c1, c2, p2);
        }
      }
    }
  }

  return new Manifold(
      {vertProperties: verts, triVerts: new Uint32Array(tris), numProp: 3} as
      any);
}

function __extrude(shape: any, height?: number, options: {
  twist?: number;
  scale?: number | number[] | undefined;
  center?: boolean;
  v?: any;
  fn?: number;
  fa?: number;
  fs?: number;
  fe?: number;
  slices?: number;
  segments?: number;
} = {}) {
  if (__isEmpty(shape)) {
    return Manifold.union([]);
  }

  // OpenSCAD ignores 3D children of 2D operations
  if (!__is2D(shape)) {
    return Manifold.union([]);
  }

  // Match OpenSCAD's extrusion vector rules: use `v` as-is, combine `v` with
  // `height` when both are given, default to `[0,0,1] * height`, and treat an
  // invalid `v` as `[0,0,1]`
  let vec: [number, number, number]|undefined;
  if (options.v !== undefined && options.v !== null) {
    const v = options.v;
    if (Array.isArray(v) && v.length === 3 &&
        v.every(c => Number.isFinite(c))) {
      vec = [v[0], v[1], v[2]];
    } else {
      vec = [0, 0, 1];
    }
    if (typeof height === 'number' && Number.isFinite(height)) {
      const len = Math.hypot(vec[0], vec[1], vec[2]);
      if (len > 0)
        vec = [
          vec[0] / len * height, vec[1] / len * height, vec[2] / len * height
        ];
    }
    height = vec[2];
  } else {
    // An invalid height defaults to OpenSCAD's linear_extrude default of 100
    height = __finiteOr(height, 100);
  }
  // OpenSCAD clamps a non-positive z extent to 0, which yields no geometry
  if (height <= 0) {
    return Manifold.union([]);
  }

  // An invalid twist means "no twist"
  const twist = __finiteOr(options.twist, 0);
  const normScale = __normalizeScale(options.scale);

  const nDivisions =
      __computeExtrudeDivisions(shape, height, {...options, scale: normScale});

  // OpenSCAD validates segments as a non-negative integer, ignoring anything
  // else
  const segments =
      (typeof options.segments === 'number' &&
       Number.isInteger(options.segments) && options.segments >= 0) ?
      options.segments :
      undefined;

  let result: any;
  if (twist !== 0 ||
      (normScale !== undefined && normScale[0] !== normScale[1])) {
    // Even without twist, non-uniform scaling uses the manual mesh builder to
    // match OpenSCAD's sliced walls and consistent quad triangulation
    result = __extrudeTwisted(
        shape,
        height,
        twist,
        nDivisions,
        normScale,
        options.center,
        options.fn ?? 0,
        options.fa ?? 12,
        options.fs ?? 2,
        segments,
    );
  } else {
    result = shape.extrude(
        height, Math.max(0, nDivisions - 1), undefined, normScale,
        options.center);
  }

  // A non-vertical extrusion vector shears the straight extrusion sideways
  if (vec && (vec[0] !== 0 || vec[1] !== 0)) {
    const kx = vec[0] / vec[2];
    const ky = vec[1] / vec[2];
    result = result.warp((p: number[]) => {
      p[0]! += kx * p[2]!;
      p[1]! += ky * p[2]!;
    });
  }
  return result;
}

function __revolve(shape: any, fn = 0, fa = 12, fs = 2, angle = 360) {
  if (__isEmpty(shape)) {
    return Manifold.union([]);
  }
  if (__is2D(shape)) {
    // NaN, ±Infinity and anything beyond ±360 all mean a full revolution
    if (typeof angle !== 'number' || !isFinite(angle) ||
        Math.abs(angle) > 360) {
      angle = 360;
    }
    if (angle === 0) {
      return Manifold.union([]);
    }

    const bounds = shape.bounds();
    // OpenSCAD supports profiles entirely on the negative X side; mirror them
    // to +X for Manifold's revolve, then rotate the result back by 180° to
    // match
    const onNegativeSide = bounds.max[0] <= 0;
    if (onNegativeSide) {
      shape = shape.mirror([1, 0]);
    }

    const absAngle = Math.abs(angle);
    // Fragments for the full circle first, then scaled by the swept angle and
    // truncated
    let full_fragments: number;
    if (fn > 0) {
      full_fragments = Math.floor(fn >= 3 ? fn : 3);
    } else {
      const r = Math.max(Math.abs(bounds.max[0]), Math.abs(bounds.min[0]));
      const N_fa = 360 / fa;
      const N_fs = 2 * Math.PI * r / fs;
      full_fragments = Math.ceil(Math.max(Math.min(N_fa, N_fs), 5));
    }
    const num_sections =
        Math.max(1, Math.floor(full_fragments * absAngle / 360));

    let revolved: any;
    if (num_sections < 3) {
      const arc = absAngle * Math.PI / 180;
      revolved = shape.extrude(1, num_sections - 1).warp((v: number[]) => {
        const phi = (1 - v[2]!) * arc;
        const x = v[0];
        const y = v[1];
        v[0] = x * Math.cos(phi);
        v[1] = x * Math.sin(phi);
        v[2] = y;
      });
    } else {
      revolved = shape.revolve(num_sections, absAngle);
    }
    if (angle < 0) revolved = revolved.mirror([0, 1, 0]);
    if (onNegativeSide) revolved = revolved.rotate([0, 0, 180]);
    return revolved;
  }
  // OpenSCAD ignores 3D children of 2D operations
  return Manifold.union([]);
}

function __sampleQuadratic(
    x0: number, y0: number, x1: number, y1: number, x2: number, y2: number,
    steps: number): [number, number][] {
  const pts: [number, number][] = [];
  for (let i = 1; i <= steps; i++) {
    const t = i / steps, mt = 1 - t;
    pts.push([
      mt * mt * x0 + 2 * mt * t * x1 + t * t * x2,
      mt * mt * y0 + 2 * mt * t * y1 + t * t * y2,
    ]);
  }
  return pts;
}

function __sampleCubic(
    x0: number, y0: number, x1: number, y1: number, x2: number, y2: number,
    x3: number, y3: number, steps: number): [number, number][] {
  const pts: [number, number][] = [];
  for (let i = 1; i <= steps; i++) {
    const t = i / steps, mt = 1 - t;
    pts.push([
      mt * mt * mt * x0 + 3 * mt * mt * t * x1 + 3 * mt * t * t * x2 +
          t * t * t * x3,
      mt * mt * mt * y0 + 3 * mt * mt * t * y1 + 3 * mt * t * t * y2 +
          t * t * t * y3,
    ]);
  }
  return pts;
}

function __pathToContours(commands: any[], fn: number): [number, number][][] {
  const steps = Math.max(2, fn > 0 ? Math.round(fn / 4) : 4);
  const contours: [number, number][][] = [];
  let current: [number, number][]|null = null;

  for (const cmd of commands) {
    switch (cmd.type) {
      case 'M':  // Move to — starts a new contour
        if (current && current.length >= 3) contours.push(current);
        current = [[cmd.x, cmd.y]];
        break;

      case 'L':  // Line to
        current?.push([cmd.x, cmd.y]);
        break;

      case 'Q': {  // Quadratic bezier
        if (!current) break;
        const prev = current[current.length - 1]!;
        const pts = __sampleQuadratic(
            prev[0], prev[1], cmd.x1, cmd.y1, cmd.x, cmd.y, steps);
        for (const [px, py] of pts) current.push([px, py]);
        break;
      }

      case 'C': {  // Cubic bezier
        if (!current) break;
        const prev = current[current.length - 1]!;
        const pts = __sampleCubic(
            prev[0], prev[1], cmd.x1, cmd.y1, cmd.x2, cmd.y2, cmd.x, cmd.y,
            steps);
        for (const [px, py] of pts) current.push([px, py]);
        break;
      }

      case 'Z':  // Close contour
        if (current && current.length >= 3) contours.push(current);
        current = null;
        break;
    }
  }
  if (current && current.length >= 3) contours.push(current);
  return contours.map(c => c.map(([x, y]): [number, number] => [x, -y]));
}

function __fontSpecToFilename(fontSpec: string): string {
  const cleaned = fontSpec.replace(/"/g, '').trim();
  const parts = cleaned.split(':');
  const family = (parts[0] || 'Liberation Sans').trim().replace(/\s+/g, '');

  let style = 'Regular';
  for (let i = 1; i < parts.length; i++) {
    const part = parts[i]!.trim();
    const match = part.match(/^style\s*=\s*(.+)$/i);
    if (match) {
      style = match[1]!.trim().replace(/\s+/g, '');
      break;
    }
  }

  return `${family}-${style}`;
}

const __opentypeFontCache = new Map<string, any>();

function __getOpentypeFont(base64DataUrl: string): any|undefined {
  const cached = __opentypeFontCache.get(base64DataUrl);
  if (cached) return cached;

  try {
    const base64 = base64DataUrl.replace(/^data:[^;]+;base64,/, '');
    const binaryStr = atob(base64);
    const bytes = new Uint8Array(binaryStr.length);
    for (let i = 0; i < binaryStr.length; i++) {
      bytes[i] = binaryStr.charCodeAt(i);
    }
    const font = opentypeParse(bytes.buffer as ArrayBuffer);
    __opentypeFontCache.set(base64DataUrl, font);
    return font;
  } catch (e) {
    console.log('err: ', e);
    return undefined;
  }
}

function __opentypeGlyphContours(
    ch: string, font: any, size: number,
    fn: number): {contours: [number, number][][]; width: number}|undefined {
  const fontSize = size * 100 / 72;

  if (ch === ' ') {
    const spaceGlyph = font.charToGlyph(' ');

    const scale = fontSize / font.unitsPerEm;

    const spaceWidth =
        (spaceGlyph?.advanceWidth ?? font.unitsPerEm * 0.25) * scale;

    return {contours: [], width: spaceWidth};
  }

  const glyph = font.charToGlyph(ch);

  if (!glyph || glyph.index === 0) {
    return undefined;
  }

  const scale = fontSize / font.unitsPerEm;

  const advance = (glyph.advanceWidth ?? 0) * scale;

  const glyphPath = glyph.getPath(0, 0, fontSize);

  const commands = glyphPath.commands;

  if (!commands || commands.length === 0) {
    return {contours: [], width: advance};
  }

  const contours = __pathToContours(commands, fn);

  return {contours, width: advance};
}

function __opentypeTextContours(
    chars: string[], size: number, spacing: number, direction: string,
    fontBase64: string, fn: number): [number, number][][]|undefined {
  const font = __getOpentypeFont(fontBase64);

  if (!font) return undefined;

  const contours: [number, number][][] = [];
  const isVertical = direction === 'ttb' || direction === 'btt';

  if (isVertical) {
    const ySign = direction === 'ttb' ? -1 : 1;
    let cursorY = 0;
    for (const ch of chars) {
      const glyph = __opentypeGlyphContours(ch, font, size, fn);
      if (!glyph) return undefined;
      const xOffset = -glyph.width / 2;
      for (const contour of glyph.contours) {
        contours.push(contour.map(
            ([x, y]): [number, number] => [x + xOffset, cursorY + y * ySign]));
      }
      cursorY += size * spacing * ySign;
    }
  } else {
    let cursorX = 0;
    for (const ch of chars) {
      const glyph = __opentypeGlyphContours(ch, font, size, fn);
      if (!glyph) return undefined;
      for (const contour of glyph.contours) {
        contours.push(
            contour.map(([x, y]): [number, number] => [x + cursorX, y]));
      }
      cursorX += glyph.width * spacing;
    }
  }

  return contours;
}

const __canvasGlyphCache =
    new Map<string, {contours: [number, number][][], width: number}>();

function __fontToCss(font: string, px: number): string {
  const spec = String(font || 'Liberation Sans');
  const family = (spec.split(':')[0] || 'Liberation Sans').replace(/"/g, '');
  const styleSpec = spec.toLowerCase();
  const weight = styleSpec.includes('bold') ? '700' : '400';
  const style = styleSpec.includes('italic') ? 'italic' : 'normal';
  return `${style} ${weight} ${px}px "${family}", Arial, sans-serif`;
}

function __canvasForText(): any {
  if (typeof document !== 'undefined' &&
      typeof document.createElement === 'function') {
    return document.createElement('canvas');
  }
  if (typeof OffscreenCanvas !== 'undefined') {
    return new OffscreenCanvas(1, 1);
  }
  return undefined;
}

function __canvasGlyphContours(ch: string, font: string, size: number):
    {contours: [number, number][][], width: number}|undefined {
  if (ch === ' ') return {contours: [], width: size * 0.35};

  const px = 128;
  const cacheKey = `${font}|${ch}|${px}`;
  const cached = __canvasGlyphCache.get(cacheKey);
  if (cached) return cached;

  const canvas = __canvasForText();
  const ctx =
      canvas?.getContext('2d', {willReadFrequently: true} as any) as any;
  if (!canvas || !ctx) return undefined;

  ctx.font = __fontToCss(font, px);
  const metrics = ctx.measureText(ch);
  const ascent = Math.ceil(metrics.actualBoundingBoxAscent || px * 0.8);
  const descent = Math.ceil(metrics.actualBoundingBoxDescent || px * 0.25);
  const leftBearing = Math.ceil(metrics.actualBoundingBoxLeft || 0);
  const rightBearing =
      Math.ceil(metrics.actualBoundingBoxRight || metrics.width || px * 0.6);
  const pad = 8;
  const widthPx = Math.max(1, Math.ceil(leftBearing + rightBearing + pad * 2));
  const heightPx = Math.max(1, ascent + descent + pad * 2);

  canvas.width = widthPx;
  canvas.height = heightPx;
  ctx.clearRect(0, 0, widthPx, heightPx);
  ctx.font = __fontToCss(font, px);
  ctx.fillStyle = '#fff';
  ctx.textBaseline = 'alphabetic';
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
  const result = {
    contours: [] as [number, number][][],
    width: Math.max(1, metrics.width) * scale
  };
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

function __canvasTextContours(
    chars: string[], size: number, spacing: number, direction: string,
    font: string): [number, number][][]|undefined {
  if (typeof document === 'undefined' && typeof OffscreenCanvas === 'undefined')
    return undefined;

  const contours: [number, number][][] = [];
  const isVertical = direction === 'ttb' || direction === 'btt';
  if (isVertical) {
    const ySign = direction === 'ttb' ? -1 : 1;
    let cursorY = 0;
    for (const ch of chars) {
      const glyph = __canvasGlyphContours(ch, font, size);
      if (!glyph) return undefined;
      const xOffset = -glyph.width / 2;
      for (const contour of glyph.contours) {
        contours.push(contour.map(
            ([x, y]): [number, number] => [x + xOffset, cursorY + y * ySign]));
      }
      cursorY += size * spacing * ySign;
    }
  } else {
    let cursorX = 0;
    for (const ch of chars) {
      const glyph = __canvasGlyphContours(ch, font, size);
      if (!glyph) return undefined;
      for (const contour of glyph.contours) {
        contours.push(
            contour.map(([x, y]): [number, number] => [x + cursorX, y]));
      }
      cursorX += glyph.width * spacing;
    }
  }

  return contours;
}

function __text(
    text: string, size: number = 10, font: string, halign: string = 'left',
    valign: string = 'baseline', spacing: number = 1, direction: string = 'ltr',
    fn: number = 0,
    fontBase64Data: string|Record<string, string>|undefined = undefined): any {
  if (!text || text.length === 0)
    return CrossSection.square([0.001, 0.001], false);

  void fn;

  const dir = (direction || 'ltr').toLowerCase();
  const chars = dir === 'rtl' ? Array.from(text).reverse() : Array.from(text);

  let contours: [number, number][][]|undefined;

  contours = __canvasTextContours(chars, size, spacing, dir, font);

  if (!contours && fontBase64Data) {
    let base64: string|undefined;
    if (typeof fontBase64Data === 'string') {
      base64 = fontBase64Data;
    } else if (typeof fontBase64Data === 'object' && fontBase64Data !== null) {
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
  if (halign === 'center')
    dx = -(left + right) / 2;
  else if (halign === 'right')
    dx = -right;
  else
    dx = 0;

  let dy = 0;
  if (valign === 'center')
    dy = -(top + bottom) / 2;
  else if (valign === 'top')
    dy = -top;
  else if (valign === 'bottom')
    dy = -bottom;

  const shifted =
      contours.map(c => c.map(([x, y]): [number, number] => [x + dx, y + dy]));
  return CrossSection.ofPolygons(shifted, 'EvenOdd');
}

function __sync_quality(fa: any, fs: any) {
  if (typeof wasm.setMinCircularAngle === 'function') {
    if (typeof fa === 'number' && fa > 0) {
      wasm.setMinCircularAngle(fa);
    }
  }
  if (typeof wasm.setMinCircularEdgeLength === 'function') {
    if (typeof fs === 'number' && fs > 0) {
      wasm.setMinCircularEdgeLength(fs);
    }
  }
}

// Only accepts actual finite numbers here; strings like "45", undef, bools,
// etc. silently become 0
function __rot_angle(x: any): number {
  return (typeof x === 'number' && isFinite(x)) ? x : 0;
}

function __rotate(shape: any, a: any, v?: any) {
  if (!shape) return shape;

  // Vector angle: XYZ euler rotation. OpenSCAD ignores 'v' entirely, uses only
  // the first three elements, and treats invalid ones as 0
  if (Array.isArray(a)) {
    const ex = __rot_angle(a[0]);
    const ey = __rot_angle(a[1]);
    const ez = __rot_angle(a[2]);
    if (__is2D(shape)) {
      return shape.rotate(ez);
    }
    return shape.rotate([ex, ey, ez]);
  }

  // Scalar angle: rotate about axis 'v'. Defaults to Z and keeps that default
  // when 'v' is not a valid 2/3-element numeric vector
  const angle = __rot_angle(a);
  let vx = 0, vy = 0, vz = 1;
  if (Array.isArray(v) && (v.length === 2 || v.length === 3) &&
      v.every((c: any) => typeof c === 'number' && isFinite(c))) {
    vx = v[0];
    vy = v[1];
    vz = v.length === 3 ? v[2] : 0;
  } else if (
      v !== undefined && v !== null && !Array.isArray(v) &&
      typeof v === 'object') {
    vx = Number(v.x) || 0;
    vy = Number(v.y) || 0;
    vz = Number(v.z) || 0;
  }

  const len = Math.sqrt(vx * vx + vy * vy + vz * vz);
  if (len < 1e-9) return shape;

  if (__is2D(shape)) {
    if (Math.abs(vz) > 1e-9) {
      return shape.rotate(angle * Math.sign(vz));
    }
    return shape;
  }

  if (vx === 0 && vy === 0) {
    return shape.rotate([0, 0, angle * Math.sign(vz)]);
  }

  const theta = angle * Math.PI / 180;
  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);
  const oneMinusCosT = 1 - cosT;

  const ux = vx / len;
  const uy = vy / len;
  const uz = vz / len;

  const R = [
    [
      oneMinusCosT * ux * ux + cosT, oneMinusCosT * ux * uy - sinT * uz,
      oneMinusCosT * ux * uz + sinT * uy, 0
    ],
    [
      oneMinusCosT * ux * uy + sinT * uz, oneMinusCosT * uy * uy + cosT,
      oneMinusCosT * uy * uz - sinT * ux, 0
    ],
    [
      oneMinusCosT * ux * uz - sinT * uy, oneMinusCosT * uy * uz + sinT * ux,
      oneMinusCosT * uz * uz + cosT, 0
    ],
    [0, 0, 0, 1]
  ];

  return __safe_transform(shape, R);
}

function __translate(shape: any, v: any) {
  if (!shape) return shape;
  if (__is2D(shape)) {
    let x = 0, y = 0;
    if (Array.isArray(v)) {
      x = Number(v[0]) || 0;
      y = Number(v[1]) || 0;
    } else if (v && typeof v === 'object') {
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
    } else if (v && typeof v === 'object') {
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
    } else if (v && typeof v === 'object') {
      x = (v.x !== undefined ? v.x : v[0]) !== undefined ?
          Number(v.x !== undefined ? v.x : v[0]) :
          1;
      y = (v.y !== undefined ? v.y : v[1]) !== undefined ?
          Number(v.y !== undefined ? v.y : v[1]) :
          1;
    } else if (typeof v === 'number' && !Number.isNaN(v)) {
      x = y = v;
    }
    return shape.scale([x, y]);
  } else {
    let x = 1, y = 1, z = 1;
    if (Array.isArray(v)) {
      x = v[0] !== undefined && v[0] !== null ? Number(v[0]) : 1;
      y = v[1] !== undefined && v[1] !== null ? Number(v[1]) : 1;
      z = v[2] !== undefined && v[2] !== null ? Number(v[2]) : 1;
    } else if (v && typeof v === 'object') {
      x = (v.x !== undefined ? v.x : v[0]) !== undefined ?
          Number(v.x !== undefined ? v.x : v[0]) :
          1;
      y = (v.y !== undefined ? v.y : v[1]) !== undefined ?
          Number(v.y !== undefined ? v.y : v[1]) :
          1;
      z = (v.z !== undefined ? v.z : v[2]) !== undefined ?
          Number(v.z !== undefined ? v.z : v[2]) :
          1;
    } else if (typeof v === 'number' && !Number.isNaN(v)) {
      x = y = z = v;
    }
    return shape.scale([x, y, z]);
  }
}

// Scale about the origin to fit `newsize`, preserving zero-sized axes unless
// `auto` is enabled, with the auto scale taken from the largest requested
// dimension
function __resize(shape: any, newsizeRaw: any, autoRaw: any) {
  if (!shape || __isEmpty(shape)) return shape;

  const newsize = [0, 0, 0];
  if (is_list_fn(newsizeRaw)) {
    for (let i = 0; i < 3 && i < newsizeRaw.length; i++) {
      const n = Number(newsizeRaw[i]);
      if (Number.isFinite(n)) newsize[i] = n;
    }
  }

  const autosize = [false, false, false];
  if (is_list_fn(autoRaw)) {
    for (let i = 0; i < 3 && i < autoRaw.length; i++)
      autosize[i] = __truthy(autoRaw[i]);
  } else if (typeof autoRaw === 'boolean') {
    autosize[0] = autosize[1] = autosize[2] = autoRaw;
  }

  const is2D = __is2D(shape);
  const dim = is2D ? 2 : 3;
  const bb = is2D ? shape.bounds() : shape.boundingBox();
  const bboxSize = [0, 0, 0];
  for (let i = 0; i < dim; i++) bboxSize[i] = bb.max[i] - bb.min[i];

  // Non-positive `newsize` components are treated as unspecified, and `auto`
  // uses the scale of the largest requested dimension
  let maxIdx = 0;
  for (let i = 1; i < 3; i++) {
    if (newsize[i]! > newsize[maxIdx]!) maxIdx = i;
  }
  const scale = [1, 1, 1];
  for (let i = 0; i < dim; i++) {
    if (newsize[i]! > 0) {
      if (bboxSize[i] === 0) {
        console.warn(
            'WARNING: Resize in direction normal to flat object is not implemented');
        return shape;
      }
      scale[i] = newsize[i]! / bboxSize[i]!;
    }
  }
  const autoscale = scale[maxIdx]!;
  for (let i = 0; i < dim; i++) {
    if (autosize[i] && !(newsize[i]! > 0)) scale[i] = autoscale;
  }

  return is2D ? shape.scale([scale[0], scale[1]]) : shape.scale(scale);
}

// Build the reflection matrix to avoid rounding artifacts, and treat a
// zero-length normal as an identity transform
function __mirror(shape: any, v: any) {
  if (!shape) return shape;
  if (__is2D(shape)) {
    let x = 0, y = 0;
    if (Array.isArray(v)) {
      x = Number(v[0]) || 0;
      y = Number(v[1]) || 0;
    } else if (v && typeof v === 'object') {
      x = Number(v.x || v[0]) || 0;
      y = Number(v.y || v[1]) || 0;
    } else {
      x = Number(v) || 0;
    }
    const normSq = x * x + y * y;
    if (normSq === 0) return shape;
    const d = 2 / normSq;
    return __safe_transform(shape, [
      [1 - d * x * x, -d * x * y, 0, 0],
      [-d * x * y, 1 - d * y * y, 0, 0],
      [0, 0, 1, 0],
    ]);
  } else {
    let x = 0, y = 0, z = 0;
    if (Array.isArray(v)) {
      x = Number(v[0]) || 0;
      y = Number(v[1]) || 0;
      z = Number(v[2]) || 0;
    } else if (v && typeof v === 'object') {
      x = Number(v.x || v[0]) || 0;
      y = Number(v.y || v[1]) || 0;
      z = Number(v.z || v[2]) || 0;
    } else {
      x = Number(v) || 0;
    }
    const normSq = x * x + y * y + z * z;
    if (normSq === 0) return shape;
    const d = 2 / normSq;
    return __safe_transform(shape, [
      [1 - d * x * x, -d * x * y, -d * x * z, 0],
      [-d * x * y, 1 - d * y * y, -d * y * z, 0],
      [-d * x * z, -d * y * z, 1 - d * z * z, 0],
    ]);
  }
}

function __cube(size: any, center = false) {
  // Invalid or `undef` `size` uses the default (1,1,1), while only valid but
  // degenerate sizes produce empty geometry
  let v: number[] = [1, 1, 1];
  if (size !== undefined && size !== null) {
    if (typeof size === 'number') {
      v = [size, size, size];
    } else if (
        is_list_fn(size) && size.length === 3 &&
        size.every((x: any) => typeof x === 'number')) {
      v = [size[0], size[1], size[2]];
    }
  }
  // A non-finite or non-positive dimension yields no geometry instead of
  // crashing
  if (v.some((x) => !Number.isFinite(x) || x <= 0)) {
    return Manifold.union([]);
  }
  return Manifold.cube(v as [number, number, number], center);
}

function __square(size: any, center = false) {
  // Invalid or `undef` `size` uses the default (1,1), while only valid but
  // degenerate sizes produce empty geometry
  let v: number[] = [1, 1];
  if (size !== undefined && size !== null) {
    if (typeof size === 'number') {
      v = [size, size];
    } else if (
        is_list_fn(size) && size.length === 2 &&
        size.every((x: any) => typeof x === 'number')) {
      v = [size[0], size[1]];
    }
  }
  // A non-finite or non-positive dimension yields no geometry instead of
  // crashing
  if (v.some((x) => !Number.isFinite(x) || x <= 0)) {
    return CrossSection.square(0);
  }
  return CrossSection.square(v as [number, number], center);
}

function __sphere(radius: number, fn = 0, fa = 12, fs = 2) {
  // A non-finite (or non-positive) size produces no geometry instead of
  // crashing
  if (!Number.isFinite(radius) || radius <= 0) {
    return Manifold.union([]);
  }
  let N: number;
  if (fn > 0) {
    // Takes the explicit-$fn path whenever $fn > 0, clamping to a minimum of 3
    // - a non-finite $fn is clamped to that minimum
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
  const def = (x: any) => x !== undefined && x !== null &&
      !(typeof x === 'number' && Number.isNaN(x));
  if (def(dSpec)) return dSpec / 2;
  if (def(rSpec)) return rSpec;
  if (def(dGen)) return dGen / 2;
  if (def(rGen)) return rGen;
  return dflt;
}

function __cylinder(
    height: number, radiusLow: number, radiusHigh = -1.0, fn = 0,
    center = false, fa = 12, fs = 2) {
  // Non-finite dimensions produce no geometry
  if (!Number.isFinite(height) || !Number.isFinite(radiusLow) ||
      (radiusHigh >= 0 && !Number.isFinite(radiusHigh))) {
    return Manifold.union([]);
  }
  let segs = fn;
  if (segs > 0) {
    // OpenSCAD clamps $fn (including non-finite) to a minimum of 3 fragments
    // and truncates
    segs = Number.isFinite(segs) && segs >= 3 ? Math.floor(segs) : 3;
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

function __forceWinding(
    contour: [number, number][], ccw: boolean): [number, number][] {
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
    return CrossSection.square(0);
  }

  // A point with a non-finite coordinate yields no geometry
  if (points.some(
          (p: any) => Array.isArray(p) &&
              p.some((c: any) => !Number.isFinite(Number(c))))) {
    return CrossSection.square(0);
  }

  if (paths === undefined || paths === null) {
    const ccwPoints = __forceWinding([...points], true);
    return CrossSection.ofPolygons([ccwPoints]);
  }

  if (Array.isArray(paths) && paths.length > 0 && !Array.isArray(paths[0])) {
    const contour =
        paths.map((idx: any) => points[Number(idx) || 0]).filter(Boolean);
    const ccwContour = __forceWinding(contour, true);
    return CrossSection.ofPolygons([ccwContour]);
  }

  if (Array.isArray(paths)) {
    const contours =
        paths
            .map((path: any) => {
              let contour: any[] = [];
              if (Array.isArray(path)) {
                contour = path.map((idx: any) => points[Number(idx) || 0])
                              .filter(Boolean);
              } else if (typeof path === 'number') {
                contour = [points[path]];
              }
              return contour;
            })
            .filter((c: any) => c.length > 0);
    // Use EvenOdd fill rule - OpenSCAD polygon() does not assume outer/hole
    // winding
    return CrossSection.ofPolygons(contours, 'EvenOdd');
  }

  const ccwPoints = __forceWinding([...points], true);
  return CrossSection.ofPolygons([ccwPoints]);
}

function __polyhedron(points: any, faces: any) {
  // A point with a non-finite coordinate yields no geometry
  if (Array.isArray(points) &&
      points.some(
          (p: any) => Array.isArray(p) &&
              p.some((c: any) => !Number.isFinite(Number(c))))) {
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
      const idx = face.map((i: any) => Number(i));
      if (idx.length === 3) {
        tris.push(idx[2]!, idx[1]!, idx[0]!);
        continue;
      }

      // Properly triangulate concave faces to avoid overlapping triangles,
      // falling back to a simple fan only for degenerate faces
      let done = false;
      const pts = idx.map(
          (i) =>
              [verts[i * 3] ?? 0, verts[i * 3 + 1] ?? 0,
               verts[i * 3 + 2] ?? 0]);
      // Newell normal follows the face's winding order
      let nx = 0, ny = 0, nz = 0;
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i]!, q = pts[(i + 1) % pts.length]!;
        nx += (p[1]! - q[1]!) * (p[2]! + q[2]!);
        ny += (p[2]! - q[2]!) * (p[0]! + q[0]!);
        nz += (p[0]! - q[0]!) * (p[1]! + q[1]!);
      }
      const nLen = Math.hypot(nx, ny, nz);
      if (nLen > 1e-12) {
        nx /= nLen;
        ny /= nLen;
        nz /= nLen;
        let ux: number, uy: number, uz: number;
        if (Math.abs(nx) > Math.abs(nz)) {
          ux = -ny;
          uy = nx;
          uz = 0;
        } else {
          ux = 0;
          uy = -nz;
          uz = ny;
        }
        const uLen = Math.hypot(ux, uy, uz);
        ux /= uLen;
        uy /= uLen;
        uz /= uLen;
        const vx = ny * uz - nz * uy, vy = nz * ux - nx * uz,
              vz = nx * uy - ny * ux;
        const poly2d = pts.map(
            (p) =>
                [p[0]! * ux + p[1]! * uy + p[2]! * uz,
                 p[0]! * vx + p[1]! * vy + p[2]! * vz,
        ]);
        try {
          for (const t of wasm.triangulate([poly2d])) {
            tris.push(idx[t[2]!]!, idx[t[1]!]!, idx[t[0]!]!);
          }
          done = true;
        } catch { /* self-intersecting or otherwise invalid face: use the fan */
        }
      }
      if (!done) {
        for (let i = 1; i + 1 < idx.length; i++) {
          tris.push(idx[i + 1]!, idx[i]!, idx[0]!);
        }
      }
    }
  }
  const mesh = new wasm.Mesh({
    numProp: 3,
    vertProperties: new Float32Array(verts),
    triVerts: new Uint32Array(tris),
  });
  // OpenSCAD allows duplicate per-face vertices, but Manifold requires shared
  // edges, so weld coincident vertices before building the solid.
  mesh.merge();
  try {
    return new Manifold(mesh);
  } catch (e: any) {
    // OpenSCAD treats a non-manifold polyhedron as a soft error, produces empty
    // geometry
    console.warn('WARNING: Polyhedron is not manifold, object will be empty');
    return Manifold.union([]);
  }
}

function __parse_color_for_scope(c: any, alpha: any): any {
  const base = __parse_color_value(c);
  if (!base) return undefined;
  const a = (alpha !== undefined && alpha !== null &&
             Number.isFinite(Number(alpha))) ?
      Number(alpha) :
      base[3];
  return [base[0], base[1], base[2], a];
}


async function gridFromImage(dataUrl: string, invert: boolean): Promise<{
  width: number; height: number; Z: (x: number, y: number) => number;
  minVal: number
}> {
  const {width, height, data} = await decodeImageToPixels(dataUrl);
  // Flip the image vertically and apply `invert` as `1 - pixel`, even if it
  // produces negative values
  const Z = (x: number, y: number): number => {
    const i = ((height - 1 - y) * width + x) * 4;
    const gray =
        0.2126 * data![i]! + 0.7152 * data![i + 1]! + 0.0722 * data![i + 2]!;
    return (100 / 255) * (invert ? 1 - gray : gray);
  };
  let minVal = 200;
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) minVal = Math.min(minVal, Z(x, y));
  return {width, height, Z, minVal};
}

function gridFromText(text: string): {
  width: number; height: number; Z: (x: number, y: number) => number;
  minVal: number
} {
  const rows: number[][] = [];
  // Only parsed values affect the minimum height, so zero-filled gaps never
  // raise the floor above z=0
  let minVal = 1;
  for (const line of text.split(/\r?\n/)) {
    const t = line.trim();
    // skip blanks, '#' and '%' comment lines
    if (t === '' || t.startsWith('#') || t.startsWith('%')) continue;
    const vals = t.split(/\s+/).map(Number).filter(v => Number.isFinite(v));
    if (vals.length) {
      rows.push(vals);
      for (const v of vals) minVal = Math.min(minVal, v);
    }
  }
  if (rows.length === 0) throw new Error('__surface: empty data file');
  const height = rows.length;
  const width = Math.max(...rows.map(r => r.length));
  // Reads row 0 as the first data row; row index -> Y, column index -> X. Value
  // used directly as Z (no normalization)
  const Z = (x: number, y: number): number => rows[y]?.[x] ?? 0;
  return {width, height, Z, minVal};
}

async function __surface(source: string, opts: {
  center?: boolean;
  invert?: boolean;
  kind?: 'image' | 'text';
  fn?: number;
  fa?: number;
  fs?: number
} = {}) {
  const {center = false, invert = false, kind = 'image'} = opts;
  // OpenSCAD only honors a bool invert, and only for images
  const grid = kind === 'text' ? gridFromText(source) :
                                 await gridFromImage(source, invert === true);
  return buildSurfaceMesh(grid, center);
}

function buildSurfaceMesh(
    {width, height, Z, minVal}: {
      width: number; height: number; Z: (x: number, y: number) => number;
      minVal: number
    },
    center: boolean) {
  const ox = center ? -(width - 1) / 2 : 0;
  const oy = center ? -(height - 1) / 2 : 0;

  const zFloor = minVal - 1;

  const numTop = width * height;
  const numQuads = (width - 1) * (height - 1);
  const vertProps: number[] = [];
  const tris: number[] = [];

  const topIdx = (x: number, y: number) => y * width + x;
  const centerIdx = (x: number, y: number) => numTop + y * (width - 1) + x;
  const botIdx = (x: number, y: number) => numTop + numQuads + y * width + x;

  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) vertProps.push(x + ox, y + oy, Z(x, y));

  for (let y = 0; y < height - 1; y++)
    for (let x = 0; x < width - 1; x++) {
      const zc = (Z(x, y) + Z(x + 1, y) + Z(x, y + 1) + Z(x + 1, y + 1)) / 4;
      vertProps.push(x + 0.5 + ox, y + 0.5 + oy, zc);
    }

  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) vertProps.push(x + ox, y + oy, zFloor);

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
    tris.push(
        topIdx(x, 0), botIdx(x, 0), topIdx(x + 1, 0), topIdx(x + 1, 0),
        botIdx(x, 0), botIdx(x + 1, 0));
  const yb = height - 1;
  for (let x = 0; x < width - 1; x++)
    tris.push(
        topIdx(x, yb), topIdx(x + 1, yb), botIdx(x, yb), topIdx(x + 1, yb),
        botIdx(x + 1, yb), botIdx(x, yb));
  for (let y = 0; y < height - 1; y++)
    tris.push(
        topIdx(0, y), topIdx(0, y + 1), botIdx(0, y), topIdx(0, y + 1),
        botIdx(0, y + 1), botIdx(0, y));
  const xr = width - 1;
  for (let y = 0; y < height - 1; y++)
    tris.push(
        topIdx(xr, y), botIdx(xr, y), topIdx(xr, y + 1), topIdx(xr, y + 1),
        botIdx(xr, y), botIdx(xr, y + 1));

  return new Manifold(new wasm.Mesh({
    vertProperties: new Float32Array(vertProps),
    triVerts: new Uint32Array(tris),
    numProp: 3,
  }));
}


async function decodeImageToPixels(dataUrl: string):
    Promise<{width: number; height: number; data: Uint8ClampedArray;}> {
  if (typeof OffscreenCanvas !== 'undefined') {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const canvas = new OffscreenCanvas(img.width, img.height);
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(img, 0, 0);
        const {data, width, height} =
            ctx.getImageData(0, 0, img.width, img.height);
        resolve({width, height, data});
      };
      img.onerror = () =>
          reject(new Error('__surface: failed to decode image'));
      img.src = dataUrl;
    });
  } else {
    const img = await loadImage(dataUrl);
    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    const {data, width, height} = ctx.getImageData(0, 0, img.width, img.height);
    return {width, height, data: data as unknown as Uint8ClampedArray};
  }
}

function pow_fn(base: any, exp: any) {
  return Math.pow(base, exp);
}

// Export all runtime symbols for compiled code
export {
  Manifold,
  CrossSection,
  wasm,
  __cube,
  __square,
  __sphere,
  __cylinder,
  __circle,
  __radius,
  __rotate,
  __polygon,
  __polyhedron,
  __translate,
  __scale,
  __mirror,
  __resize,
  is_undef_fn,
  is_bool_fn,
  is_num_fn,
  is_string_fn,
  is_list_fn,
  is_function_fn,
  __unknown_fn,
  sin_fn,
  cos_fn,
  tan_fn,
  asin_fn,
  acos_fn,
  atan_fn,
  atan2_fn,
  abs_fn,
  sign_fn,
  floor_fn,
  ceil_fn,
  round_fn,
  sqrt_fn,
  exp_fn,
  ln_fn,
  log_fn,
  pow_fn,
  min_fn,
  max_fn,
  norm_fn,
  cross_fn,
  len_fn,
  str_fn,
  chr_fn,
  ord_fn,
  concat_fn,
  search_fn,
  lookup_fn,
  rands_fn,
  openscad_assert_fn,
  __truthy,
  __eq,
  __lt,
  __gt,
  __le,
  __ge,
  __add,
  __sub,
  __mul,
  __div,
  __mod,
  __band,
  __bor,
  __shl,
  __shr,
  __bnot,
  __neg,
  __pos,
  __index,
  version_fn,
  version_num_fn,
  PI,
  INF,
  NAN,
  undef,
  _EPSILON,
  __ctx,
  __withSpecials,
  __children_stack,
  __with_children,
  __pick_children,
  parent_module_fn,
  __is_finite_matrix4,
  __to_manifold_mat4,
  __safe_transform,
  __identity4,
  __safe_offset2d,
  __safe_project3d,
  __apply_color,
  __each,
  __flat_map_iter,
  __range,
  __rangeCount,
  __is2D,
  __union2d3d,
  __difference2d3d,
  __intersection2d3d,
  __hull2d3d,
  __minkowski2d3d,
  __rootMod,
  __applyRoot,
  __extrude,
  __revolve,
  __text,
  __parse_color_for_scope,
  __surface,
  __echo,
  __oecho,
  __fnlit,
  __tc,
  __call
};
