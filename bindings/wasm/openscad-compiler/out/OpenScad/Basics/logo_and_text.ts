import * as __rt from "../../runtime.js";
const { Manifold, CrossSection, wasm, is_undef_fn, is_bool_fn, is_num_fn, is_string_fn, is_list_fn, is_function_fn, sin_fn, cos_fn, tan_fn, asin_fn, acos_fn, atan_fn, atan2_fn, abs_fn, sign_fn, floor_fn, ceil_fn, round_fn, sqrt_fn, exp_fn, ln_fn, log_fn, min_fn, max_fn, norm_fn, cross_fn, len_fn, str_fn, chr_fn, ord_fn, concat_fn, search_fn, lookup_fn, openscad_assert_fn, __eq, __add, __sub, __mul, __div, __mod, __neg, __pos, version_fn, version_num_fn, __children_stack, __with_children, __is_finite_matrix4, __to_manifold_mat4, __safe_transform, __identity4, __safe_attach_transform, __safe_offset2d, __safe_project3d, __apply_color, __flat_map_iter, __range, __union2d3d, __difference2d3d, __intersection2d3d, __hull2d3d, __minkowski2d3d, __extrude, __revolve } = __rt;
var PI: any = __rt.PI;
var INF: any = __rt.INF;
var NAN: any = __rt.NAN;
var undef: any = __rt.undef;
var _EPSILON: any = __rt._EPSILON;

// The $fn parameter will influence all objects inside this module
// It can, optionally, be overridden when instantiating the module
function Logo$mod(size: any = 50, $fn: any = 100): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  // Temporary variables
  var hole: any = __div(size, 2);
  var cylinderHeight: any = __mul(size, 1.25);
  // One positive object (sphere) and three negative objects (cylinders)
  __items.push(__difference2d3d(Manifold.sphere((size) / 2), [
  Manifold.cylinder(cylinderHeight, (hole) / 2, (hole) / 2, 0, true),
  // The '#' operator highlights the object
Manifold.cylinder(cylinderHeight, (hole) / 2, (hole) / 2, 0, true).rotate([90, 0, 0]),
  Manifold.cylinder(cylinderHeight, (hole) / 2, (hole) / 2, 0, true).rotate([0, 90, 0])
]));
  return __union2d3d(__items);
}
// Set the initial viewport parameters
var $vpr: any = [90, 0, 0];
var $vpt: any = [300, 0, 80];
var $vpd: any = 1600;
var logosize: any = 120;
// Helper to create 3D text with correct font and orientation
function t$mod(t: any, s: any = 18, style: any = ":style=Bold", spacing: any = 1): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(__extrude(((__t: any, __s: any) => CrossSection.square([Math.max(0.001, len_fn(str_fn(__t)) * __s * 0.6), Math.max(0.001, __s)], false))(t, s), 1).rotate([90, 0, 0]));
  return __union2d3d(__items);
}
// Color helpers
function green$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(__apply_color(children(), [__div(157, 255), __div(203, 255), __div(81, 255)], undefined));
  return __union2d3d(__items);
}
function corn$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(__apply_color(children(), [__div(249, 255), __div(210, 255), __div(44, 255)], undefined));
  return __union2d3d(__items);
}
function black$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(__apply_color(children(), [0, 0, 0], undefined));
  return __union2d3d(__items);
}

var $fn: any = 0, $fa: any = 12, $fs: any = 2;
var $vpr: any = [0, 0, 0], $vpt: any = [0, 0, 0], $vpd: any = 500;
var $parent_modules: any = 0;
var _NO_ARG: any = Symbol("NO_ARG");
const __result_items: any[] = [];
__result_items.push(__union2d3d([
  __with_children(() => Manifold.union([]), 0, () => Logo$mod(logosize)).rotate([25, 25, __neg(40)]).translate([0, 0, 30]),
  (() => { const __childFns = [
  () => (__with_children(() => Manifold.union([]), 0, () => t$mod("Open", /* s = */ 42, undefined, /* spacing = */ 1.05)))
]; return __with_children((i) => (i === undefined ? __union2d3d(__childFns.map(fn => fn())) : ((i >= 0 && i < __childFns.length) ? __childFns[i]() : Manifold.union([]))), __childFns.length, () => green$mod()); })().translate([100, 0, 40]),
  (() => { const __childFns = [
  () => (__with_children(() => Manifold.union([]), 0, () => t$mod("SCAD", /* s = */ 42, undefined, /* spacing = */ 0.9)))
]; return __with_children((i) => (i === undefined ? __union2d3d(__childFns.map(fn => fn())) : ((i >= 0 && i < __childFns.length) ? __childFns[i]() : Manifold.union([]))), __childFns.length, () => corn$mod()); })().translate([247, 0, 40]),
  (() => { const __childFns = [
  () => (__with_children(() => Manifold.union([]), 0, () => t$mod("The Programmers")))
]; return __with_children((i) => (i === undefined ? __union2d3d(__childFns.map(fn => fn())) : ((i >= 0 && i < __childFns.length) ? __childFns[i]() : Manifold.union([]))), __childFns.length, () => black$mod()); })().translate([100, 0, 0]),
  (() => { const __childFns = [
  () => (__with_children(() => Manifold.union([]), 0, () => t$mod("Solid 3D CAD Modeller")))
]; return __with_children((i) => (i === undefined ? __union2d3d(__childFns.map(fn => fn())) : ((i >= 0 && i < __childFns.length) ? __childFns[i]() : Manifold.union([]))), __childFns.length, () => black$mod()); })().translate([160, 0, __neg(30)])
]).translate([110, 0, 80]));
__result_items.push((console.log("version = ", version_fn()), Manifold.union([])));
export const result = __union2d3d(__result_items);