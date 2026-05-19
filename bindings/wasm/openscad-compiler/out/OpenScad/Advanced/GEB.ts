import * as __rt from "../../runtime.js";
const { Manifold, CrossSection, wasm, is_undef_fn, is_bool_fn, is_num_fn, is_string_fn, is_list_fn, is_function_fn, sin_fn, cos_fn, tan_fn, asin_fn, acos_fn, atan_fn, atan2_fn, abs_fn, sign_fn, floor_fn, ceil_fn, round_fn, sqrt_fn, exp_fn, ln_fn, log_fn, min_fn, max_fn, norm_fn, cross_fn, len_fn, str_fn, chr_fn, ord_fn, concat_fn, search_fn, lookup_fn, openscad_assert_fn, __eq, __add, __sub, __mul, __div, __mod, __neg, __pos, version_fn, version_num_fn, __children_stack, __with_children, __is_finite_matrix4, __to_manifold_mat4, __safe_transform, __identity4, __safe_attach_transform, __safe_offset2d, __safe_project3d, __apply_color, __flat_map_iter, __range, __union2d3d, __difference2d3d, __intersection2d3d, __hull2d3d, __minkowski2d3d, __extrude, __revolve } = __rt;
var PI: any = __rt.PI;
var INF: any = __rt.INF;
var NAN: any = __rt.NAN;
var undef: any = __rt.undef;
var _EPSILON: any = __rt._EPSILON;

var font: any = "Liberation Sans";
// Nicer, but not generally installed:
// font = "Bank Gothic";
function G$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(__safe_offset2d(((__t: any, __s: any) => CrossSection.square([Math.max(0.001, len_fn(str_fn(__t)) * __s * 0.6), Math.max(0.001, __s)], false))("G", 10), 0.3));
  return __union2d3d(__items);
}
function E$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(__safe_offset2d(((__t: any, __s: any) => CrossSection.square([Math.max(0.001, len_fn(str_fn(__t)) * __s * 0.6), Math.max(0.001, __s)], false))("E", 10), 0.3));
  return __union2d3d(__items);
}
function B$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(__safe_offset2d(((__t: any, __s: any) => CrossSection.square([Math.max(0.001, len_fn(str_fn(__t)) * __s * 0.6), Math.max(0.001, __s)], false))("B", 10), 0.5));
  return __union2d3d(__items);
}
var $fn: any = 64;
function GEB$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(__intersection2d3d([
  __extrude(__with_children(() => Manifold.union([]), 0, () => B$mod()), 20),
  __extrude(__with_children(() => Manifold.union([]), 0, () => E$mod()), 20).rotate([90, 0, 0]),
  __extrude(__with_children(() => Manifold.union([]), 0, () => G$mod()), 20).rotate([90, 0, 90])
]));
  return __union2d3d(__items);
}

var $fn: any = 0, $fa: any = 12, $fs: any = 2;
var $vpr: any = [0, 0, 0], $vpt: any = [0, 0, 0], $vpd: any = 500;
var $parent_modules: any = 0;
var _NO_ARG: any = Symbol("NO_ARG");
const __result_items: any[] = [];
__result_items.push(__apply_color(__with_children(() => Manifold.union([]), 0, () => GEB$mod()), "Ivory", undefined));
__result_items.push(__apply_color(__extrude(__difference2d3d(((__s: any) => CrossSection.square((is_list_fn(__s) ? __s : [__s, __s]), true))(40), [
  __safe_project3d(__with_children(() => Manifold.union([]), 0, () => GEB$mod()))
]), 1).translate([0, 0, __neg(20)]), "MediumOrchid", undefined));
__result_items.push(__apply_color(__extrude(__difference2d3d(((__s: any) => CrossSection.square((is_list_fn(__s) ? __s : [__s, __s]), true))([40, 39]).translate([0, 0.5]), [
  __safe_project3d(__with_children(() => Manifold.union([]), 0, () => GEB$mod()).rotate([__neg(90), 0, 0]))
]), 1).translate([0, 0, __neg(20)]).rotate([90, 0, 0]), "DarkMagenta", undefined));
__result_items.push(__apply_color(__extrude(__difference2d3d(((__s: any) => CrossSection.square((is_list_fn(__s) ? __s : [__s, __s]), true))([39, 39]).translate([__neg(0.5), 0.5]), [
  __safe_project3d(__with_children(() => Manifold.union([]), 0, () => GEB$mod()).rotate([0, __neg(90), __neg(90)]))
]), 1).translate([0, 0, __neg(20)]).rotate([90, 0, 90]), "MediumSlateBlue", undefined));
__result_items.push((console.log("version = ", version_fn()), Manifold.union([])));
export const result = __union2d3d(__result_items);