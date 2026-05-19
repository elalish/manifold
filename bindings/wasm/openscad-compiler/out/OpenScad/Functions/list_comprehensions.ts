import * as __rt from "../../runtime.js";
const { Manifold, CrossSection, wasm, is_undef_fn, is_bool_fn, is_num_fn, is_string_fn, is_list_fn, is_function_fn, sin_fn, cos_fn, tan_fn, asin_fn, acos_fn, atan_fn, atan2_fn, abs_fn, sign_fn, floor_fn, ceil_fn, round_fn, sqrt_fn, exp_fn, ln_fn, log_fn, min_fn, max_fn, norm_fn, cross_fn, len_fn, str_fn, chr_fn, ord_fn, concat_fn, search_fn, lookup_fn, openscad_assert_fn, __eq, __add, __sub, __mul, __div, __mod, __neg, __pos, version_fn, version_num_fn, __children_stack, __with_children, __is_finite_matrix4, __to_manifold_mat4, __safe_transform, __identity4, __safe_attach_transform, __safe_offset2d, __safe_project3d, __apply_color, __flat_map_iter, __range, __union2d3d, __difference2d3d, __intersection2d3d, __hull2d3d, __minkowski2d3d, __extrude, __revolve } = __rt;
var PI: any = __rt.PI;
var INF: any = __rt.INF;
var NAN: any = __rt.NAN;
var undef: any = __rt.undef;
var _EPSILON: any = __rt._EPSILON;

// list_comprehensions.scad - Examples of list comprehension usage
// Basic list comprehension:
// Returns a 2D vertex per iteration of the for loop
// Note: subsequent assignments inside the for loop is allowed
function ngon$mod(num: any, r: any): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(CrossSection.ofPolygons([...((() => { const __r = []; for (let i = 0; i <= __sub(num, 1); i += 1) __r.push(...(__flat_map_iter(__div(__mul(i, 360), num), (a) => [[__mul(r, cos_fn(a)), __mul(r, sin_fn(a))]]))); return __r; })())]));
  return __union2d3d(__items);
}
// More complex list comprehension:
// Similar to ngon(), but uses an inner function to calculate
// the vertices. the let() keyword allows assignment of temporary variables.
function rounded_ngon$mod(num: any, r: any, rounding: any = 0): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  function v_fn(a: any): any {
    return ((d) => (((v) => (__mul((__sub(r, rounding)), [cos_fn(v), sin_fn(v)])))(__mul(floor_fn(__div((__add(a, __div(d, 2))), d)), d))))(__div(360, num));
  }
  __items.push(CrossSection.ofPolygons([...((() => { const __r = []; for (let a = 0; a <= __sub(360, 1); a += 1) __r.push(...([__add(v_fn(a), __mul(rounding, [cos_fn(a), sin_fn(a)]))])); return __r; })())]));
  return __union2d3d(__items);
}
// Gear/star generator
// Uses a list comprehension taking a list of radii to generate a star shape
function star$mod(num: any, radii: any): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(CrossSection.ofPolygons([...((() => { const __r = []; for (let i = 0; i <= __sub(num, 1); i += 1) __r.push(...(__flat_map_iter(__div(__mul(i, 360), num), (a) => __flat_map_iter(radii[__mod(i, len_fn(radii))], (r) => [[__mul(r, cos_fn(a)), __mul(r, sin_fn(a))]])))); return __r; })())]));
  return __union2d3d(__items);
}

var $fn: any = 0, $fa: any = 12, $fs: any = 2;
var $vpr: any = [0, 0, 0], $vpt: any = [0, 0, 0], $vpd: any = 500;
var $parent_modules: any = 0;
var _NO_ARG: any = Symbol("NO_ARG");
const __result_items: any[] = [];
__result_items.push(__with_children(() => Manifold.union([]), 0, () => ngon$mod(3, 10)));
__result_items.push(__with_children(() => Manifold.union([]), 0, () => ngon$mod(6, 8)).translate([20, 0]));
__result_items.push(__with_children(() => Manifold.union([]), 0, () => ngon$mod(10, 6)).translate([36, 0]));
__result_items.push(__with_children(() => Manifold.union([]), 0, () => rounded_ngon$mod(3, 10, 5)).translate([0, 22]));
__result_items.push(__with_children(() => Manifold.union([]), 0, () => rounded_ngon$mod(6, 8, 4)).translate([20, 22]));
__result_items.push(__with_children(() => Manifold.union([]), 0, () => rounded_ngon$mod(10, 6, 3)).translate([36, 22]));
__result_items.push(__with_children(() => Manifold.union([]), 0, () => star$mod(20, [6, 10])).translate([0, 44]));
__result_items.push(__with_children(() => Manifold.union([]), 0, () => star$mod(40, [6, 8, 8, 6])).translate([20, 44]));
__result_items.push(__with_children(() => Manifold.union([]), 0, () => star$mod(30, [3, 4, 5, 6, 5, 4])).translate([36, 44]));
__result_items.push((console.log("version = ", version_fn()), Manifold.union([])));
export const result = __union2d3d(__result_items);