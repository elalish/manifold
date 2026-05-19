import * as __rt from "../../runtime.js";
const { Manifold, CrossSection, wasm, is_undef_fn, is_bool_fn, is_num_fn, is_string_fn, is_list_fn, is_function_fn, sin_fn, cos_fn, tan_fn, asin_fn, acos_fn, atan_fn, atan2_fn, abs_fn, sign_fn, floor_fn, ceil_fn, round_fn, sqrt_fn, exp_fn, ln_fn, log_fn, min_fn, max_fn, norm_fn, cross_fn, len_fn, str_fn, chr_fn, ord_fn, concat_fn, search_fn, lookup_fn, openscad_assert_fn, __eq, __add, __sub, __mul, __div, __mod, __neg, __pos, version_fn, version_num_fn, __children_stack, __with_children, __is_finite_matrix4, __to_manifold_mat4, __safe_transform, __identity4, __safe_attach_transform, __safe_offset2d, __safe_project3d, __apply_color, __flat_map_iter, __range, __union2d3d, __difference2d3d, __intersection2d3d, __hull2d3d, __minkowski2d3d, __extrude, __revolve } = __rt;
var PI: any = __rt.PI;
var INF: any = __rt.INF;
var NAN: any = __rt.NAN;
var undef: any = __rt.undef;
var _EPSILON: any = __rt._EPSILON;

// One shape with corresponding text
function shapeWithArea$mod(num: any, r: any): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(CrossSection.ofPolygons(ngon_fn(num, r)));
  __items.push(__apply_color(((__t: any, __s: any) => CrossSection.square([Math.max(0.001, len_fn(str_fn(__t)) * __s * 0.6), Math.max(0.001, __s)], false))(str_fn(round_fn(area_fn(ngon_fn(num, r)))), 8), "Cyan", undefined).translate([0, __neg(20)]));
  return __union2d3d(__items);
}
// Simple list comprehension for creating N-gon vertices
function ngon_fn(num: any, r: any): any {
  return [...((() => { const __r = []; for (let i = 0; i <= __sub(num, 1); i += 1) __r.push(...(__flat_map_iter(__div(__mul(i, 360), num), (a) => [[__mul(r, cos_fn(a)), __mul(r, sin_fn(a))]]))); return __r; })())];
}
// Area of a triangle with the 3rd vertex in the origin
function triarea_fn(v0: any, v1: any): any {
  return __div(cross_fn(v0, v1), 2);
}
// Area of a polygon using the Shoelace formula
function area_fn(vertices: any): any {
  return ((areas) => (sum_fn(areas)))([...(((num) => ((() => { const __r = []; for (let i = 0; i <= __sub(num, 1); i += 1) __r.push(...([triarea_fn(vertices[i], vertices[__mod((__add(i, 1)), num)])])); return __r; })()))(len_fn(vertices)))]);
}
// Recursive helper function: Sums all values in a list.
// In this case, sum all partial areas into the final area.
function sum_fn(values: any, s: any = 0): any {
  return (__eq(s, __sub(len_fn(values), 1)) ? values[s] : __add(values[s], sum_fn(values, __add(s, 1))));
}

var $fn: any = 0, $fa: any = 12, $fs: any = 2;
var $vpr: any = [0, 0, 0], $vpt: any = [0, 0, 0], $vpd: any = 500;
var $parent_modules: any = 0;
var _NO_ARG: any = Symbol("NO_ARG");
const __result_items: any[] = [];
// polygon_areas.scad: Another recursion example 
// Draw all geometry
__result_items.push(__apply_color(((__t: any, __s: any) => CrossSection.square([Math.max(0.001, len_fn(str_fn(__t)) * __s * 0.6), Math.max(0.001, __s)], false))("Areas:", 8), "Red", undefined).translate([0, 20]));
__result_items.push(__with_children(() => Manifold.union([]), 0, () => shapeWithArea$mod(3, 10)).translate([__neg(44), 0]));
__result_items.push(__with_children(() => Manifold.union([]), 0, () => shapeWithArea$mod(4, 10)).translate([__neg(22), 0]));
__result_items.push(__with_children(() => Manifold.union([]), 0, () => shapeWithArea$mod(6, 10)).translate([0, 0]));
__result_items.push(__with_children(() => Manifold.union([]), 0, () => shapeWithArea$mod(10, 10)).translate([22, 0]));
__result_items.push(__with_children(() => Manifold.union([]), 0, () => shapeWithArea$mod(360, 10)).translate([44, 0]));
__result_items.push((console.log("version = ", version_fn()), Manifold.union([])));
export const result = __union2d3d(__result_items);