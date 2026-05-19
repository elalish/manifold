import * as __rt from "../../runtime.js";
const { Manifold, CrossSection, wasm, is_undef_fn, is_bool_fn, is_num_fn, is_string_fn, is_list_fn, is_function_fn, sin_fn, cos_fn, tan_fn, asin_fn, acos_fn, atan_fn, atan2_fn, abs_fn, sign_fn, floor_fn, ceil_fn, round_fn, sqrt_fn, exp_fn, ln_fn, log_fn, min_fn, max_fn, norm_fn, cross_fn, len_fn, str_fn, chr_fn, ord_fn, concat_fn, search_fn, lookup_fn, openscad_assert_fn, __eq, __add, __sub, __mul, __div, __mod, __neg, __pos, version_fn, version_num_fn, __children_stack, __with_children, __is_finite_matrix4, __to_manifold_mat4, __safe_transform, __identity4, __safe_attach_transform, __safe_offset2d, __safe_project3d, __apply_color, __flat_map_iter, __range, __union2d3d, __difference2d3d, __intersection2d3d, __hull2d3d, __minkowski2d3d, __extrude, __revolve } = __rt;
var PI: any = __rt.PI;
var INF: any = __rt.INF;
var NAN: any = __rt.NAN;
var undef: any = __rt.undef;
var _EPSILON: any = __rt._EPSILON;

// These are examples for the `roof` function, which builds
// "roofs" on top of 2d sketches. A roof can be constructed using
// either straight skeleton or Voronoi diagram (see below).
//
// Under the hood, to construct straight skeletons we use cgal,
// while for Voronoi diagrams we use boost::polygon.
//
// With the current implementation, computation of Voronoi diagrams
// is much faster (10x - 100x) than that of straight skeletons.
// some 2d sketch
function sketch$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(CrossSection.ofPolygons([[__neg(5), __neg(1)], [__neg(0.15), __neg(1)], [0, 0], [0.15, __neg(1)], [5, __neg(1)], [5, __neg(0.1)], [4, 0], [5, 0.1], [5, 1], [__neg(5), 1]]));
  return __union2d3d(__items);
}

var $fn: any = 0, $fa: any = 12, $fs: any = 2;
var $vpr: any = [0, 0, 0], $vpt: any = [0, 0, 0], $vpd: any = 500;
var $parent_modules: any = 0;
var _NO_ARG: any = Symbol("NO_ARG");
const __result_items: any[] = [];
// This file is a part of openscad. Everything implied is implied.
// Author: Alexey Korepanov <kaikaikai@yandex.ru>
__result_items.push((console.log("version = ", version_fn()), Manifold.union([])));
// straight skeleton roof
__result_items.push(Manifold.union([]));
// Voronoi diagram roof (the default)
__result_items.push(Manifold.union([]).translate([0, __neg(5), 0]));
// Voronoi diagram respects discretization parameters
// $fa, $fs and $fn:
__result_items.push(Manifold.union([]).translate([0, __neg(8), 0]));
// A nice application is beveling of fonts:
__result_items.push(Manifold.union([]).translate([6, 0, 0]));
__result_items.push(Manifold.union([]).translate([6, __neg(7), 0]));
export const result = __union2d3d(__result_items);