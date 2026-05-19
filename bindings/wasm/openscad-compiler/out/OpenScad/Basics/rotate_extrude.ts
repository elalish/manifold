import * as __rt from "../../runtime.js";
const { Manifold, CrossSection, wasm, is_undef_fn, is_bool_fn, is_num_fn, is_string_fn, is_list_fn, is_function_fn, sin_fn, cos_fn, tan_fn, asin_fn, acos_fn, atan_fn, atan2_fn, abs_fn, sign_fn, floor_fn, ceil_fn, round_fn, sqrt_fn, exp_fn, ln_fn, log_fn, min_fn, max_fn, norm_fn, cross_fn, len_fn, str_fn, chr_fn, ord_fn, concat_fn, search_fn, lookup_fn, openscad_assert_fn, __eq, __add, __sub, __mul, __div, __mod, __neg, __pos, version_fn, version_num_fn, __children_stack, __with_children, __is_finite_matrix4, __to_manifold_mat4, __safe_transform, __identity4, __safe_attach_transform, __safe_offset2d, __safe_project3d, __apply_color, __flat_map_iter, __range, __union2d3d, __difference2d3d, __intersection2d3d, __hull2d3d, __minkowski2d3d, __extrude, __revolve } = __rt;
var PI: any = __rt.PI;
var INF: any = __rt.INF;
var NAN: any = __rt.NAN;
var undef: any = __rt.undef;
var _EPSILON: any = __rt._EPSILON;

var $fn: any = 0, $fa: any = 12, $fs: any = 2;
var $vpr: any = [0, 0, 0], $vpt: any = [0, 0, 0], $vpd: any = 500;
var $parent_modules: any = 0;
var _NO_ARG: any = Symbol("NO_ARG");
const __result_items: any[] = [];
__result_items.push((console.log("version = ", version_fn()), Manifold.union([])));
// rotate_extrude() rotates a 2D shape around the Z axis. 
// Note that the 2D shape must be either completely on the 
// positive or negative side of the X axis.
__result_items.push(__apply_color(__revolve(((__s: any) => CrossSection.square((is_list_fn(__s) ? __s : [__s, __s]), false))(5).translate([10, 0]), 0, 360), "red", undefined));
// rotate_extrude() uses the global $fn/$fa/$fs settings, but
// it's possible to give a different value as parameter.
__result_items.push(__apply_color(__revolve(((__t: any, __s: any) => CrossSection.square([Math.max(0.001, len_fn(str_fn(__t)) * __s * 0.6), Math.max(0.001, __s)], false))("  J", 10), 80, 360).translate([40, 0, 0]), "cyan", undefined));
// Using a shape that touches the X axis is allowed and produces
// 3D objects that don't have a hole in the center.
__result_items.push(__apply_color(__revolve(CrossSection.ofPolygons([[0, 0], [8, 4], [4, 8], [4, 12], [12, 16], [0, 20]]), 80, 360).translate([0, 30, 0]), "green", undefined));
// By default rotate_extrude forms a full 360 degree circle, 
// but a partial rotation can be performed by using the angle parameter.
// Positive angles create an arc starting from the X axis, going counter-clockwise.
// Negative angles generate an arc in the clockwise direction.
__result_items.push(__apply_color(__union2d3d([
  __revolve(((__s: any) => CrossSection.square((is_list_fn(__s) ? __s : [__s, __s]), false))(5).translate([12.5, 0]), 0, 180),
  __revolve(((__s: any) => CrossSection.square((is_list_fn(__s) ? __s : [__s, __s]), false))(5).translate([5, 0]), 0, 180).translate([7.5, 0]),
  __revolve(((__s: any) => CrossSection.square((is_list_fn(__s) ? __s : [__s, __s]), false))(5).translate([5, 0]), 0, __neg(180)).translate([__neg(7.5), 0])
]).translate([40, 40]), "magenta", undefined));
export const result = __union2d3d(__result_items);