import * as __rt from "../../runtime.js";
const { Manifold, CrossSection, wasm, is_undef_fn, is_bool_fn, is_num_fn, is_string_fn, is_list_fn, is_function_fn, sin_fn, cos_fn, tan_fn, asin_fn, acos_fn, atan_fn, atan2_fn, abs_fn, sign_fn, floor_fn, ceil_fn, round_fn, sqrt_fn, exp_fn, ln_fn, log_fn, min_fn, max_fn, norm_fn, cross_fn, len_fn, str_fn, chr_fn, ord_fn, concat_fn, search_fn, lookup_fn, openscad_assert_fn, __eq, __add, __sub, __mul, __div, __mod, __neg, __pos, version_fn, version_num_fn, __children_stack, __with_children, __is_finite_matrix4, __to_manifold_mat4, __safe_transform, __identity4, __safe_attach_transform, __safe_offset2d, __safe_project3d, __apply_color, __flat_map_iter, __range, __union2d3d, __difference2d3d, __intersection2d3d, __hull2d3d, __minkowski2d3d, __extrude, __revolve } = __rt;
var PI: any = __rt.PI;
var INF: any = __rt.INF;
var NAN: any = __rt.NAN;
var undef: any = __rt.undef;
var _EPSILON: any = __rt._EPSILON;

var font: any = "Liberation Sans"; //["Liberation Sans", "Liberation Sans:style=Bold", "Liberation Sans:style=Italic", "Liberation Mono", "Liberation Serif"]
var cube_size: any = 60;
var letter_size: any = 50;
var letter_height: any = 5;
var o: any = __sub(__div(cube_size, 2), __div(letter_height, 2));
function letter$mod(l: any): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  // Use linear_extrude() to make the letters 3D objects as they
  // are only 2D shapes when only using text()
  __items.push(__extrude(((__t: any, __s: any) => CrossSection.square([Math.max(0.001, len_fn(str_fn(__t)) * __s * 0.6), Math.max(0.001, __s)], false))(l, letter_size), letter_height));
  return __union2d3d(__items);
}

var $fn: any = 0, $fa: any = 12, $fs: any = 2;
var $vpr: any = [0, 0, 0], $vpt: any = [0, 0, 0], $vpd: any = 500;
var $parent_modules: any = 0;
var _NO_ARG: any = Symbol("NO_ARG");
const __result_items: any[] = [];
// text_on_cube.scad - Example for text() usage in OpenSCAD
__result_items.push((console.log("version = ", version_fn()), Manifold.union([])));
__result_items.push(__difference2d3d(__union2d3d([
  __apply_color(((__s: any) => Manifold.cube((is_list_fn(__s) ? __s : [__s, __s, __s]), true))(cube_size), "gray", undefined),
  __with_children(() => Manifold.union([]), 0, () => letter$mod("C")).rotate([90, 0, 0]).translate([0, __neg(o), 0]),
  __with_children(() => Manifold.union([]), 0, () => letter$mod("U")).rotate([90, 0, 90]).translate([o, 0, 0]),
  __with_children(() => Manifold.union([]), 0, () => letter$mod("B")).rotate([90, 0, 180]).translate([0, o, 0]),
  __with_children(() => Manifold.union([]), 0, () => letter$mod("E")).rotate([90, 0, __neg(90)]).translate([__neg(o), 0, 0])
]), [
  // Put some symbols on top and bottom using symbols from the
// Unicode symbols table.
// (see https://en.wikipedia.org/wiki/Miscellaneous_Symbols)
//
// Note that depending on the font used, not all the symbols
// are actually available.
__with_children(() => Manifold.union([]), 0, () => letter$mod("u263A")).translate([0, 0, o]),
  __with_children(() => Manifold.union([]), 0, () => letter$mod("u263C")).translate([0, 0, __sub(__neg(o), letter_height)])
]));
export const result = __union2d3d(__result_items);