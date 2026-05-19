import * as __rt from "../../runtime.js";
const { Manifold, CrossSection, wasm, is_undef_fn, is_bool_fn, is_num_fn, is_string_fn, is_list_fn, is_function_fn, sin_fn, cos_fn, tan_fn, asin_fn, acos_fn, atan_fn, atan2_fn, abs_fn, sign_fn, floor_fn, ceil_fn, round_fn, sqrt_fn, exp_fn, ln_fn, log_fn, min_fn, max_fn, norm_fn, cross_fn, len_fn, str_fn, chr_fn, ord_fn, concat_fn, search_fn, lookup_fn, openscad_assert_fn, __eq, __add, __sub, __mul, __div, __mod, __neg, __pos, version_fn, version_num_fn, __children_stack, __with_children, __is_finite_matrix4, __to_manifold_mat4, __safe_transform, __identity4, __safe_attach_transform, __safe_offset2d, __safe_project3d, __apply_color, __flat_map_iter, __range, __union2d3d, __difference2d3d, __intersection2d3d, __hull2d3d, __minkowski2d3d, __extrude, __revolve } = __rt;
var PI: any = __rt.PI;
var INF: any = __rt.INF;
var NAN: any = __rt.NAN;
var undef: any = __rt.undef;
var _EPSILON: any = __rt._EPSILON;

// offset.scad - Example for offset() usage in OpenSCAD
var $fn: any = 40;
var foot_height: any = 20;
function outline$mod(wall: any = 1): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(__difference2d3d(__safe_offset2d(children(), __div(wall, 2)), [
  __safe_offset2d(children(), __div(__neg(wall), 2))
]));
  return __union2d3d(__items);
}

var $fn: any = 0, $fa: any = 12, $fs: any = 2;
var $vpr: any = [0, 0, 0], $vpt: any = [0, 0, 0], $vpd: any = 500;
var $parent_modules: any = 0;
var _NO_ARG: any = Symbol("NO_ARG");
const __result_items: any[] = [];
__result_items.push((console.log("version = ", version_fn()), Manifold.union([])));
// offsetting with a positive value allows to create rounded corners easily
__result_items.push(__extrude(__safe_offset2d(((__s: any) => CrossSection.square((is_list_fn(__s) ? __s : [__s, __s]), true))(50), 10), foot_height));
__result_items.push(__extrude((() => { const __childFns = [
  () => (CrossSection.circle(15))
]; return __with_children((i) => (i === undefined ? __union2d3d(__childFns.map(fn => fn())) : ((i >= 0 && i < __childFns.length) ? __childFns[i]() : Manifold.union([]))), __childFns.length, () => outline$mod(/* wall = */ 2)); })(), 20).translate([0, 0, foot_height]));
__result_items.push(Manifold.cylinder(100, 14, 14));
__result_items.push(Manifold.sphere(30).translate([0, 0, 100]));
export const result = __union2d3d(__result_items);