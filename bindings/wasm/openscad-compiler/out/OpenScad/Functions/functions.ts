import * as __rt from "../../runtime.js";
const { Manifold, CrossSection, wasm, is_undef_fn, is_bool_fn, is_num_fn, is_string_fn, is_list_fn, is_function_fn, sin_fn, cos_fn, tan_fn, asin_fn, acos_fn, atan_fn, atan2_fn, abs_fn, sign_fn, floor_fn, ceil_fn, round_fn, sqrt_fn, exp_fn, ln_fn, log_fn, min_fn, max_fn, norm_fn, cross_fn, len_fn, str_fn, chr_fn, ord_fn, concat_fn, search_fn, lookup_fn, openscad_assert_fn, __eq, __add, __sub, __mul, __div, __mod, __neg, __pos, version_fn, version_num_fn, __children_stack, __with_children, __is_finite_matrix4, __to_manifold_mat4, __safe_transform, __identity4, __safe_attach_transform, __safe_offset2d, __safe_project3d, __apply_color, __flat_map_iter, __range, __union2d3d, __difference2d3d, __intersection2d3d, __hull2d3d, __minkowski2d3d, __extrude, __revolve } = __rt;
var PI: any = __rt.PI;
var INF: any = __rt.INF;
var NAN: any = __rt.NAN;
var undef: any = __rt.undef;
var _EPSILON: any = __rt._EPSILON;

// Functions can be defined to simplify code using lots of
// calculations.
// Simple example with a single function argument (which should
// be a number) and returning a number calculated based on that.
function f_fn(x: any): any {
  return __add(__mul(0.5, x), 1);
}
// Functions can call other functions and return complex values
// too. In this case a 3 element vector is returned which can
// be used as point in 3D space or as vector (in the mathematical
// meaning) for translations and other transformations.
function g_fn(x: any): any {
  return [__add(__mul(5, x), 20), __sub(__mul(f_fn(x), f_fn(x)), 50), 0];
}

var $fn: any = 0, $fa: any = 12, $fs: any = 2;
var $vpr: any = [0, 0, 0], $vpt: any = [0, 0, 0], $vpd: any = 500;
var $parent_modules: any = 0;
var _NO_ARG: any = Symbol("NO_ARG");
const __result_items: any[] = [];
__result_items.push((console.log("version = ", version_fn()), Manifold.union([])));
__result_items.push(__apply_color((() => {
  const __items = [];
  {
    const __step_0: any = 5;
    for (let a: any = __neg(100); (__step_0 >= 0) ? a <= 100 : a >= 100; a += __step_0) {
      __items.push(((__s: any) => Manifold.cube((is_list_fn(__s) ? __s : [__s, __s, __s]), true))(2).translate([a, f_fn(a), 0]));
    }
  }
  return __union2d3d(__items);
})(), "red", undefined));
__result_items.push(__apply_color((() => {
  const __items = [];
  {
    const __step_0: any = 10;
    for (let a: any = __neg(200); (__step_0 >= 0) ? a <= 200 : a >= 200; a += __step_0) {
      __items.push(Manifold.sphere(1).translate(g_fn(__div(a, 8))));
    }
  }
  return __union2d3d(__items);
})(), "green", undefined));
export const result = __union2d3d(__result_items);