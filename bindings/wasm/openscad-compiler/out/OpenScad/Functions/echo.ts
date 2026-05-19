import * as __rt from "../../runtime.js";
const { Manifold, CrossSection, wasm, is_undef_fn, is_bool_fn, is_num_fn, is_string_fn, is_list_fn, is_function_fn, sin_fn, cos_fn, tan_fn, asin_fn, acos_fn, atan_fn, atan2_fn, abs_fn, sign_fn, floor_fn, ceil_fn, round_fn, sqrt_fn, exp_fn, ln_fn, log_fn, min_fn, max_fn, norm_fn, cross_fn, len_fn, str_fn, chr_fn, ord_fn, concat_fn, search_fn, lookup_fn, openscad_assert_fn, __eq, __add, __sub, __mul, __div, __mod, __neg, __pos, version_fn, version_num_fn, __children_stack, __with_children, __is_finite_matrix4, __to_manifold_mat4, __safe_transform, __identity4, __safe_attach_transform, __safe_offset2d, __safe_project3d, __apply_color, __flat_map_iter, __range, __union2d3d, __difference2d3d, __intersection2d3d, __hull2d3d, __minkowski2d3d, __extrude, __revolve } = __rt;
var PI: any = __rt.PI;
var INF: any = __rt.INF;
var NAN: any = __rt.NAN;
var undef: any = __rt.undef;
var _EPSILON: any = __rt._EPSILON;

// Using echo() in expression context can help with debugging
// recursive functions. See console window for output of the
// examples below.
// Simple example just outputting the function input parameters.
function f1_fn(x: any, y: any): any {
  return (console.log("f1: ", x, y), __add(__add(__mul(__mul(0.5, x), x), __mul(4, y)), 1));
}
var r1: any = f1_fn(3, 5);
// To output the result, there are multiple possibilities, the
// easiest is to use let() to assign the result to a variable
// (y here) which is used for both echo() output and result.
function f2_fn(x: any): any {
  return ((y) => ((console.log("f2: ", y), y)))(pow_fn(x, 3));
}
var r2: any = f2_fn(4);
// Another option is using a helper function where the argument
// is evaluated first and then passed to the result() helper
// where it's printed using echo() and returned as result.
function result_fn(x: any): any {
  return (console.log("f3: ", x), x);
}
function f3_fn(x: any): any {
  return result_fn(__sub(__mul(x, x), 5));
}
var r3: any = f3_fn(5);
// A more complex example is a recursive function. Combining
// the two different ways of printing values before and after
// evaluation it's possible to output the input value x when
// descending into the recursion and the result y collected
// when returning.
function f4_fn(x: any): any {
  return (console.log("f4: ", "x = ", x), ((y) => ((console.log("f4: ", "y = ", y), y)))((__eq(x, 1) ? 1 : __mul(x, f4_fn(__sub(x, 1))))));
}
var r4: any = f4_fn(5);

var $fn: any = 0, $fa: any = 12, $fs: any = 2;
var $vpr: any = [0, 0, 0], $vpt: any = [0, 0, 0], $vpd: any = 500;
var $parent_modules: any = 0;
var _NO_ARG: any = Symbol("NO_ARG");
const __result_items: any[] = [];
__result_items.push((console.log("version = ", version_fn()), Manifold.union([])));
export const result = __union2d3d(__result_items);