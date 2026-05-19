import * as __rt from "../../runtime.js";
const { Manifold, CrossSection, wasm, is_undef_fn, is_bool_fn, is_num_fn, is_string_fn, is_list_fn, is_function_fn, sin_fn, cos_fn, tan_fn, asin_fn, acos_fn, atan_fn, atan2_fn, abs_fn, sign_fn, floor_fn, ceil_fn, round_fn, sqrt_fn, exp_fn, ln_fn, log_fn, min_fn, max_fn, norm_fn, cross_fn, len_fn, str_fn, chr_fn, ord_fn, concat_fn, search_fn, lookup_fn, openscad_assert_fn, __eq, __add, __sub, __mul, __div, __mod, __neg, __pos, version_fn, version_num_fn, __children_stack, __with_children, __is_finite_matrix4, __to_manifold_mat4, __safe_transform, __identity4, __safe_attach_transform, __safe_offset2d, __safe_project3d, __apply_color, __flat_map_iter, __range, __union2d3d, __difference2d3d, __intersection2d3d, __hull2d3d, __minkowski2d3d, __extrude, __revolve } = __rt;
var PI: any = __rt.PI;
var INF: any = __rt.INF;
var NAN: any = __rt.NAN;
var undef: any = __rt.undef;
var _EPSILON: any = __rt._EPSILON;

// Recursive calls of modules can generate complex geometry, especially
// fractal style objects.
// The example uses a recursive module to generate a random tree as
// described in http://natureofcode.com/book/chapter-8-fractals/
// number of levels for the recursion
var levels: any = 10; // [1:1:14]
// length of the first segment
var len: any = 100; // [10:10:200]
// thickness of the first segment
var thickness: any = 5; //[1:1:20]
// the identity matrix
var identity: any = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
// random generator, to generate always the same output for the example,
// this uses a seed for rands() and stores the array of random values in
// the random variable. To generate different output, remove the seed or
// replace the function rnd() to just call rands(s, e, 1)[0].
var rcnt: any = 1000;
var random: any = rands_fn(0, 1, rcnt, 18);
function rnd_fn(s: any, e: any, r: any): any {
  return __add(__mul(random[__mod(r, rcnt)], (__sub(e, s))), s);
}
// generate 4x4 translation matrix
function mt_fn(x: any, y: any): any {
  return [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]];
}
// generate 4x4 rotation matrix around Z axis
function mr_fn(a: any): any {
  return [[cos_fn(a), __neg(sin_fn(a)), 0, 0], [sin_fn(a), cos_fn(a), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
}
function tree$mod(length: any, thickness: any, count: any, m: any = identity, r: any = 1): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(__apply_color(__safe_transform(((__s: any) => CrossSection.square((is_list_fn(__s) ? __s : [__s, __s]), false))([thickness, length]), m), [0, __sub(1, (__mul(__div(0.8, levels), count))), 0], undefined));
  __items.push((() => {
  if ((count > 0)) {
    return __union2d3d([
  __with_children(() => Manifold.union([]), 0, () => tree$mod(__mul(rnd_fn(0.6, 0.8, r), length), __mul(0.8, thickness), __sub(count, 1), __mul(__mul(m, mt_fn(0, length)), mr_fn(rnd_fn(20, 35, __add(r, 1)))), __mul(8, r))),
  __with_children(() => Manifold.union([]), 0, () => tree$mod(__mul(rnd_fn(0.6, 0.8, __add(r, 1)), length), __mul(0.8, thickness), __sub(count, 1), __mul(__mul(m, mt_fn(0, length)), mr_fn(__neg(rnd_fn(20, 35, __add(r, 3))))), __add(__mul(8, r), 4)))
]);
  }
  return Manifold.union([]);
})());
  return __union2d3d(__items);
}

var $fn: any = 0, $fa: any = 12, $fs: any = 2;
var $vpr: any = [0, 0, 0], $vpt: any = [0, 0, 0], $vpd: any = 500;
var $parent_modules: any = 0;
var _NO_ARG: any = Symbol("NO_ARG");
const __result_items: any[] = [];
__result_items.push(__with_children(() => Manifold.union([]), 0, () => tree$mod(len, thickness, levels)));
__result_items.push((console.log("version = ", version_fn()), Manifold.union([])));
export const result = __union2d3d(__result_items);