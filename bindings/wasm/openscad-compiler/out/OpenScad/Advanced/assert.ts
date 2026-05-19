import * as __rt from "../../runtime.js";
const { Manifold, CrossSection, wasm, is_undef_fn, is_bool_fn, is_num_fn, is_string_fn, is_list_fn, is_function_fn, sin_fn, cos_fn, tan_fn, asin_fn, acos_fn, atan_fn, atan2_fn, abs_fn, sign_fn, floor_fn, ceil_fn, round_fn, sqrt_fn, exp_fn, ln_fn, log_fn, min_fn, max_fn, norm_fn, cross_fn, len_fn, str_fn, chr_fn, ord_fn, concat_fn, search_fn, lookup_fn, openscad_assert_fn, __eq, __add, __sub, __mul, __div, __mod, __neg, __pos, version_fn, version_num_fn, __children_stack, __with_children, __is_finite_matrix4, __to_manifold_mat4, __safe_transform, __identity4, __safe_attach_transform, __safe_offset2d, __safe_project3d, __apply_color, __flat_map_iter, __range, __union2d3d, __difference2d3d, __intersection2d3d, __hull2d3d, __minkowski2d3d, __extrude, __revolve } = __rt;
var PI: any = __rt.PI;
var INF: any = __rt.INF;
var NAN: any = __rt.NAN;
var undef: any = __rt.undef;
var _EPSILON: any = __rt._EPSILON;

function size_fn(x: any): any {
  return (openscad_assert_fn(__eq(__mod(x, 2), 0), "Size must be an even number"), x);
}
function ring$mod(r: any = 10, cnt: any = 3, s: any = 6): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push((openscad_assert_fn((r >= 10), "Parameter r must be >= 10"), Manifold.union([])));
  __items.push((openscad_assert_fn(((cnt >= 3) && (cnt <= 20)), "Parameter cnt must be between 3 and 20 (inclusive"), Manifold.union([])));
  __items.push((() => {
  const __items = [];
  {
    const __step_0: any = 1;
    for (let a: any = 0; (__step_0 >= 0) ? a <= __sub(cnt, 1) : a >= __sub(cnt, 1); a += __step_0) {
      __items.push(((__s: any) => Manifold.cube((is_list_fn(__s) ? __s : [__s, __s, __s]), true))(size_fn(s)).translate([r, 0, 0]).rotate(__div(__mul(a, 360), cnt)));
    }
  }
  return __union2d3d(__items);
})());
  return __union2d3d(__items);
}

var $fn: any = 0, $fa: any = 12, $fs: any = 2;
var $vpr: any = [0, 0, 0], $vpt: any = [0, 0, 0], $vpd: any = 500;
var $parent_modules: any = 0;
var _NO_ARG: any = Symbol("NO_ARG");
const __result_items: any[] = [];
__result_items.push((console.log("version = ", version_fn()), Manifold.union([])));
// ring(5, 5, 4); // trigger assertion for parameter r
// ring(10, 2, 4); // trigger assertion for parameter cnt
// ring(10, 3, 5); // trigger assertion in function size()
__result_items.push(__apply_color(__with_children(() => Manifold.union([]), 0, () => ring$mod(10, 3, 4)), "red", undefined));
__result_items.push(__apply_color(__with_children(() => Manifold.union([]), 0, () => ring$mod(25, 9, 6)), "green", undefined));
__result_items.push(__apply_color(__with_children(() => Manifold.union([]), 0, () => ring$mod(40, 20, 8)), "blue", undefined));
export const result = __union2d3d(__result_items);