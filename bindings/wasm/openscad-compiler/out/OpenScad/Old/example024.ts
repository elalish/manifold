import * as __rt from "../../runtime.js";
const { Manifold, CrossSection, wasm, is_undef_fn, is_bool_fn, is_num_fn, is_string_fn, is_list_fn, is_function_fn, sin_fn, cos_fn, tan_fn, asin_fn, acos_fn, atan_fn, atan2_fn, abs_fn, sign_fn, floor_fn, ceil_fn, round_fn, sqrt_fn, exp_fn, ln_fn, log_fn, min_fn, max_fn, norm_fn, cross_fn, len_fn, str_fn, chr_fn, ord_fn, concat_fn, search_fn, lookup_fn, openscad_assert_fn, __eq, __add, __sub, __mul, __div, __mod, __neg, __pos, version_fn, version_num_fn, __children_stack, __with_children, __is_finite_matrix4, __to_manifold_mat4, __safe_transform, __identity4, __safe_attach_transform, __safe_offset2d, __safe_project3d, __apply_color, __flat_map_iter, __range, __union2d3d, __difference2d3d, __intersection2d3d, __hull2d3d, __minkowski2d3d, __extrude, __revolve } = __rt;
var PI: any = __rt.PI;
var INF: any = __rt.INF;
var NAN: any = __rt.NAN;
var undef: any = __rt.undef;
var _EPSILON: any = __rt._EPSILON;

// Menger Sponge
// Size of edge of sponge
var D: any = 100;
// Fractal depth (number of iterations)
var n: any = 3;
function menger$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(__difference2d3d(((__s: any) => Manifold.cube((is_list_fn(__s) ? __s : [__s, __s, __s]), true))(D), [
  (() => {
  const __items = [];
  {
    const __iter_0: any = [[0, 0, 0], [0, 0, 90], [0, 90, 0]];
    for (const v of __iter_0) {
      __items.push(__with_children(() => Manifold.union([]), 0, () => menger_negative$mod(/* side = */ D, /* maxside = */ D, /* level = */ n)).rotate(v));
    }
  }
  return __union2d3d(__items);
})()
]));
  return __union2d3d(__items);
}
function menger_negative$mod(side: any = 1, maxside: any = 1, level: any = 1): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  var l: any = __div(side, 3);
  __items.push(((__s: any) => Manifold.cube((is_list_fn(__s) ? __s : [__s, __s, __s]), true))([__mul(maxside, 1.1), l, l]));
  __items.push((() => {
  if ((level > 1)) {
    return (() => {
  const __items = [];
  {
    const __step_0: any = 1;
    for (let i: any = __neg(1); (__step_0 >= 0) ? i <= 1 : i >= 1; i += __step_0) {
      {
        const __step_1: any = 1;
        for (let j: any = __neg(1); (__step_1 >= 0) ? j <= 1 : j >= 1; j += __step_1) {
          __items.push((() => {
  if ((i || j)) {
    return __with_children(() => Manifold.union([]), 0, () => menger_negative$mod(/* side = */ l, /* maxside = */ maxside, /* level = */ __sub(level, 1))).translate([0, __mul(i, l), __mul(j, l)]);
  }
  return Manifold.union([]);
})());
        }
      }
    }
  }
  return __union2d3d(__items);
})();
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
__result_items.push((console.log("version = ", version_fn()), Manifold.union([])));
__result_items.push(__difference2d3d(__with_children(() => Manifold.union([]), 0, () => menger$mod()).rotate([45, atan_fn(__div(1, sqrt_fn(2))), 0]), [
  ((__s: any) => Manifold.cube((is_list_fn(__s) ? __s : [__s, __s, __s]), true))(__mul(2, D)).translate([0, 0, __neg(D)])
]));
export const result = __union2d3d(__result_items);