import * as __rt from "../../runtime.js";
const { Manifold, CrossSection, wasm, is_undef_fn, is_bool_fn, is_num_fn, is_string_fn, is_list_fn, is_function_fn, sin_fn, cos_fn, tan_fn, asin_fn, acos_fn, atan_fn, atan2_fn, abs_fn, sign_fn, floor_fn, ceil_fn, round_fn, sqrt_fn, exp_fn, ln_fn, log_fn, min_fn, max_fn, norm_fn, cross_fn, len_fn, str_fn, chr_fn, ord_fn, concat_fn, search_fn, lookup_fn, openscad_assert_fn, __eq, __add, __sub, __mul, __div, __mod, __neg, __pos, version_fn, version_num_fn, __children_stack, __with_children, __is_finite_matrix4, __to_manifold_mat4, __safe_transform, __identity4, __safe_attach_transform, __safe_offset2d, __safe_project3d, __apply_color, __flat_map_iter, __range, __union2d3d, __difference2d3d, __intersection2d3d, __hull2d3d, __minkowski2d3d, __extrude, __revolve } = __rt;
var PI: any = __rt.PI;
var INF: any = __rt.INF;
var NAN: any = __rt.NAN;
var undef: any = __rt.undef;
var _EPSILON: any = __rt._EPSILON;

function screw$mod(type: any = 2, r1: any = 15, r2: any = 20, n: any = 7, h: any = 100, t: any = 8): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(__extrude(__difference2d3d(CrossSection.circle(r2), [
  (() => {
  const __items = [];
  {
    const __step_0: any = 1;
    for (let i: any = 0; (__step_0 >= 0) ? i <= __sub(n, 1) : i >= __sub(n, 1); i += __step_0) {
      __items.push(__union2d3d([
  (() => {
  if (__eq(type, 1)) {
    return CrossSection.ofPolygons([[__mul(2, r2), 0], [r2, 0], [__mul(r1, cos_fn(__div(180, n))), __mul(r1, sin_fn(__div(180, n)))], [__mul(r2, cos_fn(__div(360, n))), __mul(r2, sin_fn(__div(360, n)))], [__mul(__mul(2, r2), cos_fn(__div(360, n))), __mul(__mul(2, r2), sin_fn(__div(360, n)))]]).rotate(__div(__mul(i, 360), n));
  }
  return Manifold.union([]);
})(),
  (() => {
  if (__eq(type, 2)) {
    return CrossSection.ofPolygons([[__mul(2, r2), 0], [r2, 0], [__mul(r1, cos_fn(__div(90, n))), __mul(r1, sin_fn(__div(90, n)))], [__mul(r1, cos_fn(__div(180, n))), __mul(r1, sin_fn(__div(180, n)))], [__mul(r2, cos_fn(__div(270, n))), __mul(r2, sin_fn(__div(270, n)))], [__mul(__mul(2, r2), cos_fn(__div(270, n))), __mul(__mul(2, r2), sin_fn(__div(270, n)))]]).rotate(__div(__mul(i, 360), n));
  }
  return Manifold.union([]);
})()
]));
    }
  }
  return __union2d3d(__items);
})()
]), h, { twist: __div(__mul(360, t), n) }));
  return __union2d3d(__items);
}
function nut$mod(type: any = 2, r1: any = 16, r2: any = 21, r3: any = 30, s: any = 6, n: any = 7, h: any = __div(100, 5), t: any = __div(8, 5)): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(__difference2d3d(Manifold.cylinder(h, r3, r3, s), [
  __with_children(() => Manifold.union([]), 0, () => screw$mod(type, r1, r2, n, __mul(h, 2), __mul(t, 2))).translate([0, 0, __div(__neg(h), 2)])
]));
  return __union2d3d(__items);
}
function spring$mod(r1: any = 100, r2: any = 10, h: any = 100, hr: any = 12): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  var stepsize: any = __div(1, 16);
  function segment$mod(i1: any, i2: any): any {
    var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
    var $children: any = __c.count;
    function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
    const __items: any[] = [];
    var alpha1: any = __div(__mul(__mul(i1, 360), r2), hr);
    var alpha2: any = __div(__mul(__mul(i2, 360), r2), hr);
    var len1: any = __mul(sin_fn(acos_fn(__sub(__mul(i1, 2), 1))), r2);
    var len2: any = __mul(sin_fn(acos_fn(__sub(__mul(i2, 2), 1))), r2);
    __items.push((() => {
    if ((len1 < 0.01)) {
      return CrossSection.ofPolygons([[__mul(cos_fn(alpha1), r1), __mul(sin_fn(alpha1), r1)], [__mul(cos_fn(alpha2), (__sub(r1, len2))), __mul(sin_fn(alpha2), (__sub(r1, len2)))], [__mul(cos_fn(alpha2), (__add(r1, len2))), __mul(sin_fn(alpha2), (__add(r1, len2)))]]);
    }
    return Manifold.union([]);
  })());
    __items.push((() => {
    if ((len2 < 0.01)) {
      return CrossSection.ofPolygons([[__mul(cos_fn(alpha1), (__add(r1, len1))), __mul(sin_fn(alpha1), (__add(r1, len1)))], [__mul(cos_fn(alpha1), (__sub(r1, len1))), __mul(sin_fn(alpha1), (__sub(r1, len1)))], [__mul(cos_fn(alpha2), r1), __mul(sin_fn(alpha2), r1)]]);
    }
    return Manifold.union([]);
  })());
    __items.push((() => {
    if (((len1 >= 0.01) && (len2 >= 0.01))) {
      return CrossSection.ofPolygons([[__mul(cos_fn(alpha1), (__add(r1, len1))), __mul(sin_fn(alpha1), (__add(r1, len1)))], [__mul(cos_fn(alpha1), (__sub(r1, len1))), __mul(sin_fn(alpha1), (__sub(r1, len1)))], [__mul(cos_fn(alpha2), (__sub(r1, len2))), __mul(sin_fn(alpha2), (__sub(r1, len2)))], [__mul(cos_fn(alpha2), (__add(r1, len2))), __mul(sin_fn(alpha2), (__add(r1, len2)))]]);
    }
    return Manifold.union([]);
  })());
    return __union2d3d(__items);
  }
  __items.push(__extrude((() => {
  const __items = [];
  {
    const __step_0: any = stepsize;
    for (let i: any = stepsize; (__step_0 >= 0) ? i <= __add(1, __div(stepsize, 2)) : i >= __add(1, __div(stepsize, 2)); i += __step_0) {
      __items.push(__with_children(() => Manifold.union([]), 0, () => segment$mod(__sub(i, stepsize), min_fn(i, 1))));
    }
  }
  return __union2d3d(__items);
})(), 100, { twist: __div(__mul(180, h), hr) }));
  return __union2d3d(__items);
}

var $fn: any = 0, $fa: any = 12, $fs: any = 2;
var $vpr: any = [0, 0, 0], $vpt: any = [0, 0, 0], $vpd: any = 500;
var $parent_modules: any = 0;
var _NO_ARG: any = Symbol("NO_ARG");
const __result_items: any[] = [];
__result_items.push((console.log("version = ", version_fn()), Manifold.union([])));
__result_items.push(__with_children(() => Manifold.union([]), 0, () => screw$mod()).translate([__neg(30), 0, 0]));
__result_items.push(__with_children(() => Manifold.union([]), 0, () => nut$mod()).translate([30, 0, 0]));
__result_items.push(__with_children(() => Manifold.union([]), 0, () => spring$mod()));
export const result = __union2d3d(__result_items);