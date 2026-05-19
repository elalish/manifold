import * as __rt from "../../runtime.js";
const { Manifold, CrossSection, wasm, is_undef_fn, is_bool_fn, is_num_fn, is_string_fn, is_list_fn, is_function_fn, sin_fn, cos_fn, tan_fn, asin_fn, acos_fn, atan_fn, atan2_fn, abs_fn, sign_fn, floor_fn, ceil_fn, round_fn, sqrt_fn, exp_fn, ln_fn, log_fn, min_fn, max_fn, norm_fn, cross_fn, len_fn, str_fn, chr_fn, ord_fn, concat_fn, search_fn, lookup_fn, openscad_assert_fn, __eq, __add, __sub, __mul, __div, __mod, __neg, __pos, version_fn, version_num_fn, __children_stack, __with_children, __is_finite_matrix4, __to_manifold_mat4, __safe_transform, __identity4, __safe_attach_transform, __safe_offset2d, __safe_project3d, __apply_color, __flat_map_iter, __range, __union2d3d, __difference2d3d, __intersection2d3d, __hull2d3d, __minkowski2d3d, __extrude, __revolve } = __rt;
var PI: any = __rt.PI;
var INF: any = __rt.INF;
var NAN: any = __rt.NAN;
var undef: any = __rt.undef;
var _EPSILON: any = __rt._EPSILON;

// size is a vector [w, h, d]
function roundedBox$mod(size: any, radius: any, sidesonly: any): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  var rot: any = [[0, 0, 0], [90, 0, 90], [90, 90, 0]];
  __items.push((() => {
  if (sidesonly) {
    return __union2d3d([
  ((__s: any) => Manifold.cube((is_list_fn(__s) ? __s : [__s, __s, __s]), true))(__sub(size, [__mul(2, radius), 0, 0])),
  ((__s: any) => Manifold.cube((is_list_fn(__s) ? __s : [__s, __s, __s]), true))(__sub(size, [0, __mul(2, radius), 0])),
  (() => {
  const __items = [];
  {
    const __iter_0: any = [__sub(radius, __div(size[0], 2)), __add(__neg(radius), __div(size[0], 2))];
    for (const x of __iter_0) {
      {
        const __iter_1: any = [__sub(radius, __div(size[1], 2)), __add(__neg(radius), __div(size[1], 2))];
        for (const y of __iter_1) {
          __items.push(Manifold.cylinder(size[2], radius, radius, 0, true).translate([x, y, 0]));
        }
      }
    }
  }
  return __union2d3d(__items);
})()
]);
  }
  else {
    return __union2d3d([
  ((__s: any) => Manifold.cube((is_list_fn(__s) ? __s : [__s, __s, __s]), true))([size[0], __sub(size[1], __mul(radius, 2)), __sub(size[2], __mul(radius, 2))]),
  ((__s: any) => Manifold.cube((is_list_fn(__s) ? __s : [__s, __s, __s]), true))([__sub(size[0], __mul(radius, 2)), size[1], __sub(size[2], __mul(radius, 2))]),
  ((__s: any) => Manifold.cube((is_list_fn(__s) ? __s : [__s, __s, __s]), true))([__sub(size[0], __mul(radius, 2)), __sub(size[1], __mul(radius, 2)), size[2]]),
  (() => {
  const __items = [];
  {
    const __step_0: any = 1;
    for (let axis: any = 0; (__step_0 >= 0) ? axis <= 2 : axis >= 2; axis += __step_0) {
      __items.push((() => {
  const __items = [];
  {
    const __iter_0: any = [__sub(radius, __div(size[axis], 2)), __add(__neg(radius), __div(size[axis], 2))];
    for (const x of __iter_0) {
      {
        const __iter_1: any = [__sub(radius, __div(size[__mod((__add(axis, 1)), 3)], 2)), __add(__neg(radius), __div(size[__mod((__add(axis, 1)), 3)], 2))];
        for (const y of __iter_1) {
          __items.push(Manifold.cylinder(__sub(size[__mod((__add(axis, 2)), 3)], __mul(2, radius)), radius, radius, 0, true).translate([x, y, 0]).rotate(rot[axis]));
        }
      }
    }
  }
  return __union2d3d(__items);
})());
    }
  }
  return __union2d3d(__items);
})(),
  (() => {
  const __items = [];
  {
    const __iter_0: any = [__sub(radius, __div(size[0], 2)), __add(__neg(radius), __div(size[0], 2))];
    for (const x of __iter_0) {
      {
        const __iter_1: any = [__sub(radius, __div(size[1], 2)), __add(__neg(radius), __div(size[1], 2))];
        for (const y of __iter_1) {
          {
            const __iter_2: any = [__sub(radius, __div(size[2], 2)), __add(__neg(radius), __div(size[2], 2))];
            for (const z of __iter_2) {
              __items.push(Manifold.sphere(radius).translate([x, y, z]));
            }
          }
        }
      }
    }
  }
  return __union2d3d(__items);
})()
]);
  }
})());
  return __union2d3d(__items);
}

var $fn: any = 0, $fa: any = 12, $fs: any = 2;
var $vpr: any = [0, 0, 0], $vpt: any = [0, 0, 0], $vpd: any = 500;
var $parent_modules: any = 0;
var _NO_ARG: any = Symbol("NO_ARG");
const __result_items: any[] = [];
__result_items.push((console.log("version = ", version_fn()), Manifold.union([])));
__result_items.push(__with_children(() => Manifold.union([]), 0, () => roundedBox$mod([20, 30, 40], 5, true)).translate([__neg(15), 0, 0]));
__result_items.push(__with_children(() => Manifold.union([]), 0, () => roundedBox$mod([20, 30, 40], 5, false)).translate([15, 0, 0]));
export const result = __union2d3d(__result_items);