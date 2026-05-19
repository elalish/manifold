import * as __rt from "../../runtime.js";
const { Manifold, CrossSection, wasm, is_undef_fn, is_bool_fn, is_num_fn, is_string_fn, is_list_fn, is_function_fn, sin_fn, cos_fn, tan_fn, asin_fn, acos_fn, atan_fn, atan2_fn, abs_fn, sign_fn, floor_fn, ceil_fn, round_fn, sqrt_fn, exp_fn, ln_fn, log_fn, min_fn, max_fn, norm_fn, cross_fn, len_fn, str_fn, chr_fn, ord_fn, concat_fn, search_fn, lookup_fn, openscad_assert_fn, __eq, __add, __sub, __mul, __div, __mod, __neg, __pos, version_fn, version_num_fn, __children_stack, __with_children, __is_finite_matrix4, __to_manifold_mat4, __safe_transform, __identity4, __safe_attach_transform, __safe_offset2d, __safe_project3d, __apply_color, __flat_map_iter, __range, __union2d3d, __difference2d3d, __intersection2d3d, __hull2d3d, __minkowski2d3d, __extrude, __revolve } = __rt;
var PI: any = __rt.PI;
var INF: any = __rt.INF;
var NAN: any = __rt.NAN;
var undef: any = __rt.undef;
var _EPSILON: any = __rt._EPSILON;

function align_in_grid_and_add_text$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push((() => {
  if (__eq($children, 0)) {
    return __extrude(((__t: any, __s: any) => CrossSection.square([Math.max(0.001, len_fn(str_fn(__t)) * __s * 0.6), Math.max(0.001, __s)], false))("Nothing...", 6), 1);
  }
  else {
    return ((t: any) => (__union2d3d([
  __extrude(((__t: any, __s: any) => CrossSection.square([Math.max(0.001, len_fn(str_fn(__t)) * __s * 0.6), Math.max(0.001, __s)], false))(t, 6), 1),
  (() => {
  const __items = [];
  {
    const __step_0: any = 1;
    for (let y: any = 0; (__step_0 >= 0) ? y <= __sub($children, 1) : y >= __sub($children, 1); y += __step_0) {
      __items.push((() => {
  const __items = [];
  {
    const __step_0: any = 1;
    for (let x: any = 0; (__step_0 >= 0) ? x <= __sub($children, 1) : x >= __sub($children, 1); x += __step_0) {
      __items.push(children(y).scale(__add(1, __div(x, $children))).translate([__mul(15, (__sub(x, __div((__sub($children, 1)), 2)))), __add(__mul(20, y), 40), 0]));
    }
  }
  return __union2d3d(__items);
})());
    }
  }
  return __union2d3d(__items);
})()
])))((__eq($children, 1) ? "one object" : str_fn($children, " objects ")));
  }
})());
  return __union2d3d(__items);
}

var $fn: any = 0, $fa: any = 12, $fs: any = 2;
var $vpr: any = [0, 0, 0], $vpt: any = [0, 0, 0], $vpd: any = 500;
var $parent_modules: any = 0;
var _NO_ARG: any = Symbol("NO_ARG");
const __result_items: any[] = [];
// children_indexed.scad - Usage of indexed children()
// children() with a parameter allows access to a specific child
// object with children(0) being the first one. In addition the
// $children variable is automatically set to the number of child
// objects.
__result_items.push(__apply_color(__with_children(() => Manifold.union([]), 0, () => align_in_grid_and_add_text$mod()).translate([__neg(100), __neg(20), 0]), "red", undefined));
__result_items.push(__apply_color((() => { const __childFns = [
  () => (((__s: any) => Manifold.cube((is_list_fn(__s) ? __s : [__s, __s, __s]), true))(5))
]; return __with_children((i) => (i === undefined ? __union2d3d(__childFns.map(fn => fn())) : ((i >= 0 && i < __childFns.length) ? __childFns[i]() : Manifold.union([]))), __childFns.length, () => align_in_grid_and_add_text$mod()); })().translate([__neg(50), __neg(20), 0]), "yellow", undefined));
__result_items.push(__apply_color((() => { const __childFns = [
  () => (((__s: any) => Manifold.cube((is_list_fn(__s) ? __s : [__s, __s, __s]), true))(5)),
  () => (Manifold.sphere(4))
]; return __with_children((i) => (i === undefined ? __union2d3d(__childFns.map(fn => fn())) : ((i >= 0 && i < __childFns.length) ? __childFns[i]() : Manifold.union([]))), __childFns.length, () => align_in_grid_and_add_text$mod()); })().translate([0, __neg(20), 0]), "cyan", undefined));
__result_items.push(__apply_color((() => { const __childFns = [
  () => (((__s: any) => Manifold.cube((is_list_fn(__s) ? __s : [__s, __s, __s]), true))(5)),
  () => (Manifold.sphere(4)),
  () => (Manifold.cylinder(5, 4, 4))
]; return __with_children((i) => (i === undefined ? __union2d3d(__childFns.map(fn => fn())) : ((i >= 0 && i < __childFns.length) ? __childFns[i]() : Manifold.union([]))), __childFns.length, () => align_in_grid_and_add_text$mod()); })().translate([50, __neg(20), 0]), "green", undefined));
__result_items.push((console.log("version = ", version_fn()), Manifold.union([])));
export const result = __union2d3d(__result_items);