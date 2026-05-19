import * as __rt from "../../runtime.js";
const { Manifold, CrossSection, wasm, is_undef_fn, is_bool_fn, is_num_fn, is_string_fn, is_list_fn, is_function_fn, sin_fn, cos_fn, tan_fn, asin_fn, acos_fn, atan_fn, atan2_fn, abs_fn, sign_fn, floor_fn, ceil_fn, round_fn, sqrt_fn, exp_fn, ln_fn, log_fn, min_fn, max_fn, norm_fn, cross_fn, len_fn, str_fn, chr_fn, ord_fn, concat_fn, search_fn, lookup_fn, openscad_assert_fn, __eq, __add, __sub, __mul, __div, __mod, __neg, __pos, version_fn, version_num_fn, __children_stack, __with_children, __is_finite_matrix4, __to_manifold_mat4, __safe_transform, __identity4, __safe_attach_transform, __safe_offset2d, __safe_project3d, __apply_color, __flat_map_iter, __range, __union2d3d, __difference2d3d, __intersection2d3d, __hull2d3d, __minkowski2d3d, __extrude, __revolve } = __rt;
var PI: any = __rt.PI;
var INF: any = __rt.INF;
var NAN: any = __rt.NAN;
var undef: any = __rt.undef;
var _EPSILON: any = __rt._EPSILON;

function make_ring_of$mod(radius: any, count: any): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push((() => {
  const __items = [];
  {
    const __step_0: any = 1;
    for (let a: any = 0; (__step_0 >= 0) ? a <= __sub(count, 1) : a >= __sub(count, 1); a += __step_0) {
      __items.push(((angle: any) => (children().rotate([0, 0, angle]).translate(__mul(radius, [sin_fn(angle), __neg(cos_fn(angle)), 0]))))(__div(__mul(a, 360), count)));
    }
  }
  return __union2d3d(__items);
})());
  return __union2d3d(__items);
}
function something$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(((__s: any) => Manifold.cube((is_list_fn(__s) ? __s : [__s, __s, __s]), true))(10));
  __items.push(Manifold.cylinder(12, 2, 2, 40));
  __items.push(__extrude(((__t: any, __s: any) => CrossSection.square([Math.max(0.001, len_fn(str_fn(__t)) * __s * 0.6), Math.max(0.001, __s)], false))("SCAD", 8), 2).rotate([90, 0, 0]).translate([0, 0, 12]));
  __items.push(((__s: any) => Manifold.cube((is_list_fn(__s) ? __s : [__s, __s, __s]), true))([22, 1.6, 0.4]).translate([0, 0, 12]));
  return __union2d3d(__items);
}

var $fn: any = 0, $fa: any = 12, $fs: any = 2;
var $vpr: any = [0, 0, 0], $vpt: any = [0, 0, 0], $vpd: any = 500;
var $parent_modules: any = 0;
var _NO_ARG: any = Symbol("NO_ARG");
const __result_items: any[] = [];
// children.scad - Usage of children()
// The use of children() allows to write generic modules that
// modify child modules regardless of how the child geometry
// is created.
__result_items.push(__apply_color((() => { const __childFns = [
  () => (((__s: any) => Manifold.cube((is_list_fn(__s) ? __s : [__s, __s, __s]), true))(8))
]; return __with_children((i) => (i === undefined ? __union2d3d(__childFns.map(fn => fn())) : ((i >= 0 && i < __childFns.length) ? __childFns[i]() : Manifold.union([]))), __childFns.length, () => make_ring_of$mod(/* radius = */ 15, /* count = */ 6)); })(), "red", undefined));
__result_items.push(__apply_color((() => { const __childFns = [
  () => (__difference2d3d(Manifold.sphere(5), [
  Manifold.cylinder(12, 2, 2, 0, true)
]))
]; return __with_children((i) => (i === undefined ? __union2d3d(__childFns.map(fn => fn())) : ((i >= 0 && i < __childFns.length) ? __childFns[i]() : Manifold.union([]))), __childFns.length, () => make_ring_of$mod(/* radius = */ 30, /* count = */ 12)); })(), "green", undefined));
__result_items.push(__apply_color((() => { const __childFns = [
  () => (__with_children(() => Manifold.union([]), 0, () => something$mod()))
]; return __with_children((i) => (i === undefined ? __union2d3d(__childFns.map(fn => fn())) : ((i >= 0 && i < __childFns.length) ? __childFns[i]() : Manifold.union([]))), __childFns.length, () => make_ring_of$mod(/* radius = */ 50, /* count = */ 4)); })(), "cyan", undefined));
__result_items.push((console.log("version = ", version_fn()), Manifold.union([])));
export const result = __union2d3d(__result_items);