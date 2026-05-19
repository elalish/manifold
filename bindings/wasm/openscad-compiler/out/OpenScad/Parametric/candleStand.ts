import * as __rt from "../../runtime.js";
const { Manifold, CrossSection, wasm, is_undef_fn, is_bool_fn, is_num_fn, is_string_fn, is_list_fn, is_function_fn, sin_fn, cos_fn, tan_fn, asin_fn, acos_fn, atan_fn, atan2_fn, abs_fn, sign_fn, floor_fn, ceil_fn, round_fn, sqrt_fn, exp_fn, ln_fn, log_fn, min_fn, max_fn, norm_fn, cross_fn, len_fn, str_fn, chr_fn, ord_fn, concat_fn, search_fn, lookup_fn, openscad_assert_fn, __eq, __add, __sub, __mul, __div, __mod, __neg, __pos, version_fn, version_num_fn, __children_stack, __with_children, __is_finite_matrix4, __to_manifold_mat4, __safe_transform, __identity4, __safe_attach_transform, __safe_offset2d, __safe_project3d, __apply_color, __flat_map_iter, __range, __union2d3d, __difference2d3d, __intersection2d3d, __hull2d3d, __minkowski2d3d, __extrude, __revolve } = __rt;
var PI: any = __rt.PI;
var INF: any = __rt.INF;
var NAN: any = __rt.NAN;
var undef: any = __rt.undef;
var _EPSILON: any = __rt._EPSILON;

/*[ Candle Stand ]*/
//Length of candle stand
var length: any = 50; // [70:large,50:medium,30:small]
//Radius of ring of stand
var radius: any = 25;
/* [ Number of candle holders ]*/
// Number of candle holders
var count: any = 7; //[3:14]
//Do you want center Candle
var centerCandle: any = true;
/* [ Candle Holder ]*/
//Length of candle holder
var candleSize: any = 7;
//Width of candle holder
var width: any = 4;
//Size of hole for candle holder
var holeSize: any = 3;
var CenterCandleWidth: any = 4;
/*[Properties of support]*/
var heightOfSupport: any = 3;
var widthOfSupport: any = 3;
/*[Properties of Ring]*/
var heightOfRing: any = 4;
var widthOfRing: any = 23;
//make ring with candle holders
function make$mod(radius: any, count: any, candleSize: any, length: any): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  var __save_$fa: any = $fa;
  var __save_$fs: any = $fs;
  $fa = 0.5;
  $fs = 0.5;
  __items.push(__difference2d3d(__union2d3d([
  //making holders
(() => { const __childFns = [
  () => (Manifold.cylinder(candleSize, width, width))
]; return __with_children((i) => (i === undefined ? __union2d3d(__childFns.map(fn => fn())) : ((i >= 0 && i < __childFns.length) ? __childFns[i]() : Manifold.union([]))), __childFns.length, () => make_ring_of$mod(radius, count)); })(),
  //Attaching holders to stand
(() => {
  const __items = [];
  {
    const __step_0: any = 1;
    for (let a: any = 0; (__step_0 >= 0) ? a <= __sub(count, 1) : a >= __sub(count, 1); a += __step_0) {
      __items.push(((__s: any) => Manifold.cube((is_list_fn(__s) ? __s : [__s, __s, __s]), false))([radius, widthOfSupport, heightOfSupport]).translate([0, __div(__neg(width), 2), 0]).rotate(__div(__mul(a, 360), count)));
    }
  }
  return __union2d3d(__items);
})(),
  // make ring
__extrude(__difference2d3d(CrossSection.circle(radius), [
  CrossSection.circle(widthOfRing)
]), heightOfRing)
]), [
  //Making holes in candle holder
(() => { const __childFns = [
  () => (Manifold.cylinder(__add(candleSize, 1), holeSize, holeSize))
]; return __with_children((i) => (i === undefined ? __union2d3d(__childFns.map(fn => fn())) : ((i >= 0 && i < __childFns.length) ? __childFns[i]() : Manifold.union([]))), __childFns.length, () => make_ring_of$mod(radius, count)); })()
]));
  try {
    return __union2d3d(__items);
  } finally {
    $fa = __save_$fa;
    $fs = __save_$fs;
  }
}
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
      __items.push(((angle: any) => (children().translate(__mul(radius, [cos_fn(angle), __neg(sin_fn(angle)), 0]))))(__div(__mul(a, 360), count)));
    }
  }
  return __union2d3d(__items);
})());
  return __union2d3d(__items);
}

var $fn: any = 0, $fa: any = 12, $fs: any = 2;
var $vpr: any = [0, 0, 0], $vpt: any = [0, 0, 0], $vpd: any = 500;
var $parent_modules: any = 0;
var $fa: any;
var $fs: any;
var _NO_ARG: any = Symbol("NO_ARG");
const __result_items: any[] = [];
// Center stand
__result_items.push(Manifold.cylinder(length, __sub(width, 2), __sub(width, 2)));
//Create center candle
__result_items.push((() => {
  if (centerCandle) {
    return (() => {
  var $fn: any = 360;
  return __union2d3d([
    Manifold.cylinder(candleSize, CenterCandleWidth, CenterCandleWidth),
    Manifold.cylinder(__add(candleSize, 1), __sub(CenterCandleWidth, 2), __sub(CenterCandleWidth, 2))
  ]);
})();
  }
  else {
    return Manifold.sphere(CenterCandleWidth);
  }
})().translate([0, 0, __sub(length, __div(candleSize, 2))]));
//make ring 
__result_items.push(__union2d3d([
  __with_children(() => Manifold.union([]), 0, () => make$mod(radius, count, candleSize, length)),
  //make bottom cover for candle holders
(() => { const __childFns = [
  () => (Manifold.cylinder(1, width, width))
]; return __with_children((i) => (i === undefined ? __union2d3d(__childFns.map(fn => fn())) : ((i >= 0 && i < __childFns.length) ? __childFns[i]() : Manifold.union([]))), __childFns.length, () => make_ring_of$mod(radius, count)); })()
]).translate([0, 0, __sub(length, __div(candleSize, 2))]));
//Base of candle stand
__result_items.push((() => {
  const __items = [];
  {
    const __step_0: any = 1;
    for (let a: any = 0; (__step_0 >= 0) ? a <= __sub(count, 1) : a >= __sub(count, 1); a += __step_0) {
      __items.push(((__s: any) => Manifold.cube((is_list_fn(__s) ? __s : [__s, __s, __s]), false))([radius, widthOfSupport, heightOfSupport]).translate([0, __div(__neg(width), 2), 0]).rotate(__div(__mul(a, 360), count)));
    }
  }
  return __union2d3d(__items);
})());
export const result = __union2d3d(__result_items);