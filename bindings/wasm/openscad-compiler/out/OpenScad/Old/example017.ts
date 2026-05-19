import * as __rt from "../../runtime.js";
const { Manifold, CrossSection, wasm, is_undef_fn, is_bool_fn, is_num_fn, is_string_fn, is_list_fn, is_function_fn, sin_fn, cos_fn, tan_fn, asin_fn, acos_fn, atan_fn, atan2_fn, abs_fn, sign_fn, floor_fn, ceil_fn, round_fn, sqrt_fn, exp_fn, ln_fn, log_fn, min_fn, max_fn, norm_fn, cross_fn, len_fn, str_fn, chr_fn, ord_fn, concat_fn, search_fn, lookup_fn, openscad_assert_fn, __eq, __add, __sub, __mul, __div, __mod, __neg, __pos, version_fn, version_num_fn, __children_stack, __with_children, __is_finite_matrix4, __to_manifold_mat4, __safe_transform, __identity4, __safe_attach_transform, __safe_offset2d, __safe_project3d, __apply_color, __flat_map_iter, __range, __union2d3d, __difference2d3d, __intersection2d3d, __hull2d3d, __minkowski2d3d, __extrude, __revolve } = __rt;
var PI: any = __rt.PI;
var INF: any = __rt.INF;
var NAN: any = __rt.NAN;
var undef: any = __rt.undef;
var _EPSILON: any = __rt._EPSILON;

// To render the DXF file from the command line:
// openscad -o example017.dxf -D'mode="parts"' example017.scad
//Mode can be either "parts",  "exploded" or "assembled".
var mode: any = "assembled"; // ["parts",  "exploded", "assembled"]
var thickness: any = 6;
var locklen1: any = 15;
var locklen2: any = 10;
var boltlen: any = 15;
var midhole: any = 10;
var inner1_to_inner2: any = 50;
var total_height: any = 80;
function shape_tripod$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  var x1: any = 0;
  var x2: any = __add(x1, thickness);
  var x3: any = __add(x2, locklen1);
  var x4: any = __add(x3, thickness);
  var x5: any = __add(x4, inner1_to_inner2);
  var x6: any = __sub(x5, thickness);
  var x7: any = __sub(x6, locklen2);
  var x8: any = __sub(x7, thickness);
  var x9: any = __sub(x8, thickness);
  var x10: any = __sub(x9, thickness);
  var y1: any = 0;
  var y2: any = __add(y1, thickness);
  var y3: any = __add(y2, thickness);
  var y4: any = __add(y3, thickness);
  var y5: any = __sub(__add(y3, total_height), __mul(3, thickness));
  var y6: any = __add(y5, thickness);
  __items.push(__union2d3d([
  __difference2d3d(CrossSection.ofPolygons([[x1, y2], [x2, y2], [x2, y1], [x3, y1], [x3, y2], [x4, y2], [x4, y1], [x5, y1], [__add(x5, thickness), y3], [x5, y4], [x5, y5], [x6, y5], [x6, y6], [x7, y6], [x7, y5], [x8, y5], [x8, y6], [x9, y5], [x9, y4], [x10, y3], [x2, y3]]), [
  CrossSection.circle(thickness).translate([x10, y4]),
  CrossSection.circle(thickness).translate([__add(x5, thickness), y4])
]),
  ((__s: any) => CrossSection.square((is_list_fn(__s) ? __s : [__s, __s]), false))([__sub(boltlen, thickness), __mul(thickness, 2)]).translate([x5, y1]),
  CrossSection.circle(thickness).translate([__sub(__add(x5, boltlen), thickness), y2]),
  __intersection2d3d([
  CrossSection.circle(thickness),
  ((__s: any) => CrossSection.square((is_list_fn(__s) ? __s : [__s, __s]), false))(__mul(thickness, 2)).translate([__mul(__neg(thickness), 2), 0])
]).translate([x2, y2]),
  __intersection2d3d([
  CrossSection.circle(thickness),
  ((__s: any) => CrossSection.square((is_list_fn(__s) ? __s : [__s, __s]), false))(__mul(thickness, 2)).translate([__mul(__neg(thickness), 2), 0])
]).translate([x8, y5])
]));
  return __union2d3d(__items);
}
function shape_inner_disc$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(__difference2d3d(CrossSection.circle(__add(__add(__add(midhole, boltlen), __mul(2, thickness)), locklen2)), [
  (() => {
  const __items = [];
  {
    const __iter_0: any = [0, 120, 240];
    for (const alpha of __iter_0) {
      __items.push(((__s: any) => CrossSection.square((is_list_fn(__s) ? __s : [__s, __s]), true))([thickness, locklen2]).translate([0, __add(__add(__add(midhole, boltlen), thickness), __div(locklen2, 2))]).rotate(alpha));
    }
  }
  return __union2d3d(__items);
})(),
  CrossSection.circle(__add(midhole, boltlen))
]));
  return __union2d3d(__items);
}
function shape_outer_disc$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(__difference2d3d(CrossSection.circle(__add(__add(__add(__add(midhole, boltlen), inner1_to_inner2), __mul(2, thickness)), locklen1)), [
  (() => {
  const __items = [];
  {
    const __iter_0: any = [0, 120, 240];
    for (const alpha of __iter_0) {
      __items.push(((__s: any) => CrossSection.square((is_list_fn(__s) ? __s : [__s, __s]), true))([thickness, locklen1]).translate([0, __add(__add(__add(__add(midhole, boltlen), inner1_to_inner2), thickness), __div(locklen1, 2))]).rotate(alpha));
    }
  }
  return __union2d3d(__items);
})(),
  CrossSection.circle(__add(__add(midhole, boltlen), inner1_to_inner2))
]));
  return __union2d3d(__items);
}
function parts$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  var tripod_x_off: any = __add(__sub(locklen1, locklen2), inner1_to_inner2);
  var tripod_y_off: any = max_fn(__add(__add(__add(__add(midhole, boltlen), inner1_to_inner2), __mul(4, thickness)), locklen1), total_height);
  __items.push(__with_children(() => Manifold.union([]), 0, () => shape_inner_disc$mod()));
  __items.push(__with_children(() => Manifold.union([]), 0, () => shape_outer_disc$mod()));
  __items.push((() => {
  const __items = [];
  {
    const __iter_0: any = [[1, 1], [__neg(1), 1], [1, __neg(1)]];
    for (const s of __iter_0) {
      __items.push(__with_children(() => Manifold.union([]), 0, () => shape_tripod$mod()).translate([tripod_x_off, __neg(tripod_y_off)]).scale(s));
    }
  }
  return __union2d3d(__items);
})());
  return __union2d3d(__items);
}
function exploded$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(__extrude(__with_children(() => Manifold.union([]), 0, () => shape_inner_disc$mod()), thickness).translate([0, 0, __add(total_height, __mul(2, thickness))]));
  __items.push(__extrude(__with_children(() => Manifold.union([]), 0, () => shape_outer_disc$mod()), thickness));
  __items.push(__apply_color((() => {
  const __items = [];
  {
    const __iter_0: any = [0, 120, 240];
    for (const alpha of __iter_0) {
      __items.push(__extrude(__with_children(() => Manifold.union([]), 0, () => shape_tripod$mod()), thickness).rotate([90, 0, __neg(90)]).translate([0, __add(__add(__add(__add(__mul(thickness, 2), locklen1), inner1_to_inner2), boltlen), midhole), __mul(1.5, thickness)]).rotate(alpha));
    }
  }
  return __union2d3d(__items);
})(), [0.7, 0.7, 1], undefined));
  return __union2d3d(__items);
}
function bottle$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  var r: any = __add(boltlen, midhole);
  var h: any = __sub(total_height, __mul(thickness, 2));
  __items.push(__revolve(__union2d3d([
  ((__s: any) => CrossSection.square((is_list_fn(__s) ? __s : [__s, __s]), false))([r, h]),
  __intersection2d3d([
  ((__s: any) => CrossSection.square((is_list_fn(__s) ? __s : [__s, __s]), false))([r, r]),
  CrossSection.circle(r).scale([1, 0.7])
]).translate([0, h]),
  __intersection2d3d([
  ((__s: any) => CrossSection.square((is_list_fn(__s) ? __s : [__s, __s]), false))([__div(r, 2), r]).translate([0, __div(__neg(r), 2)]),
  CrossSection.circle(__div(r, 2))
]).translate([0, __add(h, r)])
]), 0, 360));
  return __union2d3d(__items);
}
function assembled$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(__extrude(__with_children(() => Manifold.union([]), 0, () => shape_inner_disc$mod()), thickness).translate([0, 0, __sub(total_height, thickness)]));
  __items.push(__extrude(__with_children(() => Manifold.union([]), 0, () => shape_outer_disc$mod()), thickness));
  __items.push(__apply_color((() => {
  const __items = [];
  {
    const __iter_0: any = [0, 120, 240];
    for (const alpha of __iter_0) {
      __items.push(__extrude(__with_children(() => Manifold.union([]), 0, () => shape_tripod$mod()), thickness).rotate([90, 0, __neg(90)]).translate([0, __add(__add(__add(__add(__mul(thickness, 2), locklen1), inner1_to_inner2), boltlen), midhole), 0]).rotate(alpha));
    }
  }
  return __union2d3d(__items);
})(), [0.7, 0.7, 1], undefined));
  __items.push(__with_children(() => Manifold.union([]), 0, () => bottle$mod()).translate([0, 0, __mul(thickness, 2)]));
  return __union2d3d(__items);
}

var $fn: any = 0, $fa: any = 12, $fs: any = 2;
var $vpr: any = [0, 0, 0], $vpt: any = [0, 0, 0], $vpd: any = 500;
var $parent_modules: any = 0;
var _NO_ARG: any = Symbol("NO_ARG");
const __result_items: any[] = [];
__result_items.push((console.log("version = ", version_fn()), Manifold.union([])));
__result_items.push((() => {
  if (__eq(mode, "parts")) {
    return __with_children(() => Manifold.union([]), 0, () => parts$mod());
  }
  return Manifold.union([]);
})());
__result_items.push((() => {
  if (__eq(mode, "exploded")) {
    return __with_children(() => Manifold.union([]), 0, () => exploded$mod());
  }
  return Manifold.union([]);
})());
__result_items.push((() => {
  if (__eq(mode, "assembled")) {
    return __with_children(() => Manifold.union([]), 0, () => assembled$mod());
  }
  return Manifold.union([]);
})());
export const result = __union2d3d(__result_items);