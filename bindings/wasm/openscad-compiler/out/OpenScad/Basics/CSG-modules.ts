import * as __rt from "../../runtime.js";
const { Manifold, CrossSection, wasm, is_undef_fn, is_bool_fn, is_num_fn, is_string_fn, is_list_fn, is_function_fn, sin_fn, cos_fn, tan_fn, asin_fn, acos_fn, atan_fn, atan2_fn, abs_fn, sign_fn, floor_fn, ceil_fn, round_fn, sqrt_fn, exp_fn, ln_fn, log_fn, min_fn, max_fn, norm_fn, cross_fn, len_fn, str_fn, chr_fn, ord_fn, concat_fn, search_fn, lookup_fn, openscad_assert_fn, __eq, __add, __sub, __mul, __div, __mod, __neg, __pos, version_fn, version_num_fn, __children_stack, __with_children, __is_finite_matrix4, __to_manifold_mat4, __safe_transform, __identity4, __safe_attach_transform, __safe_offset2d, __safe_project3d, __apply_color, __flat_map_iter, __range, __union2d3d, __difference2d3d, __intersection2d3d, __hull2d3d, __minkowski2d3d, __extrude, __revolve } = __rt;
var PI: any = __rt.PI;
var INF: any = __rt.INF;
var NAN: any = __rt.NAN;
var undef: any = __rt.undef;
var _EPSILON: any = __rt._EPSILON;

// CSG-modules.scad - Basic usage of modules, if, color, $fs/$fa
// Change this to false to remove the helper geometry
var debug: any = true;
// Global resolution
var $fs: any = 0.1; // Don't generate smaller facets than 0.1 mm
var $fa: any = 5; // Don't generate larger angles than 5 degrees
// Core geometric primitives.
// These can be modified to create variations of the final object
function body$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(__apply_color(Manifold.sphere(10), "Blue", undefined));
  return __union2d3d(__items);
}
function intersector$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(__apply_color(((__s: any) => Manifold.cube((is_list_fn(__s) ? __s : [__s, __s, __s]), true))(15), "Red", undefined));
  return __union2d3d(__items);
}
function holeObject$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(__apply_color(Manifold.cylinder(20, 5, 5, 0, true), "Lime", undefined));
  return __union2d3d(__items);
}
// Various modules for visualizing intermediate components
function intersected$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(__intersection2d3d([
  __with_children(() => Manifold.union([]), 0, () => body$mod()),
  __with_children(() => Manifold.union([]), 0, () => intersector$mod())
]));
  return __union2d3d(__items);
}
function holeA$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(__with_children(() => Manifold.union([]), 0, () => holeObject$mod()).rotate([0, 90, 0]));
  return __union2d3d(__items);
}
function holeB$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(__with_children(() => Manifold.union([]), 0, () => holeObject$mod()).rotate([90, 0, 0]));
  return __union2d3d(__items);
}
function holeC$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(__with_children(() => Manifold.union([]), 0, () => holeObject$mod()));
  return __union2d3d(__items);
}
function holes$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(__union2d3d([
  __with_children(() => Manifold.union([]), 0, () => holeA$mod()),
  __with_children(() => Manifold.union([]), 0, () => holeB$mod()),
  __with_children(() => Manifold.union([]), 0, () => holeC$mod())
]));
  return __union2d3d(__items);
}
function helpers$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  // Inner module since it's only needed inside helpers
  function line$mod(): any {
    var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
    var $children: any = __c.count;
    function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
    const __items: any[] = [];
    __items.push(__apply_color(Manifold.cylinder(10, 1, 1, 0, true), "Black", undefined));
    return __union2d3d(__items);
  }
  __items.push(__union2d3d([
  __union2d3d([
  __with_children(() => Manifold.union([]), 0, () => intersected$mod()),
  __with_children(() => Manifold.union([]), 0, () => body$mod()).translate([__neg(15), 0, __neg(35)]),
  __with_children(() => Manifold.union([]), 0, () => intersector$mod()).translate([15, 0, __neg(35)]),
  __with_children(() => Manifold.union([]), 0, () => line$mod()).rotate([0, 30, 0]).translate([__neg(7.5), 0, __neg(17.5)]),
  __with_children(() => Manifold.union([]), 0, () => line$mod()).rotate([0, __neg(30), 0]).translate([7.5, 0, __neg(17.5)])
]).translate([__neg(30), 0, __neg(40)]),
  __union2d3d([
  __with_children(() => Manifold.union([]), 0, () => holes$mod()),
  __with_children(() => Manifold.union([]), 0, () => holeA$mod()).translate([__neg(10), 0, __neg(35)]),
  __with_children(() => Manifold.union([]), 0, () => holeB$mod()).translate([10, 0, __neg(35)]),
  __with_children(() => Manifold.union([]), 0, () => holeC$mod()).translate([30, 0, __neg(35)]),
  __with_children(() => Manifold.union([]), 0, () => line$mod()).rotate([0, __neg(20), 0]).translate([5, 0, __neg(17.5)]),
  __with_children(() => Manifold.union([]), 0, () => line$mod()).rotate([0, 30, 0]).translate([__neg(5), 0, __neg(17.5)]),
  __with_children(() => Manifold.union([]), 0, () => line$mod()).rotate([0, __neg(45), 0]).translate([15, 0, __neg(17.5)])
]).translate([30, 0, __neg(40)]),
  __with_children(() => Manifold.union([]), 0, () => line$mod()).rotate([0, 45, 0]).translate([__neg(20), 0, __neg(22.5)]),
  __with_children(() => Manifold.union([]), 0, () => line$mod()).rotate([0, __neg(45), 0]).translate([20, 0, __neg(22.5)])
]).scale(0.5));
  return __union2d3d(__items);
}

var $fn: any = 0, $fa: any = 12, $fs: any = 2;
var $vpr: any = [0, 0, 0], $vpt: any = [0, 0, 0], $vpd: any = 500;
var $parent_modules: any = 0;
var _NO_ARG: any = Symbol("NO_ARG");
const __result_items: any[] = [];
// Main geometry
__result_items.push(__difference2d3d(__intersection2d3d([
  __with_children(() => Manifold.union([]), 0, () => body$mod()),
  __with_children(() => Manifold.union([]), 0, () => intersector$mod())
]), [
  __with_children(() => Manifold.union([]), 0, () => holes$mod())
]));
// Helpers
__result_items.push((() => {
  if (debug) {
    return __with_children(() => Manifold.union([]), 0, () => helpers$mod());
  }
  return Manifold.union([]);
})());
__result_items.push((console.log("version = ", version_fn()), Manifold.union([])));
export const result = __union2d3d(__result_items);