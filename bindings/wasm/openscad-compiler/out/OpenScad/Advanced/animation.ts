import * as __rt from "../../runtime.js";
const { Manifold, CrossSection, wasm, is_undef_fn, is_bool_fn, is_num_fn, is_string_fn, is_list_fn, is_function_fn, sin_fn, cos_fn, tan_fn, asin_fn, acos_fn, atan_fn, atan2_fn, abs_fn, sign_fn, floor_fn, ceil_fn, round_fn, sqrt_fn, exp_fn, ln_fn, log_fn, min_fn, max_fn, norm_fn, cross_fn, len_fn, str_fn, chr_fn, ord_fn, concat_fn, search_fn, lookup_fn, openscad_assert_fn, __eq, __add, __sub, __mul, __div, __mod, __neg, __pos, version_fn, version_num_fn, __children_stack, __with_children, __is_finite_matrix4, __to_manifold_mat4, __safe_transform, __identity4, __safe_attach_transform, __safe_offset2d, __safe_project3d, __apply_color, __flat_map_iter, __range, __union2d3d, __difference2d3d, __intersection2d3d, __hull2d3d, __minkowski2d3d, __extrude, __revolve } = __rt;
var PI: any = __rt.PI;
var INF: any = __rt.INF;
var NAN: any = __rt.NAN;
var undef: any = __rt.undef;
var _EPSILON: any = __rt._EPSILON;

// animation.scad - Demo of animation usage
// The animation functionality is based simply on a variable $t
// that is changed automatically by OpenSCAD while repeatedly
// showing the model.
// To activate animation, select "View->Animate" from the
// menu; this will cause three fields to appear
// underneath the Preview console: Time, FPS & Steps.
// To commence animation, enter values into the FPS and Steps input
// fields (e.g. 5 FPS and 200 Steps for this animation).
// This is not intended to directly produce real-time animations
// but the image sequence can be exported to generate videos of
// the animation.
// Length of the 2 arm segments, change to see the effects on
// the arm movements.
//length of the red arm
var arm1_length: any = 70;
//length of the green arm
var arm2_length: any = 50;
var r: any = 2;
var $fn: any = 30;
var pos: any = position_fn($t);
// Function describing the X/Y position that should be traced
// by the arm over time.
// The $t variable will be used as parameter for this function
// so the range for t is [0..1].
function position_fn(t: any): any {
  return ((t < 0.5) ? [__sub(__mul(200, t), 50), __add(__mul(30, sin_fn(__mul(__mul(5, 360), t))), 60)] : [__mul(50, cos_fn(__mul(360, (__sub(t, 0.5))))), __add(__mul(100, __neg(sin_fn(__mul(360, (__sub(t, 0.5)))))), 60)]);
}
// Inverse kinematics functions for a scara style arm
// See http://forums.reprap.org/read.php?185,283327
function sq_fn(x: any, y: any): any {
  return __add(__mul(x, x), __mul(y, y));
}
function angB_fn(x: any, y: any, l1: any, l2: any): any {
  return __sub(180, acos_fn(__div((__sub(__add(__mul(l2, l2), __mul(l1, l1)), sq_fn(x, y))), (__mul(__mul(2, l1), l2)))));
}
function ang2_fn(x: any, y: any, l1: any, l2: any): any {
  return __sub(__sub(90, acos_fn(__div((__add(__sub(__mul(l2, l2), __mul(l1, l1)), sq_fn(x, y))), (__mul(__mul(2, l2), sqrt_fn(sq_fn(x, y))))))), atan2_fn(x, y));
}
function ang1_fn(x: any, y: any, l1: any, l2: any): any {
  return __add(ang2_fn(x, y, l1, l2), angB_fn(x, y, l1, l2));
}
// Draw an arm segment with the given color and length.
function segment$mod(col: any, l: any): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(__apply_color(__hull2d3d([
  Manifold.sphere(r),
  Manifold.sphere(r).translate([l, 0, 0])
]), col, undefined));
  return __union2d3d(__items);
}
// Draw the whole 2 segmented arm trying to reach position x/y.
// Parameters l1 and l2 are the length of the two arm segments.
function arm$mod(x: any, y: any, l1: any, l2: any): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  var a1: any = ang1_fn(x, y, l1, l2);
  var a2: any = ang2_fn(x, y, l1, l2);
  __items.push(Manifold.sphere(__mul(2, r)));
  __items.push(Manifold.cylinder(__mul(6, r), 2, 2, 0, true));
  __items.push(__with_children(() => Manifold.union([]), 0, () => segment$mod("red", l1)).rotate([0, 0, a1]));
  __items.push(__union2d3d([
  Manifold.sphere(__mul(2, r)),
  __with_children(() => Manifold.union([]), 0, () => segment$mod("green", l2)).rotate([0, 0, a2])
]).translate(__mul(l1, [cos_fn(a1), sin_fn(a1), 0])));
  __items.push(Manifold.cylinder(__mul(4, r), 0, r, 0, true).translate([x, y, __div(__neg(r), 2)]));
  return __union2d3d(__items);
}
function curve$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(CrossSection.ofPolygons([...((() => { const __r = []; for (let a = 0; a <= 1; a += 0.004) __r.push(...([position_fn(a)])); return __r; })())]));
  return __union2d3d(__items);
}
// Draws the plate and the traced function using small black cubes.
function plate$mod(): any {
  var __c: any = __children_stack.length > 0 ? __children_stack[__children_stack.length - 1] : { fn: undefined, count: 0 };
  var $children: any = __c.count;
  function children(i: any): any { return __c.fn ? __c.fn(i) : Manifold.union([]); }
  const __items: any[] = [];
  __items.push(__union2d3d([
  ((__s: any) => Manifold.cube((is_list_fn(__s) ? __s : [__s, __s, __s]), true))([150, 150, 0.1]).translate([0, 25, 0]),
  __apply_color(__extrude(__difference2d3d(__with_children(() => Manifold.union([]), 0, () => curve$mod()), [
  __safe_offset2d(__with_children(() => Manifold.union([]), 0, () => curve$mod()), __neg(1))
]), 0.1), "Black", undefined)
]).translate([0, 0, __mul(__neg(3), r)]));
  return __union2d3d(__items);
}

var $fn: any = 0, $fa: any = 12, $fs: any = 2;
var $vpr: any = [0, 0, 0], $vpt: any = [0, 0, 0], $vpd: any = 500;
var $parent_modules: any = 0;
var _NO_ARG: any = Symbol("NO_ARG");
const __result_items: any[] = [];
__result_items.push(__with_children(() => Manifold.union([]), 0, () => plate$mod()));
__result_items.push(__with_children(() => Manifold.union([]), 0, () => arm$mod(pos[0], pos[1], arm1_length, arm2_length)));
__result_items.push((console.log("version = ", version_fn()), Manifold.union([])));
export const result = __union2d3d(__result_items);