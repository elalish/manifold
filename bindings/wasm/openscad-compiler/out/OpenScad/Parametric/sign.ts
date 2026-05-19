import * as __rt from "../../runtime.js";
const { Manifold, CrossSection, wasm, is_undef_fn, is_bool_fn, is_num_fn, is_string_fn, is_list_fn, is_function_fn, sin_fn, cos_fn, tan_fn, asin_fn, acos_fn, atan_fn, atan2_fn, abs_fn, sign_fn, floor_fn, ceil_fn, round_fn, sqrt_fn, exp_fn, ln_fn, log_fn, min_fn, max_fn, norm_fn, cross_fn, len_fn, str_fn, chr_fn, ord_fn, concat_fn, search_fn, lookup_fn, openscad_assert_fn, __eq, __add, __sub, __mul, __div, __mod, __neg, __pos, version_fn, version_num_fn, __children_stack, __with_children, __is_finite_matrix4, __to_manifold_mat4, __safe_transform, __identity4, __safe_attach_transform, __safe_offset2d, __safe_project3d, __apply_color, __flat_map_iter, __range, __union2d3d, __difference2d3d, __intersection2d3d, __hull2d3d, __minkowski2d3d, __extrude, __revolve } = __rt;
var PI: any = __rt.PI;
var INF: any = __rt.INF;
var NAN: any = __rt.NAN;
var undef: any = __rt.undef;
var _EPSILON: any = __rt._EPSILON;

// First example of parameteric model
//   
//    syntax: 
//        //Description
//        variable=value; //Parameter
//        
//        This type of comment tells the name of group to which parameters below
//        this comment will belong 
//    
//       /*[ group name ]*/ 
//
//Below comment tells the group to which a variable will belong
/*[ properties of Sign]*/
//The resolution of the curves. Higher values give smoother curves but may increase the model render time.
var resolution: any = 10; //[10, 20, 30, 50, 100]
//The horizontal radius of the outer ellipse of the sign.
var radius: any = 80; //[60 : 200]
//Total height of the sign
var height: any = 2; //[1 : 10]
/*[ Content To be written ] */
//Message to be write 
var Message: any = "Welcome to..."; //["Welcome to...", "Happy Birthday!", "Happy Anniversary", "Congratulations", "Thank You"]
//Name of Person, company etc.
var To: any = "Parametric Designs";
var $fn: any = resolution;

var $fn: any = 0, $fa: any = 12, $fs: any = 2;
var $vpr: any = [0, 0, 0], $vpt: any = [0, 0, 0], $vpd: any = 500;
var $parent_modules: any = 0;
var _NO_ARG: any = Symbol("NO_ARG");
const __result_items: any[] = [];
__result_items.push(__difference2d3d(Manifold.cylinder(__mul(2, height), radius, radius, 0, true), [
  Manifold.cylinder(__add(height, 1), __sub(radius, 10), __sub(radius, 10), 0, true).translate([0, 0, height])
]).scale([1, 0.5]));
__result_items.push(__extrude(__union2d3d([
  ((__t: any, __s: any) => CrossSection.square([Math.max(0.001, len_fn(str_fn(__t)) * __s * 0.6), Math.max(0.001, __s)], false))(Message, 10).translate([0, __neg(__neg(4))]),
  ((__t: any, __s: any) => CrossSection.square([Math.max(0.001, len_fn(str_fn(__t)) * __s * 0.6), Math.max(0.001, __s)], false))(To, 10).translate([0, __neg(16)])
]), height));
export const result = __union2d3d(__result_items);