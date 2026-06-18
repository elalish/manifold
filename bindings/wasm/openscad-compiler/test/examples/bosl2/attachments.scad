// Volume: 206002.28497672992, SurfaceArea: 32859.826660373765
include <BOSL2/std.scad>


$fn=32;

cuboid([60,40,40], rounding=5, edges="Z", anchor=BOTTOM) {
    attach(TOP, BOTTOM) prismoid([60,40],[20,20], h=50, rounding1=5, rounding2=10) {
        attach(TOP) cylinder(d=20, h=30, center=false) {
            attach(TOP) cylinder(d1=50, d2=30, h=12, center=false);
        }
        attach([FRONT, BACK, LEFT, RIGHT]) cylinder(d1=14, d2=5, h=20) {
            attach(TOP, LEFT, overlap=5) prismoid([30,20], [20,20], h=10, shift=[-7,0]);
        }
    }
}


// vim: expandtab tabstop=4 shiftwidth=4 softtabstop=4 nowrap
