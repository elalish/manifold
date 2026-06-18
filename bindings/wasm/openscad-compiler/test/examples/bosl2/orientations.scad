// Volume: 32016.145972482103, SurfaceArea: 9753.328364927065
include <BOSL2/std.scad>

// Shows all the orientations on cubes in their correct rotations.

orientations = [
    RIGHT,  BACK,    UP,
    LEFT,   FWD,     DOWN,
];
axiscolors = ["red", "forestgreen", "dodgerblue"];
axisdiam = 0.5;
axislen = 12;
axislbllen = 15;

module orient_cube(ang) {
    color("lightgray") cube(20, center=true);
    color(axiscolors.x) up  ((20-1)/2+0.01) back ((20-1)/2+0.01) cube([18,1,1], center=true);
    color(axiscolors.y) up  ((20-1)/2+0.01) right((20-1)/2+0.01) cube([1,18,1], center=true);
    color(axiscolors.z) back((20-1)/2+0.01) right((20-1)/2+0.01) cube([1,1,18], center=true);
    for (axis=[0:2], neg=[0:1]) {
        idx = axis + 3*neg;
        labels = [
            "RIGHT",  "BACK",    "UP",
            "LEFT",   "FWD",     "DOWN"
        ];
        rot(ang, from=UP, to=orientations[idx]) {
            up(10) {
                back(4) color("black") text3d(text=str("spin=",ang), size=2.5);
                fwd(2) color(axiscolors[axis]) text3d(text="orient=", size=2.5);
                fwd(6) color(axiscolors[axis]) text3d(text=labels[idx], size=2.5);
            }
        }
    }
}


module text3d(text, h=0.01, size=3) {
    linear_extrude(height=h, convexity=10) {
        text(text=text, size=size, valign="center", halign="center");
    }
}

module dottedline(l, d) for(y = [0:d*3:l]) up(y) sphere(d=d);

module orient_cubes() {
    // X axis
    color(axiscolors[0]) {
        yrot( 90) cylinder(h=axislen, d=axisdiam, center=false);
        right(axislbllen) rot([90,0,0]) text3d(text="X+");
        yrot(-90) dottedline(l=axislen, d=axisdiam);
        left(axislbllen) rot([90,0,180]) text3d(text="X-");
    }
    // Y axis
    color(axiscolors[1]) {
        xrot(-90) cylinder(h=axislen, d=axisdiam, center=false);
        back(axislbllen) rot([90,0,90]) text3d(text="Y+");
        xrot( 90) dottedline(l=axislen, d=axisdiam);
        fwd(axislbllen) rot([90,0,-90]) text3d(text="Y-");
    }
    // Z axis
    color(axiscolors[2])  {
        cylinder(h=axislen, d=axisdiam, center=false);
        up(axislbllen) rot([0,-90,90+$vpr[2]]) text3d(text="Z+");
        xrot(180) dottedline(l=axislen, d=axisdiam);
        down(axislbllen) rot([0,90,-90+$vpr[2]]) text3d(text="Z-");
    }

    for (ang = [0:90:270]) {
        off = rot(p=40*BACK,ang);
        translate(off) {
            orient_cube(ang);
        }
    }
}


orient_cubes();



// vim: expandtab tabstop=4 shiftwidth=4 softtabstop=4 nowrap
