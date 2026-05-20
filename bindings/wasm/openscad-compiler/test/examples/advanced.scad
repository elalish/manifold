wall = 2;
size = 30;
hole_r = 10;

// Difference: cube with cylindrical hole
difference() {
    cube([size, size, size], center=true);
    cylinder(h=size + 1, r=hole_r, center=true, $fn=64);
}
