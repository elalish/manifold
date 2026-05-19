// Boolean operations and transforms
$fn = 48;

union() {
    // Main body
    sphere(r=20);

    // Arms
    rotate([0, 90, 0])
        cylinder(h=60, r=5, center=true);

    // Hat
    translate([0, 0, 18])
        cylinder(h=15, r1=12, r2=2);
}
