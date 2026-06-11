// Volume: 31945.078370375744, SurfaceArea: 9186.79158096168
// Viewport test
$vpr = [75, 0, 45];   // rotated 75° on X, 45° on Z — angled view
$vpt = [0, 0, 10];    // look at a point slightly above the origin
$vpd = 200;            // camera distance
$vpf = 30;             // narrower field of view than default

// Simple scene — a few shapes at different positions so the camera angle is obvious
difference() {
  cube([40, 40, 20], center = true);
  cylinder(h = 25, r = 10, center = true, $fn = 48);
}

translate([25, 0, 0])
  sphere(r = 8, $fn = 48);

translate([-25, 0, 0])
  sphere(r = 8, $fn = 48);

translate([0, 25, 0])
  cylinder(h = 30, r = 5, $fn = 32);