// Volume: 4112.861654538725, SurfaceArea: 1245.2051191189146
module all_transparent() {
 %children();
}

module first_transparent() {
 %children(0);
}

difference() {
  sphere(r=10);
  all_transparent() cylinder(h=30, r=6, center=true);
  first_transparent() rotate([90,0,0]) cylinder(h=30, r=6, center=true);
}
