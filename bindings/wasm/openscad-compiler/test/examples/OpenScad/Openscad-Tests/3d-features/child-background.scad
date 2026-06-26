// Volume: 4112.861654538725, SurfaceArea: 1245.2051191189146
module transparent() {
 %children();
}

difference() {
  sphere(r=10);
  transparent() cylinder(h=30, r=6, center=true);
}
