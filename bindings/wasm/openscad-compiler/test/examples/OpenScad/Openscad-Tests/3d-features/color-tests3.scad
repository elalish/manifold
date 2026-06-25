// Volume: 2284.7681905214977, SurfaceArea: 1341.1703131100762
difference() {
  sphere(10);
  color("red") translate([0,0,10]) cube(16, center=true);
  translate([0,0,5]) sphere(8);
}