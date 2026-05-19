// 2D geometry - union of circle and square (CrossSection, not thin extrusion)
union() {
    circle(r=5);
    square([8, 4], center=true);
}
