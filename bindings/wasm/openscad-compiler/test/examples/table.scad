// Volume: 16494.879516308847, SurfaceArea: 11791.08604753301
module leg(h = 40, r = 3) {
    cylinder(h=h, r=r, $fn=32);
}

module table(w = 80, d = 50, h = 40, top_t = 3) {
    // tabletop
    translate([0, 0, h])
        cube([w, d, top_t]);

    // four legs
    for (x = [0, w - 6], y = [0, d - 6]) {
        translate([x + 3, y + 3, 0])
            leg(h=h);
    }
}

table();
