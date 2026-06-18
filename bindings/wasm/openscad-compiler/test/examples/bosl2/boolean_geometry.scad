// Volume: 5472.755824197527, SurfaceArea: 93535.30697105842
include <BOSL2/std.scad>

$fn = 36;

rgn1 = [
    square(100),
    move([50,50], p=circle(d=60)),
    [[35,35],[35,65],[65,65]]
];

rgn2 = [
    [[0,0], [100,100], [100,0]],
    [[27,10], [90,73], [90,10]],
    move([70,30], p=circle(d=25))
];


polycolor=[0.5,1,0.5];
outlinecolor="black";


module showit(label, rgn, poly=polycolor, outline=outlinecolor, width=0.5) {
    move([-50,-50]) {
        if(outline) color(outline) linear_extrude(height=max(0.1,1-width)) for(path=rgn) stroke(path, width=width, closed=true);
        if(poly) color(poly) linear_extrude(height=0.1) region(rgn);
        color("black") right(50) fwd(7) linear_extrude(height=0.1) text(text=label, size=8, halign="center", valign="center");
    }
}


ydistribute(-125) {
    xdistribute(120) {
        showit("Region A", rgn1, poly=[1,0,0,0.5]);
        showit("Region B", rgn2, poly=[0,0,1,0.5]);
        union() {
            showit("A and B Overlaid", rgn1, poly=[1,0,0,0.5]);
            showit("", rgn2, poly=[0,0,1,0.5]);
        }
    }
    xdistribute(120) {
        showit("Union A+B", union(rgn1, rgn2));
        showit("Difference A-B", difference(rgn1, rgn2));
        showit("Intersection A&B", intersection(rgn1, rgn2));
        showit("Exclusive OR A^B", exclusive_or(rgn1, rgn2));
    }
}

// vim: expandtab tabstop=4 shiftwidth=4 softtabstop=4 nowrap
