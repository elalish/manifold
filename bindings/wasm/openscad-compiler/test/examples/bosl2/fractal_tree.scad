// Volume: 239715547.43132645, SurfaceArea: 30055378.305346366
include <BOSL2/std.scad>
module tree(l=1500, sc=0.7, depth=10)
    recolor("lightgray")
    cylinder(h=l, d1=l/5, d2=l/5*sc)
        attach(TOP)
            if (depth>0)
                zrot(90)
                zrot_copies(n=2)
                yrot(30) tree(depth=depth-1, l=l*sc, sc=sc);
            else
                recolor("springgreen")
                yscale(0.67)
                teardrop(d=l*3, l=1, anchor=BOT, spin=90);
tree();


// vim: expandtab tabstop=4 shiftwidth=4 softtabstop=4 nowrap
