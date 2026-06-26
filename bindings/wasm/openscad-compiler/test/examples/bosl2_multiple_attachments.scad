// Volume: 25188.426357159336, SurfaceArea: 7634.940891372288
include <BOSL2/std.scad>;

diff()
cuboid([60,40,10])
{
    attach(TOP+LEFT) cyl(h=20,r=3);
    attach(TOP+RIGHT) cyl(h=20,r=3);
    attach(BOT+LEFT) cyl(h=20,r=3);
    attach(BOT+RIGHT) cyl(h=20,r=3);
}