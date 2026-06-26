include <BOSL2/std.scad>
include <BOSL2/beziers.scad>

$fn = 24;


piece_size = 40;
pieces = [5,5];
seed = 8885;

difference() {
    square(pieces*piece_size, center=true);
    jigsaw_puzzle_cuts(piece_size, pieces, seed=seed);
}


module jigsaw_puzzle_cuts(piece_size=50, pieces=[5,5], kerf=0.2, seed=8889)
{
    function js_sect(
        size,  // 0.5 to 2
        shift, // -10 to 10
        skew,  // -15 to 15
        pinch, // 0.5 to 2
        flip   // boolean
    ) = let(
        jigx = 0.3 * piece_size,
        jigy = 0.2 * piece_size,
        pad = (piece_size - jigx) / 2,
        descr = str(
            "jigsaw",
            " ", format_int(jigx*size), "x", format_int(jigy*size),
            round(skew)==0? "" : str(
                " skew:", format_int(skew)
            ),
            round(pinch*100)==100? "" : str(
                " pinch:", format_int(pinch*100)
            ),
            flip? " yflip" : ""
        )
    ) [pad+shift, descr, pad-shift];

    function jig_path(line_seed) =
        let(
            sw = piece_size * 0.05,
            seeds = rands(1, 999999, 5, seed_value=line_seed),
            size  = rands(0.8, 1.5, 1, seed_value=seeds[0])[0],
            shift = rands(-sw, +sw, 1, seed_value=seeds[1])[0],
            skew  = rands(-15, +15, 1, seed_value=seeds[2])[0],
            pinch = rands(0.75, 1.25, 1, seed_value=seeds[3])[0],
            flip  = rands(0, 1, 1, seed_value=seeds[4])[0] >= 0.5
        )
        js_sect(size, shift, skew, pinch, flip);

    function jig_altpath(w,n,seed) =
        let(
            variance = piece_size / 10,
            randoms = rands(-variance, variance, n*3+1, seed_value=seed),
            dx = w/n/3,
            bez = [
                [-w/2, randoms[0]],
                [-w/2+dx, randoms[0]],
                for (i = [1:1:n-1]) let(
                        x = i * w / n -w/2
                    ) each [
                        [x - dx, randoms[i]],
                        [x, randoms[i]],
                        [x + dx, randoms[i]],
                    ],
                [+w/2-dx, last(randoms)],
                [+w/2, last(randoms)]
            ],
            path = bezpath_curve(bez, splinesteps=8)
        ) path;

    seed_set = rands(0, 999999, pieces.x + pieces.y + 2, seed_value=seed);

    for (j = [0:1:pieces.x-2]) {
        line_seeds = rands(0, 999999, pieces.y + 1, seed_value=seed_set[j]);
        pathdesc = [
            piece_size/2,
            for (i = [0:1:pieces.y-1])
                each jig_path(line_seeds[i]),
            piece_size/2,
        ];
        altpath = jig_altpath(pieces.y*piece_size, pieces.y, seed_set[j]+8);
        path = partition_path(pathdesc, altpath=altpath);
        fwd((pieces.x-1)*piece_size/2)
            back((j+0.5)*piece_size)
                stroke(path, dots="butt", width=kerf);
    }

    for (j = [0:1:pieces.y-2]) {
        line_seeds = rands(0, 999999, pieces.x + 1, seed_value=seed_set[j+pieces.x]);
        pathdesc = [
            piece_size/2,
            for (i = [0:1:pieces.x-1])
                each jig_path(line_seeds[i]),
            piece_size/2,
        ];
        altpath = jig_altpath(pieces.x*piece_size, pieces.x, seed_set[j]+8);
        path = partition_path(pathdesc, altpath=altpath);
        left((pieces.y-1)*piece_size/2)
            right((j+0.5)*piece_size)
                zrot(90)
                    stroke(path, dots="butt", width=kerf);
    }
}


