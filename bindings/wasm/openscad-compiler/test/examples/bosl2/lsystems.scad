include <BOSL2/std.scad>

function _lsystem_recurse(s, rules, lev) =
    lev<=0? s : _lsystem_recurse([
        for (
            i = 0,
            slen = len(s),
            sout = "";

            i <= slen;

            ch = s[i],
            found = search([ch], rules)[0],
            sout = str(sout, i==slen? "" : found==[]? ch : rules[found][1]),
            i = i + 1
        ) if (i==slen) sout
    ][0], rules, lev-1);


function _lsystem_to_turtle(s, step=1, angle=90, startang=0) =
    concat(
        startang? ["left", startang] : [],
        ["angle", angle, "length", step],
        [
            for (
                i = 0,
                slen = len(s);

                i <= slen;

                ch = s[i],
                cmd = (ch=="A" || ch=="B" || ch=="F")? ["move"] :
                    (ch=="+")? ["left"] :
                    (ch=="-")? ["right"] :
                    [],
                i=i+1
            ) if(i>0 && cmd!=[]) each cmd
        ]
    );


function lsystem_turtle(basis, rules, levels=5, step=1, angle=90, startang=0) =
    turtle(_lsystem_to_turtle(_lsystem_recurse(basis, rules, levels), step=step, angle=angle, startang=startang));


function dragon_curve        (levels=9,  step=1) = lsystem_turtle(levels=levels, step=step, angle=90,  "FX",        [["X", "X+YF+"], ["Y", "-FX-Y"]]);
function terdragon_curve     (levels=7,  step=1) = lsystem_turtle(levels=levels, step=step, angle=120, "F",         [["F", "F+F-F"]]);
function twindragon_curve    (levels=11, step=1) = lsystem_turtle(levels=levels, step=step, angle=90,  "FX+FX+",    [["X", "X+YF"], ["Y","FX-Y"]]);
function moore_curve         (levels=4,  step=1) = lsystem_turtle(levels=levels, step=step, angle=90,  "LFL+F+LFL", [["L", "-RF+LFL+FR-"], ["R", "+LF-RFR-FL+"]]);
function hilbert_curve       (levels=4,  step=1) = lsystem_turtle(levels=levels, step=step, angle=90,  "X",         [["X","-YF+XFX+FY-"], ["Y","+XF-YFY-FX+"]]);
function gosper_curve        (levels=4,  step=1) = lsystem_turtle(levels=levels, step=step, angle=60,  "A",         [["A", "A-B--B+A++AA+B-"], ["B", "+A-BB--B-A++A+B"]]);
function quadratic_gosper    (levels=2,  step=1) = lsystem_turtle(levels=levels, step=step, angle=90,  "-YF",       [["X", "XFX-YF-YF+FX+FX-YF-YFFX+YF+FXFXYF-FX+YF+FXFX+YF-FXYF-YF-FX+FX+YFYF-"], ["Y", "+FXFX-YF-YF+FX+FXYF+FX-YFYF-FX-YF+FXYFYF-FX-YFFX+FX+YF-YF-FX+FX+YFY"]]);
function peano_curve         (levels=4,  step=1) = lsystem_turtle(levels=levels, step=step, angle=90,  "X",         [["X","XFYFX+F+YFXFY-F-XFYFX"], ["Y","YFXFY-F-XFYFX+F+YFXFY"]]);
function koch_snowflake      (levels=4,  step=1) = lsystem_turtle(levels=levels, step=step, angle=60,  "F++F++F",   [["F","F-F++F-F"]]);
function sierpinski_arrowhead(levels=6,  step=1) = lsystem_turtle(levels=levels, step=step, angle=60,  "A",         [["A", "B-A-B"], ["B","A+B+A"]]);
function sierpinski_triangle (levels=4,  step=1) = lsystem_turtle(levels=levels, step=step, angle=120, "A-B-B",     [["A","A-B+A+B-A"], ["B","BB"]]);
function square_sierpinski   (levels=5,  step=1) = lsystem_turtle(levels=levels, step=step, angle=90,  "F+XF+F+XF", [["X","XF-F+F-XF+F+XF-F+F-X"]]);
function cesaro_curve        (levels=4,  step=1) = lsystem_turtle(levels=levels, step=step, angle=85,  "F",         [["F","F+F--F+F"]]);
function paul_bourke1        (levels=3,  step=1) = lsystem_turtle(levels=levels, step=step, angle=90,  "F+F+F+F+",  [["F","F+F-F-FF+F+F-F"]]);
function paul_bourke_triangle(levels=6,  step=1) = lsystem_turtle(levels=levels, step=step, angle=120, "F+F+F",     [["F","F-F+F"]]);
function paul_bourke_crystal (levels=4,  step=1) = lsystem_turtle(levels=levels, step=step, angle=90,  "F+F+F+F",   [["F","FF+F++F+F"]]);
function space_filling_tree  (levels=4,  step=1) = lsystem_turtle(levels=levels, step=step, angle=90,  "X",         [["X","FX++F-FX++F-FX++F-FX++F-"],["F", "FF"]], startang=45);
function krishna_anklets     (levels=6,  step=1) = lsystem_turtle(levels=levels, step=step, angle=45,  "-X--X",     [["X","XFX--XFX"]]);


points = hilbert_curve(levels=5, step=100/pow(2,5));
stroke(points, width=1);


// vim: expandtab tabstop=4 shiftwidth=4 softtabstop=4 nowrap
