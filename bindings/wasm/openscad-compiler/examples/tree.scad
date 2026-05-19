 module simple_tree(size, dna, n) {   
        if (n > 0) {
            // trunk
            cylinder(r1=size/10, r2=size/12, h=size, $fn=24);
            // branches
            translate([0,0,size])
                for(bd = dna) {
                    angx = bd[0];
                    angz = bd[1];
                    scal = bd[2];
                        rotate([angx,0,angz])
                            simple_tree(scal*size, dna, n-1);
                }
        }
        else { // leaves
            color("green")
            scale([1,1,3])
                translate([0,0,size/6]) 
                    rotate([90,0,0]) 
                        cylinder(r=size/6,h=size/10);
        }
    }
    // dna is a list of branching data bd of the tree:
    //      bd[0] - inclination of the branch
    //      bd[1] - Z rotation angle of the branch
    //      bd[2] - relative scale of the branch
    dna = [ [12,  80, 0.85], [55,    0, 0.6], 
            [62, 125, 0.6], [57, -125, 0.6] ];
    simple_tree(50, dna, 5);