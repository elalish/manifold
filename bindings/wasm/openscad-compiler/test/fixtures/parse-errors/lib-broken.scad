// Parses cleanly, but is defined before the syntax error below.
module good() { cube(3); }
module bad() { translate([1,0,0) cube(1); }
