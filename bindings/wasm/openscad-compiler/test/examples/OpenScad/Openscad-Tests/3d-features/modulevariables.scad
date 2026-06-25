// Volume: 8929.807099769992, SurfaceArea: 3653.061817190116
module mymodule(modparam) {
  inner_variable = 23;
  inner_variable2 = modparam * 2;
  cylinder(r1=inner_variable, r2=inner_variable2, h=10);
}

mymodule(5);
