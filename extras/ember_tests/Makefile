man_props: man_props.o
	c++  -lmanifold -L../manifold/build/src -o man_props man_props.o

man_props.o: man_props.cpp
	c++ -c --std=c++17 -I../manifold/include   man_props.cpp

run_manprops: man_props
	DYLD_LIBRARY_PATH=../manifold/build/src ./man_props

man_big: man_big.o
	c++  -lmanifold -L../manifold/build/src -o man_big man_big.o

man_big.o: man_big.cpp
	c++ -c --std=c++17 -I../manifold/include   man_big.cpp

run_manbig: man_big
	DYLD_LIBRARY_PATH=../manifold/build/src ./man_big

man_bench: man_bench.o
	c++  -lmanifold -lassimp -L../manifold/build/src -o man_bench man_bench.o

man_bench.o: man_bench.cpp
	c++ -c -std=c++17 -g -I../manifold/include -I../manifold/build/_deps/tbb-src/include  man_bench.cpp

run_man_bench: man_bench
	DYLD_LIBRARY_PATH=../manifold/build/src \
	 ./man_bench testfiles/472053.stl testfiles/130046.stl \
	 -t1 -0.00802736 0.00669899 0.000370525 0.00147035 0.00119281 0.0102892 \
          0.00654613 0.00794688 -0.00185673 55.0852 -51.0913, -19.5047 \
	 -t2 -0.000726775 -0.00187347 0.0230987 -0.0109203 0.0204111 0.00131189 \
         -0.0204403 -0.0108381 -0.00152218 1.05276 -0.042012 -4.89386 \
	 --threads 4


