# Ember Benchmark

The paper EMBER: Exact Mesh Boolean via Efficient & Robust
Local Arrangments (see https://dl.acm.org/doi/10.1145/3528223.3530181)
has supplemental material that describes 1000 tests using meshes
from the Thinki10K dataset (see https://ten-thousand-models.appspot.com/)

## Dataset

Download the dataset from https://ten-thousand-models.appspot.com ,
decompress and put the `raw_meshes` directory inside `testfiles`.

## Running the benchmark

1. Download the dataset.
2. Build the binary with `MANIFOLD_PAR` and `MANIFOLD_EXPORT` enabled. The
   binary should be inside `build/extras` (relative to the project root).
3. Run `python do_ember_tests.py`. This will take 10 minutes to an hour
   depending on your machine. At the end, a `benchmark.csv` file will be
   generated.
4. Run `python analyze_ember_tests.py` to analyze the result.

## Files

- `man_bench` is a binary that reads two .stl files and has corresponding
  transform arguments. It converts those to MeshGL, then to Manifold,
  then runs a boolean difference, and converts the result back to MeshGL.
  It times the total time take to do this (not including loading the
  stl files into MeshGL format), and appends a line to a benchmarks.csv
  file with the argument and timing information.
- `testfiles/test_file/ember-benchmark-cases.json`: JSON file describing each
  test case.
- `do_ember_tests.py` reads the description of the files and transforms
  from the supplemental material of the Ember paper, and executes
  man_bench on each pair.  This creates a benchmark.csv file (remember
  to remove that file again if you want a clean new run of data).
- `analyze_ember_tests.py` reads the benchmark.csv file and calculates
  statistics and does a plot.
- `bug-case.json` is a single test case that takes abnormally long.
- `m3max_benchmarks.csv` is the benchmarks file gotten by running on a
  Macbook Pro M3 Max (12 performance cores, 4 efficiency cores).

