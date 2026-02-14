#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "../meshIO.h"
#include "manifold/manifold.h"

#ifdef CONTROL_PARALLELISM
#include "tbb/global_control.h"

static int global_thread_limit = 0;
#endif

using namespace manifold;

using high_resolution_clock = std::chrono::high_resolution_clock;
using time_point = high_resolution_clock::time_point;

const char* benchmark_filename = "benchmark.csv";

/* Append a stats line to the file with name benchmark_filename.
 * If the file doesn't exist yet, write a CSV header line first.
 * Assume that conversion from MeshGL to Manifold of the two args
 * happens between start and before_boolean,
 * that the actual boolean operation in Manifold space happens
 * between before_boolean and after_boolean,
 * and that the conversion of the result from Manifold back to
 * MeshGL happens between after_boolean and end.
 */
static void record_time(const std::string& file1, const std::string& file2,
                        time_point start, time_point before_boolean,
                        time_point after_boolean, time_point end) {
  using duration = std::chrono::duration<double, std::milli>;
  duration total_ms = end - start;
  duration convert_in_ms = before_boolean - start;
  duration manifold_boolean_ms = after_boolean - before_boolean;
  duration convert_out_ms = end - after_boolean;

  std::fstream stats_file;
  if (!std::filesystem::exists(benchmark_filename)) {
    /* First time: write CSV header line. */
    stats_file.open(benchmark_filename, std::ios::out);
    stats_file << "file1,file2,total (ms),to manifold (ms),boolean (ms),from "
                  "manifold (ms)";
#ifdef CONTROL_PARALLELISM
    stats_file << ",threads";
#endif
    stats_file << std::endl;
  } else {
    stats_file.open(benchmark_filename, std::ios::app);
  }
  stats_file << file1 << "," << file2 << "," << std::fixed
             << std::setprecision(3) << total_ms.count() << ","
             << convert_in_ms.count() << "," << manifold_boolean_ms.count()
             << "," << convert_out_ms.count();
#ifdef CONTROL_PARALLELISM
  stats_file << "," << global_thread_limit;
#endif
  stats_file << std::endl;
}

/* Convert a MeshGL to Manifold and check status.
 * If there is an error status, throw an exception.
 * The file argument is inserted into the exception message. */
static Manifold manifold_from_meshgl(const MeshGL& meshgl,
                                     const std::string& file) {
  Manifold man = Manifold(meshgl);
  if (man.Status() != Manifold::Error::NoError) {
    if (man.Status() == Manifold::Error::NotManifold) {
      throw std::invalid_argument(file + " not manifold");
    }
    throw std::invalid_argument("error converting " + file + " to manifold");
  }
  return man;
}

/* Return true if the meshes are identical, including element order. */
bool compare_mgls(const MeshGL& mgl1, const MeshGL& mgl2) {
  bool verbose = true;
  if (mgl1.NumVert() != mgl2.NumVert() || mgl1.NumTri() != mgl2.NumTri() ||
      mgl1.numProp != mgl2.numProp) {
    if (verbose) {
      std::cout << "basic sizes differ\n";
    }
    return false;
  }
  int ntri = mgl1.NumTri();
  for (int t = 0; t < ntri; t++) {
    for (int i = 0; i < 3; i++) {
      if (mgl1.triVerts[3 * t + i] != mgl2.triVerts[3 * t + i]) {
        if (verbose) {
          std::cout << "tri verts differ at t = " << t << "\n";
        }
        return false;
      }
    }
  }
  int nprop = mgl1.numProp;
  int nvert = mgl1.NumVert();
  for (int v = 0; v < nvert; v++) {
    for (int p = 0; p < nprop; p++) {
      if (mgl1.vertProperties[v * nprop + p] !=
          mgl2.vertProperties[v * nprop + p]) {
        if (verbose) {
          std::cout << "vert props differ at v = " << v << "\n";
        }
        return false;
      }
    }
  }
  return true;
}

/* Do a Difference boolean between the meshes in file1 and file2,
 * which are assumed to be in a format that assimp handles.
 * Before doing the boolean, transform1 and transform2 are
 * applied to their respective meshes.
 * Throws exception if there is a problem with the files or the
 * meshes are not manifold or a runtime error in Manifold boolean
 * happens.
 * Time the various phases of the boolean (excluding the reading of
 * the files and conversion to MeshGL), and use record_time() to
 * append a stats line to the benchmark file.
 */
static MeshGL do_boolean(const std::string& file1, const std::string& file2,
                         const mat3x4& transform1, const mat3x4& transform2) {
  MeshGL meshgl1 = ImportMesh(file1, true);
  MeshGL meshgl2 = ImportMesh(file2, true);
  time_point time0 = high_resolution_clock::now();
  Manifold man1 = manifold_from_meshgl(meshgl1, file1).Transform(transform1);
  Manifold man2 = manifold_from_meshgl(meshgl2, file2).Transform(transform2);
  time_point time1 = high_resolution_clock::now();
  Manifold man_ans = man1 - man2;
  /* Checking Status() here will force evaluation. */
  if (man_ans.Status() != Manifold::Error::NoError) {
    throw std::runtime_error("runtime error doing boolean");
  }
  time_point time2 = high_resolution_clock::now();
  MeshGL meshgl_ans = man_ans.GetMeshGL();
  time_point time3 = high_resolution_clock::now();

  MeshGL meshgl_ans2 = (man1 - man2).GetMeshGL();
  if (!compare_mgls(meshgl_ans, meshgl_ans2)) {
    throw std::runtime_error("non-deterministic");
  }

  record_time(file1, file2, time0, time1, time2, time3);
  return meshgl_ans;
}

static void usage() {
  std::cout << "Usage: %s file1 file2 [-t1 <12 floats>] [-t2 <12 floats>] "
               "[--threads <int>]\n";
}

int main(int argc, const char** argv) {
  if (argc < 3) {
    usage();
    return 1;
  }
  constexpr bool export_output = false;
  constexpr bool verbose = false;
  std::string file1(argv[1]);
  std::string file2(argv[2]);
  mat3x4 transform1 = linalg::identity;
  mat3x4 transform2 = linalg::identity;
  int threads = 1000;
  if (argc > 3) {
    int argi = 3;
    while (argi < argc) {
      if (strcmp(argv[argi], "-t1") == 0 || strcmp(argv[argi], "-t2") == 0) {
        if (argi + 12 >= argc) {
          usage();
          return 1;
        }
        mat3x4& t = argv[argi][2] == '1' ? transform1 : transform2;
        for (int i = 0; i < 12; i++) {
          try {
            double v = std::stod(argv[argi + 1 + i]);
            t[i / 3][i % 3] = v;
          } catch (...) {
            usage();
            return 1;
          }
        }
        argi += 13;
      } else if (strcmp(argv[argi], "--threads") == 0) {
        if (argi + 1 >= argc) {
          usage();
          return 1;
        }
        try {
          threads = std::stoi(argv[argi + 1]);
        } catch (...) {
          usage();
          return 1;
        }
        argi += 2;
      } else {
        usage();
        return 1;
      }
    }
  }
  if (verbose) {
    std::cout << "files: " << file1 << ", " << file2 << "\n";
    if (threads > 0) {
      std::cout << "threads: " << threads << "\n";
    }
    for (int i : {0, 1}) {
      std::cout << "transform " << i << "\n";
      const mat3x4& t = i == 0 ? transform1 : transform2;
      for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 4; col++) {
          std::cout << t[col][row] << " ";
        }
        std::cout << std::endl;
      }
    }
  }

#ifdef CONTROL_PARALLELISM
  global_thread_limit = threads;
  tbb::global_control control(
      tbb::global_control::parameter::max_allowed_parallelism, threads);
#endif
  MeshGL ans = do_boolean(file1, file2, transform1, transform2);
  if (export_output) {
    ExportOptions opts;
    ExportMesh("man_bench_out.obj", ans, opts);
  }
  return 0;
}
