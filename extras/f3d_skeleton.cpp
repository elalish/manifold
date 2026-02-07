/**
 * If you want to iterate on some C++ code creating a Manifold, this can help.
 *
 * Launches f3d on your Manifold and exits.
 * Installing f3d and being present on the path is an exercise left to the
 reader.
 * Won't work on Windows, except in cygwin, because of how we run f3d. Feel free
 to fix it.
 *
 * Instructions:
 * Copy this into your project.
 * Ensure it builds and runs.
 * Comment out `getManifold` and implement it yourself.
 * Enjoy.

 * Or even quicker if you've built Manifold yourself,
 * simply implement it right here, and run
 * `cmake --build build` or `cmake --build build -- f3dNoop/fast`
 * from the repo root, then `build/extras/f3dNoop` to run.
 */

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "manifold/manifold.h"
#include "manifold/meshIO.h"

using namespace manifold;

// Implement this yourself:
const Manifold getManifold() { return Manifold(); }

namespace fs = std::filesystem;

/**
 * Generates a path in the system temp folder with a microsecond timestamp.
 */
std::string get_temp_path() {
  auto now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  auto micros =
      std::chrono::duration_cast<std::chrono::microseconds>(duration).count();

  fs::path temp_dir = fs::temp_directory_path();
  std::string filename = "manifold_" + std::to_string(micros) + ".glb";

  return (temp_dir / filename).string();
}

/**
 * Exports your Manifold to a temporary glTF file, launches f3d to open it, and
 * exits.
 */
int main(int argc, char** argv) {
  manifold::ManifoldParams().verbose = true;

  const Manifold manifold = getManifold();

  if (manifold.Status() != Manifold::Error::NoError) {
    std::cout << "Could not make a valid manifold, error: "
              << (int)manifold.Status() << std::endl;
    return 2;
  }

  const std::vector<Manifold> parts = manifold.Decompose();
  std::cout << parts.size() << " objects:" << std::endl;
  for (const Manifold& part : parts) {
    std::cout << part.NumVert() << " vertices, " << part.NumTri()
              << " triangles, volume = " << part.Volume()
              << ", surface area = " << part.SurfaceArea() << std::endl;
  }

  const std::string filepath = get_temp_path();

  MeshGL mesh = manifold.GetMeshGL();
  try {
    ExportMesh(filepath, mesh, {});
    std::cout << "Exported to: " << filepath << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Export failed: " << e.what() << std::endl;
    return 1;
  }

  // Without --hdri-ambient it's really dark.
  // --edges shows edges.
  // Can change defaults. `h` also brings up in-app menu.
  std::string command = "f3d --hdri-ambient --edges \"" + filepath + "\" &";

  std::cout << "Launching f3d..." << std::endl;
  int result = std::system(command.c_str());

  if (result != 0) {
    std::cerr << "Failed to launch f3d." << std::endl;
  }

  return 0;
}
