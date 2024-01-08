{
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "nixpkgs/nixos-unstable";
  inputs.gtest-src = {
    url = "github:google/googletest/v1.14.0";
    flake = false;
  };
  outputs = { self, nixpkgs, flake-utils, gtest-src }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };
          thrust-210 = pkgs.nvidia-thrust.overrideAttrs (_: _: {
            version = "24486a1";
            src = builtins.fetchGit {
              url = "https://github.com/NVIDIA/thrust.git";
              ref = "main";
              rev = "24486a169a62a58ef8f824d3dc9613c006b6f5a7";
              submodules = true;
            };
            cmakeFlags = [ "-DTHRUST_ENABLE_HEADER_TESTING=OFF" "-DTHRUST_ENABLE_TESTING=OFF" "-DTHRUST_ENABLE_EXAMPLES=OFF" "-DTHRUST_DEVICE_SYSTEM=CPP" ];
            fixupPhase = ''
              cat <<EOT > $out/lib/cmake/thrust/thrust-header-search.cmake
              # Parse version information from version.h in source tree
              set(_THRUST_VERSION_INCLUDE_DIR "$out/include")
              if(EXISTS "\''${_THRUST_VERSION_INCLUDE_DIR}/thrust/version.h")
                set(_THRUST_VERSION_INCLUDE_DIR "\''${_THRUST_VERSION_INCLUDE_DIR}" CACHE FILEPATH "" FORCE) # Clear old result
                set_property(CACHE _THRUST_VERSION_INCLUDE_DIR PROPERTY TYPE INTERNAL)
              endif()
              EOT
              cat <<EOT > $out/lib/cmake/libcudacxx/libcudacxx-header-search.cmake
              # Parse version information from version header:
              unset(_libcudacxx_VERSION_INCLUDE_DIR CACHE) # Clear old result to force search
              find_path(_libcudacxx_VERSION_INCLUDE_DIR cuda/std/detail/__config
                NO_DEFAULT_PATH # Only search explicit paths below:
                PATHS
                  "\''${CMAKE_CURRENT_LIST_DIR}/../../../include" # Source tree
              )
              set_property(CACHE _libcudacxx_VERSION_INCLUDE_DIR PROPERTY TYPE INTERNAL)
              EOT
            '';
            enableParallelBuilding = false;
          });
          manifold =
            { parallel-backend ? "none"
            , doCheck ? true
            , build-tools ? [ ]
            , ...
            }: pkgs.stdenv.mkDerivation {
              inherit doCheck;
              pname = "manifold-${parallel-backend}";
              version = "2.3.1";
              src = self;
              nativeBuildInputs = (with pkgs; [
                cmake
                thrust-210
                glm
                ninja
                (python39.withPackages
                  (ps: with ps; [ trimesh pytest ]))
                gtest
              ]) ++ build-tools;
              cmakeFlags = [
                "-DMANIFOLD_PYBIND=ON"
                "-DMANIFOLD_CBIND=ON"
                "-DBUILD_SHARED_LIBS=ON"
                "-DMANIFOLD_PAR=${pkgs.lib.strings.toUpper parallel-backend}"
              ];
              checkPhase = ''
                cd test
                ./manifold_test
                cd ../../
                PYTHONPATH=$PYTHONPATH:$(pwd)/build/bindings/python python3 bindings/python/examples/run_all.py
                PYTHONPATH=$PYTHONPATH:$(pwd)/build/bindings/python python3 -m pytest
                cd build
              '';
            };
          parallelBackends = [
            { parallel-backend = "none"; }
            {
              parallel-backend = "tbb";
              build-tools = with pkgs; [ tbb_2021_8 pkg-config ];
            }
          ];
          devShell = { additional ? [ ] }: pkgs.mkShell {
            buildInputs = with pkgs; [
              cmake
              tbb_2021_8
              thrust-210
              gtest
            ] ++ additional;
          };
        in
        {
          packages = (builtins.listToAttrs
            (map
              (x: {
                name = "manifold-" + x.parallel-backend;
                value = manifold x;
              })
              parallelBackends)) // {
            manifold-js = pkgs.buildEmscriptenPackage {
              name = "manifold-js";
              version = "2.3.1";
              src = self;
              nativeBuildInputs = (with pkgs; [ cmake python39 ]);
              buildInputs = [ pkgs.nodejs ];
              configurePhase = ''
                mkdir -p .emscriptencache
                export EM_CACHE=$(pwd)/.emscriptencache
                mkdir build
                cd build
                emcmake cmake -DCMAKE_BUILD_TYPE=Release \
                -DFETCHCONTENT_SOURCE_DIR_GLM=${pkgs.glm.src} \
                -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${gtest-src} \
                -DFETCHCONTENT_SOURCE_DIR_THRUST=${thrust-210.src} ..
              '';
              buildPhase = ''
                emmake make -j''${NIX_BUILD_CORES}
              '';
              checkPhase = ''
                cd test
                node manifold_test.js
                cd ../
              '';
              installPhase = ''
                mkdir -p $out
                cp {extras,wasm}/*.js $out/
                cp {extras,wasm}/*.wasm $out/
              '';
            };
            # but how should we make it work with other python versions?
            manifold3d = with pkgs.python3Packages; buildPythonPackage {
              pname = "manifold3d";
              version = "2.3.1";
              src = self;
              propagatedBuildInputs = [
                numpy
              ];
              buildInputs = with pkgs; [
                tbb_2021_8
                thrust-210
                glm
              ];
              nativeBuildInputs = with pkgs; [
                cmake
                ninja
                setuptools
                scikit-build-core
                pyproject-metadata
                pathspec
                pkg-config
              ];
              checkInputs = [
                trimesh
                pytest
              ];
              format = "pyproject";
              dontUseCmakeConfigure = true;
              doCheck = true;
              checkPhase = ''
                python3 bindings/python/examples/run_all.py
                python3 -m pytest
              '';
            };
          };
          devShell = devShell { };
        }
      );
}
