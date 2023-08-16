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
          manifold =
            { parallel-backend ? "none"
            , doCheck ? true
            , build-tools ? [ ]
            , ...
            }: pkgs.stdenv.mkDerivation {
              inherit doCheck;
              pname = "manifold-${parallel-backend}";
              version = "beta";
              src = self;
              nativeBuildInputs = (with pkgs; [
                cmake
                ninja
                (python39.withPackages
                  (ps: with ps; [ trimesh ]))
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
                cd build
              '';
            };
          parallelBackends = [
            { parallel-backend = "none"; }
            {
              parallel-backend = "tbb";
              build-tools = with pkgs; [ tbb pkg-config ];
            }
          ];
          devShell = { additional ? [ ] }: pkgs.mkShell {
            buildInputs = with pkgs; [
              cmake
              llvmPackages_13.openmp
              clang-tools
              clang_13
              emscripten
              tbb
              lcov
              gtest
              tracy
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
              version = "beta";
              src = self;
              nativeBuildInputs = (with pkgs; [ cmake python39 ]);
              buildInputs = [ pkgs.nodejs ];
              configurePhase = ''
                mkdir -p .emscriptencache
                export EM_CACHE=$(pwd)/.emscriptencache
                mkdir build
                cd build
                emcmake cmake -DCMAKE_BUILD_TYPE=Release -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${gtest-src} ..
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
          };
          devShell = devShell { };
        }
      );
}
