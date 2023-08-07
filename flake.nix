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
            , cuda-support ? false
            , doCheck ? true
            , build-tools ? [ ]
            , ...
            }: pkgs.stdenv.mkDerivation {
              inherit doCheck;
              pname =
                if cuda-support then
                  "manifold-${parallel-backend}-cuda"
                else
                  "manifold-${parallel-backend}";
              version = "beta";
              src = self;
              nativeBuildInputs = (with pkgs; [ cmake (python39.withPackages (ps: with ps; [ trimesh ])) gtest ]) ++ build-tools ++
                (if cuda-support then with pkgs.cudaPackages; [ cuda_nvcc cuda_cudart cuda_cccl pkgs.addOpenGLRunpath ] else [ ]);
              cmakeFlags = [
                "-DMANIFOLD_PYBIND=ON"
                "-DMANIFOLD_PAR=${pkgs.lib.strings.toUpper parallel-backend}"
                "-DMANIFOLD_USE_CUDA=${if cuda-support then "ON" else "OFF"}"
              ];
              checkPhase = ''
                cd test
                ./manifold_test
                cd ../../
                PYTHONPATH=$PYTHONPATH:$(pwd)/build/bindings/python python3 bindings/python/examples/run_all.py
                cd build
              '';
              installPhase = ''
                mkdir -p $out
                cp src/manifold/libmanifold.a $out/
                cp extras/perfTest $out
                cp bindings/python/manifold3d* $out
              '';
            };
          parallelBackends = [
            { parallel-backend = "none"; }
            {
              parallel-backend = "omp";
              build-tools = [ pkgs.llvmPackages_13.openmp ];
            }
            {
              parallel-backend = "tbb";
              build-tools = with pkgs; [ tbb pkg-config ];
            }
          ];
          buildMatrix = with pkgs; with lib; lists.flatten (map
            (env: map
              (x: x // env)
              parallelBackends) [
            { cuda-support = false; }
            {
              cuda-support = true;
            }
          ]);
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
                name = "manifold-" + x.parallel-backend + (if
                  x.cuda-support then "-cuda" else "");
                value = manifold x;
              })
              buildMatrix)) // {
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
          devShells.cuda = devShell {
            additional = with pkgs.cudaPackages; [
              cuda_nvcc
              cuda_cudart
              cuda_cccl
            ];
          };
        }
      );
}
