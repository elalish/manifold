{
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "nixpkgs/nixos-unstable";
  inputs.gtest-src = {
    url = "github:google/googletest/v1.14.0";
    flake = false;
  };
  inputs.clipper2-src = {
    url = "github:AngusJohnson/Clipper2";
    flake = false;
  };
  outputs = { self, nixpkgs, flake-utils, gtest-src, clipper2-src }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };
          clipper2 = pkgs.clipper2.overrideAttrs (_: _: {
            version = "14052024";
            src = clipper2-src;
          });
          manifold =
            { parallel-backend ? "none"
            , doCheck ? true
            , build-tools ? [ ]
            , ...
            }: pkgs.stdenv.mkDerivation {
              inherit doCheck;
              pname = "manifold-${parallel-backend}";
              version = "2.5.1";
              src = self;
              nativeBuildInputs = (with pkgs; [
                cmake
                ninja
                (python3.withPackages
                  (ps: with ps; [ nanobind trimesh pytest ]))
                gtest
                pkg-config
              ]) ++ build-tools;
              buildInputs = with pkgs; [
                clipper2
                assimp
              ];
              cmakeFlags = [
                "-DMANIFOLD_CBIND=ON"
                "-DMANIFOLD_EXPORT=ON"
                "-DBUILD_SHARED_LIBS=ON"
                "-DMANIFOLD_PAR=${pkgs.lib.strings.toUpper parallel-backend}"
              ];
              checkPhase = ''
                cd test
                ./manifold_test
                cd ../
              '';
            };
          parallelBackends = [
            { parallel-backend = "none"; }
            {
              parallel-backend = "tbb";
              build-tools = with pkgs; [ tbb pkg-config ];
            }
          ];
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
              version = "2.5.1";
              src = self;
              nativeBuildInputs = (with pkgs; [ cmake python39 ]);
              buildInputs = [ pkgs.nodejs ];
              configurePhase = ''
                cp -r ${clipper2-src} clipper2
                chmod -R +w clipper2
                mkdir -p .emscriptencache
                export EM_CACHE=$(pwd)/.emscriptencache
                mkdir build
                cd build
                emcmake cmake -DCMAKE_BUILD_TYPE=Release \
                -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${gtest-src} \
                -DFETCHCONTENT_SOURCE_DIR_CLIPPER2=../clipper2 ..
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
              version = "2.5.1";
              src = self;
              propagatedBuildInputs = [
                numpy
              ];
              buildInputs = with pkgs; [
                tbb
                clipper2
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
                nanobind
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
          devShell = pkgs.mkShell {
            buildInputs = with pkgs; [
              cmake
              tbb
              gtest
            ];
          };
        }
      );
}
