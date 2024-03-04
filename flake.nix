{
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "nixpkgs/nixos-unstable";
  inputs.gtest-src = {
    url = "github:google/googletest/v1.14.0";
    flake = false;
  };
  inputs.thrust-src = {
    url = "git+https://github.com/NVIDIA/thrust.git?submodules=1";
    flake = false;
  };
  inputs.clipper2-src = {
    url = "github:AngusJohnson/Clipper2/Clipper2_1.3.0";
    flake = false;
  };
  outputs = { self, nixpkgs, flake-utils, gtest-src, thrust-src, clipper2-src }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };
          clipper2 = pkgs.clipper2.overrideAttrs (_: _: {
            version = "1.3.0";
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
              version = "2.3.1";
              src = self;
              nativeBuildInputs = (with pkgs; [
                cmake
                ninja
                (python3.withPackages
                  (ps: with ps; [ trimesh pytest ]))
                gtest
                pkg-config
              ]) ++ build-tools;
              buildInputs = with pkgs; [
                glm
                clipper2
                assimp
              ];
              cmakeFlags = [
                "-DMANIFOLD_PYBIND=ON"
                "-DMANIFOLD_CBIND=ON"
                "-DBUILD_SHARED_LIBS=ON"
                "-DFETCHCONTENT_SOURCE_DIR_THRUST=${thrust-src}"
                "-DMANIFOLD_PAR=${pkgs.lib.strings.toUpper parallel-backend}"
              ];
              prePatch = ''
                substituteInPlace bindings/python/CMakeLists.txt \
                  --replace 'DESTINATION ''${Python_SITEARCH}' 'DESTINATION "${placeholder "out"}/${pkgs.python3.sitePackages}"'
              '';
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
                cp -r ${clipper2-src} clipper2
                chmod -R +w clipper2
                mkdir -p .emscriptencache
                export EM_CACHE=$(pwd)/.emscriptencache
                mkdir build
                cd build
                emcmake cmake -DCMAKE_BUILD_TYPE=Release \
                -DFETCHCONTENT_SOURCE_DIR_GLM=${pkgs.glm.src} \
                -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${gtest-src} \
                -DFETCHCONTENT_SOURCE_DIR_THRUST=${thrust-src} \
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
              version = "2.3.1";
              src = self;
              propagatedBuildInputs = [
                numpy
              ];
              buildInputs = with pkgs; [
                tbb_2021_8
                glm
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
              SKBUILD_CMAKE_DEFINE = "FETCHCONTENT_SOURCE_DIR_THRUST=${thrust-src}";
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
          devShell = pkgs.mkShell {
            buildInputs = with pkgs; [
              cmake
              tbb_2021_8
              gtest
            ];
          };
        }
      );
}
