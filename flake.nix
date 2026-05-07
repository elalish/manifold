{
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "nixpkgs/nixos-unstable";
  inputs.clipper2-src = {
    url = "github:AngusJohnson/Clipper2";
    flake = false;
  };
  outputs =
    { self
    , nixpkgs
    , flake-utils
    , clipper2-src
    }:
    flake-utils.lib.eachDefaultSystem
      (system:
      let
        manifold-version = "3.4.1";
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            (final: prev: {
              clipper2 = prev.clipper2.overrideAttrs (_: rec {
                version = clipper2-src.rev;
                src = clipper2-src;
                sourceRoot = "source/CPP";
              });
            })
          ];
        };
        manifold =
          { parallel ? true }: pkgs.stdenv.mkDerivation {
            pname = "manifold-${if parallel then "tbb" else "none"}";
            version = manifold-version;
            src = self;
            nativeBuildInputs = (with pkgs; [
              cmake
              ninja
              (python3.withPackages
                (ps: with ps; [ nanobind trimesh pytest ]))
              gtest
            ]) ++ (if parallel then [ pkgs.onetbb ] else [ ]);
            buildInputs = with pkgs; [
              clipper2
            ];
            cmakeFlags = [
              "-DMANIFOLD_STRICT=ON"
              "-DMANIFOLD_CBIND=ON"
              "-DBUILD_SHARED_LIBS=ON"
              "-DMANIFOLD_PAR=${if parallel then "ON" else "OFF"}"
            ];
            doCheck = true;
            checkPhase = ''
              cd test
              ./manifold_test
              cd ../
            '';
          };
        manifold-emscripten = { doCheck ? true, parallel ? false }: pkgs.buildEmscriptenPackage {
          name = "manifold-js";
          version = manifold-version;
          src = self;
          nativeBuildInputs = (with pkgs; [ cmake python3 ]);
          buildInputs = [ pkgs.nodejs ];
          configurePhase = ''
            cp -r ${clipper2-src} clipper2
            chmod -R +w clipper2
            mkdir -p .emscriptencache
            export EM_CACHE=$(pwd)/.emscriptencache
            mkdir build
            cd build
            emcmake cmake -DCMAKE_BUILD_TYPE=MinSizeRel \
            -DMANIFOLD_STRICT=ON \
            -DMANIFOLD_PAR=${if parallel then "ON" else "OFF"} \
            -DMANIFOLD_USE_BUILTIN_TBB=${if parallel then "ON" else "OFF"} \
            -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${pkgs.gtest.src} \
            -DFETCHCONTENT_SOURCE_DIR_TBB=${pkgs.onetbb.src} \
            -DFETCHCONTENT_SOURCE_DIR_CLIPPER2=../clipper2 ..
          '';
          buildPhase = ''
            emmake make -j''${NIX_BUILD_CORES}
          '';
          checkPhase =
            if doCheck then ''
              cd test
              node manifold_test.js
              cd ../
            '' else "";
          installPhase = ''
            mkdir -p $out
            cp bindings/wasm/manifold.* $out/
          '';
        };
      in
      {
        packages = {
          manifold-tbb = manifold { };
          manifold-none = manifold { parallel = false; };
          manifold-js = manifold-emscripten { };
          manifold-js-tbb = manifold-emscripten {
            parallel = true;
            doCheck =
              false;
          };
          # but how should we make it work with other python versions?
          manifold3d = with pkgs.python3Packages; buildPythonPackage {
            pname = "manifold3d";
            version = manifold-version;
            src = self;
            propagatedBuildInputs = [ numpy ];
            buildInputs = with pkgs; [ clipper2 onetbb ];
            nativeBuildInputs = with pkgs; [
              cmake
              ninja
              setuptools
              scikit-build-core
              pyproject-metadata
              pathspec
            ];
            checkInputs = [ nanobind trimesh pytest ];
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
            (python3.withPackages (ps: with ps; [
              # test dependencies
              trimesh
              numpy
              pytest

              # formatting tools
              black

              # misc
              matplotlib
            ]))

            gersemi
            ninja
            cmake
            onetbb
            gtest
            assimp
            clipper2
            pkg-config

            # useful tools
            clang_18
            llvmPackages_18.clang-tools
            llvmPackages_18.bintools
            tracy
            f3d
          ];
        };
      }
      );
}
