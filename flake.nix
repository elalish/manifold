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
  inputs.onetbb-src = {
    url = "github:oneapi-src/oneTBB/v2022.0.0";
    flake = false;
  };
  inputs.gersemi-src = {
    url = "github:BlankSpruce/gersemi/0.17.0";
    flake = false;
  };
  outputs =
    { self
    , nixpkgs
    , flake-utils
    , gtest-src
    , clipper2-src
    , onetbb-src
    , gersemi-src
    }:
    flake-utils.lib.eachDefaultSystem
      (system:
      let
        manifold-version = "3.2.0";
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            (final: prev: {
              clipper2 = prev.clipper2.overrideAttrs (_: {
                version = clipper2-src.rev;
                src = clipper2-src;
              });
            })
          ];
        };
        gersemi = with pkgs.python3Packages; buildPythonPackage {
          pname = "gersemi";
          version = "0.17.0";
          src = gersemi-src;
          propagatedBuildInputs = [
            appdirs
            lark
            pyyaml
          ];
          doCheck = true;
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
            ]) ++ (if parallel then [ pkgs.tbb_2021_11 ] else [ ]);
            buildInputs = with pkgs; [
              clipper2
              assimp
            ];
            cmakeFlags = [
              "-DMANIFOLD_CBIND=ON"
              "-DMANIFOLD_EXPORT=ON"
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
          nativeBuildInputs = (with pkgs; [ cmake python39 ]);
          buildInputs = [ pkgs.nodejs ];
          configurePhase = ''
            cp -r ${clipper2-src} clipper2
            chmod -R +w clipper2
            mkdir -p .emscriptencache
            export EM_CACHE=$(pwd)/.emscriptencache
            mkdir build
            cd build
            emcmake cmake -DCMAKE_BUILD_TYPE=MinSizeRel \
            -DMANIFOLD_PAR=${if parallel then "ON" else "OFF"} \
            -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${gtest-src} \
            -DFETCHCONTENT_SOURCE_DIR_TBB=${onetbb-src} \
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
            buildInputs = with pkgs; [ clipper2 tbb_2021_11 ];
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
              gersemi
              black

              # misc
              matplotlib
            ]))

            ninja
            cmake
            tbb_2021_11
            gtest
            assimp
            clipper2
            pkg-config

            # useful tools
            clang-tools_18
            clang_18
            llvmPackages_18.bintools
            tracy
          ];
        };
      }
      );
}
