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
        manifold-version = "3.0.0";
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            (final: prev: {
              clipper2 = prev.clipper2.overrideAttrs (_: {
                version = clipper2-src.rev;
                src = clipper2-src;
              });
              # https://github.com/NixOS/nixpkgs/pull/343743#issuecomment-2424163602
              binaryen =
                let
                  testsuite = final.fetchFromGitHub {
                    owner = "WebAssembly";
                    repo = "testsuite";
                    rev = "e05365077e13a1d86ffe77acfb1a835b7aa78422";
                    hash = "sha256-yvZ5AZTPUA6nsD3xpFC0VLthiu2CxVto66RTXBXXeJM=";
                  };
                in
                prev.binaryen.overrideAttrs (_: rec {
                  version = "119";
                  src = pkgs.fetchFromGitHub {
                    owner = "WebAssembly";
                    repo = "binaryen";
                    rev = "version_${version}";
                    hash = "sha256-JYXtN3CW4qm/nnjGRvv3GxQ0x9O9wHtNYQLqHIYTTOA=";
                  };
                  preConfigure = ''
                    if [ $doCheck -eq 1 ]; then
                      sed -i '/googletest/d' third_party/CMakeLists.txt
                      rmdir test/spec/testsuite
                      ln -s ${testsuite} test/spec/testsuite
                    else
                      cmakeFlagsArray=($cmakeFlagsArray -DBUILD_TESTS=0)
                    fi
                  '';
                });
            })
          ];
        };
        onetbb = pkgs.tbb_2021_11.overrideAttrs (_: {
          version = onetbb-src.rev;
          src = onetbb-src;
        });
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
            ]) ++ (if parallel then [ onetbb ] else [ ]);
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
            checkPhase = ''
              cd test
              ./manifold_test
              cd ../
            '';
          };
        manifold-emscripten = { doCheck ? true }: pkgs.buildEmscriptenPackage {
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
            emcmake cmake -DCMAKE_BUILD_TYPE=Release \
            -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${gtest-src} \
            -DFETCHCONTENT_SOURCE_DIR_CLIPPER2=../clipper2 ..
          '';
          buildPhase = ''
            emmake make -j''${NIX_BUILD_CORES}
          '';
          checkPhase = if doCheck then ''
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
          # but how should we make it work with other python versions?
          manifold3d = with pkgs.python3Packages; buildPythonPackage {
            pname = "manifold3d";
            version = manifold-version;
            src = self;
            propagatedBuildInputs = [ numpy ];
            buildInputs = [ onetbb pkgs.clipper2 ];
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
            ]))

            ninja
            cmake
            onetbb
            gtest
            assimp
            clipper2
            pkg-config

            # useful tools
            clang-tools_18
            clang_18
            tracy
          ];
        };
      }
      );
}
