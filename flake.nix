{
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "nixpkgs/nixos-21.11";

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };
          manifold = { backend ? "CPP", doCheck ? true, build-tools ? [ ], runtime ? [ ] }: pkgs.stdenv.mkDerivation {
            inherit doCheck;
            pname = "manifold-${backend}";
            version = "beta";
            src = self;
            patches = [ ./assimp.diff ];
            nativeBuildInputs = (with pkgs; [ cmake python38 ]) ++ build-tools;
            buildInputs = runtime;
            cmakeFlags = [ "-DTHRUST_BACKEND=${backend}" ];
            checkPhase = ''
              cd test
              ./manifold_test
              cd ../
            '';
            installPhase = ''
              mkdir -p $out
              cp manifold/libmanifold.a $out/
              cp meshIO/libmeshIO.a $out/
              cp tools/loadMesh $out
              cp tools/perfTest $out
            '';
          };
          devShell = { additional ? [ ] }: pkgs.mkShell {
            buildInputs = with pkgs; [
              cmake
              ccls
              llvmPackages.openmp
              clang-tools
              clang_13
              emscripten
              tbb
            ] ++ additional;
          };
        in
        {
          packages.manifold-cpp = manifold { };
          packages.manifold-omp = manifold { backend = "OMP"; runtime = [ pkgs.llvmPackages.openmp ]; };
          packages.manifold-tbb = manifold { backend = "TBB"; runtime = [ pkgs.tbb pkgs.pkg-config ]; };
          packages.manifold-cuda = manifold {
            backend = "CUDA";
            runtime = [
              pkgs.cudaPackages.cudatoolkit_11
            ];
            doCheck = false;
          };
          packages.manifold-js = pkgs.buildEmscriptenPackage {
            name = "manifold-js";
            version = "beta";
            src = self;
            patches = [ ./assimp.diff ];
            nativeBuildInputs = (with pkgs; [ cmake python38 ]);
            buildInputs = [ pkgs.nodejs ];
            configurePhase = ''
              mkdir build
              cd build
              emcmake cmake -DCMAKE_BUILD_TYPE=Release ..
            '';
            buildPhase = ''
              emmake make
            '';
            checkPhase = ''
              cd test
              node manifold_test.js
              cd ../
            '';
            installPhase = ''
              mkdir -p $out
              cd tools
              cp *.js $out/
              cp *.wasm $out/
            '';
          };
          devShell = devShell { };
          devShells.cuda = devShell {
            additional = [
              pkgs.cudaPackages.cudatoolkit_11
            ];
          };
        }
      );
}
