#!/bin/bash

if [ ! -d clipper2 ]; then
  git clone https://github.com/AngusJohnson/Clipper2.git clipper2
fi

cd clipper2
git fetch --all --tags
git checkout 46f639177fe418f9689e8ddb74f08a870c71f5b4
cd ..
if [ ! -d nanobind ]; then
  git clone https://github.com/wjakob/nanobind.git nanobind
fi

cd nanobind
git fetch --all --tags
git checkout v2.12.0
cd ..

#rm -rf build
mkdir build
cd build
cmake \
  -DMANIFOLD_CROSS_SECTION=ON \
  -DMANIFOLD_USE_BUILTIN_CLIPPER2=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DASSIMP_ENABLE=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DMANIFOLD_DEBUG=OFF \
  -DMANIFOLD_ASSERT=OFF \
  -DMANIFOLD_TEST=OFF \
  -DMANIFOLD_EXTRAS=ON \
  -DMANIFOLD_EXPORT=OFF \
  -DMANIFOLD_PAR=ON \
  -DMANIFOLD_USE_BUILTIN_TBB=ON \
  -DFETCHCONTENT_SOURCE_DIR_CLIPPER2=../clipper2 \
  -DFETCHCONTENT_SOURCE_DIR_NANOBIND=../nanobind \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_BUILD_RPATH='$ORIGIN' \
  -DCMAKE_OSX_ARCHITECTURES=x86_64 ..
make
cd ..
mkdir -p ./bindings/java/src/main/resources/manifold3d/natives/mac-x86_64/
rm -rf ./bindings/java/src/main/resources/manifold3d/natives/mac-x86_64/*
cp ./build/src/libmanifold.dylib ./bindings/java/src/main/resources/manifold3d/natives/mac-x86_64/
ln -sf libmanifold.dylib ./bindings/java/src/main/resources/manifold3d/natives/mac-x86_64/libmanifold.3.dylib
ln -sf libmanifold.3.dylib ./bindings/java/src/main/resources/manifold3d/natives/mac-x86_64/libmanifold.3.5.1.dylib
cp ./build/bindings/c/libmanifoldc.dylib ./bindings/java/src/main/resources/manifold3d/natives/mac-x86_64/
ln -sf libmanifoldc.dylib ./bindings/java/src/main/resources/manifold3d/natives/mac-x86_64/libmanifoldc.3.dylib
ln -sf libmanifoldc.3.dylib ./bindings/java/src/main/resources/manifold3d/natives/mac-x86_64/libmanifoldc.3.5.1.dylib
mvn test --file bindings/java/pom.xml --no-transfer-progress
