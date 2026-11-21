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
  -DCMAKE_BUILD_RPATH='$ORIGIN' ..
make
cd ..
mkdir -p ./bindings/java/src/main/resources/manifold3d/natives/linux-arm64/
rm -rf ./bindings/java/src/main/resources/manifold3d/natives/linux-arm64/*
cp ./build/src/libmanifold.so ./bindings/java/src/main/resources/manifold3d/natives/linux-arm64/
ln -sf libmanifold.so ./bindings/java/src/main/resources/manifold3d/natives/linux-arm64/libmanifold.so.3
ln -sf libmanifold.so.3 ./bindings/java/src/main/resources/manifold3d/natives/linux-arm64/libmanifold.so.3.5.1
cp ./build/bindings/c/libmanifoldc.so ./bindings/java/src/main/resources/manifold3d/natives/linux-arm64/
ln -sf libmanifoldc.so ./bindings/java/src/main/resources/manifold3d/natives/linux-arm64/libmanifoldc.so.3
ln -sf libmanifoldc.so.3 ./bindings/java/src/main/resources/manifold3d/natives/linux-arm64/libmanifoldc.so.3.5.1
mvn test --file bindings/java/pom.xml --no-transfer-progress
