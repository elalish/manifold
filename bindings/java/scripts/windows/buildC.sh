#!/bin/bash


#rm -rf build
mkdir build

cmake . \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DMANIFOLD_STRICT=ON \
  -DMANIFOLD_USE_BUILTIN_TBB=ON \
  -DMANIFOLD_DEBUG=OFF \
  -DMANIFOLD_ASSERT=OFF \
  -DMANIFOLD_CROSS_SECTION=ON \
  -DMANIFOLD_EXPORT=OFF \
  -DMANIFOLD_PAR=ON \
  -DFETCHCONTENT_SOURCE_DIR_TBB=tbb \
  -DFETCHCONTENT_SOURCE_DIR_CLIPPER2=clipper2 \
  -DFETCHCONTENT_SOURCE_DIR_NANOBIND=nanobind \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=gtest \
  -A x64 -B build
cmake --build build --target ALL_BUILD --config Release
mkdir -p ./bindings/java/src/main/resources/manifold3d/natives/win-x86_64/
rm -rf ./bindings/java/src/main/resources/manifold3d/natives/win-x86_64/*
cp ./build/lib/Release/manifold.dll ./bindings/java/src/main/resources/manifold3d/natives/win-x86_64/
cp ./build/lib/Release/manifoldc.dll ./bindings/java/src/main/resources/manifold3d/natives/win-x86_64/
mvn test --file bindings/java/pom.xml --no-transfer-progress
