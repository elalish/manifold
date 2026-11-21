## Manifold3d Java Bindings

[![Clojars Project](https://img.shields.io/clojars/v/com.github.madhephaestus/manifold3d.svg)](https://clojars.org/com.github.madhephaestus/manifold3d)

## Developemnt

First build the C code:

#### Linux

```
        bash bindings/java/scripts/linux/buildC.sh
        cd bindings/java/
        mvn test --file pom.xml --no-transfer-progress
```
       

#### MacOS

```
      mkdir build
        cd build
        cmake  -DMANIFOLD_CROSS_SECTION=ON -DMANIFOLD_USE_BUILTIN_CLIPPER2=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DMANIFOLD_STRICT=ON -DMANIFOLD_USE_BUILTIN_TBB=ON -DMANIFOLD_DEBUG=ON -DMANIFOLD_ASSERT=ON -DMANIFOLD_PYBIND=OFF -DMANIFOLD_PAR=ON .. && make
        cd ..
        mkdir -p ./bindings/java/src/main/resources/manifold3d/natives/mac-arm64/
        ls -al ./build/src/
        cp ./build/src/libmanifold.dylib ./bindings/java/src/main/resources/manifold3d/natives/mac-arm64/
        cp ./build/bindings/c/libmanifoldc.dylib ./bindings/java/src/main/resources/manifold3d/natives/mac-arm64/
        cd bindings/java/
        mvn test --file pom.xml --no-transfer-progress
```
        
#### Windows


```
        cmake . -DMANIFOLD_CROSS_SECTION=ON -DMANIFOLD_USE_BUILTIN_CLIPPER2=ON  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DMANIFOLD_DEBUG=ON -DMANIFOLD_STRICT=ON -DMANIFOLD_PAR=ON -DMANIFOLD_EXPORT=ON -DMANIFOLD_USE_BUILTIN_TBB=ON -DMANIFOLD_ASSERT=ON -DMANIFOLD_PAR=ON -A x64 -B build
        cd build
        cmake --build . --target ALL_BUILD --config Release
        cd ..
        mkdir -p bindings\java\src\main\resources\manifold3d\natives\win-x86_64\
        cp build\lib\Release\manifold.dll bindings\java\src\main\resources\manifold3d\natives\win-x86_64\manifold.dll
        cp build\lib\Release\manifoldc.dll bindings\java\src\main\resources\manifold3d\natives\win-x86_64\manifoldc.dll
        cd bindings\java
        mvn test --file pom.xml --no-transfer-progress
```
        
#### Maven install

for local install to maven of a snapshot run:

```
bash bindings/java/scripts/localSnapshotInstall.sh 

```

and see the printouts:

```
Clojar Java /home/hephaestus/bin/java21/
Clojar Version: 3.4.0-46a5518e-SNAPSHOT
```

That means you can run a local build with version `3.4.0-46a5518e-SNAPSHOT` which means it is commit 46a5518e which is ahead of the tag 3.4.0. 