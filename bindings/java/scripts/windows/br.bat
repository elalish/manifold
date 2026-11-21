#!/bin/bash

set -e
javac --enable-preview --source 21 --add-modules javafx.controls -cp "/d/git/java/bin/javafx/*" -d target/classes src/main/java/com/cad/*.java

java --enable-preview --enable-native-access=ALL-UNNAMED --add-modules javafx.controls -cp "target/classes;/d/git/java/bin/javafx/*" -Djava.library.path=/d/git/manifold-c/cad-app/src/resources/natives/win-x86_64 com.cad.ManifoldCADApp
