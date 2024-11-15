#!/usr/bin/env bash
sed -i 's/var workerOptions={type:"module",workerData:"em-pthread",name:"em-pthread"};//g' ./examples/built/manifold.js
sed -i 's/workerOptions/{type:"module",workerData:"em-pthread",name:"em-pthread"}/g' ./examples/built/manifold.js
