#/usr/bin/env bash
if test -f $1/manifold.js; then
  # I still have no idea why this is needed...
  sed -i 's/new Worker/new (ENVIRONMENT_IS_NODE ? global.Worker : Worker)/g' $1/manifold.js
fi
sed -i 's/var nodeWorkerThreads=require("worker_threads");/const{createRequire:createRequire}=await import("module");var require=createRequire(import.meta.url);var nodeWorkerThreads=require("worker_threads");/g' $1/*.worker.js
sed -i 's/__filename/import.meta.url/g' $1/*.worker.js
