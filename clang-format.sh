#!/usr/bin/env -S bash -e
for f in {collider,manifold,meshIO,polygon,samples,utilities}/**/*.{h,cpp} {test,tools}/*.cpp; do
  clang-format --dry-run --Werror $f &
  pids[${i}]=$!
done

for pid in ${pids[*]}; do
    wait $pid
done
