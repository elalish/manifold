#!/usr/bin/env -S bash -e
for f in {bindings,samples,extras,src,test}/**/*.{h,cpp}; do
  clang-format --dry-run --Werror $f &
  pids[${i}]=$!
done

for pid in ${pids[*]}; do
    wait $pid
done
