#!/usr/bin/env python3
import sys

if len(sys.argv) != 2:
    print("Usage: python fixup.py <SOURCE_DIR>")

filename = sys.argv[1]
with open(filename, "r") as file:
    data = file.read()

data = data.replace(
    'var workerOptions={type:"module",workerData:"em-pthread",name:"em-pthread"};', ""
)
data = data.replace(
    "workerOptions", '{type:"module",workerData:"em-pthread",name:"em-pthread"}'
)

with open(filename, "w") as file:
    file.write(data)
