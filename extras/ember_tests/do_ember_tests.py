import os
import json
import subprocess
import sys
import pathlib

ember_tests = pathlib.Path(__file__).parent
root = ember_tests.parent.parent

meshdir = ember_tests.joinpath("testfiles/raw_meshes")
testspecfile = ember_tests.joinpath("testfiles/ember-benchmark-cases.json")
man_bench = root.joinpath("build/extras/man_bench")

threads_cases = [1, 2, 4, 6, 8, 10, 12, 14, 16]

meshfiles = os.listdir(meshdir)

id_to_file = {}
for f in meshfiles:
    base, ext = os.path.splitext(f)
    if base and ext:
        try:
            baseid = int(base)
            id_to_file[baseid] = f
        except:
            pass


def file_for_id(id):
    if id in id_to_file:
        file = id_to_file[id]
        fullfile = meshdir.joinpath(file)
        return fullfile
    return None


with open(testspecfile, "r") as file:
    spec_data = json.load(file)

for threads in threads_cases:
    spec_index = 0
    for spec in spec_data:
        id1 = spec["id_a"]
        id2 = spec["id_b"]
        tr1 = spec["transform_a"]
        tr2 = spec["transform_b"]
        file1 = file_for_id(id1)
        if not file1:
            print("no test file for ", id1)
        file2 = file_for_id(id2)
        if not file2:
            print("no test file for ", id2)
        args = [man_bench, file1, file2]
        for i in [1, 2]:
            args.append("-t%d" % i)
            tr = tr1 if i == 1 else tr2
            for col in ["col0", "col1", "col2", "col3"]:
                for row in ["x", "y", "z"]:
                    v = tr[col][row]
                    args.append("%f" % v)
            if threads != 0:
                args.append("--threads")
                args.append("%d" % threads)
            print("test", spec_index, "threads", threads)
        cp = subprocess.run(args)
        if cp.returncode != 0:
            print("return code", cp.returncode)
            sys.exit(1)
        spec_index = spec_index + 1
