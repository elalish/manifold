from os.path import dirname
import re

base = dirname(dirname(dirname(__file__)))


def snake_case(name):
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def python_param_modifier(comment):
    # p = f":{snake_case(m[0][1:])}:"
    comment = re.sub(r"@(param \w+)", lambda m: f":{snake_case(m[1])}:", comment)
    # python API renames `MeshGL` to `Mesh`
    comment = re.sub("mesh_gl", "mesh", comment)
    comment = re.sub("MeshGL", "Mesh", comment)
    return comment


def method_key(name):
    name = re.sub(r"\+", "_plus", name)
    name = re.sub(r"\-", "_minus", name)
    name = re.sub(r"\^", "_xor", name)
    name = re.sub(r"\=", "_eq", name)
    name = re.sub(r"\:", "_", name)
    name = re.sub(r"\~", "destroy_", name)
    return name


parens_re = re.compile(r"[^(]+\(([^(]*(\(.*\))*[^(]*\))", flags=re.DOTALL)
args_re = re.compile(
    r"^[^,^\(^\)]*(\(.*\))*[^,^\(^\)]*[\s\&\*]([0-9\w]+)\s*[,\)]", flags=re.DOTALL
)


def parse_args(s):
    par = parens_re.match(s)
    if not par:
        return None
    out = []
    arg_str = par[1]
    while m := re.search(args_re, arg_str):
        out += [snake_case(m[2])]
        arg_str = arg_str[m.span()[1] :]
    return out


def collect(fname, matcher, param_modifier=python_param_modifier):
    comment = ""
    with open(fname) as f:
        for line in f:
            line = line.lstrip()
            if line.startswith("/**"):
                comment = ""
            elif line.startswith("*/"):
                pass
            elif line.startswith("*") and comment is not None:
                comment += line[1:].lstrip()
            elif comment and (m := matcher(line)):
                while (args := parse_args(line)) is None:
                    line += next(f)
                    if len(line) > 500:
                        break

                method = method_key(snake_case(m[1]))
                # comment = re.sub(param_re, param_modifier, comment)
                comment = param_modifier(comment)
                method = "__".join([method, *args])
                assert method not in comments
                comments[method] = comment
                comment = ""


comments = {}

method_re = re.compile(r"(\w+::[\w\-\+\^\=\:]+)\(")
function_re = re.compile(r"([\w\-\+\^\=\:]+)\(")


# we don't handle inline functions in classes properly
# so instead just white-list functions we want
def select_functions(s):
    m = function_re.search(s)
    if m and "Triangulate" in m[0]:
        return m
    if m and "Circular" in m[0]:
        return m
    return None


collect(f"{base}/src/manifold.cpp", lambda s: method_re.search(s))
collect(f"{base}/src/constructors.cpp", lambda s: method_re.search(s))
collect(f"{base}/src/sort.cpp", lambda s: method_re.search(s))
collect(f"{base}/src/sdf.cpp", lambda s: method_re.search(s))
collect(f"{base}/src/cross_section/cross_section.cpp", lambda s: method_re.search(s))
collect(f"{base}/src/polygon.cpp", select_functions)
collect(f"{base}/include/manifold/common.h", select_functions)

comments = dict(sorted(comments.items()))

with open(f"{base}/bindings/python/docstring_override.txt") as f:
    key = ""
    for l in f:
        if l.startswith("  "):
            comments[key] += l[2:]
        else:
            key = l[:-2]
            if key not in comments.keys():
                print(f"Error, unknown docstring override key {key}")
                exit(-1)
            comments[key] = ""

gen_h = "autogen_docstrings.inl"
with open(gen_h, "w") as f:
    f.write("#pragma once\n\n")
    f.write("// --- AUTO GENERATED ---\n")
    f.write("// gen_docs.py is run by cmake build\n\n")
    f.write("namespace manifold_docstrings {\n")
    for key, doc in comments.items():
        f.write(f'const char* {key} = R"___({doc.strip()})___";\n')
    f.write("} // namespace manifold_docs")
