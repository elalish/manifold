
# %%
from os.path import dirname
import re

base = dirname(dirname(dirname(__file__)))

manifold_h = f'{base}/src/manifold/include/manifold.h'
public_h = f'{base}/src/utilities/include/public.h'
polygon_cpp = f'{base}/src/polygon/src/polygon.cpp'
manifold_cpp = f'{base}/src/manifold/src/manifold.cpp'

method_re = re.compile(r'Manifold::([\w\-\+\^\=\:]+)\(')
param_re = re.compile(r'@param (\w+)')
comments = {}

def snake_case(name):
    # name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

def method_key(name):
    name = re.sub('\+', '_plus', name)
    name = re.sub('\-', '_minus', name)
    name = re.sub('\^', '_xor', name)
    name = re.sub('\=', '_eq', name)
    name = re.sub('\:', '_', name)
    name = re.sub('\~', 'destroy_', name)
    return name

comment = ''
with open(manifold_cpp) as f:
    for line in f:
        line = line.lstrip()
        if line.startswith('/**'):
            comment = ''
        elif line.startswith('*/'):
            pass
        elif line.startswith('*') and comment is not None:
            comment += line[1:].lstrip()
        elif m := method_re.search(line):
            method = 'manifold_'+method_key(snake_case(m[1]))
            splices = []
            last = 0
            for m in param_re.finditer(comment):
                b, e = m.span()
                var = snake_case(m[1])
                splices += [
                    comment[last:b],
                    f':param {var}:'
                ]
                last = e
            splices += [comment[last:]]
            comment = ''.join(splices)
            orig_method = method
            n = 1
            while method in comments:
                method = f'{orig_method}_{n}'
                n += 1
            comments[method] = comment
            comment = ''

comments
# %%

# comments = {}

qual_re = re.compile(r'([\w\-\+\^\=\:]+)\(')
comment = ''
patterns = ["Circular", "Triangulate"]
for fname in [polygon_cpp, public_h]:
    with open(fname) as f:
        for line in f:
            line = line.lstrip()
            if line.startswith('/**'):
                comment = ''
            elif line.startswith('*/'):
                pass
            elif line.startswith('*') and comment is not None:
                comment += line[1:].lstrip()
            elif m := qual_re.search(line):
                allow = sum(p in line for p in patterns)
                if allow:
                    method = method_key(snake_case(m[1]))
                    splices = []
                    last = 0
                    for m in param_re.finditer(comment):
                        b, e = m.span()
                        var = snake_case(m[1])
                        splices += [
                            comment[last:b],
                            f':param {var}:'
                        ]
                        last = e
                    splices += [comment[last:]]
                    comment = ''.join(splices)

                    def re_snake_case(m):
                        return (m[1] or '')+snake_case(m[2])
                    comment = re.sub(r'(\w+::)*(\w+)\(\)', re_snake_case, comment)

                    comments[method] = comment
                comment = ''

comments
# %%

gen_h = f'{base}/bindings/python/docstrings.inl'
with open(gen_h, 'w') as f:
    f.write('#pragma once\n')
    f.write('namespace manifold_docs {\n')
    for key, doc in comments.items():
        f.write(f'const auto {key} = R"___({doc.strip()})___";\n')
    f.write('} // namespace manifold_docs')
# %%
