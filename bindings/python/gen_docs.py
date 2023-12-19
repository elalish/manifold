# %%

from os.path import dirname
from hashlib import md5
import re

base = dirname(dirname(dirname(__file__)))

def snake_case(name):
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

def param_snake_case(m):
    return f':{snake_case(m[0][1:])}:'
    
def method_key(name):
    name = re.sub('\+', '_plus', name)
    name = re.sub('\-', '_minus', name)
    name = re.sub('\^', '_xor', name)
    name = re.sub('\=', '_eq', name)
    name = re.sub('\:', '_', name)
    name = re.sub('\~', 'destroy_', name)
    return name

def close_paren(s):
    cnt = 0
    for i in range(len(s)):
        if s[i] == '(':
            cnt += 1
        elif s[i] == ')':
            cnt -= 1
            if cnt == 0:
                return i
    return -1

def collect(fname, matcher):
    param_re = re.compile(r'@param (\w+)')
    comment = ''
    with open(fname) as f:
        for line in f:
            line = line.lstrip()
            if line.startswith('/**'):
                comment = ''
            elif line.startswith('*/'):
                pass
            elif line.startswith('*') and comment is not None:
                comment += line[1:].lstrip()
            elif comment and (m := matcher(line)):
                while (close := close_paren(line)) < 0:
                    line += next(f)
                line = re.sub(r'\s', '', line[:close+1])
                method = method_key(snake_case(m[1]))
                comment = re.sub(param_re, param_snake_case, comment)
                lhash = md5(line.encode("utf-8")).hexdigest()
                method = f'{method}_{lhash[:12]}'
                comments[method] = comment
                comment = ''

comments = {}

method_re = re.compile(r'(\w+::[\w\-\+\^\=\:]+)\(')
function_re = re.compile(r'([\w\-\+\^\=\:]+)\(')

# we don't handle inline functions in classes very well
# so instead just white-list functions we want
def select_functions(s):
    m = function_re.search(s)
    if m and 'Triangulate' in m[0]:
        return m
    if m and 'Circular' in m[0]:
        return m
    return None

collect(f'{base}/src/manifold/src/manifold.cpp', lambda s: method_re.search(s))
collect(f'{base}/src/manifold/src/constructors.cpp', lambda s: method_re.search(s))
collect(f'{base}/src/cross_section/src/cross_section.cpp', lambda s: method_re.search(s))
collect(f'{base}/src/polygon/src/polygon.cpp', select_functions)
collect(f'{base}/src/utilities/include/public.h', select_functions)

comments = dict(sorted(comments.items()))
comments
# %%

gen_h = f'{base}/bindings/python/docstrings.inl'
with open(gen_h, 'w') as f:
    f.write('#pragma once\n')
    f.write('#include <string>\n')
    f.write('namespace manifold_docs {\n')
    for key, doc in comments.items():
        f.write(f'const auto {key} = R"___({doc.strip()})___";\n')
    f.write('} // namespace manifold_docs')

# %%
