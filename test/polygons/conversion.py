# Usage:
# python conversion.py file.cpp output.txt
import sys
import re

TEST_PATTERN = re.compile(r'TEST\(\w+,\s?(\w+)\)')
POLYS_PUSHBACK = re.compile(r'\s+polys.push_back')
VERTEX = re.compile(r'\s+\{(-?[0-9\.ef+-]+),\s?(-?[0-9\.ef+-]+)\}')
TEST_POLY = re.compile(r'\s+TestPoly\(polys,\s?(\d+)(,\s?(-?[0-9\.ef+-]+))?\)')
WEIRD_FORMAT = re.compile(r'\d+\.f')

if len(sys.argv) != 3:
    print('Usage: python conversion.py file.cpp output.txt')
    exit(1)

with open(sys.argv[1]) as f:
    lines = f.readlines()

output_lines = []

def sanitize(f):
    m = WEIRD_FORMAT.match(f)
    if m:
        return f[:-1] + '0'
    if f[-1] == 'f':
        return f[:-1]
    return f

name = ""
polys = []
start = False
for l in lines:
    if not start:
        m = TEST_PATTERN.match(l)
        if m:
            name = m.group(1)
            polys = []
            start = True
            print(name)
        else:
            continue
    m = POLYS_PUSHBACK.match(l)
    if m:
        polys.append([])
        continue
    m = VERTEX.match(l)
    if m:
        polys[-1].append((sanitize(m.group(1)), sanitize(m.group(2))))
        continue
    m = TEST_POLY.match(l)
    if m:
        expected = m.group(1)
        precision = m.group(3)
        if precision is None:
            precision = -1.0

        output_lines.append(f'{name} {expected} {precision} {len(polys)}')
        for poly in polys:
            output_lines.append(f'{len(poly)}')
            assert len(poly) > 0
            for v in poly:
                output_lines.append(f'{v[0]} {v[1]}')
        start = False

with open(sys.argv[2], 'w') as f:
    f.write('\n'.join(output_lines))
