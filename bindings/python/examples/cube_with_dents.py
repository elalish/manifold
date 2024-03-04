"""
 Copyright 2022 The Manifold Authors.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

from manifold3d import Manifold
from functools import reduce

# https://gist.github.com/ochafik/2db96400e3c1f73558fcede990b8a355#file-cube-with-half-spheres-dents-scad


def run(n=5, overlap=True):
    a = Manifold.cube([n, n, 0.5]).translate([-0.5, -0.5, -0.5])

    spheres = [
        Manifold.sphere(0.45 if overlap else 0.55, 50).translate([i, j, 0])
        for i in range(n)
        for j in range(n)
    ]
    # spheres = reduce(lambda a, b: a + b, spheres)

    return a - sum(spheres, Manifold())
