"""
 Copyright 2023 The Manifold Authors.

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

import math
import numpy as np
from manifold3d import Manifold


def gyroid(x, y, z):
    xi = x - math.pi / 4.0
    yi = y - math.pi / 4.0
    zi = z - math.pi / 4.0
    return (
        math.cos(xi) * math.sin(yi)
        + math.cos(yi) * math.sin(zi)
        + math.cos(zi) * math.sin(xi)
    )


def gyroid_levelset(level, period, size, n):
    return Manifold.level_set(
        gyroid,
        [-period, -period, -period, period, period, period],
        period / n,
        level,
    ).scale([size / period] * 3)


def rhombic_dodecahedron(size):
    box = Manifold.cube(size * math.sqrt(2.0) * np.array([1, 1, 2]), True)
    result = box.rotate([90, 45, 0]) ^ box.rotate([90, 45, 90])
    return result ^ box.rotate([0, 0, 45])


def gyroid_module(size=20, n=15):
    period = math.pi * 2.0
    result = (
        gyroid_levelset(-0.4, period, size, n) ^ rhombic_dodecahedron(size)
    ) - gyroid_levelset(0.4, period, size, n)
    return result.rotate([-45, 0, 90]).translate([0, 0, size / math.sqrt(2.0)])


def run(size=20, n=15):
    return gyroid_module(size, n)
