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
from manifold3d import Manifold, Mesh

def sphere_implicit(x, y, z):
    sphere1 = 0.5 - math.sqrt((x*x) + (y*y) + (z*z))
    x += 0.25; y += 0.25; z -= 0.25
    sphere2 = 0.5 - math.sqrt((x*x) + (y*y) + (z*z))
    return min(sphere1, -sphere2)

def run():
    levelset_mesh = Mesh.levelset(sphere_implicit, [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5], 0.05, 0.0)
    model = Manifold.from_mesh(levelset_mesh)
    return model
