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
from manifold3d import Manifold, Mesh

def sphere_implicit(x, y, z):
    sphere1 = 0.5 - math.sqrt((x*x) + (y*y) + (z*z))
    x += 0.25; y += 0.25; z -= 0.25
    sphere2 = 0.5 - math.sqrt((x*x) + (y*y) + (z*z))
    return min(sphere1, -sphere2)

def sphere_implicit_batched(coords):
    coords = coords.copy()
    x = coords[:, 0]; y = coords[:, 1]; z = coords[:, 2]
    sphere1 = 0.5 - np.sqrt((x*x) + (y*y) + (z*z))
    x += 0.25; y += 0.25; z -= 0.25
    sphere2 = 0.5 - np.sqrt((x*x) + (y*y) + (z*z))
    return np.minimum(sphere1, -sphere2)

def run():
    levelset_mesh_slow = Mesh.levelset      (sphere_implicit,         [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5], 0.05, 0.0)
    levelset_mesh_fast = Mesh.levelset_batch(sphere_implicit_batched, [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5], 0.05, 0.0, False)
    model_slow = Manifold.from_mesh(levelset_mesh_slow)
    model_fast = Manifold.from_mesh(levelset_mesh_fast)
    assert abs(model_slow.get_volume() - model_fast.get_volume()) < 0.0000001
    return model_slow
