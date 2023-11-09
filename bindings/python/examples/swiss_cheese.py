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
 
import time
import random
from manifold3d import Manifold, OpType
from functools import reduce

def run(n=10):
    a = Manifold.cube(1, 1, 1).translate(-0.5, -0.5, -0.5)

    spheres = [Manifold.sphere(0.01+random.random()*0.2, 50).translate(random.random()-0.5, random.random()-0.5, random.random()-0.5)
               for i in range(n) for j in range(n)]

    t0 = time.perf_counter()
    combined_spheres = reduce(lambda a, b: a + b, spheres)
    slow_cheese = a - combined_spheres
    print("Individual cheese:", (time.perf_counter() - t0)*1000.0, "ms")

    t0 = time.perf_counter()
    fast_cheese = Manifold.batch_boolean([a]+spheres, OpType.Subtract)
    print("Batch cheese: %s" % (time.perf_counter() - t0))

    return fast_cheese