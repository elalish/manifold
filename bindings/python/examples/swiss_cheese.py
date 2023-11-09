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
 
import random
from manifold3d import Manifold, OpType
from functools import reduce

def run(n=10):
    block = Manifold.cube(1, 1, 1).translate(-0.5, -0.5, -0.5)

    holes = [Manifold.sphere(0.01+random.random()*0.2, 50).translate(random.random()-0.5, random.random()-0.5, random.random()-0.5)
               for i in range(n) for j in range(n)]

    # These techniques should be equivalent
    cheese_1 = block - reduce(lambda a, b: a + b, holes)
    cheese_2 = Manifold.batch_boolean([block]+holes, OpType.Subtract)

    assert cheese_1.num_vert        () == cheese_2.num_vert        ()
    assert cheese_1.num_edge        () == cheese_2.num_edge        ()
    assert cheese_1.num_tri         () == cheese_2.num_tri         ()
    assert cheese_1.genus           () == cheese_2.genus           ()
    assert cheese_1.get_volume      () == cheese_2.get_volume      ()
    assert cheese_1.get_surface_area() == cheese_2.get_surface_area()

    return cheese_1
