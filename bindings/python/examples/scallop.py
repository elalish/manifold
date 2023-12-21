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

from manifold3d import Manifold, Mesh
import numpy as np


def run():
    # A smoothed manifold demonstrating selective edge sharpening with
    # smooth() and refine(), see more details at:
    # https://elalish.blogspot.com/2022/03/smoothing-triangle-meshes.html

    height = 10
    radius = 30
    offset = 20
    wiggles = 12
    sharpness = 0.8
    n = 50

    triangles = []
    positions = [[-offset, 0, height], [-offset, 0, -height]]
    sharpenedEdges = []

    delta = np.pi / wiggles
    for i in range(2 * wiggles):
        theta = (i - wiggles) * delta
        amp = 0.5 * height * max(np.cos(0.8 * theta), 0)

        positions.append(
            [
                radius * np.cos(theta),
                radius * np.sin(theta),
                amp * (1 if i % 2 == 0 else -1),
            ]
        )

        j = i + 1
        if j == 2 * wiggles:
            j = 0

        smoothness = 1 - sharpness * np.cos((theta + delta / 2) / 2)
        halfedge = 3 * len(triangles) + 1
        sharpenedEdges.append((halfedge, smoothness))
        triangles.append([0, 2 + i, 2 + j])

        halfedge = 3 * len(triangles) + 1
        sharpenedEdges.append((halfedge, smoothness))
        triangles.append([1, 2 + j, 2 + i])

    scallop = Mesh(
        tri_verts=np.array(triangles, np.int32),
        vert_properties=np.array(positions, np.float32),
    )

    def colorCurvature(_pos, oldProp):
        a = max(0, min(1, oldProp[0] / 3 + 0.5))
        b = a * a * (3 - 2 * a)
        red = [1, 0, 0]
        blue = [0, 0, 1]
        return [(1 - b) * blue[i] + b * red[i] for i in range(3)]

    edges, smoothing = zip(*sharpenedEdges)
    return (
        Manifold.smooth(scallop, edges, smoothing)
        .refine(n)
        .calculate_curvature(-1, 0)
        .set_properties(3, colorCurvature)
    )
