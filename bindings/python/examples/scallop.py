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
    # A smoothed manifold demonstrating manual smoothing by assigning tangents to
    # halfedges. Use refine() before export to see the curvature.

    height = 10
    radius = 30
    offset = 20
    wiggles = 12
    lean = 0.3
    frontal_sharpness = 1
    n = 50

    def lerp(a, b, t):
        return np.array(a) + (np.array(b) - np.array(a)) * t

    triangles = []
    positions = [[-offset, 0, height], [-offset, 0, -height]]
    halfedge_tangent = []

    len_ = np.pi * radius / (2 * wiggles)
    top_normal = np.array([-lean, 0, np.sqrt(1 - lean * lean)])
    delta = np.pi / wiggles
    center_tangents = [None] * (2 * wiggles)
    edge_tangents = [None] * (2 * wiggles)

    for i in range(2 * wiggles):
        theta = i * delta
        amp = 0.5 * height * (np.cos(theta) + 1) / 2
        v = np.array(
            [
                radius * np.cos(theta),
                radius * np.sin(theta),
                amp * (1 if i % 2 == 0 else -1),
            ],
            dtype=np.float64,
        )

        positions.append(v.tolist())

        center_tan = np.array([(v[0] + offset) / 2, v[1] / 2, (v[2] - height) / 2])
        center_tangents[i] = center_tan - np.dot(center_tan, top_normal) * top_normal
        edge_tangents[i] = np.array([-len_ * np.sin(theta), len_ * np.cos(theta), 0.0])

    for i in range(2 * wiggles):
        next_i = 0 if i + 1 == 2 * wiggles else i + 1

        radial = center_tangents[i]
        next_radial = center_tangents[next_i]
        edge = edge_tangents[i]
        next_edge = -edge_tangents[next_i]
        sharpness = frontal_sharpness * (np.cos(i * delta) + 1) / 4
        up = lerp(
            np.array([0, 0, len_]),
            np.array([next_edge[1], -next_edge[0], 0]),
            sharpness,
        )
        down = lerp(
            np.array([0, 0, -len_]), np.array([-edge[1], edge[0], 0]), sharpness
        )

        triangles.append([0, 2 + i, 2 + next_i])
        halfedge_tangent.extend([*radial.tolist(), 1])
        halfedge_tangent.extend([*edge.tolist(), 1])
        halfedge_tangent.extend([*up.tolist(), 1])

        triangles.append([1, 2 + next_i, 2 + i])
        halfedge_tangent.extend([next_radial[0], next_radial[1], -next_radial[2], 1])
        halfedge_tangent.extend([*next_edge.tolist(), 1])
        halfedge_tangent.extend([*down.tolist(), 1])

    scallop = Mesh(
        tri_verts=np.array(triangles, np.int32),
        vert_properties=np.array(positions, np.float32),
        halfedge_tangent=np.array(halfedge_tangent, np.float32).reshape(-1, 3, 4),
    )

    def colorCurvature(_pos, oldProp):
        a = max(0, min(1, oldProp[0] / 3 + 0.5))
        b = a * a * (3 - 2 * a)
        red = [1, 0, 0]
        blue = [0, 0, 1]
        return [(1 - b) * blue[i] + b * red[i] for i in range(3)]

    return (
        Manifold(scallop)
        .refine(n)
        .calculate_curvature(-1, 0)
        .set_properties(3, colorCurvature)
    )
