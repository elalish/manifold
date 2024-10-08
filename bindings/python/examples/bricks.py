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

# https://gist.github.com/deckar01/ef11def51de7e71d9f288c6e5819fdb7

INCHES = 25.4

brick_depth = (3 + 5 / 8) * INCHES
brick_height = (2 + 1 / 4) * INCHES
brick_length = (7 + 5 / 8) * INCHES

mortar_gap = (3 / 8) * INCHES


def brick():
    return Manifold.cube([brick_length, brick_depth, brick_height])


def halfbrick():
    return Manifold.cube([(brick_length - mortar_gap) / 2, brick_depth, brick_height])


def row(length):
    bricks = [
        brick().translate([(brick_length + mortar_gap) * x, 0, 0])
        for x in range(length)
    ]
    return sum(bricks, Manifold())


def wall(length, height, alternate=0):
    bricks = [
        row(length).translate(
            [
                ((z + alternate) % 2) * (brick_depth + mortar_gap),
                0,
                (brick_height + mortar_gap) * z,
            ]
        )
        for z in range(height)
    ]
    return sum(bricks, Manifold())


def walls(length, width, height):
    return sum(
        [
            wall(length, height),
            wall(width, height, 1).rotate([0, 0, 90]).translate([brick_depth, 0, 0]),
            wall(length, height, 1).translate(
                [0, (width) * (brick_length + mortar_gap), 0]
            ),
            wall(width, height)
            .rotate([0, 0, 90])
            .translate(
                [(length + 0.5) * (brick_length + mortar_gap) - mortar_gap, 0, 0]
            ),
        ],
        Manifold(),
    )


def floor(length, width):
    results = [walls(length, width, 1)]
    if length > 1 and width > 1:
        results.append(
            floor(length - 1, width - 1).translate(
                [brick_depth + mortar_gap, brick_depth + mortar_gap, 0]
            )
        )
    if length == 1 and width > 1:
        results.append(row(width - 1).rotate((0, 0, 90)))
    if width == 1 and length > 1:
        results.append(
            row(length - 1).translate(
                [2 * (brick_depth + mortar_gap), brick_depth + mortar_gap, 0]
            )
        )
    results.append(
        halfbrick().translate([brick_depth + mortar_gap, brick_depth + mortar_gap, 0])
    )
    return sum(results, Manifold())


def run(width=10, length=10, height=10):
    return walls(length, width, height) + floor(length, width)
