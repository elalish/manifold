// Copyright 2022 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

(function(exports) {

exports.functions = {
  Intro: function() {
    // Write code in JavaScript or TypeScript and this editor will show the API
    // docs. Type e.g. "box." to see the Manifold API. Type "Module." to see the
    // static API - these functions can also be used bare. Use console.log() to
    // print output (lower-right). This editor defines Z as up and units of mm.

    const box = cube([100, 100, 100], true);
    const ball = sphere(60, 100);
    // You must name your final output "result".
    const result = box.subtract(ball);

    // All changes are automatically saved and restored between sessions.
    // This app is purely local - there is no server communication.
    // This means it will work equally well offline once loaded.
    // To share your code with another browser/device, simply copy the text to a
    // file.

    // See the script drop-down above ("Intro") for usage examples.
    return result;
  },

  TetrahedronPuzzle: function() {
    // A tetrahedron cut into two identical halves that can screw together as a
    // puzzle. This only outputs one of the halves. This demonstrates how
    // redundant points along a polygon can be used to make twisted extrusions
    // smoother. Based on the screw puzzle by George Hart:
    // https://www.thingiverse.com/thing:186372

    const edgeLength = 50;  // Length of each edge of the overall tetrahedron.
    const gap = 0.2;        // Spacing between the two halves to allow sliding.
    const nDivisions =
        50;  // Number of divisions (both ways) in the screw surface.

    const scale = edgeLength / (2 * Math.sqrt(2));

    const tet = tetrahedron().scale(scale);

    const box = [];
    box.push([2, -2], [2, 2]);
    for (let i = 0; i <= nDivisions; ++i) {
      box.push([gap / (2 * scale), 2 - i * 4 / nDivisions]);
    }

    const screw = extrude(box, 2, nDivisions, 270)
                      .rotate([0, 0, -45])
                      .translate([0, 0, -1])
                      .scale(scale);

    const result = tet.intersect(screw);
    return result;
  },

  RoundedFrame: function() {
    // Demonstrates how at 90-degree intersections, the sphere and cylinder
    // facets match up perfectly, for any choice of global resolution
    // parameters.

    function roundedFrame(edgeLength, radius, circularSegments = 0) {
      const edge = cylinder(edgeLength, radius, -1, circularSegments);
      const corner = sphere(radius, circularSegments);

      let edge1 = union(corner, edge);
      edge1 = edge1.rotate([-90, 0, 0]).translate([
        -edgeLength / 2, -edgeLength / 2, 0
      ]);

      let edge2 = edge1.rotate([0, 0, 180]);
      edge2 = union(edge2, edge1);
      edge2 =
          union(edge2, edge.translate([-edgeLength / 2, -edgeLength / 2, 0]));

      let edge4 = edge2.rotate([0, 0, 90]);
      edge4 = union(edge4, edge2);

      let frame = edge4.translate([0, 0, -edgeLength / 2]);
      frame = union(frame, frame.rotate([180, 0, 0]));

      return frame;
    }

    setMinCircularAngle(3);
    setMinCircularEdgeLength(0.5);
    const result = roundedFrame(100, 10);
    return result;
  },

  Heart: function() {
    // Smooth, complex manifolds can be created using the warp() function. This
    // example recreates the Exploitable Heart by Emmett Lalish:
    // https://www.thingiverse.com/thing:6190

    const func = (v) => {
      const x2 = v[0] * v[0];
      const y2 = v[1] * v[1];
      const z = v[2];
      const z2 = z * z;
      const a = x2 + 9 / 4 * y2 + z2;
      const b = z * z2 * (x2 + 9 / 80 * y2);
      const a2 = a * a;
      const a3 = a * a2;

      const step = (r) => {
        const r2 = r * r;
        const r4 = r2 * r2;
        // Taubin's function: https://mathworld.wolfram.com/HeartSurface.html
        const f = a3 * r4 * r2 - b * r4 * r - 3 * a2 * r4 + 3 * a * r2 - 1;
        // Derivative
        const df = 6 * a3 * r4 * r - 5 * b * r4 - 12 * a2 * r2 * r + 6 * a * r;
        return f / df;
      };
      // Newton's method for root finding
      let r = 1.5;
      let dr = 1;
      while (Math.abs(dr) > 0.0001) {
        dr = step(r);
        r -= dr;
      }
      // Update radius
      v[0] *= r;
      v[1] *= r;
      v[2] *= r;
    };

    const ball = sphere(1, 200);
    const heart = ball.warp(func);
    const box = heart.boundingBox();
    const result = heart.scale(100 / (box.max[0] - box.min[0]));
    return result;
  },

  Scallop: function() {
    // A smoothed manifold demonstrating selective edge sharpening with smooth()
    // and refine(), see more details at:
    // https://elalish.blogspot.com/2022/03/smoothing-triangle-meshes.html

    const height = 10;
    const radius = 30;
    const offset = 20;
    const wiggles = 12;
    const sharpness = 0.8;
    const n = 50;

    const vertPos = [];
    const triVerts = [];
    vertPos.push(-offset, 0, height, -offset, 0, -height);
    const sharpenedEdges = [];

    const delta = 3.14159 / wiggles;
    for (let i = 0; i < 2 * wiggles; ++i) {
      const theta = (i - wiggles) * delta;
      const amp = 0.5 * height * Math.max(Math.cos(0.8 * theta), 0);

      vertPos.push(
          radius * Math.cos(theta), radius * Math.sin(theta),
          amp * (i % 2 == 0 ? 1 : -1));
      let j = i + 1;
      if (j == 2 * wiggles) j = 0;

      const smoothness = 1 - sharpness * Math.cos((theta + delta / 2) / 2);
      let halfedge = triVerts.length + 1;
      sharpenedEdges.push({halfedge, smoothness});
      triVerts.push(0, 2 + i, 2 + j);

      halfedge = triVerts.length + 1;
      sharpenedEdges.push({halfedge, smoothness});
      triVerts.push(1, 2 + j, 2 + i);
    }

    const scallop = new Mesh();
    scallop.triVerts = Uint32Array.from(triVerts);
    scallop.vertPos = Float32Array.from(vertPos);
    const result = smooth(scallop, sharpenedEdges).refine(n);
    return result;
  },

  TorusKnot: function() {
    // Creates a classic torus knot, defined as a string wrapping periodically
    // around the surface of an imaginary donut. If p and q have a common factor
    // then you will get multiple separate, interwoven knots. This is an example
    // of using the warp() method, thus avoiding any direct handling of
    // triangles.

    // @param p The number of times the thread passes through the donut hole.
    // @param q The number of times the thread circles the donut.
    // @param majorRadius Radius of the interior of the imaginary donut.
    // @param minorRadius Radius of the small cross-section of the imaginary
    //   donut.
    // @param threadRadius Radius of the small cross-section of the actual
    //   object.
    // @param circularSegments Number of linear segments making up the
    //   threadRadius circle. Default is getCircularSegments(threadRadius).
    // @param linearSegments Number of segments along the length of the knot.
    //   Default makes roughly square facets.

    function torusKnot(
        p, q, majorRadius, minorRadius, threadRadius, circularSegments = 0,
        linearSegments = 0) {
      function gcd(a, b) {
        return b == 0 ? a : gcd(b, a % b);
      }

      const kLoops = gcd(p, q);
      p /= kLoops;
      q /= kLoops;
      const n = circularSegments > 2 ? circularSegments :
                                       getCircularSegments(threadRadius);
      const m = linearSegments > 2 ? linearSegments :
                                     n * q * majorRadius / threadRadius;

      const circle = [];
      const dPhi = 2 * 3.14159 / n;
      const offset = 2;
      for (let i = 0; i < n; ++i) {
        circle.push([Math.cos(dPhi * i) + offset, Math.sin(dPhi * i)]);
      }

      const point = new THREE.Vector3();
      const axis = new THREE.Vector3();

      const func =
          (v) => {
            const psi = q * Math.atan2(v[0], v[1]);
            const theta = psi * p / q;
            const x1 = Math.sqrt(v[0] * v[0] + v[1] * v[1]);
            const phi = Math.atan2(x1 - offset, v[2]);
            point.set(Math.cos(phi), 0, Math.sin(phi))
                .multiplyScalar(threadRadius);
            const r = majorRadius + minorRadius * Math.cos(theta);
            point.applyAxisAngle(
                axis.set(1, 0, 0), -Math.atan2(p * minorRadius, q * r));
            point.x += minorRadius;
            point.applyAxisAngle(axis.set(0, 1, 0), theta);
            point.x += majorRadius;
            point.applyAxisAngle(axis.set(0, 0, 1), psi);
            v[0] = point.x;
            v[1] = point.y;
            v[2] = point.z;
          }

      let knot = revolve(circle, m).warp(func);

      if (kLoops > 1) {
        const knots = [];
        for (let k = 0; k < kLoops; ++k) {
          knots.push(knot.rotate([0, 0, 360 * (k / kLoops) * (q / p)]));
        }
        knot = compose(knots);
      }

      return knot;
    }

    // This recreates Matlab Knot by Emmett Lalish:
    // https://www.thingiverse.com/thing:7080

    const result = torusKnot(1, 3, 25, 10, 3.75);
    return result;
  },

  MengerSponge: function() {
    // This example demonstrates how symbolic perturbation correctly creates
    // holes even though the subtracted objects are exactly coplanar.

    function vec2add(a, b) {
      return [a[0] + b[0], a[1] + b[1]];
    }

    function fractal(holes, hole, w, position, depth, maxDepth) {
      w /= 3;
      holes.push(
          hole.scale([w, w, 1.0]).translate([position[0], position[1], 0.0]));
      if (depth == maxDepth) return;
      const offsets = [
        [-w, -w], [-w, 0.0], [-w, w], [0.0, w], [w, w], [w, 0.0], [w, -w],
        [0.0, -w]
      ];
      for (let offset of offsets)
        fractal(holes, hole, w, vec2add(position, offset), depth + 1, maxDepth);
    }

    function mengerSponge(n) {
      let result = cube([1, 1, 1], true);
      const holes = [];
      fractal(holes, result, 1.0, [0.0, 0.0], 1, n);

      const hole = compose(holes);

      result = difference(result, hole);
      result = difference(result, hole.rotate([90, 0, 0]));
      result = difference(result, hole.rotate([0, 90, 0]));

      return result;
    }

    const result = mengerSponge(3).scale(100);
    return result;
  },

  StretchyBracelet: function() {
    // Recreates Stretchy Bracelet by Emmett Lalish:
    // https://www.thingiverse.com/thing:13505

    function base(
        width, radius, decorRadius, twistRadius, nDecor, innerRadius,
        outerRadius, cut, nCut, nDivision) {
      function rotate(v, theta) {
        return [
          v[0] * Math.cos(theta) - v[1] * Math.sin(theta),
          v[0] * Math.sin(theta) + v[1] * Math.cos(theta)
        ];
      }

      let b = cylinder(width, radius + twistRadius / 2);
      const circle = [];
      const dPhiDeg = 180 / nDivision;
      for (let i = 0; i < 2 * nDivision; ++i) {
        circle.push([
          decorRadius * Math.cos(dPhiDeg * i * Math.PI / 180) + twistRadius,
          decorRadius * Math.sin(dPhiDeg * i * Math.PI / 180)
        ]);
      }
      let decor =
          extrude(circle, width, nDivision, 180).scale([1, 0.5, 1]).translate([
            0, radius, 0
          ]);
      for (let i = 0; i < nDecor; i++)
        b = b.add(decor.rotate([0, 0, (360.0 / nDecor) * i]));
      const stretch = [];
      const dPhiRad = 2 * Math.PI / nCut;

      const p0 = [outerRadius, 0];
      const p1 = [innerRadius, -cut];
      const p2 = [innerRadius, cut];
      for (let i = 0; i < nCut; ++i) {
        stretch.push(rotate(p0, dPhiRad * i));
        stretch.push(rotate(p1, dPhiRad * i));
        stretch.push(rotate(p2, dPhiRad * i));
        stretch.push(rotate(p0, dPhiRad * i));
      }
      b = intersection(extrude(stretch, width), b);
      return b;
    }

    function stretchyBracelet(
        radius = 30, height = 8, width = 15, thickness = 0.4, nDecor = 20,
        nCut = 27, nDivision = 30) {
      const twistRadius = Math.PI * radius / nDecor;
      const decorRadius = twistRadius * 1.5;
      const outerRadius = radius + (decorRadius + twistRadius) * 0.5;
      const innerRadius = outerRadius - height;
      const cut = 0.5 * (Math.PI * 2 * innerRadius / nCut - thickness);
      const adjThickness = 0.5 * thickness * height / cut;

      return difference(
          base(
              width, radius, decorRadius, twistRadius, nDecor,
              innerRadius + thickness, outerRadius + adjThickness,
              cut - adjThickness, nCut, nDivision),
          base(
              width, radius - thickness, decorRadius, twistRadius, nDecor,
              innerRadius, outerRadius + 3 * adjThickness, cut, nCut,
              nDivision));
    }

    const result = stretchyBracelet();
    return result;
  },

  GyroidModule: function() {
    // Recreates Modular Gyroid Puzzle by Emmett Lalish:
    // https://www.thingiverse.com/thing:25477. This sample demonstrates the use
    // of a Signed Distance Function (SDF) to create smooth, complex manifolds.

    const size = 20;
    const n = 20;
    const pi = 3.14159;

    function gyroid(p) {
      const x = p[0] - pi / 4;
      const y = p[1] - pi / 4;
      const z = p[2] - pi / 4;
      return Math.cos(x) * Math.sin(y) + Math.cos(y) * Math.sin(z) +
          Math.cos(z) * Math.sin(x);
    }

    function gyroidOffset(level) {
      const period = 2 * pi;
      const box = {
        min: [-period, -period, -period],
        max: [period, period, period]
      };
      return levelSet(gyroid, box, period / n, level).scale(size / period);
    };

    function rhombicDodecahedron() {
      const box = cube([1, 1, 2], true).scale(size * Math.sqrt(2));
      const result =
          box.rotate([90, 45, 0]).intersect(box.rotate([90, 45, 90]));
      return result.intersect(box.rotate([0, 0, 45]));
    }

    let result = rhombicDodecahedron().intersect(gyroidOffset(-0.4))
    result = result.subtract(gyroidOffset(0.4));
    result = result.rotate([-45, 0, 90]).translate([0, 0, size / Math.sqrt(2)]);
    return result;
  }
};

exports.functionBodies = new Map();

for (const [func, code] of Object.entries(exports.functions)) {
  const whole = code.toString();
  const lines = whole.split('\n');
  lines.splice(0, 1);   // remove first line
  lines.splice(-2, 2);  // remove last two lines
  // remove first four leading spaces
  const body = '\n' + lines.map(l => l.slice(4)).join('\n');

  const name =
      func.replace(/([a-z])([A-Z])/g, '$1 $2');  // Add spaces between words
  exports.functionBodies.set(name, body);
};
})(typeof exports == 'undefined' ? this['examples'] = {} : exports);