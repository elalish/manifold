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

export const examples = {
  functions: {
    Intro: function() {
      // Write code in JavaScript or TypeScript and this editor will show the
      // API docs. Type e.g. "box." to see the Manifold API. Type
      // "CrossSection." or "Manifold." to list the 2D and 3D constructors,
      // respectively. Type "module." to see the static API - these functions
      // can also be used bare. Use console.log() to print output (lower-right).
      // This editor defines Z as up and units of mm.
      const {cube, sphere} = Manifold;
      const box = cube([100, 100, 100], true);
      const ball = sphere(60, 100);
      // You must name your final output "result", or create at least one
      // GLTFNode - see Menger Sponge and Gyroid Module examples.
      const result = box.subtract(ball);

      // For visual debug, wrap any shape with show() and it and all of its
      // copies will be shown in transparent red, akin to # in OpenSCAD. Or try
      // only() to ghost out everything else, akin to * in OpenSCAD.

      // All changes are automatically saved and restored between sessions.
      // This PWA is purely local - there is no server communication.
      // This means it will work equally well offline once loaded.
      // Consider installing it (icon in the search bar) for easy access.

      // See the script drop-down above ("Intro") for usage examples. The
      // gl-matrix package from npm is automatically imported for convenience -
      // its API is available in the top-level glMatrix object.

      // Use GLTFNode for disjoint manifolds rather than compose(), as this will
      // keep them better organized in the GLB. This will also allow you to
      // specify material properties, and even vertex colors via
      // setProperties(). See Tetrahedron Puzzle example.
      return result;
    },

    TetrahedronPuzzle: function() {
      // A tetrahedron cut into two identical halves that can screw together as
      // a puzzle. This only outputs one of the halves. This demonstrates how
      // redundant points along a polygon can be used to make twisted extrusions
      // smoother. Based on the screw puzzle by George Hart:
      // https://www.thingiverse.com/thing:186372

      const edgeLength = 50;  // Length of each edge of the overall tetrahedron.
      const gap = 0.2;  // Spacing between the two halves to allow sliding.
      const nDivisions = 50;  // Divisions (both ways) in the screw surface.

      const scale = edgeLength / (2 * Math.sqrt(2));

      const tet = Manifold.tetrahedron().scale(scale);

      const box = [];
      box.push([1, -1], [1, 1]);
      for (let i = 0; i <= nDivisions; ++i) {
        box.push([gap / (4 * scale), 1 - i * 2 / nDivisions]);
      }

      const cyan = [0, 1, 1];
      const magenta = [1, 0, 1];
      const fade = (color, pos) => {
        for (let i = 0; i < 3; ++i) {
          color[i] = cyan[i] * pos[2] + magenta[i] * (1 - pos[2]);
        }
      };

      // setProperties(3, fade) creates three channels of vertex properties
      // according to the above fade function. setMaterial assigns these
      // channels as colors, and sets the factor to white, since our default is
      // yellow.
      const screw = setMaterial(
          Manifold.extrude(box, 1, nDivisions, 270).setProperties(3, fade),
          {baseColorFactor: [1, 1, 1], attributes: ['COLOR_0']});

      const result = tet.intersect(
          screw.rotate([0, 0, -45]).translate([0, 0, -0.5]).scale(2 * scale));

      // Assigned materials are only applied to a GLTFNode. Note that material
      // definitions cascade, applying recursively to all child surfaces, but
      // overridden by any materials defined lower down. Our default material:
      // {
      //   roughness = 0.2,
      //   metallic = 1,
      //   baseColorFactor = [1, 1, 0],
      //   alpha = 1,
      //   unlit = false,
      //   name = ''
      // }
      const node = new GLTFNode();
      node.manifold = result;
      return result;
    },

    RoundedFrame: function() {
      // Demonstrates how at 90-degree intersections, the sphere and cylinder
      // facets match up perfectly, for any choice of global resolution
      // parameters.
      const {sphere, cylinder, union} = Manifold;

      function roundedFrame(edgeLength, radius, circularSegments = 0) {
        const edge = cylinder(edgeLength, radius, -1, circularSegments);
        const corner = sphere(radius, circularSegments);

        const edge1 = union(corner, edge).rotate([-90, 0, 0]).translate([
          -edgeLength / 2, -edgeLength / 2, 0
        ]);

        const edge2 = union(
            union(edge1, edge1.rotate([0, 0, 180])),
            edge.translate([-edgeLength / 2, -edgeLength / 2, 0]));

        const edge4 = union(edge2, edge2.rotate([0, 0, 90])).translate([
          0, 0, -edgeLength / 2
        ]);

        return union(edge4, edge4.rotate([180, 0, 0]));
      }

      setMinCircularAngle(3);
      setMinCircularEdgeLength(0.5);
      const result = roundedFrame(100, 10);
      return result;
    },

    Heart: function() {
      // Smooth, complex manifolds can be created using the warp() function.
      // This example recreates the Exploitable Heart by Emmett Lalish:
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
          const df =
              6 * a3 * r4 * r - 5 * b * r4 - 12 * a2 * r2 * r + 6 * a * r;
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

      const ball = Manifold.sphere(1, 200);
      const heart = ball.warp(func);
      const box = heart.boundingBox();
      const result = heart.scale(100 / (box.max[0] - box.min[0]));
      return result;
    },

    Scallop: function() {
      // A smoothed manifold demonstrating selective edge sharpening with
      // smooth() and refine(), see more details at:
      // https://elalish.blogspot.com/2022/03/smoothing-triangle-meshes.html

      const height = 10;
      const radius = 30;
      const offset = 20;
      const wiggles = 12;
      const sharpness = 0.8;
      const n = 50;

      const positions = [];
      const triangles = [];
      positions.push(-offset, 0, height, -offset, 0, -height);
      const sharpenedEdges = [];

      const delta = 3.14159 / wiggles;
      for (let i = 0; i < 2 * wiggles; ++i) {
        const theta = (i - wiggles) * delta;
        const amp = 0.5 * height * Math.max(Math.cos(0.8 * theta), 0);

        positions.push(
            radius * Math.cos(theta), radius * Math.sin(theta),
            amp * (i % 2 == 0 ? 1 : -1));
        let j = i + 1;
        if (j == 2 * wiggles) j = 0;

        const smoothness = 1 - sharpness * Math.cos((theta + delta / 2) / 2);
        let halfedge = triangles.length + 1;
        sharpenedEdges.push({halfedge, smoothness});
        triangles.push(0, 2 + i, 2 + j);

        halfedge = triangles.length + 1;
        sharpenedEdges.push({halfedge, smoothness});
        triangles.push(1, 2 + j, 2 + i);
      }

      const triVerts = Uint32Array.from(triangles);
      const vertProperties = Float32Array.from(positions);
      const scallop = new Mesh({numProp: 3, triVerts, vertProperties});

      const colorCurvature = (color, pos, oldProp) => {
        const a = Math.max(0, Math.min(1, oldProp[0] / 3 + 0.5));
        const b = a * a * (3 - 2 * a);
        const red = [1, 0, 0];
        const blue = [0, 0, 1];
        for (let i = 0; i < 3; ++i) {
          color[i] = (1 - b) * blue[i] + b * red[i];
        }
      };
      const result = Manifold.smooth(scallop, sharpenedEdges)
                         .refine(n)
                         .calculateCurvature(-1, 0)
                         .setProperties(3, colorCurvature);

      const node = new GLTFNode();
      node.manifold = result;
      node.material = {
        baseColorFactor: [1, 1, 1],
        metallic: 0,
        attributes: ['COLOR_0']
      };
      return result;
    },

    TorusKnot: function() {
      // Creates a classic torus knot, defined as a string wrapping periodically
      // around the surface of an imaginary donut. If p and q have a common
      // factor then you will get multiple separate, interwoven knots. This is
      // an example of using the warp() method, thus avoiding any direct
      // handling of triangles.

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
        const {vec3} = glMatrix;

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

        const offset = 2
        const circle = CrossSection.circle(1, n).translate([offset, 0]);

        const func = (v) => {
          const psi = q * Math.atan2(v[0], v[1]);
          const theta = psi * p / q;
          const x1 = Math.sqrt(v[0] * v[0] + v[1] * v[1]);
          const phi = Math.atan2(x1 - offset, v[2]);
          vec3.set(
              v, threadRadius * Math.cos(phi), 0, threadRadius * Math.sin(phi));
          const center = vec3.fromValues(0, 0, 0);
          const r = majorRadius + minorRadius * Math.cos(theta);
          vec3.rotateX(v, v, center, -Math.atan2(p * minorRadius, q * r));
          v[0] += minorRadius;
          vec3.rotateY(v, v, center, theta);
          v[0] += majorRadius;
          vec3.rotateZ(v, v, center, psi);
        };

        let knot = Manifold.revolve(circle, m).warp(func);

        if (kLoops > 1) {
          const knots = [];
          for (let k = 0; k < kLoops; ++k) {
            knots.push(knot.rotate([0, 0, 360 * (k / kLoops) * (q / p)]));
          }
          knot = Manifold.compose(knots);
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
      const {vec2} = glMatrix;

      function fractal(holes, hole, w, position, depth, maxDepth) {
        w /= 3;
        holes.push(
            hole.scale([w, w, 1.0]).translate([position[0], position[1], 0.0]));
        if (depth == maxDepth) return;
        const offsets = [
          vec2.fromValues(-w, -w), vec2.fromValues(-w, 0.0),
          vec2.fromValues(-w, w), vec2.fromValues(0.0, w),
          vec2.fromValues(w, w), vec2.fromValues(w, 0.0),
          vec2.fromValues(w, -w), vec2.fromValues(0.0, -w)
        ];
        for (let offset of offsets)
          fractal(
              holes, hole, w, vec2.add(offset, position, offset), depth + 1,
              maxDepth);
      }

      function mengerSponge(n) {
        let result = Manifold.cube([1, 1, 1], true);
        const holes = [];
        fractal(holes, result, 1.0, [0.0, 0.0], 1, n);

        const hole = Manifold.compose(holes);

        result = Manifold.difference(
            result,
            hole,
            hole.rotate([90, 0, 0]),
            hole.rotate([0, 90, 0]),
        );
        return result;
      }

      const posColors = (newProp, pos) => {
        for (let i = 0; i < 3; ++i) {
          newProp[i] = (1 - pos[i]) / 2;
        }
      };

      const result = mengerSponge(3)
                         .trimByPlane([1, 1, 1], 0)
                         .setProperties(3, posColors)
                         .scale(100);

      const node = new GLTFNode();
      node.manifold = result;
      node.material = {baseColorFactor: [1, 1, 1], attributes: ['COLOR_0']};
      return result;
    },

    StretchyBracelet: function() {
      // Recreates Stretchy Bracelet by Emmett Lalish:
      // https://www.thingiverse.com/thing:13505
      const {vec2} = glMatrix;

      function base(
          width, radius, decorRadius, twistRadius, nDecor, innerRadius,
          outerRadius, cut, nCut, nDivision) {
        let b = Manifold.cylinder(width, radius + twistRadius / 2);
        const circle = [];
        const dPhiDeg = 180 / nDivision;
        for (let i = 0; i < 2 * nDivision; ++i) {
          circle.push([
            decorRadius * Math.cos(dPhiDeg * i * Math.PI / 180) + twistRadius,
            decorRadius * Math.sin(dPhiDeg * i * Math.PI / 180)
          ]);
        }
        let decor = Manifold.extrude(circle, width, nDivision, 180)
                        .scale([1, 0.5, 1])
                        .translate([0, radius, 0]);
        for (let i = 0; i < nDecor; i++)
          b = b.add(decor.rotate([0, 0, (360.0 / nDecor) * i]));
        const stretch = [];
        const dPhiRad = 2 * Math.PI / nCut;

        const o = vec2.fromValues(0, 0);
        const p0 = vec2.fromValues(outerRadius, 0);
        const p1 = vec2.fromValues(innerRadius, -cut);
        const p2 = vec2.fromValues(innerRadius, cut);
        for (let i = 0; i < nCut; ++i) {
          stretch.push(vec2.rotate([0, 0], p0, o, dPhiRad * i));
          stretch.push(vec2.rotate([0, 0], p1, o, dPhiRad * i));
          stretch.push(vec2.rotate([0, 0], p2, o, dPhiRad * i));
          stretch.push(vec2.rotate([0, 0], p0, o, dPhiRad * i));
        }
        const result =
            Manifold.intersection(Manifold.extrude(stretch, width), b);
        return result;
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

        return Manifold.difference(
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
      // https://www.thingiverse.com/thing:25477. This sample demonstrates the
      // use of a Signed Distance Function (SDF) to create smooth, complex
      // manifolds.
      const {vec3} = glMatrix;

      // number of modules along pyramid edge (use 1 for print orientation)
      const m = 4;
      // module size
      const size = 20;
      // SDF resolution
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
          min: vec3.fromValues(-period, -period, -period),
          max: vec3.fromValues(period, period, period)
        };
        return Manifold.levelSet(gyroid, box, period / n, level)
            .scale(size / period);
      };

      function rhombicDodecahedron() {
        const box = Manifold.cube([1, 1, 2], true).scale(size * Math.sqrt(2));
        const result =
            box.rotate([90, 45, 0]).intersect(box.rotate([90, 45, 90]));
        return result.intersect(box.rotate([0, 0, 45]));
      }

      const gyroidModule = rhombicDodecahedron()
                               .intersect(gyroidOffset(-0.4))
                               .subtract(gyroidOffset(0.4));

      if (m > 1) {
        for (let i = 0; i < m; ++i) {
          for (let j = i; j < m; ++j) {
            for (let k = j; k < m; ++k) {
              const node = new GLTFNode();
              node.manifold = gyroidModule;
              node.translation =
                  [(k + i - j) * size, (k - i) * size, (-j) * size];
              node.material = {
                baseColorFactor:
                    [(k + i - j + 1) / m, (k - i + 1) / m, (j + 1) / m]
              };
            }
          }
        }
      }

      const result = gyroidModule.rotate([-45, 0, 90]).translate([
        0, 0, size / Math.sqrt(2)
      ]);
      return result;
    }
  },

  functionBodies: new Map()
};

for (const [func, code] of Object.entries(examples.functions)) {
  const whole = code.toString();
  const lines = whole.split('\n');
  lines.splice(0, 1);   // remove first line
  lines.splice(-2, 2);  // remove last two lines
  // remove first six leading spaces
  const body = '\n' + lines.map(l => l.slice(6)).join('\n');

  const name =
      func.replace(/([a-z])([A-Z])/g, '$1 $2');  // Add spaces between words
  examples.functionBodies.set(name, body);
};

if (typeof self !== 'undefined') {
  self.examples = examples;
}
