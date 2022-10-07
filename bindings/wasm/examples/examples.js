const exampleFunctions = {
  Intro: function () {
    // Write code in JavaScript or TypeScript and this editor will show the API docs.
    // Manifold constructors include "cube", "cylinder", "sphere", "extrude", "revolve".
    // Type e.g. "box." to see the Manifold API.
    // Use console.log() to print output (lower-right).
    // This editor defines Z as up and units of mm.
    const box = cube([100, 100, 100], true);
    const ball = sphere(60, 100);
    // You must name your final output "result".
    const result = box.subtract(ball);
    return result;
  },

  Warp: function () {
    const ball = sphere(60, 100);
    const func = (v) => {
      v[2] /= 2;
    };
    const result = ball.warp(func);
    return result;
  },

  RoundedFrame: function () {
    function roundedFrame(edgeLength, radius, circularSegments = 0) {
      const edge = cylinder(edgeLength, radius, -1, circularSegments);
      const corner = sphere(radius, circularSegments);

      let edge1 = union(corner, edge);
      edge1 = edge1.rotate([-90, 0, 0]).translate([-edgeLength / 2, -edgeLength / 2, 0]);

      let edge2 = edge1.rotate([0, 0, 180]);
      edge2 = union(edge2, edge1);
      edge2 = union(edge2, edge.translate([-edgeLength / 2, -edgeLength / 2, 0]));

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

  MengerSponge: function () {
    function vec2add(a, b) {
      return [a[0] + b[0], a[1] + b[1]];
    }

    function fractal(holes, hole, w, position, depth, maxDepth) {
      w /= 3;
      holes.push(
        hole.scale([w, w, 1.0]).translate([position[0], position[1], 0.0]));
      if (depth == maxDepth) return;
      const offsets = [
        [-w, -w], [-w, 0.0], [-w, w], [0.0, w], [w, w], [w, 0.0], [w, -w], [0.0, -w]
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

    const result = mengerSponge(3).scale([100, 100, 100]);
    return result;
  },

  StretchyBracelet: function () {
    function base(
      width, radius, decorRadius, twistRadius,
      nDecor, innerRadius, outerRadius, cut,
      nCut, nDivision) {
      function rotate(v, theta) {
        return [
          v[0] * Math.cos(theta) - v[1] * Math.sin(theta),
          v[0] * Math.sin(theta) + v[1] * Math.cos(theta)
        ];
      }

      let b = cylinder(width, radius + twistRadius / 2);
      const circle = [[]];
      const dPhiDeg = 180 / nDivision;
      for (let i = 0; i < 2 * nDivision; ++i) {
        circle[0].push([
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
      const stretch = [[]];
      const dPhiRad = 2 * Math.PI / nCut;

      const p0 = [outerRadius, 0];
      const p1 = [innerRadius, -cut];
      const p2 = [innerRadius, cut];
      for (let i = 0; i < nCut; ++i) {
        stretch[0].push(rotate(p0, dPhiRad * i));
        stretch[0].push(rotate(p1, dPhiRad * i));
        stretch[0].push(rotate(p2, dPhiRad * i));
        stretch[0].push(rotate(p0, dPhiRad * i));
      }
      b = intersection(extrude(stretch, width), b);
      return b;
    }

    function stretchyBracelet(
      radius = 30, height = 8, width = 15,
      thickness = 0.4, nDecor = 20, nCut = 27,
      nDivision = 30) {
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
          innerRadius, outerRadius + 3 * adjThickness, cut, nCut, nDivision));
    }

    const result = stretchyBracelet();
    return result;
  }
};

const examples = new Map();

for (const [name, func] of Object.entries(exampleFunctions)) {
  const whole = func.toString();
  const lines = whole.split('\n');
  lines.splice(0, 1);// remove first line
  lines.splice(-2, 2);// remove last two lines
  for (const line of lines) {
    line.substring(2);// remove first two leading spaces
  }
  const body = '\n' + lines.join('\n');

  name.replace(/([A-Z])/g, " $1");// Add spaces between words
  examples.set(name, body);
};
