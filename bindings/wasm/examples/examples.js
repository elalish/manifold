const examples = new Map();

function appendExample(name, func) {
  const whole = func.toString();
  const lines = whole.split('\n');
  lines.splice(0, 1);// remove first line
  lines.splice(-2, 2);// remove last two lines
  for (const line of lines) {
    line.substring(2);// remove first two leading spaces
  }
  const body = '\n' + lines.join('\n');
  examples.set(name, body);
}

function Intro() {
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
}
appendExample('Intro', Intro);

function Warp() {
  const ball = sphere(60, 100);
  const func = (v) => {
    v[2] /= 2;
  };
  const result = ball.warp(func);
  return result;
}
appendExample('Warp', Warp);
