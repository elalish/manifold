---
title: Tips and Tricks
---
## Degrees vs. Radians

Manifold's rotation API takes **degrees**, not radians. This is a deliberate design choice: Manifold can eliminate floating point error entirely for multiples of 90°, and uses more efficient code paths when it detects such rotations.

```js
// Pass degrees directly
box.rotate([90, 0, 0]);
box.rotate([0, 45, 0]);
```

If your own code works in radians (e.g. when using `Math.atan2`), convert to degrees before passing to Manifold:

```js
const toDeg = (rad) => rad * (180 / Math.PI);

const angle = Math.atan2(y, x); // radians
box.rotate([0, 0, toDeg(angle)]);
```

For common angles, always prefer the exact degree value (`90`, `45`, `30`, etc.) over a computed approximation to get the precision guarantee. See [issue #1262](https://github.com/elalish/manifold/issues/1262) for a real-world example where a rotation of `-89.999999999999` instead of `-90` caused mesh cracks.
