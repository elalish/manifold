# Get Started


## Installation

In your project root folder run:

```bash
npm i manifold-3d
```

To start using Manifold, import it and initialize it:

```js
import Module from 'manifold-3d';

const wasm = await Module();
wasm.setup();
const { Manifold } = wasm;
```

Intro example

```js
const {cube, sphere} = Manifold;
const box = cube([100, 100, 100], true);
const ball = sphere(60, 100);
const result = box.subtract(ball);
```

## Next steps

In order to visualize Manifold mesh using Three.js library please check out our example [here](https://github.com/elalish/manifold/blob/master/bindings/wasm/examples/three.ts) 