# OpenSCAD to TypeScript Compiler with Manifold

A compiler that translates OpenSCAD source into TypeScript/JavaScript
modules built on top of [Manifold](https://github.com/elalish/manifold)'s WASM bindings
(`manifold-3d`).

This tool lowers `.scad` files to plain ES modules. Each compiled file exports a
`result` (a `Manifold` or `CrossSection`) that can be evaluated in Node or in the
browser, so an OpenSCAD model becomes an ordinary JavaScript dependency.

```scad
// test/examples/cube.scad
translate([1, 2, 3]) cube(5);
```

compiles to

```ts
import * as __rt from '../../runtime/runtime.js';
const {__cube, __translate, __union2d3d, __applyRoot, /* ... */} = __rt;

const __result_items: any[] = [];
__result_items.push(__translate(__cube(5, false), [1, 2, 3]));
export const result = __union2d3d(__applyRoot(__result_items));
export const background = __union2d3d(__applyRoot(__background_items, true));
export const __viewport = {vpr: __ctx.$vpr, vpt: __ctx.$vpt, vpd: __ctx.$vpd, vpf: __ctx.$vpf};
```

## Architecture

```
.scad --> lexer --> parser --> AST --> resolver --> IR --> emitter --> .ts --> runtime --> Manifold
```

| Path | Role |
| --- | --- |
| `core/lexer.ts` | Tokenizer; every token carries a `SourceRange` (offset/line/column). |
| `core/parser.ts` | Recursive-descent parser producing the AST in `core/ast.ts`. |
| `core/resolver.ts` | Resolves `include <>` / `use <>`, walks the file closure, applies `use`-scope privatization, and finds library roots from `OPENSCADPATH`. |
| `core/ir.ts` | Small intermediate representation for geometry nodes. |
| `core/compiler.ts` | Emitter: name mangling, scoping, module/function lowering, tail-call elimination, font and surface-data embedding. |
| `core/orchestrate.ts` | Drives consumer + external-library compilation and the library cache. |
| `runtime/runtime.ts` | The runtime every compiled file imports: primitives, transforms, booleans, extrusions, OpenSCAD value semantics (`__add`, `__eq`, `__index`, …), `echo`, `rands`, `text`, `surface`. |
| `commands/` | `commander` subcommands (`compile`, `compile-all`). |
| `index.ts` | CLI entry point (`openscad-to-manifold`). |
| `viewer.html` | Three.js viewer that loads a compiled `.ts` module in the browser. |

Source locations survive the whole pipeline: the emitted TypeScript is annotated with the
originating `.scad` file and comments, which is what `test/source-location.test.ts`
verifies.

### External libraries

`include <BOSL2/std.scad>` does *not* get inlined into your output. Libraries found on the
search path are compiled **once** into `runtime/libraries/<name-lowercased>/`, alongside a
`.manifest.json`, and the consumer file imports from there. The cache is reused on
subsequent builds and recompiled when the manifest is missing entries the current program
needs.

> On change of compiler version or deletion of `runtime/libraries/` the cached library output is regenerated with the new codegen.

## Setup

### 1. Install dependencies

Requires Node.js 25+

```bash
npm install
```

### 2. Configure paths

Copy `.env.example` to `.env` and fill in what you need. Values are also read from the
real environment; `.env` only fills in variables that aren't already set.

```ini
OPENSCADPATH=/path/to/openscad/libraries   # for include/use of BOSL2, MCAD, ...
FONTPATH=./bundle/fonts                    # TTF/OTF fonts for text()
IMAGEBASEPATH=./bundle/images              # base dir for surface() PNG/DAT files
```

`OPENSCADPATH` accepts multiple `path.delimiter`-separated entries. The standard per-OS
user library directories (`~/Documents/OpenSCAD/libraries`,
`~/.local/share/OpenSCAD/libraries`, …) are appended automatically when they exist. The
directory of the file being compiled and the current working directory are always
searched first.

`bundle/fonts` ships Liberation Sans (regular/bold) and `bundle/images` ships the
surface fixtures used by the test corpus, so pointing at those is enough to run
everything in this repo.

### 3. Build

```bash
npm run build
```

This does two things: bundles the CLI to `dist/index.cjs`, and compiles
`runtime/runtime.ts` to `runtime/runtime.js` — the file that compiled output imports.
**`runtime/runtime.js` is gitignored, so a build is mandatory before anything runs.**

Optionally expose the CLI globally:

```bash
npm link
```

## Usage

### Compile a single file

```bash
npx tsx index.ts compile path/to/model.scad --output out/model.ts
# or, after npm link:
openscad-to-manifold compile path/to/model.scad --output out/model.ts
```

`--output` must end in `.ts`. Without it, output goes to
`test/out/<basename>.ts`. The command reports the resolved file count, any external
libraries used, and the size of the generated code.

### Compile the whole example corpus

```bash
npm run compile-all
```

Walks `test/examples/**` recursively and mirrors the tree into `test/out/`, reporting a
list of failures and exiting non-zero if any file fails.

### Run compiled output

A compiled module is a normal ES module - importing it evaluates the model.

```ts
import {result} from './out/model.js';

console.log(result.volume(), result.surfaceArea());
```

```bash
npx tsx out/model.ts
```

Named exports:

| Export | Meaning |
| --- | --- |
| `result` | The model geometry (`Manifold` for 3D, `CrossSection` for 2D). |
| `background` | Geometry under the `%` modifier. |
| `__viewport` | `$vpr` / `$vpt` / `$vpd` / `$vpf` at end of evaluation. |

### View in the browser

```bash
npm run serve
```

Then open `viewer.html` and enter a path to a compiled module (e.g. `test/out/cube.ts`).
The viewer strips TypeScript annotations on the fly, imports the module, renders it with
Three.js, and applies the exported viewport variables.

## Language support

**Primitives** — `cube`, `sphere`, `cylinder`, `circle`, `square`, `polygon`,
`polyhedron`, `text`, `surface`

**Transforms** — `translate`, `rotate`, `scale`, `mirror`, `resize`, `multmatrix`,
`offset`, `color`

**Booleans / operators** — `union`, `difference`, `intersection`, `hull`, `minkowski`
(3D), `projection`, `render`, `group`

**Extrusions** — `linear_extrude` (including `v`, `scale`, `twist`, `slices`, `segments`,
`$fe`), `rotate_extrude`

**Control flow** — `for`, `intersection_for`, `if`/`else`, `let`, `each`, list
comprehensions (`for` / `if` / `let` / C-style `for`), ranges with OpenSCAD's iteration
and overflow semantics

**Abstraction** — modules and functions with default arguments and named parameters,
`children()` / `$children` (special variables are read live, not snapshotted), function
literals and `let`-bound recursive lambdas, `include <>` / `use <>` with `use`-scope
variable privatization

**Recursion** — self tail calls in named functions become loops; mutual and lambda tail
calls go through a trampoline; unconditionally infinite module/function recursion is a
compile-time error rather than a stack overflow

**Modifiers** — `*` (disable, dropped at compile time), `!` (root, first wins),
`%` (background, exported separately as `background`), `#` (parsed and accepted; it only
affects preview highlighting in OpenSCAD, so it has no effect on the emitted geometry)

**Builtin functions** — `is_undef`, `is_bool`, `is_num`, `is_string`, `is_list`,
`is_function`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `abs`, `sign`,
`floor`, `ceil`, `round`, `sqrt`, `exp`, `ln`, `log`, `pow`, `min`, `max`, `norm`,
`cross`, `len`, `str`, `chr`, `ord`, `concat`, `search`, `lookup`, `rands`, `version`,
`version_num`, `parent_module`, plus `echo` and `assert` (both as statements and as
expression modifiers)

`rands()` reproduces OpenSCAD byte-for-byte (MT19937 + the same seed hash and
`generate_canonical` draw), so seeded models match upstream exactly.

### Special variables

| Variable | Default | Notes |
| --- | --- | --- |
| `$fn`, `$fa`, `$fs` | `0`, `12`, `2` | Mesh resolution. |
| `$t` | `0` | Animation time, `0`–`1`. |
| `$preview` | `false` | Preview-mode flag. |
| `$vpr`, `$vpt`, `$vpd`, `$vpf` | `[0,0,0]`, `[0,0,0]`, `500`, `22.5` | Viewport; re-exported as `__viewport`. |
| `$children` | — | Child count inside a module. |
| `$parent_modules` | `0` | Depth of the module call stack. |
| `$idx` | — | Index within `for`-generated children. |
| `$color` | — | Color inherited from an enclosing `color()`. |

To animate a compiled model, assign `$t` before `result` is evaluated.

### Not supported yet

- `import()` of external geometry (STL/OFF/3MF/DXF/SVG)
- 2D `minkowski()`
- `resize()` along an axis normal to a flat object (warns and skips)
- `is_object()` and other experimental/opt-in builtins

## Testing

```bash
npm test
```

`pretest` builds, links, and runs `compile-all` first, so the suites always test freshly
generated output. The three suites can also be run individually:

| Command | What it checks |
| --- | --- |
| `npm run test:geometry` | Volume and surface area of every compiled model against an OpenSCAD baseline, within 0.1% relative tolerance. Each model runs in a forked worker with a 60s timeout. |
| `npm run test:semantic` | `echo()` output of compiled models against pre-recorded OpenSCAD output in `test/echo-results/`. |
| `npm run test:source-location` | Structural invariants on every AST node's `SourceRange` (non-negative, well-ordered, nested inside its parent, slices back to real source). |

### Test corpus

`test/examples/` holds 308 `.scad` files: the OpenSCAD regression suite
(`OpenScad/Openscad-Tests/`, plus `Basics`, `Advanced`, `Functions`, `Parametric`,
`Old`), BOSL2-heavy models (`bosl2/`), and 135 echo-semantics files (`echo/`).

Geometry baselines live in a `// Volume: …, SurfaceArea: …` comment on the first line of
each `.scad` file - 123 files currently carry one, and files without one are skipped by
the geometry suite. Baselines are produced with a local OpenSCAD install
(`openscad -o output.3mf --backend=manifold`), then read back through
`manifold-3d`'s `importManifold`:

```bash
cd test && npx tsx parameter-calculator.ts examples/path/to/model.scad   # one file
npm run compute-parameters                                              # bulk
```
Echo baselines in `test/echo-results/*.echo` are pre-generated so CI does not need an
OpenSCAD binary.

### Benchmarks

```bash
npm run benchmark                  # end-to-end: our compile+run vs. openscad CLI
npx tsx test/benchmark-phases.ts   # splits runtime init / logical evaluation / mesh eval
```

Both need `openscad` on `PATH` for the comparison numbers; `benchmark-phases` accepts
`--dir=<path>` and a run count.

## CI

The `OpenSCAD Compiler` job in `.github/workflows/manifold.yml` clones BOSL2 and MCAD into
a scratch directory, points `OPENSCADPATH`/`FONTPATH`/`IMAGEBASEPATH` at them and at
`bundle/`, and runs `npm test`. The main WASM job explicitly excludes
`openscad-compiler/**` so the two never run together.

## See also

- [`comparison.md`](comparison.md) — side-by-side renders of OpenSCAD output vs. compiled
  Manifold output for a handful of models.
