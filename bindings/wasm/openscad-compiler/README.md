# OpenSCAD to Manifold.js Prototype Compiler

Early-stage experimental prototype for:
https://github.com/opencax/GSoC/issues/117

## Current Project State

This project parses OpenSCAD source, resolves recursive include/use dependencies, and compiles to JavaScript that runs with manifold-3d.

Pipeline:

```text
OpenSCAD (.scad) -> Lexer -> Parser -> AST -> Compiler -> Manifold JS (.js)
```

Generated files in out/ export:

```js
export const result = ...
```

Each compiled file imports out/runtime.js.

## Supported Constructs

### Geometry and Modules
- Primitives: cube, sphere, cylinder, circle, square, polygon, polyhedron, text
- Transforms: translate, rotate, scale, mirror, multmatrix, resize
- Boolean ops: union, difference, intersection, hull, minkowski
- Extrusion: linear_extrude, rotate_extrude
- Other modules: projection, offset, color, render, children, echo, assert, let

### Language Features
- Variables, functions, modules
- for statements
- List-comprehension forms: for, C-style for, if, let, each
- if/else
- Expressions: arithmetic, comparison, logical, ternary, unary, indexing, member access
- Named arguments and default parameters
- Vectors and ranges
- Lambdas and dynamic calls, for example expr(args)
- Strings, booleans, undef, comments
- include <...> and use <...> (resolved recursively before compile)

## Include and Use Resolution

demo.ts resolves dependencies via src/resolver.ts using:
- The current file directory
- The workspace root
- OPENSCADPATH
- Default OpenSCAD user library locations by OS

include imports declarations and top-level geometry.

use imports declarations only (modules/functions).

## Setup and Run

Install dependencies:

```bash
npm install
```

Set OPENSCADPATH in your shell before running compile commands.

PowerShell example:

```powershell
$env:OPENSCADPATH = "C:\Users\<you>\Documents\OpenSCAD\libraries"
```

Bash example:

```bash
export OPENSCADPATH="$HOME/Documents/OpenSCAD/libraries"
```

Compile common examples via package scripts:

```bash
npm run dev:cube
npm run dev:advanced
npm run dev:boolean_ops
```

Compile all files under examples/:

```bash
npm run compile-all-examples
```

Compile any .scad file:

```bash
npx tsx demo.ts path/to/file.scad
```

Compiler output behavior:
- Writes generated JavaScript to out/<input-name>.js
- Prints status/progress logs to stdout (not the full generated source)

## Viewer

Start any static file server from project root, for example:

```bash
npx serve .
```

Then open:
- http://localhost:3000/viewer.html

In the viewer:
- Enter a compiled output file path such as out/cube.js
- Click Load

## Visual Comparison Assets

- BOSL2-oriented examples are in examples/
- Comparison images are in images/
- Comparison write-up is in comparison.md

## 2D Support

2D primitives (circle, square, polygon, text) and 2D operations (offset, projection, boolean ops on 2D) use Manifold CrossSection APIs. They stay 2D until linear_extrude or rotate_extrude is used.

## Current Limitations

- This is still a prototype and does not fully match OpenSCAD semantics in all cases.
- Full BOSL2 compatibility is not guaranteed.
- text() is approximated as a simple box-like CrossSection based on text length and size.
- render() and resize() are currently passthrough stubs and do not implement full OpenSCAD behavior.
- 2D minkowski() falls back to hull (CrossSection has no native Minkowski operation).
- 3D minkowski() may use approximation fallback paths depending on runtime support.
- Unknown modules are ignored with a warning and compile to empty geometry.
- Statement modifiers (#, !, %, *) are parsed but not currently applied during compilation.
- surface() and import() signatures exist, but builtin implementations are not wired yet.
- compile-all-examples currently includes examples/err.scad, which intentionally fails to parse and causes a non-zero exit code.
- OPENSCADPATH is read from the environment at runtime. A .env file exists, but it is not automatically loaded by these scripts.
- Compiled outputs import ./runtime.js, so out/runtime.js must be present.

## Future Work

- Implement full semantics for statement modifiers (#, !, %, *).
- Add builtin support for additional modules such as surface() and import().
- Improve text() geometry generation (font-aware outlines instead of box approximation).
- Improve Minkowski behavior and robustness for both 2D and 3D cases.
- Add stricter diagnostics mode (for example, fail on unknown modules instead of warning-only fallback).
- Add automated tests and CI coverage for parser, resolver, and compiler output.
- Improve compile-all-examples UX (skip known negative tests by default, or separate them into a dedicated failure test set).
- Add dedicated scripts for more examples and remove or fix outdated scripts.
- Improve documentation and contributor workflow for keeping out/runtime.js in sync.

## Project Layout

```text
src/
  lexer.ts
  parser.ts
  ast.ts
  compiler.ts
  resolver.ts
demo.ts
compile_all_examples.ts
examples/
images/
out/
  runtime.js
viewer.html
comparison.md
```
