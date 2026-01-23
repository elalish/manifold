---
title: Contributing
group: none
category: none
---

## Documentation

This guide and the ManifoldCAD User Guide are built using [TypeDoc](https://typedoc.org).

### Building Documentation

Documentation can be built using NPM:

```bash
cd bindings/wasm
npm install
npm run build
npm run docs
```

After this, the User and Developer docs will be in `docs/jsuser` and `docs/jsapi` respectively.

### Writing for the User Guide

The ManifoldCAD User Guide is generated from the declaration files present in `types/`.  These mostly re-export declarations, allowing the User Guide to inherit from the Developer Guide.  

TypeDoc reflections (comments) flagged as `@internal` will _not_ be visible in the User Guide, but will be present in the Developer Guide.  Reflections tagged as `@hidden` will not be visible in either guide.

### WASM Bindings

The manifold library itself is documented in three files:

 * `manifold-global-types.d.ts`: Defines lower level types and interfaces like {@link manifold.Vec2 | Vec2}, {@link manifold.Mat3 | Mat3}, &c.
 * `manifold-encapsulated-types.d.ts`: Declares objects and types that are implemented in C++, and compiled to WASM.  Definitions of classes like {@link manifold.Manifold | Manifold} and {@link manifold.CrossSection | CrossSection} can be found here.
 * `manifold-root.d.ts`: Defines the {@link manifold.ManifoldToplevel | ManifoldToplevel} interface, and re-exports the above two files.

These three files are consolidated into `manifold.d.ts` using [API Extractor](https://api-extractor.com/).  This is automatically run before each build as an npm lifecycle hook.  It can be run manually as `npm run prebuild`.

### Terminology

These are general guidelines, not hard rules.  The aim of this section is to encourage consistent documentation, not to institute brand standards.

* When referring to the library, manifold should generally be lowercase.  Manifold can be capitalized when grammatically correct, for example at the start of a sentence.  If unclear, `manifold` can be marked up as code to distinguish it from the manifold property.
* The Manifold class should generally be capitalized.  Where appropriate, it can be linked directly to the {@link manifold.Manifold | Manifold} class documentation.
* ManifoldCAD should be capitalized.  [ManifoldCAD.org](https://manifoldcad.org) should be a link.