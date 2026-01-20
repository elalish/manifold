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

TypeDoc reflections (comments) flagged as `@internal` will _not_ be visible in the User Guide, but will be present in the Developer Guide.

### Terminology

These are general guidelines, not hard rules.  The aim of this section is to encourage consistent documentation, not to institute brand standards.

* When referring to the library, manifold should generally be lowercase.  Manifold can be capitalized when grammatically correct, for example at the start of a sentence.  If unclear, `manifold` can be marked up as code to distinguish it from the manifold property.
* The Manifold class should generally be capitalized.  Where appropriate, it can be linked directly to the {@link manifold.Manifold | Manifold} class documentation.
* ManifoldCAD should be capitalized.  [ManifoldCAD.org](https://manifoldcad.org) should be a link.