# Manifold

This is a geometry library dedicated to maintaining manifold meshes. What does it mean to be manifold and why should you care? Here we are dealing with triangle meshes, which are a simple form of B-rep, or boundary representation, which is to say a solid that is represented implicitly by defining only its boundary, or surface. The trouble is, this implicit definition of solidity is only valid if the boundary is manifold, meaning has no gaps, holes or edges. Think of a Mobius strip: it is a surface, but it does not represent the boundary of any solid. 

## Manifoldness definition

There are many definitions of manifoldness, most of them from the realm of mathematics. Common definitions will tell you things like the set of manifold objects is not closed under Boolean operations (joining solids) because if you join two cubes along an edge, that edge will join to four surfaces instead of two, so it is no longer manifold. This is not a terribly useful definition in the realm of computational geometry. 

Instead, we choose here to separate the concepts of geometry and topology. We will define geometry as anything involving position in space. These will generally be represented in floating-point and mostly refers to vertex properties like positions, normals, etc. Topology on the other hand, will be anything related to connection. These will generally be represented as integers and applies to faces and edges, which refer to each other and vertices by index. 

The key thing to remember is that topology is exact, while geometry is inexact due to floating-point rounding. For this reason, we will choose a definition of manifoldness which relies solely on topology, so as to get consistent results. 

As such we will use the same definition of manifoldness as is used in the 3D Manufacturing Format spec: [3MF Core spec](https://github.com/3MFConsortium/spec_core/blob/master/3MF%20Core%20Specification.md), section 4.1 (disclosure: I wrote most of the 3MF core spec while I was a member of their working group). This definition has the convenient property that the set of manifold meshes IS closed under Boolean operations, since duplicate vertices are allowed as that is a geometric property. 

## What's here

Not much yet, this is an early work in progress. This library is intended to be fast with guaranteed manifold output. As such you need to input manifold solids to start, which can be hard to come by since it doesn't matter at all for 3D graphics. This library links in Assimp, which will import many kinds of 3D files, but you'll get an error if the imported mesh isn't manifold. Various automated repair tools exist online for fixing non manifold models, usually for 3D printing. 

To aid in speed, this library makes extensive use of parallelization, generally through Nvidia's thrust library. I'm mostly testing it with the CUDA backend for now, but it also supports OpenMP and TBB, so I plan to test those as well for portability. 

Not everything is so parallelizable, for instance a polygon triangulation algorithm is included which is CPU-based. It is guaranteed manifold (theoretically; I still need to write better tests for it) in that bundling the triangle half edges with the input polygon's edges together form a manifold result, which means if you triangulate all the faces of a manifold polyhedron, you'll get a manifold triangle mesh.

## About the author

This library is by [Emmett Lalish](https://www.thingiverse.com/emmett).