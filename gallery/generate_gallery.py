#!/usr/bin/env python3
"""Generate gallery data for convex decomposition benchmark.

Produces:
  gallery/data/results.json   — timing, piece counts, volume stats
  gallery/data/<Name>_original.glb
  gallery/data/<Name>_mode0.glb
"""

import json
import os
import time

import manifold3d as m3d
import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None
    print("WARNING: trimesh not installed — GLB export disabled")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Helpers ─────────────────────────────────────────────────


def is_convex(m, tol=0.001):
    vol = m.volume()
    hvol = m.hull().volume()
    return vol > 0 and abs(hvol - vol) < vol * tol


def manifold_to_trimesh(m, color=None):
    """Convert a Manifold to a trimesh.Trimesh."""
    mesh = m.to_mesh()
    verts = np.array(mesh.vert_properties)[:, :3]
    faces = np.array(mesh.tri_verts).reshape(-1, 3)
    t = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if color is not None:
        t.visual.face_colors = np.tile(color, (len(faces), 1))
    return t


def export_glb(manifolds, filepath, colors=None):
    """Export a list of manifolds as a single colored GLB scene."""
    if trimesh is None:
        return
    meshes = []
    palette = colors or [
        [31, 119, 180, 255],
        [255, 127, 14, 255],
        [44, 160, 44, 255],
        [214, 39, 40, 255],
        [148, 103, 189, 255],
        [140, 86, 75, 255],
        [227, 119, 194, 255],
        [127, 127, 127, 255],
        [188, 189, 34, 255],
        [23, 190, 207, 255],
    ]
    for i, m in enumerate(manifolds):
        c = palette[i % len(palette)]
        meshes.append(manifold_to_trimesh(m, color=c))
    scene = trimesh.Scene(meshes)
    scene.export(filepath, file_type="glb")


def export_original(m, filepath):
    if trimesh is None:
        return
    t = manifold_to_trimesh(m, color=[200, 200, 200, 180])
    trimesh.Scene([t]).export(filepath, file_type="glb")


# ── Shape definitions ───────────────────────────────────────


def make_shapes():
    """Return list of (name, manifold, is_curved) tuples."""
    shapes = []

    # LShape
    s = m3d.Manifold.cube([2, 2, 2]) - m3d.Manifold.cube([1, 1, 2]).translate([1, 1, 0])
    shapes.append(("LShape", s, False))

    # CubeSphere
    s = m3d.Manifold.cube([2, 2, 2]) - m3d.Manifold.sphere(0.8, 32)
    shapes.append(("CubeSphere", s, True))

    # CubeSphereHole (cube minus 3 axis-aligned cylinders)
    s = m3d.Manifold.cube([2, 2, 2], True)
    for axis in range(3):
        cyl = m3d.Manifold.cylinder(4, 0.5, 0.5, 32, True)
        if axis == 0:
            cyl = cyl.rotate([0, 90, 0])
        elif axis == 1:
            cyl = cyl.rotate([90, 0, 0])
        s = s - cyl
    shapes.append(("CubeSphereHole", s, True))

    # TwoSpheres
    s = m3d.Manifold.sphere(1.0, 24) + m3d.Manifold.sphere(1.0, 24).translate([1.5, 0, 0])
    shapes.append(("TwoSpheres", s, True))

    # ThreeSpheres
    s = (
        m3d.Manifold.sphere(1.0, 24)
        + m3d.Manifold.sphere(1.0, 24).translate([1.5, 0, 0])
        + m3d.Manifold.sphere(1.0, 24).translate([0.75, 1.3, 0])
    )
    shapes.append(("ThreeSpheres", s, True))

    # CubeCube
    s = m3d.Manifold.cube([2, 2, 2]) - m3d.Manifold.cube([1, 1, 1]).translate([0.5, 0.5, 0])
    shapes.append(("CubeCube", s, False))

    # FlatSlab
    slab = m3d.Manifold.cube([20, 20, 1])
    for i in range(5):
        slab -= m3d.Manifold.cube([14, 1, 0.6]).translate([3, 2.0 + i * 3.5, 0.4])
    for i in range(5):
        slab -= m3d.Manifold.cube([1, 14, 0.6]).translate([2.0 + i * 3.5, 3, 0])
    shapes.append(("FlatSlab", slab, False))

    # ThinBeam
    beam = m3d.Manifold.cube([10, 1, 1])
    s = beam + beam.rotate([0, 0, 90]).translate([0, 10, 0])
    shapes.append(("ThinBeam", s, False))

    # ThinWedge
    cube = m3d.Manifold.cube([4, 4, 4], True)
    s = cube.trim_by_plane([0.1, 1, 0], 0.5)
    shapes.append(("ThinWedge", s, False))

    # ThinFin
    fin = m3d.Manifold.cube([10, 0.1, 5])
    notch = m3d.Manifold.cube([3, 0.1, 3]).translate([3.5, 0, 0])
    s = fin - notch
    shapes.append(("ThinFin", s, False))

    # HighAspectCylinder
    cyl = m3d.Manifold.cylinder(20, 1, 1, 24)
    s = cyl - m3d.Manifold.cube([2, 2, 5]).translate([-1, -1, 7.5])
    shapes.append(("HighAspectCylinder", s, True))

    return shapes


# ── Benchmark runner ────────────────────────────────────────


def benchmark_shape(name, shape, is_curved):
    """Run decomposition and return stats dict."""
    vol = shape.volume()
    sa = shape.surface_area()
    nv = shape.num_vert()
    nt = shape.num_tri()
    genus = shape.genus()

    print(f"  {name}: {nv} verts, {nt} tris, vol={vol:.4f}, genus={genus}")

    # Export original
    export_original(shape, os.path.join(DATA_DIR, f"{name}_original.glb"))

    # Run decomposition (3 warmup-excluded runs, take median)
    times = []
    pieces_result = None
    for run in range(3):
        t0 = time.perf_counter()
        pieces = shape.convex_decomposition()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        if pieces_result is None:
            pieces_result = pieces

    median_time = sorted(times)[1]
    pieces = pieces_result

    # Stats
    n_pieces = len(pieces)
    n_convex = sum(1 for p in pieces if is_convex(p))
    n_nc = n_pieces - n_convex
    vol_sum = sum(p.volume() for p in pieces)
    vol_err = abs(vol_sum - vol) / vol if vol > 0 else 0

    # Union volume (expensive but accurate)
    if n_pieces > 0:
        union_vol = m3d.Manifold.batch_boolean(pieces, m3d.OpType.Add).volume()
    else:
        union_vol = 0
    union_err = abs(union_vol - vol) / vol if vol > 0 else 0

    print(
        f"    → {n_pieces} pieces ({n_convex} convex, {n_nc} NC), "
        f"{median_time:.1f}ms, vol_err={vol_err:.6f}"
    )

    # Export decomposition GLB
    export_glb(pieces, os.path.join(DATA_DIR, f"{name}_mode0.glb"))

    return {
        "name": name,
        "volume": vol,
        "surfaceArea": sa,
        "numVerts": nv,
        "numTris": nt,
        "genus": genus,
        "isCurved": is_curved,
        "modes": [
            {
                "mode": 0,
                "modeName": "DT",
                "glbFile": f"{name}_mode0.glb",
                "pieces": n_pieces,
                "convex": n_convex,
                "nonConvex": n_nc,
                "timeMs": round(median_time, 2),
                "volumeSum": round(vol_sum, 6),
                "volumeError": round(vol_err, 6),
                "unionVolume": round(union_vol, 6),
                "unionError": round(union_err, 6),
            }
        ],
    }


def benchmark_minkowski():
    """Run Minkowski sum benchmarks."""
    results = []

    tests = [
        (
            "LShape_sphere",
            m3d.Manifold.cube([2, 2, 2])
            - m3d.Manifold.cube([1, 1, 2]).translate([1, 1, 0]),
            m3d.Manifold.sphere(0.1, 12),
        ),
        (
            "Cube_sphere",
            m3d.Manifold.cube([2, 2, 2]),
            m3d.Manifold.sphere(0.1, 12),
        ),
    ]

    # CubeSphereHole
    csh = m3d.Manifold.cube([2, 2, 2], True)
    for axis in range(3):
        cyl = m3d.Manifold.cylinder(4, 0.5, 0.5, 32, True)
        if axis == 0:
            cyl = cyl.rotate([0, 90, 0])
        elif axis == 1:
            cyl = cyl.rotate([90, 0, 0])
        csh = csh - cyl
    tests.append(("CubeSphereHole_sphere", csh, m3d.Manifold.sphere(0.1, 12)))

    # TetTet_self
    tet = m3d.Manifold.tetrahedron().scale([1.5, 1.5, 1.5])
    tests.append(("TetTet_self", tet, tet))

    for name, shape, kernel in tests:
        print(f"  Minkowski: {name}")
        s_vol = shape.volume()
        k_vol = kernel.volume()
        s_genus = shape.genus()

        # No decompose
        t0 = time.perf_counter()
        nd = shape.minkowski_sum(kernel, decompose=False)
        t1 = time.perf_counter()
        nd_time = (t1 - t0) * 1000
        nd_simp = nd.as_original().simplify(0)

        # With decompose
        t0 = time.perf_counter()
        wd = shape.minkowski_sum(kernel, decompose=True)
        t1 = time.perf_counter()
        wd_time = (t1 - t0) * 1000
        wd_simp = wd.as_original().simplify(0)

        def mink_stats(m, glb_name):
            export_original(m, os.path.join(DATA_DIR, glb_name))
            return {
                "volume": round(m.volume(), 6),
                "genus": m.genus(),
                "numVerts": m.num_vert(),
                "numTris": m.num_tri(),
                "glbFile": glb_name,
            }

        nd_data = mink_stats(nd, f"{name}_mink_nodec.glb")
        nd_data["timeMs"] = round(nd_time, 2)
        nd_data["simplified"] = mink_stats(nd_simp, f"{name}_mink_nodec_simp.glb")

        wd_data = mink_stats(wd, f"{name}_mink_dec.glb")
        wd_data["timeMs"] = round(wd_time, 2)
        wd_data["simplified"] = mink_stats(wd_simp, f"{name}_mink_dec_simp.glb")

        print(
            f"    no_dec: {nd_time:.1f}ms, with_dec: {wd_time:.1f}ms, "
            f"speedup: {nd_time/wd_time:.1f}x"
        )

        results.append(
            {
                "name": name,
                "shapeVolume": round(s_vol, 6),
                "kernelVolume": round(k_vol, 6),
                "shapeGenus": s_genus,
                "noDecompose": nd_data,
                "withDecompose": wd_data,
            }
        )

    return results


# ── Main ────────────────────────────────────────────────────


def main():
    print("Generating gallery benchmark data...\n")

    shapes = make_shapes()
    shape_results = []

    print("=== Decomposition ===")
    for name, shape, is_curved in shapes:
        result = benchmark_shape(name, shape, is_curved)
        shape_results.append(result)

    print("\n=== Minkowski ===")
    mink_results = benchmark_minkowski()

    # CGAL data is static (can't run CGAL from Python)
    cgal_data = [
        {"name": "LShape", "pieces": 2, "timeMs": 16},
        {"name": "CubeSphere", "pieces": 71, "timeMs": 481},
        {"name": "TwoSpheres", "pieces": 57, "timeMs": 2583},
        {"name": "ThreeSpheres", "pieces": 235, "timeMs": 7310},
        {"name": "CubeCube", "pieces": 5, "timeMs": 28},
        {
            "name": "FlatSlab",
            "pieces": -1,
            "timeMs": -1,
            "error": "CRASH (assertion failure in SM_walls.h)",
        },
    ]

    results = {
        "shapes": shape_results,
        "minkowski": mink_results,
        "cgal": cgal_data,
    }

    out_path = os.path.join(DATA_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {out_path}")

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Shape':<22} {'Pcs':>4} {'NC':>3} {'Time':>10} {'VolErr':>10}")
    print("-" * 55)
    for s in shape_results:
        m = s["modes"][0]
        print(
            f"{s['name']:<22} {m['pieces']:>4} {m['nonConvex']:>3} "
            f"{m['timeMs']:>8.1f}ms {m['volumeError']:>9.6f}"
        )


if __name__ == "__main__":
    main()
