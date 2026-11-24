"""Tests for calling the bindings from several threads."""

import sys
import sysconfig
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from manifold3d import Error, ExecutionContext, Manifold, Mesh

THREADS = 4
TIMEOUT = 60


def sponge(offset=0.0):
    m = Manifold.cube((2, 2, 2), True)
    hole = Manifold.cylinder(3, 0.3, quality=48, center=True)
    for i in range(6):
        h = hole.translate((0.3 * (i - 2.5) + offset, 0, 0))
        m = m - h - h.rotate((90, 0, 0)) - h.rotate((0, 90, 0))
    return m


def evaluated(m):
    assert m.status() == Error.NoError
    return m


def run_concurrently(fn, n=THREADS):
    """Calls fn(i) for i in range(n) from n threads released together."""
    barrier = threading.Barrier(n, timeout=TIMEOUT)

    def task(i):
        barrier.wait()
        return fn(i)

    with ThreadPoolExecutor(n) as pool:
        futures = [pool.submit(task, i) for i in range(n)]
        return [f.result(timeout=TIMEOUT) for f in futures]


def mesh_key(m):
    # status() evaluates without holding the GIL, to_mesh() would hold it.
    status = m.status()
    mesh = m.to_mesh()
    return (status, mesh.vert_properties.copy(), mesh.tri_verts.copy())


def assert_same_mesh(a, b):
    assert a[0] == b[0]
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(a[2], b[2])


# Each case takes an ExecutionContext, builds its inputs, and returns a call
# that takes a few hundred ms unless cancelled.
def status_case(ctx):
    m = sponge(0.01)
    for k in range(1, 3):
        m = m + sponge(0.02 * k).translate((2.5 * k, 0, 0))
    m = m.with_context(ctx)
    return m.status


def refine_case(ctx):
    m = evaluated(sponge()).with_context(ctx)
    return lambda: m.refine(12).status()


def refine_to_length_case(ctx):
    m = evaluated(sponge()).with_context(ctx)
    return lambda: m.refine_to_length(0.01).status()


def refine_to_tolerance_case(ctx):
    m = Manifold.sphere(1, 32).calculate_normals().smooth_by_normals()
    m = evaluated(m).with_context(ctx)
    return lambda: m.refine_to_tolerance(1e-5).status()


def minkowski_sum_case(ctx):
    m = evaluated(sponge()).with_context(ctx)
    other = evaluated(Manifold.sphere(0.1, 8))
    return lambda: m.minkowski_sum(other).status()


def minkowski_difference_case(ctx):
    m = evaluated(sponge()).with_context(ctx)
    other = evaluated(Manifold.sphere(0.1, 8))
    return lambda: m.minkowski_difference(other).status()


def hull_case(ctx):
    m = evaluated(Manifold.sphere(1, 600)).with_context(ctx)
    return lambda: m.hull().status()


def from_mesh_case(ctx):
    mesh = Manifold.sphere(1, 1024).to_mesh()
    return lambda: ctx.from_mesh(mesh).status()


def from_mesh64_case(ctx):
    mesh = Manifold.sphere(1, 1024).to_mesh64()
    return lambda: ctx.from_mesh(mesh).status()


# level_set() holds the GIL, but other threads run between its callbacks.
def level_set_case(ctx):
    def sdf(x, y, z):
        return 1 - (x * x + y * y + z * z) ** 0.5

    bounds = [-1.2, -1.2, -1.2, 1.2, 1.2, 1.2]
    return lambda: ctx.level_set(sdf, bounds, 0.03).status()


@pytest.mark.parametrize(
    "case",
    [
        status_case,
        refine_case,
        refine_to_length_case,
        refine_to_tolerance_case,
        minkowski_sum_case,
        minkowski_difference_case,
        hull_case,
        from_mesh_case,
        from_mesh64_case,
        level_set_case,
    ],
)
def test_cancel_from_another_thread(case):
    ctx = ExecutionContext()
    call = case(ctx)
    started = threading.Event()
    result = {}

    def worker():
        started.set()
        result["status"] = call()

    # A long switch interval keeps the worker on the GIL from started.set()
    # until it enters the call. Without the GIL release, this thread then
    # can't run again until the call has returned, so the cancel is too late.
    interval = sys.getswitchinterval()
    sys.setswitchinterval(0.1)
    try:
        thread = threading.Thread(target=worker)
        thread.start()
        assert started.wait(TIMEOUT)
        ctx.cancel()
        thread.join(TIMEOUT)
    finally:
        sys.setswitchinterval(interval)
    assert not thread.is_alive()
    assert result["status"] == Error.Cancelled


@pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"),
    reason="needs a free-threaded Python",
)
def test_import_keeps_gil_disabled():
    assert not sys._is_gil_enabled()


def test_merge_while_others_read_mesh():
    # Normals split the vertices, so the mesh without its merge list is open
    # and merge() builds the list.
    full = sponge().calculate_normals().to_mesh()
    mesh = Mesh(np.array(full.vert_properties), np.array(full.tri_verts))
    merged = mesh_key(Manifold(full))
    before = mesh_key(Manifold(mesh))
    assert before[0] != merged[0]

    def work(i):
        if i % 2:
            return mesh_key(Manifold(mesh.merge()))
        return mesh_key(Manifold(mesh))

    for _ in range(10):
        for i, key in enumerate(run_concurrently(work)):
            assert_same_mesh(merged if i % 2 else before, key)
    assert len(mesh.merge_from_vert) == 0


def test_separate_trees_match_serial():
    expected = [mesh_key(sponge(0.01 * i)) for i in range(THREADS)]
    got = run_concurrently(lambda i: mesh_key(sponge(0.01 * i)))
    for a, b in zip(expected, got):
        assert_same_mesh(a, b)


def test_shared_tree_first_evaluation():
    expected = mesh_key(sponge())
    for _ in range(5):
        shared = sponge()
        got = run_concurrently(lambda i: mesh_key(shared))
        for key in got:
            assert_same_mesh(expected, key)


def test_callbacks_while_others_evaluate():
    def stretch(v):
        return (v[0], v[1], 1.5 * v[2])

    expected = mesh_key(sponge())
    expected_warped = mesh_key(sponge().warp(stretch))
    shared = sponge()

    def work(i):
        if i % 2:
            return mesh_key(shared)
        calls = 0

        def warp(v):
            nonlocal calls
            calls += 1
            if i == 0 and calls == 500:
                raise ValueError("callback failed")
            return stretch(v)

        try:
            return mesh_key(shared.warp(warp))
        except ValueError:
            return None

    got = run_concurrently(work)
    assert got[0] is None
    assert_same_mesh(expected_warped, got[2])
    assert_same_mesh(expected, got[1])
    assert_same_mesh(expected, got[3])
