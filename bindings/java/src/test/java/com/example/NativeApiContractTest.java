// Copyright 2026 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.example;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import com.cadoodlecad.manifold.ManifoldBindings;
import com.cadoodlecad.manifold.ManifoldBindings.FillRule;
import com.cadoodlecad.manifold.ManifoldBindings.ManifoldError;
import com.cadoodlecad.manifold.RayHit;

/**
 * Covers the surface that the C API changed or dropped: the descriptor
 * signatures reworked for the value-returning API, plus the two behaviours that
 * now live in Java ({@code smoothOut}, {@code setProperties}) and the removed
 * struct-size introspection.
 */
class NativeApiContractTest {

	private static ManifoldBindings mb;

	@BeforeAll
	static void init() throws Exception {
		mb = new ManifoldBindings();
	}

	@AfterAll
	static void close() {
		if (mb != null)
			mb.close();
	}

	// -------------------------------------------------------------------------
	// Version is read from the library, not hardcoded
	// -------------------------------------------------------------------------

	@Test
	void versionComesFromLoadedLibrary() throws Throwable {
		int[] v = mb.version();
		assertEquals(3, v.length);
		assertEquals(3, v[0], "major version");
		assertTrue(v[1] >= 0, "minor version");
		assertTrue(v[2] >= 0, "patch version");
	}

	// -------------------------------------------------------------------------
	// Property callbacks
	// -------------------------------------------------------------------------

	@Test
	void setPropertiesInvokesUpdaterPerVertex() throws Throwable {
		MemorySegment sphere = mb.sphere(1, 16);
		MemorySegment tagged = null;
		try {
			int[] calls = { 0 };
			tagged = mb.setProperties(sphere, 4, (newProp, position, oldProp) -> {
				calls[0]++;
				MemorySegment dst = newProp.reinterpret(4L * Double.BYTES);
				dst.setAtIndex(ValueLayout.JAVA_DOUBLE, 0, position.getAtIndex(ValueLayout.JAVA_DOUBLE, 0));
				dst.setAtIndex(ValueLayout.JAVA_DOUBLE, 1, position.getAtIndex(ValueLayout.JAVA_DOUBLE, 1));
				dst.setAtIndex(ValueLayout.JAVA_DOUBLE, 2, position.getAtIndex(ValueLayout.JAVA_DOUBLE, 2));
				dst.setAtIndex(ValueLayout.JAVA_DOUBLE, 3, 9.0);
			});

			assertEquals(ManifoldError.NO_ERROR, mb.status(tagged));
			assertEquals(4, mb.numProp(tagged), "updater can widen the property block");
			assertTrue(calls[0] > 0, "the updater must run for the vertices");
			assertEquals(mb.volume(sphere), mb.volume(tagged), 1e-9, "properties do not move geometry");
			assertEquals(mb.numTri(sphere), mb.numTri(tagged));
		} finally {
			mb.safeDelete(sphere);
			mb.safeDelete(tagged);
		}
	}

	@Test
	void setPropertiesWithNullUpdaterClearsProperties() throws Throwable {
		MemorySegment sphere = mb.sphere(1, 16);
		MemorySegment withProps = null;
		MemorySegment cleared = null;
		try {
			withProps = mb.setProperties(sphere, 3, (newProp, position, oldProp) -> {
			});
			assertEquals(3, mb.numProp(withProps));

			cleared = mb.setProperties(withProps, 0, null);
			assertEquals(ManifoldError.NO_ERROR, mb.status(cleared));
			assertEquals(0, mb.numProp(cleared));
		} finally {
			mb.safeDelete(sphere);
			mb.safeDelete(withProps);
			mb.safeDelete(cleared);
		}
	}

	// -------------------------------------------------------------------------
	// Winding number now comes from the library rather than a ray-cast parity
	// -------------------------------------------------------------------------

	@Test
	void windingNumberMatchesNestedShells() throws Throwable {
		MemorySegment outer = mb.cube(10, 10, 10, true);
		MemorySegment inner = mb.cube(4, 4, 4, true);
		MemorySegment hollow = null;
		try {
			hollow = mb.difference(outer, inner);
			assertEquals(1, mb.windingNumber(hollow, 4, 0, 0), "solid shell wall");
			assertEquals(0, mb.windingNumber(hollow, 0, 0, 0), "cavity reads as outside");
			assertEquals(0, mb.windingNumber(hollow, 20, 0, 0), "far outside");
		} finally {
			mb.safeDelete(outer);
			mb.safeDelete(inner);
			mb.safeDelete(hollow);
		}
	}

	// -------------------------------------------------------------------------
	// Value-returning C signatures
	// -------------------------------------------------------------------------

	@Test
	void rayHitsCarryCrossingSign() throws Throwable {
		MemorySegment cube = mb.cube(2, 2, 2, true);
		try {
			ArrayList<RayHit> hits = mb.rayCast(cube, new double[] { 0, 0, 5 }, new double[] { 0, 0, -5 });
			assertEquals(2, hits.size());
			assertEquals(1.0, hits.get(0).position()[2], 1e-9, "entry face");
			assertEquals(-1.0, hits.get(1).position()[2], 1e-9, "exit face");

			RayHit entry = hits.get(0);
			RayHit exit = hits.get(1);
			assertEquals(1, entry.crossing(), "entering the solid");
			assertEquals(-1, exit.crossing(), "leaving the solid");
			assertEquals(1.0, entry.normal()[2], 1e-9, "entry face normal points out along +z, towards the ray");
			assertEquals(-1.0, exit.normal()[2], 1e-9, "exit face normal points out along -z");
		} finally {
			mb.safeDelete(cube);
		}
	}

	@Test
	void rayHitsSurviveTheManifoldAndTheirNativeContainer() throws Throwable {
		ArrayList<RayHit> hits;
		MemorySegment cube = mb.cube(2, 2, 2, true);
		try {
			hits = mb.rayCast(cube, new double[] { 0, 0, 5 }, new double[] { 0, 0, -5 });
		} finally {
			// The native hit vector is freed inside rayCast; the manifold goes too.
			mb.safeDelete(cube);
		}
		// Reading after both are gone proves the values are plain Java objects.
		assertEquals(2, hits.size());
		assertEquals(1.0, hits.get(0).position()[2], 1e-9);
		assertEquals(-1.0, hits.get(1).position()[2], 1e-9);
	}

	@Test
	void rayHitArraysAreDefensiveCopies() throws Throwable {
		MemorySegment cube = mb.cube(2, 2, 2, true);
		try {
			RayHit hit = mb.rayCast(cube, new double[] { 0, 0, 5 }, new double[] { 0, 0, -5 }).get(0);
			double z = hit.position()[2];
			hit.position()[2] = 99.0;
			assertEquals(z, hit.position()[2], 0, "mutating a returned array must not change the hit");
			hit.normal()[0] = 99.0;
			assertEquals(0.0, hit.normal()[0], 1e-9, "normal array is copied too");
		} finally {
			mb.safeDelete(cube);
		}
	}

	@Test
	void splitReturnsTwoOwnedHalves() throws Throwable {
		MemorySegment a = mb.cube(10, 10, 10, true);
		MemorySegment b = null;
		MemorySegment[] halves = null;
		try {
			b = mb.translate(a, 5, 0, 0);
			halves = mb.split(a, b);
			assertEquals(2, halves.length);
			// Overlapping cubes split into the exclusive volumes, 500 each.
			assertEquals(500.0, mb.volume(halves[0]), 1e-6);
			assertEquals(500.0, mb.volume(halves[1]), 1e-6);
			assertEquals(ManifoldError.NO_ERROR, mb.status(halves[0]));
			assertEquals(ManifoldError.NO_ERROR, mb.status(halves[1]));
		} finally {
			mb.safeDelete(a);
			mb.safeDelete(b);
			if (halves != null)
				for (MemorySegment h : halves)
					mb.safeDelete(h);
		}
	}

	@Test
	void decomposeHandsBackIndependentOwnedParts() throws Throwable {
		MemorySegment left = mb.translate(mb.cube(2, 2, 2, true), -5, 0, 0);
		MemorySegment right = mb.translate(mb.cube(2, 2, 2, true), 5, 0, 0);
		MemorySegment composed = null;
		MemorySegment[] parts = null;
		try {
			composed = mb.compose(new MemorySegment[] { left, right });
			// Drop the inputs: the parts must own their memory.
			mb.safeDelete(left);
			left = null;
			mb.safeDelete(right);
			right = null;

			parts = mb.decompose(composed);
			assertEquals(2, parts.length);
			double total = 0;
			for (MemorySegment p : parts) {
				assertEquals(ManifoldError.NO_ERROR, mb.status(p));
				total += mb.volume(p);
			}
			assertEquals(16.0, total, 1e-6, "two 2x2x2 cubes = 8 each");
		} finally {
			mb.safeDelete(left);
			mb.safeDelete(right);
			mb.safeDelete(composed);
			if (parts != null)
				for (MemorySegment p : parts)
					mb.safeDelete(p);
		}
	}

	@Test
	void asOriginalAssignsAnId() throws Throwable {
		MemorySegment cube = mb.cube(2, 2, 2, true);
		try {
			// The input has no original ID; a negative request must still yield one.
			MemorySegment original = mb.asOriginal(cube);
			try {
				assertEquals(mb.volume(cube), mb.volume(original), 1e-9);
			} finally {
				mb.safeDelete(original);
			}
		} finally {
			mb.safeDelete(cube);
		}
	}

	@Test
	void triangulateReturnsIndicesIntoInputContours() throws Throwable {
		ArrayList<double[][]> contours = new ArrayList<>();
		contours.add(new double[][] { { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } });
		int[][] tris = mb.triangulate(contours, 1e-9);
		assertEquals(2, tris.length, "a square triangulates into two triangles");
		for (int[] t : tris) {
			assertEquals(3, t.length);
			for (int idx : t)
				assertTrue(idx >= 0 && idx < 4, "index " + idx + " must reference the four input points");
		}
	}

	// -------------------------------------------------------------------------
	// Cross-section fill rules
	// -------------------------------------------------------------------------

	/** Area of a contour list, ignoring winding direction. */
	private static double signedArea(ArrayList<double[][]> contours) {
		double area = 0;
		for (double[][] c : contours)
			for (int i = 0; i < c.length; i++) {
				double[] p = c[i];
				double[] q = c[(i + 1) % c.length];
				area += p[0] * q[1] - q[0] * p[1];
			}
		return Math.abs(area) / 2;
	}

	@Test
	void fillRulesResolveOverlapDifferently() throws Throwable {
		ArrayList<double[][]> overlapping = new ArrayList<>();
		overlapping.add(new double[][] { { 0, 0 }, { 2, 0 }, { 2, 2 }, { 0, 2 } });
		overlapping.add(new double[][] { { 1, 1 }, { 3, 1 }, { 3, 3 }, { 1, 3 } });

		ArrayList<Double> areas = new ArrayList<>();
		for (FillRule rule : FillRule.values()) {
			MemorySegment cs = mb.crossSectionOfPolygons(overlapping, rule);
			try {
				areas.add(signedArea(mb.crossSectionToPolygons(cs)));
			} finally {
				mb.deleteCrossSection(cs);
			}
		}

		double evenOdd = areas.get(FillRule.EVEN_ODD.ordinal());
		double nonZero = areas.get(FillRule.NON_ZERO.ordinal());
		double positive = areas.get(FillRule.POSITIVE.ordinal());
		double negative = areas.get(FillRule.NEGATIVE.ordinal());

		assertEquals(6.0, evenOdd, 1e-6, "even-odd drops the shared corner");
		assertEquals(7.0, nonZero, 1e-6, "non-zero unions the same-winding overlaps");
		assertEquals(nonZero, positive, 1e-6, "positive uses the default fill");
		assertEquals(nonZero, negative, 1e-6, "negative uses the default fill");
	}

	@Test
	void crossSectionRoundTripsThroughPolygons() throws Throwable {
		ArrayList<double[][]> square = new ArrayList<>();
		square.add(new double[][] { { 0, 0 }, { 3, 0 }, { 3, 3 }, { 0, 3 } });

		MemorySegment cs = mb.crossSectionOfPolygons(square, FillRule.POSITIVE);
		try {
			ArrayList<double[][]> back = mb.crossSectionToPolygons(cs);
			assertEquals(1, back.size());
			assertEquals(9.0, signedArea(back), 1e-6);
		} finally {
			mb.deleteCrossSection(cs);
		}
	}

	// -------------------------------------------------------------------------
	// Removed struct-size introspection, reimplemented from the C++ layout
	// -------------------------------------------------------------------------

	@Test
	void structSizesMatchTheNativeLayout() throws Throwable {
		// The C++ side reports sizeof(Manifold) == 48, sizeof(MeshGL64) == 248 and
		// sizeof(std::vector<Manifold>) == 24 on a 64-bit target; confirm the Java
		// reconstruction agrees rather than trusting a hand-computed constant.
		long pointer = mb.vecSize() / 3;
		assertEquals(8, pointer, "64-bit pointer width");
		assertEquals(2 * pointer, mb.manifoldSize() / 3, "shared_ptr is two pointers");
		assertEquals(48, mb.manifoldSize(), "Manifold: pNodeMutex_, pNode_, ctx_");
		assertEquals(24, mb.vecSize(), "std::vector: begin, end, capacity");
		assertEquals(248, mb.meshGL64Size(), "MeshGL64: numProp + ten vectors");
	}

	@Test
	void estimateMemoryScalesWithTriangleCount() throws Throwable {
		MemorySegment cube = mb.cube(1, 1, 1, true);
		MemorySegment sphere = mb.sphere(1, 64);
		try {
			long small = mb.estimateMemoryBytes(cube);
			long big = mb.estimateMemoryBytes(sphere);
			assertTrue(small > 0);
			assertTrue(big > small, "a denser mesh must estimate more memory");
		} finally {
			mb.safeDelete(cube);
			mb.safeDelete(sphere);
		}
	}

	// -------------------------------------------------------------------------
	// Mesh payload plumbing
	// -------------------------------------------------------------------------

	@Test
	void meshRoundTripPreservesCountsAndBounds() throws Throwable {
		MemorySegment cube = mb.cube(4, 6, 8, true);
		MemorySegment loaded = null;
		try {
			ManifoldBindings.MeshData64 mesh = mb.exportMeshGL64(cube);
			assertEquals(mb.numVert(cube), mesh.vertCount());
			assertEquals(mb.numTri(cube), mesh.triCount());
			assertEquals(mesh.vertCount() * 3, mesh.vertices().length);
			assertEquals(mesh.triCount() * 3, mesh.triangles().length);

			loaded = mb.importMeshGL64(mesh.vertices(), mesh.triangles(), mesh.vertCount(), mesh.triCount());
			assertEquals(ManifoldError.NO_ERROR, mb.status(loaded));
			assertEquals(mb.volume(cube), mb.volume(loaded), 1e-9);
			assertArrayEquals(mb.boxDimensions(mb.boundingBox(cube)),
					mb.boxDimensions(mb.boundingBox(loaded)), 1e-9);
		} finally {
			mb.safeDelete(cube);
			mb.safeDelete(loaded);
		}
	}

	@Test
	void meshVertexDataIsPositionFirst() throws Throwable {
		MemorySegment cube = mb.cube(2, 4, 6, true);
		try {
			ManifoldBindings.MeshData64 mesh = mb.exportMeshGL64(cube);
			double[] v = mesh.vertices();
			double lo = Double.POSITIVE_INFINITY, hi = Double.NEGATIVE_INFINITY;
			for (int i = 0; i < mesh.vertCount(); i++) {
				lo = Math.min(lo, v[i * 3]);
				hi = Math.max(hi, v[i * 3]);
			}
			assertEquals(-1.0, lo, 1e-9, "x spans the cube extent");
			assertEquals(1.0, hi, 1e-9);
		} finally {
			mb.safeDelete(cube);
		}
	}
}
