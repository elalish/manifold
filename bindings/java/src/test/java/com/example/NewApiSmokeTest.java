package com.example;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;

import com.cadoodlecad.manifold.ManifoldBindings;
import com.cadoodlecad.manifold.RayHit;

/**
 * Smoke tests for the API surface added on top of the v3.5.5 C bindings:
 * version, winding number, ray cast, min gap, curvature, project, extrude,
 * revolve, and triangulate.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class NewApiSmokeTest {

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

	@Test
	@Order(1)
	void versionReportsThreeComponents() throws Throwable {
		int[] v = mb.version();
		assertEquals(3, v.length);
		assertEquals(3, v[0], "major version");
		assertTrue(v[1] >= 0);
		assertTrue(v[2] >= 0);
	}

	@Test
	@Order(2)
	void windingNumberInsideAndOutside() throws Throwable {
		MemorySegment cube = mb.cube(2, 2, 2, true);
		try {
			assertEquals(0, mb.windingNumber(cube, 5, 0, 0), "point outside the cube");
			assertEquals(1, mb.windingNumber(cube, 0, 0, 0), "point inside the cube");
		} finally {
			mb.safeDelete(cube);
		}
	}

	@Test
	@Order(3)
	void rayCastHitsAndMisses() throws Throwable {
		MemorySegment cube = mb.cube(2, 2, 2, true);
		try {
			ArrayList<RayHit> hits = mb.rayCast(cube, new double[] { 0, 0, 5 },
					new double[] { 0, 0, -5 });
			assertEquals(2, hits.size(), "a ray through a cube hits entry and exit faces");
			assertTrue(hits.get(0).distance() < hits.get(1).distance(), "hits are ordered");
			assertEquals(1.0, hits.get(0).position()[2], 1e-9, "entry at z = +1");
			assertEquals(-1.0, hits.get(1).position()[2], 1e-9, "exit at z = -1");
			assertEquals(0.4, hits.get(0).distance(), 1e-9, "entry is 40% along the segment");

			ArrayList<RayHit> miss = mb.rayCast(cube, new double[] { 5, 5, 5 },
					new double[] { 5, 5, -5 });
			assertTrue(miss.isEmpty(), "a ray outside the cube misses");
		} finally {
			mb.safeDelete(cube);
		}
	}

	@Test
	@Order(4)
	void minGapBetweenDisjointCubes() throws Throwable {
		MemorySegment a = mb.cube(1, 1, 1, true);
		MemorySegment b = mb.translate(mb.cube(1, 1, 1, true), 3, 0, 0);
		try {
			double gap = mb.minGap(a, b, 5.0);
			assertEquals(2.0, gap, 1e-9, "cubes with 1mm half-widths centred 3mm apart");
		} finally {
			mb.safeDelete(a);
			mb.safeDelete(b);
		}
	}

	@Test
	@Order(5)
	void curvaturePreservesGeometry() throws Throwable {
		MemorySegment sphere = mb.sphere(1, 32);
		try {
			long tris = mb.numTri(sphere);
			MemorySegment curved = mb.calculateCurvature(sphere, 0, 1);
			assertEquals(mb.status(curved), ManifoldBindings.ManifoldError.NO_ERROR);
			assertEquals(tris, mb.numTri(curved), "curvature only adds property channels");
			assertEquals(2, mb.numProp(curved), "gaussian + mean channels");
			mb.safeDelete(curved);
		} finally {
			mb.safeDelete(sphere);
		}
	}

	private static ArrayList<double[][]> unitCircle(int n) {
		ArrayList<double[][]> contours = new ArrayList<>();
		double[][] pts = new double[n][2];
		for (int i = 0; i < n; i++) {
			double a = 2 * Math.PI * i / n;
			pts[i][0] = Math.cos(a);
			pts[i][1] = Math.sin(a);
		}
		contours.add(pts);
		return contours;
	}

	@Test
	@Order(6)
	void projectOfSphereIsDisk() throws Throwable {
		MemorySegment sphere = mb.sphere(1, 32);
		try {
			ArrayList<double[][]> contours = mb.project(sphere);
			assertEquals(1, contours.size(), "a sphere projects to one outline");
			double area = 0;
			double[][] c = contours.get(0);
			for (int i = 0; i < c.length; i++) {
				double[] p = c[i];
				double[] q = c[(i + 1) % c.length];
				area += p[0] * q[1] - q[0] * p[1];
			}
			area = Math.abs(area) / 2;
			assertEquals(Math.PI, area, 0.15, "projected disk area ~ pi r^2");
		} finally {
			mb.safeDelete(sphere);
		}
	}

	@Test
	@Order(7)
	void extrudeCylinderVolume() throws Throwable {
		MemorySegment m = mb.extrude(unitCircle(32), 2.0, 0, 0.0, 1.0);
		try {
			assertEquals(mb.status(m), ManifoldBindings.ManifoldError.NO_ERROR);
			assertEquals(Math.PI * 2.0, mb.volume(m), 0.2, "area pi * height 2");
		} finally {
			mb.safeDelete(m);
		}
	}

	@Test
	@Order(8)
	void revolveTorusGenusOne() throws Throwable {
		// A circle of radius 0.5 centred at x = 2 revolved 360° makes a torus.
		ArrayList<double[][]> contours = new ArrayList<>();
		int n = 32;
		double[][] pts = new double[n][2];
		for (int i = 0; i < n; i++) {
			double a = 2 * Math.PI * i / n;
			pts[i][0] = 2.0 + 0.5 * Math.cos(a);
			pts[i][1] = 0.5 * Math.sin(a);
		}
		contours.add(pts);
		MemorySegment m = mb.revolve(contours, 0, 360.0);
		try {
			assertEquals(mb.status(m), ManifoldBindings.ManifoldError.NO_ERROR);
			assertEquals(1, mb.genus(m), "a torus has genus 1");
		} finally {
			mb.safeDelete(m);
		}
	}

	@Test
	@Order(9)
	void triangulateSquare() throws Throwable {
		ArrayList<double[][]> contours = new ArrayList<>();
		contours.add(new double[][] { { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } });
		int[][] tris = mb.triangulate(contours, 1e-9);
		assertEquals(2, tris.length, "a square triangulates into 2 triangles");
		for (int[] t : tris) {
			assertEquals(3, t.length);
			for (int idx : t)
				assertTrue(idx >= 0 && idx < 4, "indices reference the 4 input points");
		}
	}
}
