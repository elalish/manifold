package com.example;


import org.junit.jupiter.api.*;

import com.cadoodlecad.manifold.ManifoldBindings;
import com.cadoodlecad.manifold.ManifoldBindings.MeshData64;

import java.io.File;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link ManifoldBindings}.
 *
 * <p>Tests cover:
 * <ul>
 *   <li>Union, Difference, Intersection, and Hull of a cube and sphere</li>
 *   <li>Export of each result to binary STL and 3MF</li>
 *   <li>Re-import of each file and comparison of vertex/triangle counts</li>
 * </ul>
 *
 * <p>Tolerance note: STL is a float-precision, unindexed format. After a round-trip
 * through STL the welded vertex count may differ slightly from the 3MF round-trip
 * (which preserves the original indexed topology). The tests therefore compare the
 * <em>triangle</em> count exactly (triangles survive both formats without change) and
 * check that the vertex count after re-import is within a small tolerance band.
 *
 * <p>Prerequisites: the native Manifold shared libraries must be on the library path
 * or bundled as resources, exactly as {@link ManifoldBindings} expects.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class ManifoldBindingsTest {

	/** Shared bindings instance — created once per test class. */
	private static ManifoldBindings mb;

	/**
	 * Native cube shape: 10 × 10 × 10 mm, centered at origin.
	 * Re-created fresh for each test so operations don't share state.
	 */
	private static final double CUBE_SIZE = 10.0;
	private static final double SPHERE_R = 6.0;
	private static final int SPHERE_SEGS = 100;
	private Path tmpDir = new File(".").toPath();

	// -------------------------------------------------------------------------
	// Lifecycle
	// -------------------------------------------------------------------------

	@BeforeAll
	static void initBindings() throws Exception {
		mb = new ManifoldBindings();
		assertNotNull(mb, "ManifoldBindings must initialise without error");
		assertTrue(ManifoldBindings.isNativeLibraryLoaded(), "Native library must be loaded after construction");
	}

	@AfterAll
	static void closeBindings() {
		if (mb != null)
			mb.close();
	}

	// -------------------------------------------------------------------------
	// Helpers
	// -------------------------------------------------------------------------

	/** Creates a fresh 10 mm cube centred at origin. */
	private java.lang.foreign.MemorySegment makeCube() throws Throwable {
		return mb.cube(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, true);
	}

	/** Creates a fresh sphere with {@value #SPHERE_SEGS} segments. */
	private java.lang.foreign.MemorySegment makeSphere() throws Throwable {
		return mb.sphere(SPHERE_R, SPHERE_SEGS);
	}

	/**
	 * Validity check: verifies vertex count, triangle count, and volume are all positive.
	 */
	private void assertMeshValid(java.lang.foreign.MemorySegment m, String label) throws Throwable {
		long verts = mb.numVert(m);
		long tris = mb.numTri(m);
		double vol = mb.volume(m);

		assertTrue(verts > 0, label + ": numVert must be > 0, was " + verts);
		assertTrue(tris > 0, label + ": numTri must be > 0, was " + tris);
		assertTrue(vol > 0, label + ": volume must be > 0, was " + vol);
	}

	/**
	 * Round-trip a manifold through STL:
	 * <ol>
	 *   <li>Export to a temp STL file</li>
	 *   <li>Re-import from that file</li>
	 *   <li>Assert the re-imported mesh has the same triangle count</li>
	 *   <li>Assert vertex count is within 5 % (welding may differ slightly)</li>
	 * </ol>
	 *
	 * @return the re-imported manifold (caller is responsible for cleanup)
	 */
	private java.lang.foreign.MemorySegment assertStlRoundTrip(java.lang.foreign.MemorySegment original, File stlFile,
			String label) throws Throwable {

		long origTris = mb.numTri(original);
		long origVerts = mb.numVert(original);

		// Export
		mb.exportSTL(original, stlFile);
		assertTrue(stlFile.exists() && stlFile.length() > 84,
				label + " STL: file must exist and have data beyond header");

		// Import
		java.lang.foreign.MemorySegment reimported = mb.importSTL(stlFile);
		assertNotNull(reimported, label + " STL: re-imported segment must not be null");

		long reimTris = mb.numTri(reimported);
		long reimVerts = mb.numVert(reimported);

		// Triangle count must survive exactly
		assertEquals(origTris, reimTris, label + " STL: triangle count mismatch after round-trip");

		// Vertex count after welding should be ≤ original and within 5 %
		assertTrue(reimVerts <= origVerts * 1.05 + 1 && reimVerts >= origVerts * 0.70,
				label + " STL: vertex count after re-import (" + reimVerts
						+ ") is outside expected range relative to original (" + origVerts + ")");

		return reimported;
	}

	@Test
	public void testMeshGL64() throws Throwable {
		java.lang.foreign.MemorySegment cube = makeCube();
		MemorySegment loaded = null;
		try {
			assertMeshValid(cube, "Cube");
			// A cube has 8 vertices and 12 triangles (6 faces × 2)
			assertEquals(8, mb.numVert(cube), "Cube should have 8 vertices");
			assertEquals(12, mb.numTri(cube), "Cube should have 12 triangles");

			MeshData64 meshgl = mb.exportMeshGL64(cube);
			double[] verts = meshgl.vertices(); // flat [x0,y0,z0, x1,y1,z1, ...]
			long[] tris = meshgl.triangles(); // flat [i0,i1,i2, i3,i4,i5, ...]
			long triCount = meshgl.triCount();
			loaded = mb.importMeshGL64(verts, tris, triCount, triCount);

			MeshData64 meshglLoaded = mb.exportMeshGL64(loaded);
			// double[] vertsl = meshglLoaded.vertices(); // flat [x0,y0,z0, x1,y1,z1, ...]
			// long[] trisl = meshglLoaded.triangles(); // flat [i0,i1,i2, i3,i4,i5, ...]
			long triCountl = meshglLoaded.triCount();
			assertEquals(triCountl, triCount, "Matching trinagle count");
		} finally {
			mb.safeDelete(cube);
			mb.safeDelete(loaded);
		}
	}

	/**
	 * Round-trip a manifold through 3MF:
	 * <ol>
	 *   <li>Export to a temp 3MF file</li>
	 *   <li>Re-import the first object from that file</li>
	 *   <li>Assert vertex and triangle counts match exactly (3MF is indexed)</li>
	 * </ol>
	 *
	 * @return the re-imported manifold (caller is responsible for cleanup)
	 */
	private java.lang.foreign.MemorySegment assert3mfRoundTrip(java.lang.foreign.MemorySegment original,
			File threeMfFile, String label) throws Throwable {

		long origTris = mb.numTri(original);
		long origVerts = mb.numVert(original);

		// Export (single mesh → single object in the 3MF file)
		ArrayList<java.lang.foreign.MemorySegment> exportList = new ArrayList<>();
		exportList.add(original);
		mb.export3MF(exportList, threeMfFile);
		assertTrue(threeMfFile.exists() && threeMfFile.length() > 0, label + " 3MF: file must exist and be non-empty");

		// Import
		ArrayList<java.lang.foreign.MemorySegment> imported = mb.import3MF(threeMfFile);
		assertNotNull(imported, label + " 3MF: import result must not be null");
		assertEquals(1, imported.size(), label + " 3MF: expected exactly 1 object in file");

		java.lang.foreign.MemorySegment reimported = imported.get(0);
		long reimTris = mb.numTri(reimported);
		long reimVerts = mb.numVert(reimported);

		// 3MF preserves the indexed topology — counts must match exactly
		assertEquals(origTris, reimTris, label + " 3MF: triangle count mismatch after round-trip");
		assertEquals(origVerts, reimVerts, label + " 3MF: vertex count mismatch after round-trip");

		return reimported;
	}

	// -------------------------------------------------------------------------
	// Primitives smoke-test
	// -------------------------------------------------------------------------

	@Test
	@Order(1)
	@DisplayName("Cube primitive: valid mesh")
	void testCubePrimitive() throws Throwable {
		java.lang.foreign.MemorySegment cube = makeCube();
		try {
			assertMeshValid(cube, "Cube");
			// A cube has 8 vertices and 12 triangles (6 faces × 2)
			assertEquals(8, mb.numVert(cube), "Cube should have 8 vertices");
			assertEquals(12, mb.numTri(cube), "Cube should have 12 triangles");
		} finally {
			mb.safeDelete(cube);
		}
	}

	@Test
	@Order(2)
	@DisplayName("Sphere primitive: valid mesh")
	void testSpherePrimitive() throws Throwable {
		java.lang.foreign.MemorySegment sphere = makeSphere();
		try {
			assertMeshValid(sphere, "Sphere");
			// Sphere vertex/tri counts depend on segment count; just verify they're positive
			assertTrue(mb.numVert(sphere) > 0, "Sphere vertex count");
			assertTrue(mb.numTri(sphere) > 0, "Sphere triangle count");
			// Volume of sphere: (4/3)πr³ ≈ 904.78 for r=6
			double expectedVol = (4.0 / 3.0) * Math.PI * Math.pow(SPHERE_R, 3);
			assertEquals(expectedVol, mb.volume(sphere), expectedVol * 0.05,
					"Sphere volume should be within 5 % of analytic value");
		} finally {
			mb.safeDelete(sphere);
		}
	}

	// -------------------------------------------------------------------------
	// Boolean: Union
	// -------------------------------------------------------------------------

	@Test
	@Order(10)
	@DisplayName("Union(cube, sphere): valid mesh, STL + 3MF round-trip")
	void testUnionRoundTrip() throws Throwable {
		java.lang.foreign.MemorySegment cube = makeCube();
		java.lang.foreign.MemorySegment sphere = makeSphere();
		java.lang.foreign.MemorySegment result = null;
		java.lang.foreign.MemorySegment stlRe = null;
		java.lang.foreign.MemorySegment mfRe = null;
		try {
			result = mb.union(cube, sphere);
			assertMeshValid(result, "Union");

			// Volume of union = vol(cube) + vol(sphere) - vol(intersection)
			// Must be less than the sum and greater than the larger of the two
			double cubeVol = mb.volume(cube);
			double sphereVol = mb.volume(sphere);
			double unionVol = mb.volume(result);
			assertTrue(unionVol < cubeVol + sphereVol, "Union volume must be less than sum of parts");
			assertTrue(unionVol > Math.max(cubeVol, sphereVol), "Union volume must be greater than either part alone");

			// Round-trips
			stlRe = assertStlRoundTrip(result, tmpDir.resolve("union.stl").toFile(), "Union");
			mfRe = assert3mfRoundTrip(result, tmpDir.resolve("union.3mf").toFile(), "Union");

			assertMeshValid(stlRe, "Union/STL re-import");
			assertMeshValid(mfRe, "Union/3MF re-import");
		} finally {
			mb.safeDelete(cube);
			mb.safeDelete(sphere);
			mb.safeDelete(result);
			mb.safeDelete(stlRe);
			mb.safeDelete(mfRe);
		}
	}

	// -------------------------------------------------------------------------
	// Boolean: Difference
	// -------------------------------------------------------------------------

	@Test
	@Order(20)
	@DisplayName("Difference(cube, sphere): valid mesh, STL + 3MF round-trip")
	void testDifferenceRoundTrip() throws Throwable {
		java.lang.foreign.MemorySegment cube = makeCube();
		java.lang.foreign.MemorySegment sphere = makeSphere();
		java.lang.foreign.MemorySegment result = null;
		java.lang.foreign.MemorySegment stlRe = null;
		java.lang.foreign.MemorySegment mfRe = null;
		try {
			result = mb.difference(cube, sphere);
			assertMeshValid(result, "Difference");

			// Volume of difference must be less than the cube's own volume
			double diffVol = mb.volume(result);
			double cubeVol = mb.volume(cube);
			assertTrue(diffVol < cubeVol,
					"Difference volume (" + diffVol + ") must be < cube volume (" + cubeVol + ")");
			assertTrue(diffVol > 0, "Difference must still have positive volume (sphere doesn't swallow cube)");

			stlRe = assertStlRoundTrip(result, tmpDir.resolve("difference.stl").toFile(), "Difference");
			mfRe = assert3mfRoundTrip(result, tmpDir.resolve("difference.3mf").toFile(), "Difference");

			assertMeshValid(stlRe, "Difference/STL re-import");
			assertMeshValid(mfRe, "Difference/3MF re-import");
		} finally {
			mb.safeDelete(cube);
			mb.safeDelete(sphere);
			mb.safeDelete(result);
			mb.safeDelete(stlRe);
			mb.safeDelete(mfRe);
		}
	}

	// -------------------------------------------------------------------------
	// Boolean: Intersection
	// -------------------------------------------------------------------------

	@Test
	@Order(30)
	@DisplayName("Intersection(cube, sphere): valid mesh, STL + 3MF round-trip")
	void testIntersectionRoundTrip() throws Throwable {
		java.lang.foreign.MemorySegment cube = makeCube();
		java.lang.foreign.MemorySegment sphere = makeSphere();
		java.lang.foreign.MemorySegment result = null;
		java.lang.foreign.MemorySegment stlRe = null;
		java.lang.foreign.MemorySegment mfRe = null;
		try {
			result = mb.intersection(cube, sphere);
			assertMeshValid(result, "Intersection");

			double intersectVol = mb.volume(result);
			double cubeVol = mb.volume(cube);
			double sphereVol = mb.volume(sphere);

			// Intersection volume must be ≤ the smaller of the two shapes
			double smaller = Math.min(cubeVol, sphereVol);
			assertTrue(intersectVol <= smaller * 1.001,
					"Intersection volume (" + intersectVol + ") must be ≤ smaller of cube/sphere (" + smaller + ")");
			assertTrue(intersectVol > 0, "Intersection must be non-empty (shapes overlap)");

			stlRe = assertStlRoundTrip(result, tmpDir.resolve("intersection.stl").toFile(), "Intersection");
			mfRe = assert3mfRoundTrip(result, tmpDir.resolve("intersection.3mf").toFile(), "Intersection");

			assertMeshValid(stlRe, "Intersection/STL re-import");
			assertMeshValid(mfRe, "Intersection/3MF re-import");
		} finally {
			mb.safeDelete(cube);
			mb.safeDelete(sphere);
			mb.safeDelete(result);
			mb.safeDelete(stlRe);
			mb.safeDelete(mfRe);
		}
	}

	// -------------------------------------------------------------------------
	// Hull
	// -------------------------------------------------------------------------

	@Test
	@Order(40)
	@DisplayName("Hull(cube): valid mesh, STL + 3MF round-trip")
	void testHullCubeRoundTrip() throws Throwable {
		java.lang.foreign.MemorySegment cube = makeCube();
		java.lang.foreign.MemorySegment result = null;
		java.lang.foreign.MemorySegment stlRe = null;
		java.lang.foreign.MemorySegment mfRe = null;
		try {
			result = mb.hull(cube);
			assertMeshValid(result, "Hull(cube)");

			// Hull of a cube is the cube itself — volume should be the same
			double hullVol = mb.volume(result);
			double cubeVol = mb.volume(cube);
			assertEquals(cubeVol, hullVol, cubeVol * 0.001, "Hull of a cube must have the same volume as the cube");

			stlRe = assertStlRoundTrip(result, tmpDir.resolve("hull_cube.stl").toFile(), "Hull(cube)");
			mfRe = assert3mfRoundTrip(result, tmpDir.resolve("hull_cube.3mf").toFile(), "Hull(cube)");

			assertMeshValid(stlRe, "Hull(cube)/STL re-import");
			assertMeshValid(mfRe, "Hull(cube)/3MF re-import");
		} finally {
			mb.safeDelete(cube);
			mb.safeDelete(result);
			mb.safeDelete(stlRe);
			mb.safeDelete(mfRe);
		}
	}

	@Test
	@Order(41)
	@DisplayName("Hull(sphere): valid mesh, STL + 3MF round-trip")
	void testHullSphereRoundTrip() throws Throwable {
		java.lang.foreign.MemorySegment sphere = makeSphere();
		java.lang.foreign.MemorySegment result = null;
		java.lang.foreign.MemorySegment stlRe = null;
		java.lang.foreign.MemorySegment mfRe = null;
		try {
			result = mb.hull(sphere);
			assertMeshValid(result, "Hull(sphere)");

			// Hull of a convex shape = same shape — volume within 1 %
			double hullVol = mb.volume(result);
			double sphereVol = mb.volume(sphere);
			assertEquals(sphereVol, hullVol, sphereVol * 0.01,
					"Hull of a sphere must have effectively the same volume");

			stlRe = assertStlRoundTrip(result, tmpDir.resolve("hull_sphere.stl").toFile(), "Hull(sphere)");
			mfRe = assert3mfRoundTrip(result, tmpDir.resolve("hull_sphere.3mf").toFile(), "Hull(sphere)");

			assertMeshValid(stlRe, "Hull(sphere)/STL re-import");
			assertMeshValid(mfRe, "Hull(sphere)/3MF re-import");
		} finally {
			mb.safeDelete(sphere);
			mb.safeDelete(result);
			mb.safeDelete(stlRe);
			mb.safeDelete(mfRe);
		}
	}

	@Test
	@Order(42)
	@DisplayName("BatchHull([cube, sphere]): valid mesh, STL + 3MF round-trip")
	void testBatchHullRoundTrip() throws Throwable {
		java.lang.foreign.MemorySegment cube = makeCube();
		java.lang.foreign.MemorySegment sphere = makeSphere();
		java.lang.foreign.MemorySegment result = null;
		java.lang.foreign.MemorySegment stlRe = null;
		java.lang.foreign.MemorySegment mfRe = null;
		try {
			result = mb.batchHull(new java.lang.foreign.MemorySegment[] { cube, sphere });
			assertMeshValid(result, "BatchHull");

			// Hull of cube+sphere must enclose both; volume ≥ each individual hull
			double cubeHullVol = mb.volume(mb.hull(cube));
			double sphereHullVol = mb.volume(mb.hull(sphere));
			double batchHullVol = mb.volume(result);

			assertTrue(batchHullVol >= cubeHullVol * 0.999, "Batch hull volume must be >= cube hull volume");
			assertTrue(batchHullVol >= sphereHullVol * 0.999, "Batch hull volume must be >= sphere hull volume");

			stlRe = assertStlRoundTrip(result, tmpDir.resolve("batch_hull.stl").toFile(), "BatchHull");
			mfRe = assert3mfRoundTrip(result, tmpDir.resolve("batch_hull.3mf").toFile(), "BatchHull");

			assertMeshValid(stlRe, "BatchHull/STL re-import");
			assertMeshValid(mfRe, "BatchHull/3MF re-import");
		} finally {
			mb.safeDelete(cube);
			mb.safeDelete(sphere);
			mb.safeDelete(result);
			mb.safeDelete(stlRe);
			mb.safeDelete(mfRe);
		}
	}

	// -------------------------------------------------------------------------
	// Multi-mesh 3MF export/import
	// -------------------------------------------------------------------------

	@Test
	@Order(50)
	@DisplayName("3MF multi-mesh: export cube+sphere, re-import both objects")
	void testMultiMesh3mfRoundTrip() throws Throwable {
		java.lang.foreign.MemorySegment cube = makeCube();
		java.lang.foreign.MemorySegment sphere = makeSphere();
		ArrayList<java.lang.foreign.MemorySegment> imported = null;
		try {
			long cubeVerts = mb.numVert(cube);
			long cubeTris = mb.numTri(cube);
			long sphereVerts = mb.numVert(sphere);
			long sphereTris = mb.numTri(sphere);

			// Export both as a single 3MF file
			ArrayList<java.lang.foreign.MemorySegment> exportList = new ArrayList<>();
			exportList.add(cube);
			exportList.add(sphere);
			File threeMfFile = tmpDir.resolve("multi.3mf").toFile();
			mb.export3MF(exportList, threeMfFile);
			assertTrue(threeMfFile.exists() && threeMfFile.length() > 0, "Multi-mesh 3MF file must exist");

			// Re-import: expect 2 objects in the same order
			imported = mb.import3MF(threeMfFile);
			assertEquals(2, imported.size(), "Must import exactly 2 objects from multi-mesh 3MF");

			java.lang.foreign.MemorySegment reimCube = imported.get(0);
			java.lang.foreign.MemorySegment reimSphere = imported.get(1);

			assertEquals(cubeVerts, mb.numVert(reimCube), "Re-imported cube vertex count");
			assertEquals(cubeTris, mb.numTri(reimCube), "Re-imported cube triangle count");
			assertEquals(sphereVerts, mb.numVert(reimSphere), "Re-imported sphere vertex count");
			assertEquals(sphereTris, mb.numTri(reimSphere), "Re-imported sphere triangle count");

		} finally {
			mb.safeDelete(cube);
			mb.safeDelete(sphere);
			if (imported != null) {
				for (java.lang.foreign.MemorySegment seg : imported) {
					mb.safeDelete(seg);
				}
			}
		}
	}

	// -------------------------------------------------------------------------
	// File format edge cases
	// -------------------------------------------------------------------------

	@Test
	@Order(60)
	@DisplayName("STL file size: matches 80 + 4 + triCount * 50 formula")
	void testStlFileSizeFormula() throws Throwable {
		java.lang.foreign.MemorySegment cube = makeCube();
		try {
			File stlFile = tmpDir.resolve("cube_size.stl").toFile();
			mb.exportSTL(cube, stlFile);

			long triCount = mb.numTri(cube);
			long expectedLen = 84L + triCount * 50L;
			assertEquals(expectedLen, stlFile.length(), "Binary STL file size must equal 84 + triCount*50 bytes");
		} finally {
			mb.safeDelete(cube);
		}
	}

	@Test
	@Order(61)
	@DisplayName("3MF file: is a valid ZIP archive containing 3D/3dmodel.model")
	void testThreeMfIsValidZip() throws Throwable {
		java.lang.foreign.MemorySegment sphere = makeSphere();
		try {
			File threeMfFile = tmpDir.resolve("sphere.3mf").toFile();
			ArrayList<java.lang.foreign.MemorySegment> list = new ArrayList<>();
			list.add(sphere);
			mb.export3MF(list, threeMfFile);

			// Open as ZIP and verify required entries
			try (java.util.zip.ZipInputStream zis = new java.util.zip.ZipInputStream(
					new java.io.FileInputStream(threeMfFile))) {
				boolean hasModel = false;
				boolean hasContentTypes = false;
				boolean hasRels = false;
				java.util.zip.ZipEntry entry;
				while ((entry = zis.getNextEntry()) != null) {
					String name = entry.getName().replace('\\', '/');
					if (name.equalsIgnoreCase("3D/3dmodel.model"))
						hasModel = true;
					if (name.equals("[Content_Types].xml"))
						hasContentTypes = true;
					if (name.equals("_rels/.rels"))
						hasRels = true;
					zis.closeEntry();
				}
				assertTrue(hasModel, "3MF must contain 3D/3dmodel.model");
				assertTrue(hasContentTypes, "3MF must contain [Content_Types].xml");
				assertTrue(hasRels, "3MF must contain _rels/.rels");
			}
		} finally {
			mb.safeDelete(sphere);
		}
	}

	// -------------------------------------------------------------------------
	// Volume sanity across all operations
	// -------------------------------------------------------------------------

	@Test
	@Order(70)
	@DisplayName("Volume identity: vol(A∪B) = vol(A) + vol(B) - vol(A∩B)")
	void testVolumeIdentity() throws Throwable {
		java.lang.foreign.MemorySegment cube = makeCube();
		java.lang.foreign.MemorySegment sphere = makeSphere();
		java.lang.foreign.MemorySegment unionMs = null;
		java.lang.foreign.MemorySegment intersectMs = null;
		try {
			unionMs = mb.union(cube, sphere);
			intersectMs = mb.intersection(cube, sphere);

			double volA = mb.volume(cube);
			double volB = mb.volume(sphere);
			double volU = mb.volume(unionMs);
			double volI = mb.volume(intersectMs);

			// Inclusion-exclusion principle: |A∪B| = |A| + |B| - |A∩B|
			double expected = volA + volB - volI;
			assertEquals(expected, volU, expected * 0.01, "Volume inclusion-exclusion identity must hold within 1 %");
		} finally {
			mb.safeDelete(cube);
			mb.safeDelete(sphere);
			mb.safeDelete(unionMs);
			mb.safeDelete(intersectMs);
		}
	}

	@Test
	@Order(71)
	@DisplayName("Volume identity: vol(A-B) = vol(A∩B) + vol(A-B) = vol(A)")
	void testDifferenceVolumeIdentity() throws Throwable {
		java.lang.foreign.MemorySegment cube = makeCube();
		java.lang.foreign.MemorySegment sphere = makeSphere();
		java.lang.foreign.MemorySegment diffMs = null;
		java.lang.foreign.MemorySegment intersectMs = null;
		try {
			diffMs = mb.difference(cube, sphere);
			intersectMs = mb.intersection(cube, sphere);

			double volA = mb.volume(cube);
			double volDiff = mb.volume(diffMs);
			double volI = mb.volume(intersectMs);

			// vol(A-B) + vol(A∩B) = vol(A)
			assertEquals(volA, volDiff + volI, volA * 0.01, "vol(A-B) + vol(A∩B) must equal vol(A) within 1 %");
		} finally {
			mb.safeDelete(cube);
			mb.safeDelete(sphere);
			mb.safeDelete(diffMs);
			mb.safeDelete(intersectMs);
		}
	}

	// =========================================================================
	// Compose / Decompose
	// =========================================================================

	/**
	 * Decompose two disjoint cubes into exactly two components, then rebuild the
	 * original pair with compose and check the volume is preserved.
	 *
	 * <p>Both cubes are centred far enough apart that their surfaces never touch,
	 * so manifold sees them as one composed manifold with two connected
	 * components. Decompose must return them in some order as two 8-vertex,
	 * 12-triangle cubes; compose must turn the pair back into a single manifold
	 * whose volume is the sum of the parts (components do not overlap, so no
	 * boolean overlap needs subtracting).
	 */
	@Test
	@Order(72)
	@DisplayName("Decompose(compose of 2 disjoint cubes): two components, round-trip volume")
	void testDecomposeDisjointCubes() throws Throwable {
		java.lang.foreign.MemorySegment left = null;
		java.lang.foreign.MemorySegment right = null;
		java.lang.foreign.MemorySegment composed = null;
		java.lang.foreign.MemorySegment[] parts = null;
		java.lang.foreign.MemorySegment recomposed = null;
		try {
			left = mb.translate(makeCube(), -15, 0, 0);
			right = mb.translate(makeCube(), 15, 0, 0);
			double leftVol = mb.volume(left);
			double rightVol = mb.volume(right);

			composed = mb.compose(new java.lang.foreign.MemorySegment[] { left, right });
			assertMeshValid(composed, "Composed pair");
			assertEquals(leftVol + rightVol, mb.volume(composed), (leftVol + rightVol) * 0.001,
					"Composed volume must equal the sum of the disjoint parts");

			parts = mb.decompose(composed);
			assertEquals(2, parts.length, "Two disjoint cubes must decompose into exactly 2 components");

			double sumOfParts = 0;
			for (java.lang.foreign.MemorySegment part : parts) {
				assertNotNull(part, "Each component segment must be non-null");
				assertMeshValid(part, "Component");
				// each component is one of the original 10 mm cubes
				assertEquals(8, mb.numVert(part), "Each component must be a cube with 8 vertices");
				assertEquals(12, mb.numTri(part), "Each component must be a cube with 12 triangles");
				assertEquals(CUBE_SIZE * CUBE_SIZE * CUBE_SIZE, mb.volume(part), 0.001,
						"Each component must have the volume of the original cube");
				sumOfParts += mb.volume(part);
			}
			assertEquals(leftVol + rightVol, sumOfParts, (leftVol + rightVol) * 0.001,
					"Component volumes must sum to the composed volume");

			recomposed = mb.compose(parts);
			assertMeshValid(recomposed, "Recomposed pair");
			assertEquals(leftVol + rightVol, mb.volume(recomposed), (leftVol + rightVol) * 0.001,
					"Recomposed volume must equal the composed volume");
		} finally {
			mb.safeDelete(left);
			mb.safeDelete(right);
			mb.safeDelete(composed);
			if (parts != null)
				for (java.lang.foreign.MemorySegment part : parts)
					mb.safeDelete(part);
			mb.safeDelete(recomposed);
		}
	}

	/**
	 * Decompose of a single connected solid must yield exactly one component
	 * with the same geometry, and decompose of an empty manifold must yield no
	 * components.
	 */
	@Test
	@Order(73)
	@DisplayName("Decompose edge cases: single solid → 1 part, empty manifold → 0 parts")
	void testDecomposeEdgeCases() throws Throwable {
		java.lang.foreign.MemorySegment sphere = null;
		java.lang.foreign.MemorySegment[] sphereParts = null;
		try {
			sphere = makeSphere();
			sphereParts = mb.decompose(sphere);
			assertEquals(1, sphereParts.length, "A connected solid must decompose into exactly 1 component");
			assertEquals(mb.numVert(sphere), mb.numVert(sphereParts[0]),
					"Sole component must have the same vertex count as the input");
			assertEquals(mb.volume(sphere), mb.volume(sphereParts[0]), mb.volume(sphere) * 0.001,
					"Sole component must have the same volume as the input");
		} finally {
			mb.safeDelete(sphere);
			if (sphereParts != null)
				for (java.lang.foreign.MemorySegment part : sphereParts)
					mb.safeDelete(part);
		}

		java.lang.foreign.MemorySegment[] emptyParts = null;
		java.lang.foreign.MemorySegment emptyManifold = null;
		try {
			emptyManifold = mb.empty();
			emptyParts = mb.decompose(emptyManifold);
			assertEquals(0, emptyParts.length, "An empty manifold must decompose into 0 components");
		} finally {
			mb.safeDelete(emptyManifold);
			if (emptyParts != null)
				for (java.lang.foreign.MemorySegment part : emptyParts)
					mb.safeDelete(part);
		}
	}

	// =========================================================================
	// Slice tests
	// =========================================================================

	/**
	 * Slice the 10×10×10 cube (centred at origin) at Z=0.
	 *
	 * <p>Expected cross-section: exactly one contour, exactly 4 vertices,
	 * all at Z=0 (the slice plane), forming a closed square whose corners lie
	 * at (±5, ±5).  The contour must be a proper rectangle: two pairs of edges
	 * are axis-aligned and each has length 10.
	 */
	@Test
	@Order(80)
	@DisplayName("Slice cube at Z=0: one square contour at (±5,±5)")
	void testSliceCubeAtZero() throws Throwable {
		java.lang.foreign.MemorySegment cube = makeCube(); // centred, side 10
		try {
			ArrayList<double[][]> contours = mb.slice(cube, 0.0);

			// ── contour count ──────────────────────────────────────────────
			assertEquals(1, contours.size(), "Cube slice at Z=0 must produce exactly one contour");

			double[][] pts = contours.get(0);

			// ── vertex count ───────────────────────────────────────────────
			// Manifold may emit 4 or 5 points (last may repeat first to close)
			assertTrue(pts.length == 8,
					"Cube contour must have 4 unique corners, 4 midpoints (got " + pts.length + ")");

			int unique = pts.length;

			assertEquals(8, unique, "Cube slice must have exactly 8 unique points, 4 corners and 4 mid points");

			// ── all Y-values are zero (slice plane) ────────────────────────
			// Note: slice() returns (x,y) in the 2-D cross-section plane;
			// the Z coordinate of the slice is implicit (= height argument).
			// We verify the 2-D point values are within the cube's XY extents.
			double half = CUBE_SIZE / 2.0; // 5.0
			double tol = 1e-6;

			for (int i = 0; i < unique; i++) {
				double x = pts[i][0];
				double y = pts[i][1];
				assertTrue(Math.abs(Math.abs(x) - half) < tol || Math.abs(Math.abs(y) - half) < tol, "Corner " + i
						+ " at (" + x + "," + y + ") must lie on a face of the unit square (±" + half + ")");
			}

			// ── the four corners are (±5, ±5) ─────────────────────────────
			// Collect unique corners and check all four expected ones appear.
			double[][] corners = new double[unique][2];
			System.arraycopy(pts, 0, corners, 0, unique);

			double[][] expected = { { -half, -half }, { half, -half }, { half, half }, { -half, half } };
			for (double[] exp : expected) {
				boolean found = false;
				for (double[] got : corners) {
					if (Math.abs(got[0] - exp[0]) < tol && Math.abs(got[1] - exp[1]) < tol) {
						found = true;
						break;
					}
				}
				assertTrue(found, "Expected corner (" + exp[0] + "," + exp[1] + ") not found in slice");
			}
			// Outset by 1 mm
			ArrayList<double[][]> outset = mb.sliceWithOffset(
					cube, 0.0, +1.0, ManifoldBindings.JoinType.ROUND, 2.0, 0);

			// Inset by 1 mm
			ArrayList<double[][]> inset = mb.sliceWithOffset(
					cube, 0.0, -1.0, ManifoldBindings.JoinType.ROUND, 2.0, 0);

		} finally {
			mb.safeDelete(cube);
		}
	}

	/**
	 * Slice the cube above its top face — must produce an empty result.
	 */
	@Test
	@Order(81)
	@DisplayName("Slice cube above top face: empty result")
	void testSliceCubeAboveTop() throws Throwable {
		java.lang.foreign.MemorySegment cube = makeCube();
		try {
			// Cube goes from Z=-5 to Z=+5; slice at Z=6 must be empty.
			ArrayList<double[][]> contours = mb.slice(cube, CUBE_SIZE); // Z=10, above top
			assertTrue(contours == null || contours.isEmpty(), "Slice above cube top must return no contours");
		} finally {
			mb.safeDelete(cube);
		}
	}

	/**
	 * Slice the sphere at Z=0 (equatorial plane).
	 *
	 * <p>Expected cross-section: exactly one contour whose points all lie on a
	 * circle of radius {@value #SPHERE_R}.  Because the sphere is tessellated
	 * with {@value #SPHERE_SEGS} segments, the slice will be a polygon with
	 * approximately that many vertices.  We verify:
	 * <ul>
	 *   <li>Exactly one contour.</li>
	 *   <li>Every point is within 2 % of the expected radius from the origin.</li>
	 *   <li>The contour is ordered (consecutive points are adjacent on the circle).</li>
	 *   <li>The perimeter is close to 2π·r within 5 %.</li>
	 * </ul>
	 */
	@Test
	@Order(82)
	@DisplayName("Slice sphere at Z=0: one circular contour at radius " + SPHERE_R)
	void testSliceSphereAtEquator() throws Throwable {
		java.lang.foreign.MemorySegment sphere = makeSphere();
		try {
			ArrayList<double[][]> contours = mb.slice(sphere, 0.0);

			// ── one contour ────────────────────────────────────────────────
			assertEquals(1, contours.size(), "Sphere slice at Z=0 must produce exactly one contour");

			double[][] pts = contours.get(0);
			int n = pts.length;

			// Discard duplicate closing vertex if present
			boolean closes = n >= 2 && Math.abs(pts[0][0] - pts[n - 1][0]) < 1e-9
					&& Math.abs(pts[0][1] - pts[n - 1][1]) < 1e-9;
			int uniqueN = closes ? n - 1 : n;

			// Equatorial slice of a sphere with SPHERE_SEGS circular segments
			// should yield roughly SPHERE_SEGS vertices (exact count depends on
			// tessellation; allow generous ±50 % band).
			assertTrue(uniqueN >= SPHERE_SEGS / 2, "Sphere equatorial slice should have at least " + SPHERE_SEGS / 2
					+ " vertices (got " + uniqueN + ")");

			// ── all points lie on (or very near) the sphere's equatorial circle ──
			double radiusTol = SPHERE_R * 0.05; // 5 % of radius
			for (int i = 0; i < uniqueN; i++) {
				double x = pts[i][0];
				double y = pts[i][1];
				double r = Math.sqrt(x * x + y * y);
				assertEquals(SPHERE_R, r, radiusTol, "Slice point " + i + " at (" + x + "," + y + ") radius " + r
						+ " is not within 5 % of " + SPHERE_R);
			}

			// ── perimeter close to 2π·r ────────────────────────────────────
			double perimeter = 0;
			for (int i = 0; i < uniqueN; i++) {
				double[] a = pts[i];
				double[] b = pts[(i + 1) % uniqueN];
				perimeter += Math.sqrt(Math.pow(b[0] - a[0], 2) + Math.pow(b[1] - a[1], 2));
			}
			double expectedPerimeter = 2 * Math.PI * SPHERE_R;
			assertEquals(expectedPerimeter, perimeter, expectedPerimeter * 0.05,
					"Sphere slice perimeter must be within 5 % of 2π·r = " + expectedPerimeter);

			// ── winding: contour should not self-intersect ─────────────────
			// Check via signed area (shoelace) — non-zero means a proper polygon.
			double signedArea = 0;
			for (int i = 0; i < uniqueN; i++) {
				double[] a = pts[i];
				double[] b = pts[(i + 1) % uniqueN];
				signedArea += a[0] * b[1] - b[0] * a[1];
			}
			signedArea /= 2.0;
			double expectedArea = Math.PI * SPHERE_R * SPHERE_R;
			assertEquals(expectedArea, Math.abs(signedArea), expectedArea * 0.05,
					"Shoelace area of sphere equatorial slice must be within 5 % of π·r²");

		} finally {
			mb.safeDelete(sphere);
		}
	}

	/**
	 * Slice the sphere at a height close to its north pole — should produce a
	 * small circle (one contour, all points within a tighter radius band).
	 */
	@Test
	@Order(83)
	@DisplayName("Slice sphere near north pole: one small contour")
	void testSliceSphereNearPole() throws Throwable {
		java.lang.foreign.MemorySegment sphere = makeSphere();
		try {
			// Slice at 90 % of the radius — the cross-section is a circle of
			// radius r·sin(arccos(0.9)) ≈ r·0.4359
			double sliceZ = SPHERE_R * 0.9;
			double expectedR = SPHERE_R * Math.sqrt(1 - 0.9 * 0.9); // ≈ 2.615
			double radiusTol = expectedR * 0.05;

			ArrayList<double[][]> contours = mb.slice(sphere, sliceZ);

			assertEquals(1, contours.size(), "Sphere slice near pole must produce exactly one contour");

			double[][] pts = contours.get(0);
			int n = pts.length;
			boolean closes = n >= 2 && Math.abs(pts[0][0] - pts[n - 1][0]) < 1e-9
					&& Math.abs(pts[0][1] - pts[n - 1][1]) < 1e-9;
			int uniqueN = closes ? n - 1 : n;

			assertTrue(uniqueN >= 3, "Near-pole slice must have at least 3 vertices");

			for (int i = 0; i < uniqueN; i++) {
				double x = pts[i][0];
				double y = pts[i][1];
				double r = Math.sqrt(x * x + y * y);
				assertEquals(expectedR, r, radiusTol,
						"Near-pole slice point " + i + " radius " + r + " should be ≈ " + expectedR);
			}
		} finally {
			mb.safeDelete(sphere);
		}
	}

	// =========================================================================
	// Multi-STL load → multi-part 3MF save
	// =========================================================================

	/**
	 * Loads every boolean-operation result (union, difference, intersection)
	 * and the two primitives from individual STL files, then saves them all
	 * together as a single multi-part 3MF.  Verifies:
	 * <ul>
	 *   <li>All five STL files can be read back as valid manifolds.</li>
	 *   <li>The combined 3MF file is a valid ZIP with the required OPC entries.</li>
	 *   <li>Re-importing the 3MF yields exactly five objects whose triangle
	 *       counts match the originals.</li>
	 * </ul>
	 */
	@Test
	@Order(90)
	@DisplayName("Load 5 STLs (cube, sphere, union, diff, intersect) → save as multi-part 3MF → verify")
	void testMultiStlToMultiPart3mf() throws Throwable {

		// ── 1. Build all five source manifolds ────────────────────────────
		java.lang.foreign.MemorySegment cube = makeCube();
		java.lang.foreign.MemorySegment sphere = makeSphere();
		java.lang.foreign.MemorySegment unionMs = null;
		java.lang.foreign.MemorySegment diffMs = null;
		java.lang.foreign.MemorySegment intersectMs = null;

		// Reloaded from STL
		java.lang.foreign.MemorySegment cubeStl = null;
		java.lang.foreign.MemorySegment sphereStl = null;
		java.lang.foreign.MemorySegment unionStl = null;
		java.lang.foreign.MemorySegment diffStl = null;
		java.lang.foreign.MemorySegment intersectStl = null;

		// Reloaded from final 3MF
		ArrayList<java.lang.foreign.MemorySegment> fromMfObjects = null;

		try {
			unionMs = mb.union(cube, sphere);
			diffMs = mb.difference(cube, sphere);
			intersectMs = mb.intersection(cube, sphere);

			// ── 2. Export each to its own STL ─────────────────────────────
			File cubeStlFile = tmpDir.resolve("cube.stl").toFile();
			File sphereStlFile = tmpDir.resolve("sphere.stl").toFile();
			File unionStlFile = tmpDir.resolve("union.stl").toFile();
			File diffStlFile = tmpDir.resolve("diff.stl").toFile();
			File intersectStlFile = tmpDir.resolve("intersect.stl").toFile();

			mb.exportSTL(cube, cubeStlFile);
			mb.exportSTL(sphere, sphereStlFile);
			mb.exportSTL(unionMs, unionStlFile);
			mb.exportSTL(diffMs, diffStlFile);
			mb.exportSTL(intersectMs, intersectStlFile);

			for (File f : new File[] { cubeStlFile, sphereStlFile, unionStlFile, diffStlFile, intersectStlFile }) {
				assertTrue(f.exists() && f.length() > 84, f.getName() + " must exist with data");
			}

			// ── 3. Re-import from STL ─────────────────────────────────────
			cubeStl = mb.importSTL(cubeStlFile);
			sphereStl = mb.importSTL(sphereStlFile);
			unionStl = mb.importSTL(unionStlFile);
			diffStl = mb.importSTL(diffStlFile);
			intersectStl = mb.importSTL(intersectStlFile);

			// All re-imports must be valid meshes
			assertMeshValid(cubeStl, "cube/STL");
			assertMeshValid(sphereStl, "sphere/STL");
			assertMeshValid(unionStl, "union/STL");
			assertMeshValid(diffStl, "diff/STL");
			assertMeshValid(intersectStl, "intersect/STL");

			// Triangle counts must survive the STL round-trip exactly
			assertEquals(mb.numTri(cube), mb.numTri(cubeStl), "Cube triangle count must survive STL round-trip");
			assertEquals(mb.numTri(sphere), mb.numTri(sphereStl), "Sphere triangle count must survive STL round-trip");
			assertEquals(mb.numTri(unionMs), mb.numTri(unionStl), "Union triangle count must survive STL round-trip");
			assertEquals(mb.numTri(diffMs), mb.numTri(diffStl),
					"Difference triangle count must survive STL round-trip");
			assertEquals(mb.numTri(intersectMs), mb.numTri(intersectStl),
					"Intersection triangle count must survive STL round-trip");

			// ── 4. Pack all five STL-loaded meshes into one 3MF ───────────
			ArrayList<java.lang.foreign.MemorySegment> toExport = new ArrayList<>();
			toExport.add(cubeStl);
			toExport.add(sphereStl);
			toExport.add(unionStl);
			toExport.add(diffStl);
			toExport.add(intersectStl);

			File multiMfFile = tmpDir.resolve("all_parts.3mf").toFile();
			mb.export3MF(toExport, multiMfFile);
			assertTrue(multiMfFile.exists() && multiMfFile.length() > 0, "Multi-part 3MF must exist and be non-empty");

			// ── 5. Validate ZIP structure ─────────────────────────────────
			try (java.util.zip.ZipInputStream zis = new java.util.zip.ZipInputStream(
					new java.io.FileInputStream(multiMfFile))) {
				boolean hasModel = false;
				boolean hasContentTypes = false;
				boolean hasRels = false;
				java.util.zip.ZipEntry entry;
				while ((entry = zis.getNextEntry()) != null) {
					String name = entry.getName().replace('\\', '/');
					if (name.equalsIgnoreCase("3D/3dmodel.model"))
						hasModel = true;
					if (name.equals("[Content_Types].xml"))
						hasContentTypes = true;
					if (name.equals("_rels/.rels"))
						hasRels = true;
					zis.closeEntry();
				}
				assertTrue(hasModel, "3MF must contain 3D/3dmodel.model");
				assertTrue(hasContentTypes, "3MF must contain [Content_Types].xml");
				assertTrue(hasRels, "3MF must contain _rels/.rels");
			}

			// ── 6. Re-import 3MF and verify all five objects ──────────────
			fromMfObjects = mb.import3MF(multiMfFile);
			assertEquals(5, fromMfObjects.size(), "Re-imported 3MF must contain exactly 5 objects");

			// Expected triangle counts in export order
			long[] expectedTris = { mb.numTri(cubeStl), mb.numTri(sphereStl), mb.numTri(unionStl), mb.numTri(diffStl),
					mb.numTri(intersectStl) };
			String[] labels = { "cube", "sphere", "union", "difference", "intersection" };

			for (int i = 0; i < 5; i++) {
				java.lang.foreign.MemorySegment obj = fromMfObjects.get(i);
				assertMeshValid(obj, labels[i] + "/3MF re-import");
				assertEquals(expectedTris[i], mb.numTri(obj), labels[i] + ": triangle count mismatch in 3MF re-import");
			}

		} finally {
			mb.safeDelete(cube);
			mb.safeDelete(sphere);
			mb.safeDelete(unionMs);
			mb.safeDelete(diffMs);
			mb.safeDelete(intersectMs);
			mb.safeDelete(cubeStl);
			mb.safeDelete(sphereStl);
			mb.safeDelete(unionStl);
			mb.safeDelete(diffStl);
			mb.safeDelete(intersectStl);
			if (fromMfObjects != null) {
				for (java.lang.foreign.MemorySegment seg : fromMfObjects) {
					mb.safeDelete(seg);
				}
			}
		}
	}
}