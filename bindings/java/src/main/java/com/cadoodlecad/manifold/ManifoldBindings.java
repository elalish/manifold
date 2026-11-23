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

package com.cadoodlecad.manifold;

import java.io.File;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.io.*;
import java.lang.foreign.MemorySegment;
import java.nio.*;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.zip.*;

public class ManifoldBindings {
	public enum ManifoldError {
		NO_ERROR, NON_FINITE_VERTEX, NOT_MANIFOLD, VERTEX_INDEX_OUT_OF_BOUNDS, PROPERTIES_WRONG_LENGTH,
		MISSING_POSITION_PROPERTIES, MERGE_VECTORS_DIFFERENT_LENGTHS, MERGE_INDEX_OUT_OF_BOUNDS, TRANSFORM_WRONG_LENGTH,
		RUN_INDEX_WRONG_LENGTH, FACE_ID_WRONG_LENGTH, INVALID_CONSTRUCTION, RESULT_TOO_LARGE, INVALID_TANGENTS,
		CANCELLED;

		public static ManifoldError fromInt(int code) {
			ManifoldError[] values = values();
			if (code < 0 || code >= values.length)
				throw new IllegalArgumentException("Unknown ManifoldError code: " + code);
			return values[code];
		}
	}

	/**
	 * Polygon fill rule for building a {@link FillRule} cross-section. The C API
	 * exposes dedicated entry points for even-odd and non-zero; positive and
	 * negative both map to the default (positive/add) constructor, matching the
	 * upstream C++ mapping.
	 */
	public enum FillRule {
		EVEN_ODD, NON_ZERO, POSITIVE, NEGATIVE;

		public int toInt() {
			return ordinal();
		}
	}

	/** Corner style used when offsetting cross-sections. */
	public enum JoinType {
		SQUARE, ROUND, MITER, BEVEL;

		public int toInt() {
			return ordinal();
		}
	}

	private static volatile boolean loaded = false;

	private static final Linker LINKER = Linker.nativeLinker();

	private final SymbolLookup library;
	private final Map<String, MethodHandle> functions = new HashMap<>();
	private final List<String> missingFunctions = new ArrayList<>();

	// ===== C struct layouts =====

	private static final StructLayout VEC2_LAYOUT = MemoryLayout.structLayout(ValueLayout.JAVA_DOUBLE,
			ValueLayout.JAVA_DOUBLE);

	private static final StructLayout VEC3_LAYOUT = MemoryLayout.structLayout(ValueLayout.JAVA_DOUBLE,
			ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE);

	private static final StructLayout PAIR_LAYOUT = MemoryLayout.structLayout(ValueLayout.ADDRESS.withName("first"),
			ValueLayout.ADDRESS.withName("second"));

	/** ManifoldRayHit: uint64 face_id, double distance, vec3 position, vec3 normal, int crossing (+ pad). */
	private static final StructLayout RAY_HIT_LAYOUT = MemoryLayout.structLayout(ValueLayout.JAVA_LONG.withName("face_id"),
			ValueLayout.JAVA_DOUBLE.withName("distance"), VEC3_LAYOUT.withName("position"),
			VEC3_LAYOUT.withName("normal"), ValueLayout.JAVA_INT.withName("crossing"),
			MemoryLayout.paddingLayout(4));

	private static final long RAY_HIT_FACE_ID_OFFSET = RAY_HIT_LAYOUT
			.byteOffset(MemoryLayout.PathElement.groupElement("face_id"));
	private static final long RAY_HIT_DISTANCE_OFFSET = RAY_HIT_LAYOUT
			.byteOffset(MemoryLayout.PathElement.groupElement("distance"));
	private static final long RAY_HIT_POSITION_OFFSET = RAY_HIT_LAYOUT
			.byteOffset(MemoryLayout.PathElement.groupElement("position"));
	private static final long RAY_HIT_NORMAL_OFFSET = RAY_HIT_LAYOUT
			.byteOffset(MemoryLayout.PathElement.groupElement("normal"));
	private static final long RAY_HIT_CROSSING_OFFSET = RAY_HIT_LAYOUT
			.byteOffset(MemoryLayout.PathElement.groupElement("crossing"));

	/** void (*)(double* new_prop, ManifoldVec3 position, const double* old_prop, void* ctx) */
	private static final FunctionDescriptor PROPERTY_UPDATER_DESC = FunctionDescriptor.ofVoid(ValueLayout.ADDRESS,
			VEC3_LAYOUT, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

	public static final int OPTYPE_UNION = 0;
	public static final int OPTYPE_DIFFERENCE = 1;
	public static final int OPTYPE_INTERSECTION = 2;

	private static void loadNativeLibrary(String libName, File cacheDirectory) throws Exception {
		try {
			// Detect platform
			String os = System.getProperty("os.name").toLowerCase();
			String arch = System.getProperty("os.arch").toLowerCase();
			if (arch.equals("amd64"))
				arch = "x86_64";
			if (arch.contains("aarch"))
				arch = "arm64";
			String platform;
			String extension;
			if (os.contains("win")) {
				platform = "win-" + arch;
				extension = ".dll";
				if (libName.startsWith("lib")) {
					libName = libName.substring(3, libName.length());
				}
			} else if (os.contains("mac")) {
				platform = "mac-" + arch;
				extension = ".dylib";
			} else if (os.contains("nix") || os.contains("nux") || os.contains("aix")) {
				platform = "linux-" + arch;
				extension = ".so";
			} else {
				throw new RuntimeException("Unsupported OS: " + os);
			}

			// Add library extension
			String fullName = libName + extension;

			java.io.File libsDir = nativeCacheDir(cacheDirectory);
			if (cacheDirectory == null)
				libsDir.deleteOnExit();
			if (!libsDir.isDirectory() && !libsDir.mkdirs())
				throw new RuntimeException("Cannot create native cache directory " + libsDir);

			java.io.File libFile = new java.io.File(libsDir, fullName);

			// The cache dir is version-keyed, so an existing file is this jar's own.
			if (!libFile.exists()) {
				try (java.io.InputStream in = ManifoldBindings.class
						.getResourceAsStream("/manifold3d/natives/" + platform + "/" + fullName)) {
					if (in == null)
						throw new RuntimeException("Library not found: " + fullName + " for platform " + platform);

					java.nio.file.Files.copy(in, libFile.toPath(), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
					if (cacheDirectory == null)
						libFile.deleteOnExit();
				}
			}
			System.load(libFile.getAbsolutePath());
		} catch (Exception e) {
			throw new RuntimeException("Failed to load: " + libName, e);
		}
	}

	/**
	 * Directory the natives are extracted to. A caller-supplied root gets a
	 * version-named subdirectory, so a jar upgrade never loads the previous jar's
	 * binaries against the new descriptors; no root means a per-call temp
	 * directory.
	 */
	static java.io.File nativeCacheDir(File cacheRoot) throws java.io.IOException {
		return cacheRoot == null ? Files.createTempDirectory("manifold3d").toFile()
				: new java.io.File(cacheRoot, cacheVersion());
	}

	/** Identifies this jar in the native cache path; "unknown" for a loose classpath. */
	static String cacheVersion() {
		Package pkg = ManifoldBindings.class.getPackage();
		String version = pkg == null ? null : pkg.getImplementationVersion();
		return version == null ? "unknown" : version;
	}

	/**
	 * Loads both native libraries unless they are already loaded. The flag is
	 * checked and set inside the lock so concurrent constructors neither load
	 * twice nor see a half-loaded pair, and it is only set once both loads
	 * succeed so a failure does not leave later callers believing they are ready.
	 */
	public static synchronized void loadNativeLibrarys(File cacheDirectory) throws Exception {
		if (loaded)
			return;
		loadNativeLibrary("libmanifold", cacheDirectory);
		loadNativeLibrary("libmanifoldc", cacheDirectory);
		loaded = true;
	}

	public ManifoldBindings() throws Exception {
		this((File) null);
	}

	public ManifoldBindings(File cacheDirectory) throws Exception {
		loadNativeLibrarys(cacheDirectory);

		this.library = SymbolLookup.loaderLookup();

		// ===== Construction / copy =====

		// ManifoldManifold* manifold_empty(void);
		load("manifold_empty", ValueLayout.ADDRESS);

		// ManifoldManifold* manifold_tetrahedron(void);
		load("manifold_tetrahedron", ValueLayout.ADDRESS);

		// ManifoldManifold* manifold_copy(const ManifoldManifold* m);
		load("manifold_copy", ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldManifold* manifold_cube(double x, double y, double z, int center);
		load("manifold_cube", ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE,
				ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_INT);

		// ManifoldManifold* manifold_sphere(double radius, int circular_segments);
		load("manifold_sphere", ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_INT);

		// ManifoldManifold* manifold_cylinder(double height, double radius_low, double
		// radius_high, int circular_segments, int center);
		load("manifold_cylinder", ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE,
				ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT);

		// ===== Boolean operations =====

		// ManifoldManifold* manifold_boolean(const ManifoldManifold* a, const
		// ManifoldManifold* b, ManifoldOpType op);
		load("manifold_boolean", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
				ValueLayout.JAVA_INT);

		// ManifoldManifold* manifold_union(const ManifoldManifold* a, const
		// ManifoldManifold* b);
		load("manifold_union", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldManifold* manifold_difference(const ManifoldManifold* a, const
		// ManifoldManifold* b);
		load("manifold_difference", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldManifold* manifold_intersection(const ManifoldManifold* a, const
		// ManifoldManifold* b);
		load("manifold_intersection", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldManifold* manifold_minkowski_sum(const ManifoldManifold* a, const
		// ManifoldManifold* b);
		load("manifold_minkowski_sum", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldManifold* manifold_minkowski_difference(const ManifoldManifold* a,
		// const ManifoldManifold* b);
		load("manifold_minkowski_difference", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldManifold* manifold_batch_boolean(const ManifoldManifoldVec* ms,
		// ManifoldOpType op);
		load("manifold_batch_boolean", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT);

		// ===== Manifold vectors =====

		// ManifoldManifoldVec* manifold_manifold_empty_vec(void);
		load("manifold_manifold_empty_vec", ValueLayout.ADDRESS);

		// void manifold_manifold_vec_push_back(ManifoldManifoldVec* ms, const
		// ManifoldManifold* m);
		loadVoid("manifold_manifold_vec_push_back", ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// size_t manifold_manifold_vec_length(const ManifoldManifoldVec* ms);
		load("manifold_manifold_vec_length", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);

		// ManifoldManifold* manifold_manifold_vec_get(const ManifoldManifoldVec* ms,
		// size_t idx);
		load("manifold_manifold_vec_get", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);

		// ===== Transforms =====

		// ManifoldManifold* manifold_transform(const ManifoldManifold* m, double x1,
		// double y1, double z1, ... double y4, double z4);
		load("manifold_transform", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE,
				ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE,
				ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE,
				ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE);

		// ManifoldManifold* manifold_translate(const ManifoldManifold* m, double x,
		// double y, double z);
		load("manifold_translate", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE,
				ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE);

		// ManifoldManifold* manifold_scale(const ManifoldManifold* m, double x, double
		// y, double z);
		load("manifold_scale", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE,
				ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE);

		// ManifoldManifold* manifold_mirror(const ManifoldManifold* m, double nx,
		// double ny, double nz);
		load("manifold_mirror", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE,
				ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE);

		// ManifoldManifold* manifold_rotate(const ManifoldManifold* m, double x, double
		// y, double z);
		load("manifold_rotate", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE,
				ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE);

		// ===== Refinement / smoothing =====

		// ManifoldManifold* manifold_refine(const ManifoldManifold* m, int refine);
		load("manifold_refine", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT);

		// ManifoldManifold* manifold_refine_to_length(const ManifoldManifold* m, double
		// length);
		load("manifold_refine_to_length", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE);

		// ManifoldManifold* manifold_refine_to_tolerance(const ManifoldManifold* m,
		// double tolerance);
		load("manifold_refine_to_tolerance", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE);

		// ManifoldManifold* manifold_simplify(const ManifoldManifold* m, double
		// tolerance);
		load("manifold_simplify", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE);

		// ManifoldManifold* manifold_smooth_by_normals(const ManifoldManifold* m, int
		// normalIdx);
		load("manifold_smooth_by_normals", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT);

		// ManifoldManifold* manifold_calculate_normals(const ManifoldManifold* m, int
		// normal_idx, double min_sharp_angle);
		load("manifold_calculate_normals", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
				ValueLayout.JAVA_DOUBLE);

		// ManifoldManifold* manifold_calculate_curvature(const ManifoldManifold* m, int
		// gaussian_idx, int mean_idx);
		load("manifold_calculate_curvature", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
				ValueLayout.JAVA_INT);

		// ManifoldManifold* manifold_set_properties(const ManifoldManifold* m, int
		// num_prop, void (*fun)(double*, ManifoldVec3, const double*, void*), void*
		// ctx);
		load("manifold_set_properties", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
				ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ===== Split / trim =====

		// ManifoldManifoldPair manifold_split(const ManifoldManifold* a, const
		// ManifoldManifold* b);
		load("manifold_split", PAIR_LAYOUT, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldManifold* manifold_trim_by_plane(const ManifoldManifold* m, double
		// normal_x, double normal_y, double normal_z, double offset);
		load("manifold_trim_by_plane", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE,
				ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE);

		// ===== Slice / project / polygons =====

		// ManifoldPolygons* manifold_slice(const ManifoldManifold* m, double height);
		load("manifold_slice", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE);

		// ManifoldPolygons* manifold_project(const ManifoldManifold* m);
		load("manifold_project", ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// size_t manifold_polygons_length(const ManifoldPolygons* ps);
		load("manifold_polygons_length", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);

		// size_t manifold_polygons_simple_length(const ManifoldPolygons* ps, size_t
		// idx);
		load("manifold_polygons_simple_length", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);

		// ManifoldVec2 manifold_polygons_get_point(const ManifoldPolygons* ps, size_t
		// simple_idx, size_t pt_idx);
		load("manifold_polygons_get_point", VEC2_LAYOUT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG,
				ValueLayout.JAVA_LONG);

		// ManifoldSimplePolygon* manifold_simple_polygon(const ManifoldVec2* ps, size_t
		// length);
		load("manifold_simple_polygon", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);

		// ManifoldPolygons* manifold_polygons(const ManifoldSimplePolygon* const* ps,
		// size_t length);
		load("manifold_polygons", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);

		// void manifold_delete_simple_polygon(ManifoldSimplePolygon* p);
		loadVoid("manifold_delete_simple_polygon", ValueLayout.ADDRESS);

		// void manifold_delete_polygons(ManifoldPolygons* p);
		loadVoid("manifold_delete_polygons", ValueLayout.ADDRESS);

		// ===== Analysis =====

		// double manifold_volume(const ManifoldManifold* m);
		load("manifold_volume", ValueLayout.JAVA_DOUBLE, ValueLayout.ADDRESS);

		// double manifold_surface_area(const ManifoldManifold* m);
		load("manifold_surface_area", ValueLayout.JAVA_DOUBLE, ValueLayout.ADDRESS);

		// ManifoldBox* manifold_bounding_box(const ManifoldManifold* m);
		load("manifold_bounding_box", ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldVec3 manifold_box_min(const ManifoldBox* b);
		load("manifold_box_min", VEC3_LAYOUT, ValueLayout.ADDRESS);

		// ManifoldVec3 manifold_box_max(const ManifoldBox* b);
		load("manifold_box_max", VEC3_LAYOUT, ValueLayout.ADDRESS);

		// ManifoldVec3 manifold_box_center(const ManifoldBox* b);
		load("manifold_box_center", VEC3_LAYOUT, ValueLayout.ADDRESS);

		// ManifoldVec3 manifold_box_dimensions(const ManifoldBox* b);
		load("manifold_box_dimensions", VEC3_LAYOUT, ValueLayout.ADDRESS);

		// double manifold_epsilon(const ManifoldManifold* m);
		load("manifold_epsilon", ValueLayout.JAVA_DOUBLE, ValueLayout.ADDRESS);

		// int manifold_genus(const ManifoldManifold* m);
		load("manifold_genus", ValueLayout.JAVA_INT, ValueLayout.ADDRESS);

		// double manifold_min_gap(const ManifoldManifold* m, const ManifoldManifold*
		// other, double searchLength);
		load("manifold_min_gap", ValueLayout.JAVA_DOUBLE, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
				ValueLayout.JAVA_DOUBLE);

		// int manifold_winding_number(const ManifoldManifold* m, double x, double y,
		// double z);
		load("manifold_winding_number", ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE,
				ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE);

		// ===== Ray casting =====

		// ManifoldRayHitVec* manifold_ray_cast(const ManifoldManifold* m, double
		// origin_x, double origin_y, double origin_z, double end_x, double end_y,
		// double end_z);
		load("manifold_ray_cast", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE,
				ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE,
				ValueLayout.JAVA_DOUBLE);

		// size_t manifold_ray_hit_vec_length(const ManifoldRayHitVec* v);
		load("manifold_ray_hit_vec_length", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);

		// ManifoldRayHit manifold_ray_hit_vec_get(const ManifoldRayHitVec* v, size_t
		// idx);
		load("manifold_ray_hit_vec_get", RAY_HIT_LAYOUT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);

		// void manifold_delete_ray_hit_vec(ManifoldRayHitVec* v);
		loadVoid("manifold_delete_ray_hit_vec", ValueLayout.ADDRESS);

		// ===== Composition / decomposition =====

		// ManifoldManifoldVec* manifold_decompose(const ManifoldManifold* m);
		load("manifold_decompose", ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldManifold* manifold_as_original(const ManifoldManifold* m, int id);
		load("manifold_as_original", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT);

		// ===== Hull =====

		// ManifoldManifold* manifold_hull(const ManifoldManifold* m);
		load("manifold_hull", ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldManifold* manifold_batch_hull(const ManifoldManifoldVec* ms);
		load("manifold_batch_hull", ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldManifold* manifold_hull_pts(const ManifoldVec3* ps, size_t length);
		load("manifold_hull_pts", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);

		// ===== Info =====

		// size_t manifold_num_edge(const ManifoldManifold* m);
		load("manifold_num_edge", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);

		// size_t manifold_num_prop(const ManifoldManifold* m);
		load("manifold_num_prop", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);

		// size_t manifold_num_vert(const ManifoldManifold* m);
		load("manifold_num_vert", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);

		// size_t manifold_num_tri(const ManifoldManifold* m);
		load("manifold_num_tri", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);

		// ManifoldError manifold_status(const ManifoldManifold* m);
		load("manifold_status", ValueLayout.JAVA_INT, ValueLayout.ADDRESS);

		// int manifold_is_empty(const ManifoldManifold* m);
		load("manifold_is_empty", ValueLayout.JAVA_INT, ValueLayout.ADDRESS);

		// ===== Mesh export (64-bit) =====

		// ManifoldMeshGL64* manifold_get_meshgl64(const ManifoldManifold* m);
		load("manifold_get_meshgl64", ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// size_t manifold_meshgl64_num_vert(const ManifoldMeshGL64* m);
		load("manifold_meshgl64_num_vert", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);

		// size_t manifold_meshgl64_num_tri(const ManifoldMeshGL64* m);
		load("manifold_meshgl64_num_tri", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);

		// size_t manifold_meshgl64_num_prop(const ManifoldMeshGL64* m);
		load("manifold_meshgl64_num_prop", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);

		// size_t manifold_meshgl64_vert_properties_length(const ManifoldMeshGL64* m);
		load("manifold_meshgl64_vert_properties_length", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);

		// size_t manifold_meshgl64_tri_length(const ManifoldMeshGL64* m);
		load("manifold_meshgl64_tri_length", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);

		// const double* manifold_meshgl64_vert_properties(const ManifoldMeshGL64* m);
		load("manifold_meshgl64_vert_properties", ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// const uint64_t* manifold_meshgl64_tri_verts(const ManifoldMeshGL64* m);
		load("manifold_meshgl64_tri_verts", ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldMeshGL64* manifold_meshgl64_merge(const ManifoldMeshGL64* m, double
		// tolerance);
		load("manifold_meshgl64_merge", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE);

		// ManifoldMeshGL64* manifold_meshgl64(const double* vert_props, size_t
		// n_verts, size_t n_props, const uint64_t* tri_verts, size_t n_tris);
		load("manifold_meshgl64", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG,
				ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);

		// ManifoldManifold* manifold_of_meshgl64(const ManifoldMeshGL64* mesh);
		load("manifold_of_meshgl64", ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ===== Cleanup =====

		// void manifold_delete_manifold(ManifoldManifold* m);
		loadVoid("manifold_delete_manifold", ValueLayout.ADDRESS);

		// void manifold_delete_meshgl64(ManifoldMeshGL64* m);
		loadVoid("manifold_delete_meshgl64", ValueLayout.ADDRESS);

		// void manifold_delete_manifold_vec(ManifoldManifoldVec* ms);
		loadVoid("manifold_delete_manifold_vec", ValueLayout.ADDRESS);

		// void manifold_delete_box(ManifoldBox* b);
		loadVoid("manifold_delete_box", ValueLayout.ADDRESS);

		// ===== Static quality globals =====

		// void manifold_set_circular_segments(int number);
		loadVoid("manifold_set_circular_segments", ValueLayout.JAVA_INT);

		// void manifold_set_min_circular_angle(double degrees);
		loadVoid("manifold_set_min_circular_angle", ValueLayout.JAVA_DOUBLE);

		// int manifold_get_circular_segments(double radius);
		load("manifold_get_circular_segments", ValueLayout.JAVA_INT, ValueLayout.JAVA_DOUBLE);

		// void manifold_version(int version[3]);
		loadVoid("manifold_version", ValueLayout.ADDRESS);

		// ===== Cross sections =====

		// ManifoldCrossSection* manifold_cross_section_of_polygons(const
		// ManifoldPolygons* p);
		load("manifold_cross_section_of_polygons", ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldCrossSection* manifold_cross_section_even_odd_polygons(const
		// ManifoldPolygons* p);
		load("manifold_cross_section_even_odd_polygons", ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldCrossSection* manifold_cross_section_non_zero_polygons(const
		// ManifoldPolygons* p);
		load("manifold_cross_section_non_zero_polygons", ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldCrossSection* manifold_cross_section_offset(const ManifoldCrossSection*
		// cs, double delta, ManifoldJoinType jt, double miter_limit, int
		// circular_segments);
		load("manifold_cross_section_offset", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE,
				ValueLayout.JAVA_INT, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_INT);

		// ManifoldPolygons* manifold_cross_section_to_polygons(const
		// ManifoldCrossSection* cs);
		load("manifold_cross_section_to_polygons", ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// void manifold_delete_cross_section(ManifoldCrossSection* cs);
		loadVoid("manifold_delete_cross_section", ValueLayout.ADDRESS);

		// ===== Extrude / revolve / triangulate =====

		// ManifoldManifold* manifold_extrude(const ManifoldPolygons* cs, double height,
		// int slices, double twist_degrees, double scale_x, double scale_y);
		load("manifold_extrude", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE,
				ValueLayout.JAVA_INT, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE);

		// ManifoldManifold* manifold_revolve(const ManifoldPolygons* cs, int
		// circular_segments, double revolve_degrees);
		load("manifold_revolve", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
				ValueLayout.JAVA_DOUBLE);

		// ManifoldTriangulation* manifold_triangulate(const ManifoldPolygons* ps,
		// double epsilon);
		load("manifold_triangulate", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE);

		// size_t manifold_triangulation_num_tri(const ManifoldTriangulation* m);
		load("manifold_triangulation_num_tri", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);

		// const int* manifold_triangulation_tri_verts(const ManifoldTriangulation* m);
		load("manifold_triangulation_tri_verts", ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// void manifold_delete_triangulation(ManifoldTriangulation* m);
		loadVoid("manifold_delete_triangulation", ValueLayout.ADDRESS);

		verifyAllFunctionsLoaded();
	}

	// ===== Native loader helpers =====

	private void load(String name, MemoryLayout returnLayout, MemoryLayout... argLayouts) {
		loadHandle(name, FunctionDescriptor.of(returnLayout, argLayouts));
	}

	private void loadVoid(String name, ValueLayout... argLayouts) {
		loadHandle(name, FunctionDescriptor.ofVoid(argLayouts));
	}

	private void loadHandle(String name, FunctionDescriptor desc) {
		library.find(name).ifPresentOrElse(symbol -> functions.put(name, LINKER.downcallHandle(symbol, desc)),
				() -> missingFunctions.add(name));
	}

	/**
	 * Fails if any function declared by the constructor is absent from the loaded
	 * library. Collected rather than thrown from {@link #loadHandle} so one broken
	 * build reports every renamed or dropped symbol at once, and so the failure
	 * lands in tests instead of the first caller that happens to use the function.
	 */
	private void verifyAllFunctionsLoaded() {
		if (!missingFunctions.isEmpty())
			throw new IllegalStateException(
					"Missing " + missingFunctions.size() + " native functions: " + String.join(", ", missingFunctions));
	}

	/**
	 * Returns a loaded handle, failing loudly instead of throwing an opaque NPE when
	 * a symbol was unavailable.
	 */
	private MethodHandle fn(String name) {
		MethodHandle mh = functions.get(name);
		if (mh == null)
			throw new IllegalStateException("Native function not loaded: " + name);
		return mh;
	}

	// ===== Property callbacks =====

	/** Receives one vertex's properties while {@link #setProperties} rewrites them. */
	@FunctionalInterface
	public interface PropertyUpdater {
		/**
		 * Called once per vertex. All three segments are byte-addressed and sized for
		 * indexed access: {@code newProp} holds the requested number of channels,
		 * {@code oldProp} holds the previous channel count, and {@code position} holds
		 * {x, y, z}. Channel 0..2 are the vertex position in both property rows.
		 *
		 * @param newProp  destination row of the new property block
		 * @param position vertex position, three doubles
		 * @param oldProp  source row of the previous property block
		 */
		void update(MemorySegment newProp, MemorySegment position, MemorySegment oldProp);
	}

	private static final MethodHandle PROPERTY_UPDATER_TARGET;

	static {
		try {
			PROPERTY_UPDATER_TARGET = MethodHandles.lookup().findStatic(ManifoldBindings.class,
					"dispatchPropertyUpdater",
					MethodType.methodType(void.class, PropertyUpdater.class, MemorySegment.class, MemorySegment.class,
							MemorySegment.class, MemorySegment.class));
		} catch (ReflectiveOperationException e) {
			throw new ExceptionInInitializerError(e);
		}
	}

	@SuppressWarnings("unused")
	private static void dispatchPropertyUpdater(PropertyUpdater updater, MemorySegment newProp, MemorySegment position,
			MemorySegment oldProp, MemorySegment ctx) {
		// ctx carries {new numProp, old numProp} so each buffer can be given an exact
		// size; the native callback otherwise hands back unindexable struct layouts.
		MemorySegment lengths = ctx.reinterpret(2L * Integer.BYTES);
		long newBytes = lengths.getAtIndex(ValueLayout.JAVA_INT, 0) * (long) Double.BYTES;
		long oldBytes = lengths.getAtIndex(ValueLayout.JAVA_INT, 1) * (long) Double.BYTES;
		updater.update(newProp.reinterpret(Math.max(newBytes, 0)), position.reinterpret(VEC3_LAYOUT.byteSize()),
				oldProp.reinterpret(Math.max(oldBytes, 0)));
	}

	/**
	 * Rewrites every vertex property row via {@code updater}, or clears the property
	 * block when {@code updater} is {@code null}.
	 */
	public MemorySegment setProperties(MemorySegment m, int numProp, PropertyUpdater updater) throws Throwable {
		if (m == null)
			throw new IllegalArgumentException("manifold segment must not be null");
		if (numProp < 0)
			throw new IllegalArgumentException("numProp must be non-negative: " + numProp);
		long oldNumProp = numProp(m);
		try (Arena arena = Arena.ofConfined()) {
			MemorySegment stub = MemorySegment.NULL;
			MemorySegment ctx = MemorySegment.NULL;
			if (updater != null) {
				ctx = arena.allocate(2L * Integer.BYTES);
				ctx.setAtIndex(ValueLayout.JAVA_INT, 0, numProp);
				ctx.setAtIndex(ValueLayout.JAVA_INT, 1, (int) oldNumProp);
				MethodHandle bound = MethodHandles.insertArguments(PROPERTY_UPDATER_TARGET, 0, updater);
				stub = LINKER.upcallStub(bound, PROPERTY_UPDATER_DESC, arena);
			}
			return (MemorySegment) fn("manifold_set_properties").invoke(m, numProp, stub, ctx);
		}
	}

	// ===== Static Quality Globals =====

	/**
	 * Sets the global number of circular segments used when generating cylinders,
	 * spheres, and other round primitives. Call this BEFORE creating primitives.
	 * Higher values produce smoother geometry but more triangles. A value of 0
	 * resets to the angle/length defaults.
	 */
	public void setCircularSegments(int segments) throws Throwable {
		fn("manifold_set_circular_segments").invoke(segments);
	}

	/**
	 * Sets the minimum angle (in degrees) between circular segments. The actual
	 * segment count is the maximum implied by both this angle limit and the
	 * edge-length limit. Smaller angles = more segments = smoother curves. Default
	 * is ~1 degree.
	 */
	public void setMinCircularAngle(double degrees) throws Throwable {
		fn("manifold_set_min_circular_angle").invoke(degrees);
	}

	/**
	 * Returns the number of circular segments that would be used for a circle of
	 * the given radius, given the current quality settings. Useful for debugging or
	 * verifying your quality settings before building geometry.
	 */
	public int getCircularSegments(double radius) throws Throwable {
		return (int) fn("manifold_get_circular_segments").invoke(radius);
	}

	// ===== Status =====

	public ManifoldError status(MemorySegment m) throws Throwable {
		int code = (int) fn("manifold_status").invoke(m);
		return ManifoldError.fromInt(code);
	}

	// ===== Slice =====
	/**
	 * Slices the manifold at the given Z height and returns contours offset by
	 * {@code delta} millimetres. Positive values expand outward (outset); negative
	 * values shrink inward (inset).
	 *
	 * <p>
	 * A delta of {@code 0.0} is equivalent to calling
	 * {@link #slice(MemorySegment, double)} but routed through the CrossSection
	 * pipeline (slightly more expensive). Prefer the plain {@code slice} overload
	 * when no offset is needed.
	 *
	 * <p>
	 * Inset caveat: contours narrower than {@code 2 * |delta|} will collapse and
	 * disappear — that is geometrically correct behaviour.
	 *
	 * @param m            the input manifold
	 * @param height       Z coordinate of the cutting plane
	 * @param delta        offset distance in model units (positive = outset,
	 *                     negative = inset)
	 * @param joinType     corner style for offset joins
	 * @param miterLimit   miter ratio cap — only used when {@code joinType} is
	 *                     {@link JoinType#MITER}; pass {@code 2.0} otherwise
	 * @param circularSegs arc resolution for rounded joins; {@code 0} uses the
	 *                     global quality setting (see {@link #setCircularSegments})
	 * @return list of offset contours; each contour is a {@code double[][2]} point
	 *         array
	 */
	public ArrayList<double[][]> sliceWithOffset(MemorySegment m, double height, double delta, JoinType joinType,
			double miterLimit, int circularSegs) throws Throwable {

		MemorySegment slicePolys = null;
		MemorySegment section = null;
		MemorySegment offsetSec = null;
		MemorySegment offsetPolys = null;

		try (Arena arena = Arena.ofConfined()) {

			// 1. Slice
			slicePolys = (MemorySegment) fn("manifold_slice").invoke(m, height);

			// 2. Polygons -> CrossSection (positive fill rule handles concave shapes
			// correctly)
			section = crossSectionOfPolygons(slicePolys, FillRule.POSITIVE);

			// 3. Offset
			offsetSec = (MemorySegment) fn("manifold_cross_section_offset").invoke(section, delta, joinType.toInt(),
					miterLimit, circularSegs);

			// 4. CrossSection -> Polygons -> Java
			offsetPolys = (MemorySegment) fn("manifold_cross_section_to_polygons").invoke(offsetSec);

			return polygonsToJava(offsetPolys, arena);

		} finally {
			safeDeletePolygons(slicePolys);
			safeDeleteCrossSection(section);
			safeDeleteCrossSection(offsetSec);
			safeDeletePolygons(offsetPolys);
		}
	}

	public ArrayList<double[][]> slice(MemorySegment m, double height) throws Throwable {
		MemorySegment polygons = null;
		try (Arena arena = Arena.ofConfined()) {
			polygons = (MemorySegment) fn("manifold_slice").invoke(m, height);
			return polygonsToJava(polygons, arena);
		} finally {
			safeDeletePolygons(polygons);
		}
	}

	/** Reads a native ManifoldPolygons into a Java ArrayList of contours. */
	private ArrayList<double[][]> polygonsToJava(MemorySegment polygons, Arena arena) throws Throwable {
		long numContours = (long) fn("manifold_polygons_length").invoke(polygons);
		ArrayList<double[][]> result = new ArrayList<>((int) numContours);
		MethodHandle getPoint = fn("manifold_polygons_get_point");
		for (int c = 0; c < numContours; c++) {
			long len = (long) fn("manifold_polygons_simple_length").invoke(polygons, (long) c);
			double[][] contour = new double[(int) len][2];
			for (int i = 0; i < len; i++) {
				MemorySegment pt = (MemorySegment) getPoint.invoke(arena, polygons, (long) c, (long) i);
				contour[i][0] = pt.get(ValueLayout.JAVA_DOUBLE, 0); // x
				contour[i][1] = pt.get(ValueLayout.JAVA_DOUBLE, 8); // y
			}
			result.add(contour);
		}
		return result;
	}

	private void safeDeleteCrossSection(MemorySegment cs) {
		if (cs == null)
			return;
		try {
			fn("manifold_delete_cross_section").invoke(cs);
		} catch (Throwable ignored) {
		}
	}

	private void safeDeletePolygons(MemorySegment p) {
		if (p == null)
			return;
		try {
			fn("manifold_delete_polygons").invoke(p);
		} catch (Throwable ignored) {
		}
	}

	/**
	 * Builds a cross-section from native polygons. The C API exposes dedicated
	 * constructors for even-odd and non-zero; positive and negative use the default
	 * positive fill, mirroring the upstream C++ mapping.
	 */
	private MemorySegment crossSectionOfPolygons(MemorySegment polygons, FillRule fillRule) throws Throwable {
		String f = switch (fillRule) {
		case EVEN_ODD -> "manifold_cross_section_even_odd_polygons";
		case NON_ZERO -> "manifold_cross_section_non_zero_polygons";
		case POSITIVE, NEGATIVE -> "manifold_cross_section_of_polygons";
		};
		return (MemorySegment) fn(f).invoke(polygons);
	}

	/**
	 * Builds a cross-section from Java contours with an explicit fill rule. The
	 * returned segment owns native memory and must be freed with
	 * {@code manifold_delete_cross_section}; use {@link #crossSectionToPolygons}
	 * to read it back and {@link #deleteCrossSection} to release it.
	 */
	public MemorySegment crossSectionOfPolygons(ArrayList<double[][]> contours, FillRule fillRule) throws Throwable {
		if (contours == null || contours.isEmpty())
			throw new IllegalArgumentException("contours must not be empty");
		if (fillRule == null)
			throw new IllegalArgumentException("fillRule must not be null");
		try (Arena arena = Arena.ofConfined()) {
			MemorySegment polys = javaToPolygons(arena, contours);
			try {
				return crossSectionOfPolygons(polys, fillRule);
			} finally {
				safeDeletePolygons(polys);
			}
		}
	}

	/**
	 * Reads a native cross-section back into Java contours. The cross-section stays
	 * owned by the caller.
	 */
	public ArrayList<double[][]> crossSectionToPolygons(MemorySegment crossSection) throws Throwable {
		MemorySegment polys = null;
		try (Arena arena = Arena.ofConfined()) {
			polys = (MemorySegment) fn("manifold_cross_section_to_polygons").invoke(crossSection);
			return polygonsToJava(polys, arena);
		} finally {
			safeDeletePolygons(polys);
		}
	}

	/** Releases a cross-section previously returned by {@link #crossSectionOfPolygons}. */
	public void deleteCrossSection(MemorySegment crossSection) throws Throwable {
		fn("manifold_delete_cross_section").invoke(crossSection);
	}

	// ===== New upstream capability wrappers =====

	/**
	 * Version of the manifold core this binding is talking to, as
	 * {@code {major, minor, patch}}, read from the loaded native library's
	 * {@code manifold_version}.
	 */
	public int[] version() throws Throwable {
		try (Arena arena = Arena.ofConfined()) {
			MemorySegment buf = arena.allocate(3 * Integer.BYTES);
			fn("manifold_version").invoke(buf);
			return new int[] { buf.get(ValueLayout.JAVA_INT, 0), buf.get(ValueLayout.JAVA_INT, 4),
					buf.get(ValueLayout.JAVA_INT, 8) };
		}
	}

	/**
	 * Point-containment query: returns the winding number of the manifold at the
	 * given point — {@code 0} outside, {@code 1} inside (cavities of nested shells
	 * also return 0). Unreliable exactly ON the surface.
	 */
	public int windingNumber(MemorySegment m, double x, double y, double z) throws Throwable {
		return (int) fn("manifold_winding_number").invoke(m, x, y, z);
	}

	/**
	 * Casts a ray segment from {@code origin} to {@code end} and returns every
	 * surface hit in order. An empty list means the segment misses the solid.
	 *
	 * <p>
	 * Each hit is copied into a {@link RayHit}, and the native hit vector is
	 * released before this method returns, so the results outlive the call.
	 */
	public ArrayList<RayHit> rayCast(MemorySegment m, double[] origin, double[] end) throws Throwable {
		double[] from = requirePoint(origin, "origin");
		double[] to = requirePoint(end, "end");

		// Owned native container; kept inside this method and always released.
		MemorySegment hits = (MemorySegment) fn("manifold_ray_cast").invoke(m, from[0], from[1], from[2], to[0],
				to[1], to[2]);
		try (Arena arena = Arena.ofConfined()) {
			long count = (long) fn("manifold_ray_hit_vec_length").invoke(hits);
			MethodHandle get = fn("manifold_ray_hit_vec_get");
			ArrayList<RayHit> out = new ArrayList<>((int) count);
			for (long i = 0; i < count; i++) {
				out.add(toRayHit((MemorySegment) get.invoke(arena, hits, i)));
			}
			return out;
		} finally {
			try {
				fn("manifold_delete_ray_hit_vec").invoke(hits);
			} catch (Throwable ignored) {
			}
		}
	}

	/** Copies one native {@code ManifoldRayHit} into a plain Java value. */
	private static RayHit toRayHit(MemorySegment hit) {
		return new RayHit(hit.get(ValueLayout.JAVA_LONG, RAY_HIT_FACE_ID_OFFSET),
				hit.get(ValueLayout.JAVA_DOUBLE, RAY_HIT_DISTANCE_OFFSET),
				readVec3(hit, RAY_HIT_POSITION_OFFSET), readVec3(hit, RAY_HIT_NORMAL_OFFSET),
				hit.get(ValueLayout.JAVA_INT, RAY_HIT_CROSSING_OFFSET));
	}

	private static double[] readVec3(MemorySegment struct, long offset) {
		return new double[] { struct.get(ValueLayout.JAVA_DOUBLE, offset),
				struct.get(ValueLayout.JAVA_DOUBLE, offset + 8), struct.get(ValueLayout.JAVA_DOUBLE, offset + 16) };
	}

	private static double[] requirePoint(double[] point, String name) {
		if (point == null || point.length != 3)
			throw new IllegalArgumentException(name + " must be a double[3] of {x, y, z}");
		return point;
	}

	/**
	 * Minimum gap between this manifold and another, searching along a bounding
	 * sphere hierarchy up to {@code searchLength}. Returns
	 * {@code java.lang.Double.MAX_VALUE} when the solids are farther apart than
	 * the search length.
	 */
	public double minGap(MemorySegment a, MemorySegment b, double searchLength) throws Throwable {
		return (double) fn("manifold_min_gap").invoke(a, b, searchLength);
	}

	/**
	 * Appends Gaussian ({@code gaussianIdx}) and mean ({@code meanIdx})
	 * curvature channels to the vertex properties. Indices -1 skip a channel;
	 * the output keeps everything else unchanged.
	 */
	public MemorySegment calculateCurvature(MemorySegment m, int gaussianIdx, int meanIdx) throws Throwable {
		return (MemorySegment) fn("manifold_calculate_curvature").invoke(m, gaussianIdx, meanIdx);
	}

	/**
	 * Projects the manifold's silhouette onto the XY plane. Exactly
	 * {@link #slice} at any height for a prismatic solid, but defined for any
	 * manifold: every triangle is extruded to an infinite prism and unioned.
	 */
	public ArrayList<double[][]> project(MemorySegment m) throws Throwable {
		MemorySegment polygons = null;
		try (Arena arena = Arena.ofConfined()) {
			polygons = (MemorySegment) fn("manifold_project").invoke(m);
			return polygonsToJava(polygons, arena);
		} finally {
			safeDeletePolygons(polygons);
		}
	}

	/**
	 * Extrudes a 2D cross-section (as from {@link #slice} or
	 * {@link #project}) into a prismatic solid. {@code scaleTop} shrinks or
	 * grows the top face; {@code twistDegrees} rotates it along the extrusion.
	 *
	 * Overlapping contours are resolved by the library's default (positive)
	 * fill rule; route through
	 * {@link #crossSectionOfPolygons(ArrayList, FillRule)} /
	 * {@link #crossSectionToPolygons(MemorySegment)} with an explicit
	 * {@link FillRule} when even-odd semantics are needed.
	 *
	 * @param contours    contours as {@code double[contour][points][2]}
	 * @param height      extrusion distance along +Z
	 * @param slices      number of divisions for twist/scale; 0 disables
	 * @param scaleTop    uniform top-face scale factor
	 */
	public MemorySegment extrude(ArrayList<double[][]> contours, double height, int slices, double twistDegrees,
			double scaleTop) throws Throwable {
		if (contours == null || contours.isEmpty())
			throw new IllegalArgumentException("contours must not be empty");
		try (Arena arena = Arena.ofConfined()) {
			MemorySegment polys = javaToPolygons(arena, contours);
			try {
				return (MemorySegment) fn("manifold_extrude").invoke(polys, height, slices, twistDegrees, scaleTop,
						scaleTop);
			} finally {
				safeDeletePolygons(polys);
			}
		}
	}

	/**
	 * Revolves a 2D cross-section around the Y axis. {@code revolveDegrees}
	 * below 360 produces a partial revolve; the cross-section must sit at
	 * x >= 0.
	 */
	public MemorySegment revolve(ArrayList<double[][]> contours, int circularSegments, double revolveDegrees)
			throws Throwable {
		if (contours == null || contours.isEmpty())
			throw new IllegalArgumentException("contours must not be empty");
		try (Arena arena = Arena.ofConfined()) {
			MemorySegment polys = javaToPolygons(arena, contours);
			try {
				return (MemorySegment) fn("manifold_revolve").invoke(polys, circularSegments, revolveDegrees);
			} finally {
				safeDeletePolygons(polys);
			}
		}
	}

	/**
	 * Triangulates a 2D cross-section, returning one flat index triple per
	 * triangle (referencing the contour points in input order).
	 */
	public int[][] triangulate(ArrayList<double[][]> contours, double epsilon) throws Throwable {
		if (contours == null || contours.isEmpty())
			throw new IllegalArgumentException("contours must not be empty");
		MemorySegment tri = null;
		try (Arena arena = Arena.ofConfined()) {
			MemorySegment polys = javaToPolygons(arena, contours);
			MemorySegment copy;
			try {
				tri = (MemorySegment) fn("manifold_triangulate").invoke(polys, epsilon);

				long nTri = (long) fn("manifold_triangulation_num_tri").invoke(tri);
				// Borrowed from the triangulation; valid for its lifetime only.
				MemorySegment verts = ((MemorySegment) fn("manifold_triangulation_tri_verts").invoke(tri))
						.reinterpret(nTri * 3L * Integer.BYTES);
				copy = arena.allocate(nTri * 3L * Integer.BYTES);
				copy.copyFrom(verts);
			} finally {
				safeDeletePolygons(polys);
			}
			int[][] out = new int[(int) (copy.byteSize() / (3L * Integer.BYTES))][3];
			for (int t = 0; t < out.length; t++) {
				for (int j = 0; j < 3; j++)
					out[t][j] = (int) copy.getAtIndex(ValueLayout.JAVA_INT, t * 3L + j);
			}
			return out;
		} finally {
			if (tri != null) {
				try {
					fn("manifold_delete_triangulation").invoke(tri);
				} catch (Throwable ignored) {
				}
			}
		}
	}

	/** Builds a native ManifoldPolygons from Java contour arrays (default fill). */
	private MemorySegment javaToPolygons(Arena arena, ArrayList<double[][]> contours) throws Throwable {
		// Each contour is itself a heap-allocated ManifoldSimplePolygon
		// (std::vector<vec2>), built here and destroyed in a finally block;
		// manifold_polygons copies the data, so nothing needs to outlive the call.
		ArrayList<MemorySegment> owned = new ArrayList<>(contours.size());
		MemorySegment contourPtrs = arena.allocate(ValueLayout.ADDRESS, contours.size());
		try {
			for (int c = 0; c < contours.size(); c++) {
				double[][] contour = contours.get(c);
				if (contour == null || contour.length == 0)
					throw new IllegalArgumentException("contours must not contain empty contours");
				MemorySegment pts = arena.allocate(contour.length * 2L * Double.BYTES);
				for (int i = 0; i < contour.length; i++) {
					if (contour[i] == null || contour[i].length != 2)
						throw new IllegalArgumentException("each point must be a double[2] of {x, y}");
					pts.setAtIndex(ValueLayout.JAVA_DOUBLE, i * 2, contour[i][0]);
					pts.setAtIndex(ValueLayout.JAVA_DOUBLE, i * 2 + 1, contour[i][1]);
				}
				MemorySegment sp = (MemorySegment) fn("manifold_simple_polygon").invoke(pts, (long) contour.length);
				owned.add(sp);
				contourPtrs.setAtIndex(ValueLayout.ADDRESS, c, sp);
			}
			return (MemorySegment) fn("manifold_polygons").invoke(contourPtrs, (long) contours.size());
		} finally {
			for (MemorySegment sp : owned) {
				try {
					fn("manifold_delete_simple_polygon").invoke(sp);
				} catch (Throwable ignored) {
				}
			}
		}
	}

	// ===== Primitives =====

	public MemorySegment empty() throws Throwable {
		return (MemorySegment) fn("manifold_empty").invoke();
	}

	public MemorySegment tetrahedron() throws Throwable {
		return (MemorySegment) fn("manifold_tetrahedron").invoke();
	}

	public MemorySegment cube(double x, double y, double z, boolean center) throws Throwable {
		return (MemorySegment) fn("manifold_cube").invoke(x, y, z, center ? 1 : 0);
	}

	public MemorySegment sphere(double radius, int segments) throws Throwable {
		return (MemorySegment) fn("manifold_sphere").invoke(radius, segments);
	}

	public MemorySegment cylinder(double height, double radiusLow, double radiusHigh, int segments, int center)
			throws Throwable {
		return (MemorySegment) fn("manifold_cylinder").invoke(height, radiusLow, radiusHigh, segments, center);
	}

	// ===== Transformations =====

	public MemorySegment transform(MemorySegment m, double x1, double y1, double z1, double x2, double y2, double z2,
			double x3, double y3, double z3, double x4, double y4, double z4) throws Throwable {
		return (MemorySegment) fn("manifold_transform").invoke(m, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4);
	}

	public MemorySegment translate(MemorySegment m, double x, double y, double z) throws Throwable {
		return (MemorySegment) fn("manifold_translate").invoke(m, x, y, z);
	}

	public MemorySegment scale(MemorySegment m, double x, double y, double z) throws Throwable {
		return (MemorySegment) fn("manifold_scale").invoke(m, x, y, z);
	}

	public MemorySegment rotate(MemorySegment m, double x, double y, double z) throws Throwable {
		return (MemorySegment) fn("manifold_rotate").invoke(m, x, y, z);
	}

	public MemorySegment rotateX(MemorySegment m, double degrees) throws Throwable {
		return rotate(m, degrees, 0, 0);
	}

	public MemorySegment rotateY(MemorySegment m, double degrees) throws Throwable {
		return rotate(m, 0, degrees, 0);
	}

	public MemorySegment rotateZ(MemorySegment m, double degrees) throws Throwable {
		return rotate(m, 0, 0, degrees);
	}

	public MemorySegment mirror(MemorySegment m, double nx, double ny, double nz) throws Throwable {
		return (MemorySegment) fn("manifold_mirror").invoke(m, nx, ny, nz);
	}

	// ===== Boolean operations =====

	public MemorySegment union(MemorySegment a, MemorySegment b) throws Throwable {
		return (MemorySegment) fn("manifold_union").invoke(a, b);
	}

	public MemorySegment difference(MemorySegment a, MemorySegment b) throws Throwable {
		return (MemorySegment) fn("manifold_difference").invoke(a, b);
	}

	public MemorySegment minkowski_difference(MemorySegment a, MemorySegment b) throws Throwable {
		return (MemorySegment) fn("manifold_minkowski_difference").invoke(a, b);
	}

	public MemorySegment minkowski_sum(MemorySegment a, MemorySegment b) throws Throwable {
		return (MemorySegment) fn("manifold_minkowski_sum").invoke(a, b);
	}

	public MemorySegment intersection(MemorySegment a, MemorySegment b) throws Throwable {
		return (MemorySegment) fn("manifold_intersection").invoke(a, b);
	}

	public MemorySegment booleanOp(MemorySegment a, MemorySegment b, int opType) throws Throwable {
		return (MemorySegment) fn("manifold_boolean").invoke(a, b, opType);
	}

	public MemorySegment batchUnion(MemorySegment[] shapes) throws Throwable {
		MemorySegment vec = (MemorySegment) fn("manifold_manifold_empty_vec").invoke();
		try {
			for (MemorySegment shape : shapes)
				fn("manifold_manifold_vec_push_back").invoke(vec, shape);
			return (MemorySegment) fn("manifold_batch_boolean").invoke(vec, OPTYPE_UNION);
		} finally {
			safeDeleteManifoldVec(vec);
		}
	}

	private void safeDeleteManifoldVec(MemorySegment vec) {
		if (vec == null)
			return;
		try {
			fn("manifold_delete_manifold_vec").invoke(vec);
		} catch (Throwable ignored) {
		}
	}

	// ===== Info =====

	private static final long POINTER_BYTES = ValueLayout.ADDRESS.byteSize();
	/** A {@code std::shared_ptr} is a pointer plus a control-block pointer. */
	private static final long SHARED_PTR_BYTES = 2L * POINTER_BYTES;
	/** A {@code std::vector} is begin, end and capacity pointers. */
	private static final long VECTOR_BYTES = 3L * POINTER_BYTES;

	/**
	 * Size of the native {@code manifold::Manifold} handle in bytes.
	 *
	 * <p>
	 * The C API used to report this through {@code manifold_manifold_size()}; it
	 * was dropped along with the caller-allocated buffers, so the value is derived
	 * from the C++ layout instead: three {@code std::shared_ptr} members,
	 * {@code pNodeMutex_}, {@code pNode_} and {@code ctx_}.
	 */
	public long manifoldSize() throws Throwable {
		return 3L * SHARED_PTR_BYTES;
	}

	/**
	 * Size of the native {@code manifold::MeshGL64} struct in bytes.
	 *
	 * <p>
	 * Derived from the C++ layout since {@code manifold_meshgl64_size()} was
	 * removed: one {@code size_t numProp} plus ten {@code std::vector} members
	 * ({@code vertProperties}, {@code triVerts}, {@code mergeFromVert},
	 * {@code mergeToVert}, {@code runIndex}, {@code runOriginalID},
	 * {@code runTransform}, {@code runFlags}, {@code faceID},
	 * {@code halfedgeTangent}).
	 */
	public long meshGL64Size() throws Throwable {
		return POINTER_BYTES + 10L * VECTOR_BYTES;
	}

	/**
	 * Size of the native {@code std::vector<Manifold>} in bytes, since
	 * {@code manifold_manifold_vec_size()} was removed.
	 */
	public long vecSize() throws Throwable {
		return VECTOR_BYTES;
	}

	public double volume(MemorySegment m) throws Throwable {
		return (double) fn("manifold_volume").invoke(m);
	}

	public double surfaceArea(MemorySegment m) throws Throwable {
		return (double) fn("manifold_surface_area").invoke(m);
	}

	public double epsilon(MemorySegment m) throws Throwable {
		return (double) fn("manifold_epsilon").invoke(m);
	}

	public int genus(MemorySegment m) throws Throwable {
		return (int) fn("manifold_genus").invoke(m);
	}

	public long numVert(MemorySegment m) throws Throwable {
		return (long) fn("manifold_num_vert").invoke(m);
	}

	public long numTri(MemorySegment m) throws Throwable {
		return (long) fn("manifold_num_tri").invoke(m);
	}

	public long numEdge(MemorySegment m) throws Throwable {
		return (long) fn("manifold_num_edge").invoke(m);
	}

	public long numProp(MemorySegment m) throws Throwable {
		return (long) fn("manifold_num_prop").invoke(m);
	}

	// ===== Refinement =====

	public MemorySegment refine(MemorySegment m, int level) throws Throwable {
		return (MemorySegment) fn("manifold_refine").invoke(m, level);
	}

	public MemorySegment refineToLength(MemorySegment m, double length) throws Throwable {
		return (MemorySegment) fn("manifold_refine_to_length").invoke(m, length);
	}

	public MemorySegment refineToTolerance(MemorySegment m, double tolerance) throws Throwable {
		return (MemorySegment) fn("manifold_refine_to_tolerance").invoke(m, tolerance);
	}

	public MemorySegment simplify(MemorySegment m, double tolerance) throws Throwable {
		return (MemorySegment) fn("manifold_simplify").invoke(m, tolerance);
	}

	public MemorySegment smoothByNormals(MemorySegment m, int normalIdx) throws Throwable {
		return (MemorySegment) fn("manifold_smooth_by_normals").invoke(m, normalIdx);
	}

	public MemorySegment calculateNormals(MemorySegment m, int normalIdx, double minSharpAngle) throws Throwable {
		return (MemorySegment) fn("manifold_calculate_normals").invoke(m, normalIdx, minSharpAngle);
	}

	// ===== Split/Trim =====

	public MemorySegment trimByPlane(MemorySegment m, double nx, double ny, double nz, double offset) throws Throwable {
		return (MemorySegment) fn("manifold_trim_by_plane").invoke(m, nx, ny, nz, offset);
	}

	public MemorySegment[] split(MemorySegment a, MemorySegment b) throws Throwable {
		try (Arena arena = Arena.ofConfined()) {
			// The pair is returned by value; the two handles it carries are owned
			// heap allocations independent of the arena.
			MemorySegment pair = (MemorySegment) fn("manifold_split").invoke(arena, a, b);
			return new MemorySegment[] { pair.get(ValueLayout.ADDRESS, 0), pair.get(ValueLayout.ADDRESS, 8) };
		}
	}

	// ===== Hull =====

	public MemorySegment hull(MemorySegment m) throws Throwable {
		return (MemorySegment) fn("manifold_hull").invoke(m);
	}

	/**
	 * Computes the convex hull of a set of points.
	 *
	 * Each point is supplied as a {@code double[3]} of {x, y, z}. If fewer than 4
	 * points are provided, or all points are coplanar, an empty manifold is
	 * returned.
	 */
	public MemorySegment hull(ArrayList<double[]> points) throws Throwable {
		if (points == null || points.isEmpty()) {
			return empty();
		}

		for (int i = 0; i < points.size(); i++) {
			if (points.get(i) == null || points.get(i).length != 3) {
				throw new IllegalArgumentException("Point at index " + i + " must be a double[3] of {x, y, z}");
			}
		}

		final long VEC3_BYTES = 3 * Double.BYTES;
		long count = points.size();

		try (Arena arena = Arena.ofConfined()) {
			MemorySegment ptsBuffer = arena.allocate(VEC3_BYTES * count);
			for (int i = 0; i < count; i++) {
				double[] pt = points.get(i);
				long base = i * VEC3_BYTES;
				ptsBuffer.set(ValueLayout.JAVA_DOUBLE, base, pt[0]);
				ptsBuffer.set(ValueLayout.JAVA_DOUBLE, base + Double.BYTES, pt[1]);
				ptsBuffer.set(ValueLayout.JAVA_DOUBLE, base + 2 * Double.BYTES, pt[2]);
			}
			return (MemorySegment) fn("manifold_hull_pts").invoke(ptsBuffer, count);
		}
	}

	public MemorySegment batchHull(MemorySegment[] shapes) throws Throwable {
		MemorySegment vec = (MemorySegment) fn("manifold_manifold_empty_vec").invoke();
		try {
			for (MemorySegment shape : shapes)
				fn("manifold_manifold_vec_push_back").invoke(vec, shape);
			return (MemorySegment) fn("manifold_batch_hull").invoke(vec);
		} finally {
			safeDeleteManifoldVec(vec);
		}
	}

	// ===== Compose/Decompose =====

	public MemorySegment compose(MemorySegment[] parts) throws Throwable {
		return batchUnion(parts);
	}

	/**
	 * Splits a manifold into its connected components.
	 *
	 * Each returned segment owns its native memory independently and must be
	 * freed with {@link #delete(MemorySegment)}. The elements are copies
	 * placed on the heap exactly like every other manifold handle this class
	 * hands out, so they stay valid after this call returns; the input
	 * {@code m} is untouched and remains owned by the caller.
	 *
	 * @param m the manifold to decompose; must not be null
	 * @return one segment per connected component; empty when {@code m} is an
	 *         empty manifold
	 */
	public MemorySegment[] decompose(MemorySegment m) throws Throwable {
		if (m == null)
			throw new IllegalArgumentException("manifold segment must not be null");
		MemorySegment vec = null;
		try {
			vec = (MemorySegment) fn("manifold_decompose").invoke(m);
			long count = (long) fn("manifold_manifold_vec_length").invoke(vec);
			MethodHandle vecGet = fn("manifold_manifold_vec_get");
			MemorySegment[] parts = new MemorySegment[(int) count];
			try {
				for (int i = 0; i < count; i++) {
					// manifold_manifold_vec_get hands back an owned heap copy
					parts[i] = (MemorySegment) vecGet.invoke(vec, (long) i);
				}
				return parts;
			} catch (Throwable t) {
				for (MemorySegment part : parts)
					if (part != null)
						safeDelete(part);
				throw t;
			}
		} finally {
			// the vector is a std::vector<Manifold>, so it needs the vec deleter —
			// manifold_delete_manifold would run the wrong destructor and crash
			safeDeleteManifoldVec(vec);
		}
	}

	// ===== Copy / AsOriginal =====

	public MemorySegment copy(MemorySegment m) throws Throwable {
		return (MemorySegment) fn("manifold_copy").invoke(m);
	}

	/**
	 * Returns a copy of this manifold as a new original mesh. A negative id asks
	 * the library to assign one.
	 */
	public MemorySegment asOriginal(MemorySegment m) throws Throwable {
		return (MemorySegment) fn("manifold_as_original").invoke(m, -1);
	}

	// ===== Bounding box =====

	public MemorySegment boundingBox(MemorySegment m) throws Throwable {
		return (MemorySegment) fn("manifold_bounding_box").invoke(m);
	}

	public double[] boxDimensions(MemorySegment box) throws Throwable {
		try (Arena arena = Arena.ofConfined()) {
			MemorySegment dims = (MemorySegment) fn("manifold_box_dimensions").invoke(arena, box);
			return new double[] { dims.get(ValueLayout.JAVA_DOUBLE, 0), dims.get(ValueLayout.JAVA_DOUBLE, 8),
					dims.get(ValueLayout.JAVA_DOUBLE, 16) };
		}
	}

	/**
	 * Returns the axis-aligned bounds of {@code m} as {@code {minX, minY, minZ,
	 * maxX, maxY, maxZ}}.
	 */
	public double[] bounds(MemorySegment m) throws Throwable {
		MemorySegment box = (MemorySegment) fn("manifold_bounding_box").invoke(m);
		try (Arena arena = Arena.ofConfined()) {
			MemorySegment min = (MemorySegment) fn("manifold_box_min").invoke(arena, box);
			MemorySegment max = (MemorySegment) fn("manifold_box_max").invoke(arena, box);
			return new double[] { min.get(ValueLayout.JAVA_DOUBLE, 0), min.get(ValueLayout.JAVA_DOUBLE, 8),
					min.get(ValueLayout.JAVA_DOUBLE, 16), max.get(ValueLayout.JAVA_DOUBLE, 0),
					max.get(ValueLayout.JAVA_DOUBLE, 8), max.get(ValueLayout.JAVA_DOUBLE, 16) };
		} finally {
			safeDeleteBox(box);
		}
	}

	public MemorySegment centerObject(MemorySegment m) throws Throwable {
		MemorySegment box = null;
		try {
			box = (MemorySegment) fn("manifold_bounding_box").invoke(m);
			try (Arena arena = Arena.ofConfined()) {
				MemorySegment center = (MemorySegment) fn("manifold_box_center").invoke(arena, box);
				double cx = center.get(ValueLayout.JAVA_DOUBLE, 0);
				double cy = center.get(ValueLayout.JAVA_DOUBLE, 8);
				double cz = center.get(ValueLayout.JAVA_DOUBLE, 16);
				return translate(m, -cx, -cy, -cz);
			}
		} finally {
			safeDeleteBox(box);
		}
	}

	private void safeDeleteBox(MemorySegment box) {
		if (box == null)
			return;
		try {
			fn("manifold_delete_box").invoke(box);
		} catch (Throwable ignored) {
		}
	}

	// ===== Memory estimation =====

	public long estimateMemoryBytes(MemorySegment m) throws Throwable {
		long nVert = numVert(m);
		long nTri = numTri(m);
		long nEdge = numEdge(m);
		long nProp = numProp(m);

		long vertexData = nVert * 24L;
		long halfedgeData = nEdge * 40L;
		long faceData = nTri * 8L;
		long edgeData = nEdge * 8L;
		long propertyData = nVert * nProp * 8L;

		long meshglVertexData = nVert * Math.max(3, nProp) * 8L;
		long meshglIndexData = nTri * 3L * 8L;
		long meshglStruct = meshGL64Size();

		long internalTotal = vertexData + halfedgeData + faceData + edgeData + propertyData;
		long exportTotal = meshglStruct + meshglVertexData + meshglIndexData;
		long manifoldHandle = manifoldSize();

		long bvhOverhead = (long) (internalTotal * 0.25);

		return manifoldHandle + internalTotal + exportTotal + bvhOverhead;
	}

	// ===== Cleanup =====

	public void delete(MemorySegment manifold) throws Throwable {
		fn("manifold_delete_manifold").invoke(manifold);
	}

	public void safeDelete(MemorySegment m) {
		if (m == null)
			return;
		try {
			delete(m);
		} catch (Throwable t) {
			// Already freed or invalid — swallow silently
		}
	}

	public void close() {
	}

	public static boolean isNativeLibraryLoaded() {
		return loaded;
	}

	// ===== Mesh import/export (64-bit) =====

	/**
	 * Imports a mesh from double-precision vertex and triangle arrays into a
	 * manifold.
	 *
	 * <p>
	 * Vertex positions are laid out as a flat {@code double[]} of length
	 * {@code nVerts * 3} in XYZ order. Triangle indices are a flat {@code long[]}
	 * of length {@code nTris * 3}. Duplicate vertices are welded automatically via
	 * {@code manifold_meshgl64_merge}.
	 *
	 * @param vertices  flat XYZ vertex array ({@code nVerts * 3} doubles)
	 * @param triangles flat triangle-index array ({@code nTris * 3} longs)
	 * @param nVerts    number of vertices
	 * @param nTris     number of triangles
	 * @return a fully initialised manifold {@link MemorySegment}
	 */
	public MemorySegment importMeshGL64(double[] vertices, long[] triangles, long nVerts, long nTris) throws Throwable {
		try (Arena arena = Arena.ofConfined()) {
			MemorySegment vertPtr = arena.allocate(3 * nVerts * Double.BYTES);
			vertPtr.copyFrom(MemorySegment.ofArray(vertices));

			MemorySegment triPtr = arena.allocate(3 * nTris * Long.BYTES);
			triPtr.copyFrom(MemorySegment.ofArray(triangles));

			MemorySegment meshGL = null;
			MemorySegment merged = null;
			try {
				meshGL = (MemorySegment) fn("manifold_meshgl64").invoke(vertPtr, nVerts, 3L, triPtr, nTris);
				// 0 keeps the library's baseline bounding-box merge tolerance.
				merged = (MemorySegment) fn("manifold_meshgl64_merge").invoke(meshGL, 0.0);
				return (MemorySegment) fn("manifold_of_meshgl64").invoke(merged);
			} finally {
				safeDeleteMeshGL64(meshGL);
				safeDeleteMeshGL64(merged);
			}
		}
	}

	private void safeDeleteMeshGL64(MemorySegment mesh) {
		if (mesh == null)
			return;
		try {
			fn("manifold_delete_meshgl64").invoke(mesh);
		} catch (Throwable ignored) {
		}
	}

	/**
	 * Exports a manifold to double-precision vertex and triangle arrays.
	 *
	 * @param manifold a fully initialised manifold {@link MemorySegment}
	 * @return a {@link MeshData64} record containing double vertex positions and
	 *         long triangle indices
	 */
	public MeshData64 exportMeshGL64(MemorySegment manifold) throws Throwable {
		MemorySegment meshGL = (MemorySegment) fn("manifold_get_meshgl64").invoke(manifold);

		try {
			long numVert = (long) fn("manifold_meshgl64_num_vert").invoke(meshGL);
			long numTri = (long) fn("manifold_meshgl64_num_tri").invoke(meshGL);
			long numProp = (long) fn("manifold_meshgl64_num_prop").invoke(meshGL);

			double[] vertices = new double[(int) (numVert * 3)];
			long[] triangles = new long[(int) (numTri * 3)];

			if (numVert > 0) {
				// Buffers are borrowed from meshGL and only valid while it lives.
				long vertLen = (long) fn("manifold_meshgl64_vert_properties_length").invoke(meshGL);
				MemorySegment props = ((MemorySegment) fn("manifold_meshgl64_vert_properties").invoke(meshGL))
						.reinterpret(vertLen * Double.BYTES);

				for (int i = 0; i < numVert; i++)
					for (int j = 0; (j < 3) && (j < numProp); j++)
						vertices[i * 3 + j] = props.getAtIndex(ValueLayout.JAVA_DOUBLE, i * numProp + j);
			}

			if (numTri > 0) {
				long triLen = (long) fn("manifold_meshgl64_tri_length").invoke(meshGL);
				MemorySegment idx = ((MemorySegment) fn("manifold_meshgl64_tri_verts").invoke(meshGL))
						.reinterpret(triLen * Long.BYTES);

				for (int i = 0; i < triLen; i++)
					triangles[i] = idx.getAtIndex(ValueLayout.JAVA_LONG, i);
			}

			return new MeshData64(vertices, triangles, (int) numVert, (int) numTri);

		} finally {
			safeDeleteMeshGL64(meshGL);
		}
	}
	// =========================================================================
	// Data records
	// =========================================================================

	/**
	 * Double-precision mesh data returned by {@link #exportMeshGL64(MemorySegment)}
	 * and accepted by {@link #importMeshGL(double[], long[], long, long)}.
	 *
	 * <p>
	 * {@code vertices} is a flat XYZ array of length {@code vertCount * 3}.
	 * {@code triangles} is a flat index array of length {@code triCount * 3}.
	 */
	public record MeshData64(double[] vertices, long[] triangles, int vertCount, int triCount) {
	}

	// =========================================================================
	// File I/O — STL and 3MF
	// =========================================================================

	/**
	 * Lightweight, zero-dependency library for reading and writing STL and 3MF mesh
	 * files, backed by the 64-bit mesh pipeline throughout.
	 *
	 * <h2>Export (MemorySegment → File)</h2>
	 * 
	 * <pre>{@code
	 * // STL — single mesh
	 * bindings.exportSTL(manifoldSegment, new File("out.stl"));
	 *
	 * // 3MF — one or many meshes
	 * ArrayList<MemorySegment> meshes = new ArrayList<>();
	 * meshes.add(bodySegment);
	 * meshes.add(lidSegment);
	 * bindings.export3MF(meshes, new File("assembly.3mf"));
	 * }</pre>
	 *
	 * <h2>Import (File → MemorySegment)</h2>
	 * 
	 * <pre>{@code
	 * MemorySegment manifold = bindings.importSTL(new File("model.stl"));
	 * ArrayList<MemorySegment> meshes = bindings.import3MF(new File("assembly.3mf"));
	 * }</pre>
	 *
	 * <h2>STL notes</h2>
	 * <ul>
	 * <li>Reads both binary and ASCII STL.</li>
	 * <li>Writes binary STL.</li>
	 * <li>Duplicate vertices are welded inside {@link #importMeshGL}.</li>
	 * <li>Face normals on export are computed from vertex positions.</li>
	 * </ul>
	 *
	 * <h2>3MF notes</h2>
	 * <ul>
	 * <li>Reads and writes 3MF Core Specification 1.x.</li>
	 * <li>Each {@code <object>} maps to exactly one {@link MemorySegment}.</li>
	 * <li>Indexed — vertex topology is fully preserved on round-trip.</li>
	 * </ul>
	 */

	// =========================================================================
	// Export API (MemorySegment → File)
	// =========================================================================

	/**
	 * Exports the manifold to a binary STL file.
	 *
	 * <p>
	 * Vertex positions are written as {@code float} (single-precision) because the
	 * STL binary format specifies 32-bit IEEE 754 floats. Sub-micron precision is
	 * not preserved, but this matches universal STL toolchain expectations.
	 */
	public void exportSTL(MemorySegment manifold, File file) throws Throwable {
		MeshData64 mesh = this.exportMeshGL64(manifold);
		writeBinarySTL(mesh.vertices(), mesh.triangles(), mesh.vertCount(), mesh.triCount(), file);
	}

	/**
	 * Exports one or more manifolds to a single 3MF file.
	 *
	 * <p>
	 * Each {@link MemorySegment} becomes a separate {@code <object>} element and a
	 * corresponding {@code <item>} in the {@code <build>} section.
	 */
	public void export3MF(ArrayList<MemorySegment> manifolds, File file) throws Throwable {
		if (manifolds == null || manifolds.isEmpty())
			throw new IllegalArgumentException("manifolds list must not be empty");

		List<MeshData64> meshes = new ArrayList<>(manifolds.size());
		for (MemorySegment seg : manifolds)
			meshes.add(this.exportMeshGL64(seg));

		write3MFInternal(meshes, file);
	}

	// =========================================================================
	// Import API (File → MemorySegment)
	// =========================================================================

	/**
	 * Reads an STL file (binary or ASCII) and imports it into a new manifold.
	 */
	public MemorySegment importSTL(File file) throws Throwable {
		RawMesh raw = parseSTL(file);
		return this.importMeshGL64(raw.vertices, raw.triangles, raw.vertices.length / 3L, raw.triangles.length / 3L);
	}

	/**
	 * Reads a 3MF file and imports each {@code <object>} as a separate manifold.
	 */
	public ArrayList<MemorySegment> import3MF(File file) throws Throwable {
		List<RawMesh> objects = parse3MF(file);
		ArrayList<MemorySegment> result = new ArrayList<>(objects.size());
		for (RawMesh raw : objects)
			result.add(this.importMeshGL64(raw.vertices, raw.triangles, raw.vertices.length / 3L,
					raw.triangles.length / 3L));
		return result;
	}

	// =========================================================================
	// STL – format detection
	// =========================================================================

	private static boolean looksLikeAsciiSTL(byte[] header80, long fileSize) {
		String prefix = new String(header80, 0, Math.min(5, header80.length), StandardCharsets.US_ASCII).trim()
				.toLowerCase();
		if (!prefix.startsWith("solid"))
			return false;
		if (fileSize < 134)
			return true;
		return (fileSize - 84) % 50 != 0;
	}

	// =========================================================================
	// STL – binary read (produces double[]/long[] directly)
	// =========================================================================

	private static RawMesh readBinarySTL(File file) throws IOException {
		try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(file)))) {
			dis.skipNBytes(80);
			int triCount = Integer.reverseBytes(dis.readInt());

			double[] verts = new double[triCount * 9];
			long[] tris = new long[triCount * 3];

			byte[] buf = new byte[50];
			ByteBuffer bb = ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN);

			for (int i = 0; i < triCount; i++) {
				dis.readFully(buf);
				bb.rewind();
				bb.position(12); // skip 3-float normal
				int base = i * 9;
				verts[base] = bb.getFloat();
				verts[base + 1] = bb.getFloat();
				verts[base + 2] = bb.getFloat();
				verts[base + 3] = bb.getFloat();
				verts[base + 4] = bb.getFloat();
				verts[base + 5] = bb.getFloat();
				verts[base + 6] = bb.getFloat();
				verts[base + 7] = bb.getFloat();
				verts[base + 8] = bb.getFloat();
				tris[i * 3] = i * 3;
				tris[i * 3 + 1] = i * 3 + 1;
				tris[i * 3 + 2] = i * 3 + 2;
			}
			return new RawMesh(verts, tris);
		}
	}

	// =========================================================================
	// STL – ASCII read (produces double[]/long[] directly)
	// =========================================================================

	private static RawMesh readAsciiSTL(File file) throws IOException {
		List<Double> vList = new ArrayList<>();
		List<Long> tList = new ArrayList<>();
		long vertIdx = 0;

		try (BufferedReader br = new BufferedReader(
				new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8))) {
			String line;
			long faceStart = -1;
			while ((line = br.readLine()) != null) {
				line = line.trim().toLowerCase();
				if (line.startsWith("facet normal")) {
					faceStart = vertIdx;
				} else if (line.startsWith("vertex ")) {
					String[] parts = line.split("\\s+");
					vList.add(Double.parseDouble(parts[1]));
					vList.add(Double.parseDouble(parts[2]));
					vList.add(Double.parseDouble(parts[3]));
					vertIdx++;
				} else if (line.startsWith("endfacet")) {
					tList.add(faceStart);
					tList.add(faceStart + 1);
					tList.add(faceStart + 2);
				}
			}
		}

		double[] verts = new double[vList.size()];
		for (int i = 0; i < vList.size(); i++)
			verts[i] = vList.get(i);
		long[] tris = new long[tList.size()];
		for (int i = 0; i < tList.size(); i++)
			tris[i] = tList.get(i);
		return new RawMesh(verts, tris);
	}

	// =========================================================================
	// STL – combined entry point
	// =========================================================================

	private static RawMesh parseSTL(File file) throws IOException {
		try (FileInputStream fis = new FileInputStream(file)) {
			byte[] header = fis.readNBytes(80);
			return looksLikeAsciiSTL(header, file.length()) ? readAsciiSTL(file) : readBinarySTL(file);
		}
	}

	// =========================================================================
	// STL – binary write (accepts double[]/long[])
	// =========================================================================

	private static void writeBinarySTL(double[] verts, long[] tris, int vertCount, int triCount, File file)
			throws IOException {
		ByteBuffer buf = ByteBuffer.allocate(84 + triCount * 50).order(ByteOrder.LITTLE_ENDIAN);

		byte[] header = new byte[80];
		byte[] tag = "Manifold3D-Java MeshIO STL Export".getBytes(StandardCharsets.US_ASCII);
		System.arraycopy(tag, 0, header, 0, Math.min(tag.length, 80));
		buf.put(header);
		buf.putInt(triCount);

		for (int i = 0; i < triCount; i++) {
			int i0 = (int) tris[i * 3] * 3;
			int i1 = (int) tris[i * 3 + 1] * 3;
			int i2 = (int) tris[i * 3 + 2] * 3;

			float ax = (float) verts[i0], ay = (float) verts[i0 + 1], az = (float) verts[i0 + 2];
			float bx = (float) verts[i1], by = (float) verts[i1 + 1], bz = (float) verts[i1 + 2];
			float cx = (float) verts[i2], cy = (float) verts[i2 + 1], cz = (float) verts[i2 + 2];

			float ux = bx - ax, uy = by - ay, uz = bz - az;
			float vx = cx - ax, vy = cy - ay, vz = cz - az;
			float nx = uy * vz - uz * vy;
			float ny = uz * vx - ux * vz;
			float nz = ux * vy - uy * vx;
			float len = (float) Math.sqrt(nx * nx + ny * ny + nz * nz);
			if (len > 1e-12f) {
				nx /= len;
				ny /= len;
				nz /= len;
			}

			buf.putFloat(nx);
			buf.putFloat(ny);
			buf.putFloat(nz);
			buf.putFloat(ax);
			buf.putFloat(ay);
			buf.putFloat(az);
			buf.putFloat(bx);
			buf.putFloat(by);
			buf.putFloat(bz);
			buf.putFloat(cx);
			buf.putFloat(cy);
			buf.putFloat(cz);
			buf.putShort((short) 0);
		}

		try (FileOutputStream fos = new FileOutputStream(file)) {
			fos.write(buf.array());
		}
	}

	// =========================================================================
	// 3MF – read
	// =========================================================================

	private static List<RawMesh> parse3MF(File file) throws IOException {
		return parseModelXml(extract3DModel(file));
	}

	private static String extract3DModel(File file) throws IOException {
		try (ZipInputStream zis = new ZipInputStream(new FileInputStream(file))) {
			ZipEntry entry;
			while ((entry = zis.getNextEntry()) != null) {
				if (entry.getName().replace('\\', '/').equalsIgnoreCase("3D/3dmodel.model")) {
					return new String(zis.readAllBytes(), StandardCharsets.UTF_8);
				}
				zis.closeEntry();
			}
		}
		throw new IOException("No 3D/3dmodel.model entry found in 3MF archive: " + file);
	}

	/**
	 * Hand-rolled XML pull parser for the 3MF model document. Returns one
	 * {@link RawMesh} per {@code <object>} element. Triangle indices within each
	 * object are local (start at 0).
	 */
	private static List<RawMesh> parseModelXml(String xml) {
		List<RawMesh> result = new ArrayList<>();
		List<Double> verts = null;
		List<Long> tris = null;
		int pos = 0, len = xml.length();

		while (pos < len) {
			int tagStart = xml.indexOf('<', pos);
			if (tagStart < 0)
				break;
			int tagEnd = xml.indexOf('>', tagStart);
			if (tagEnd < 0)
				break;

			String tag = xml.substring(tagStart + 1, tagEnd).trim();
			pos = tagEnd + 1;

			if (tag.startsWith("object") && !tag.startsWith("/object")) {
				verts = new ArrayList<>();
				tris = new ArrayList<>();
			} else if (tag.equals("/object")) {
				if (verts != null && tris != null && !verts.isEmpty()) {
					double[] va = new double[verts.size()];
					for (int i = 0; i < verts.size(); i++)
						va[i] = verts.get(i);
					long[] ta = new long[tris.size()];
					for (int i = 0; i < tris.size(); i++)
						ta[i] = tris.get(i);
					result.add(new RawMesh(va, ta));
				}
				verts = null;
				tris = null;
			} else if (verts != null && tag.startsWith("vertex") && !tag.startsWith("vertices")) {
				verts.add((double) attrFloat(tag, "x"));
				verts.add((double) attrFloat(tag, "y"));
				verts.add((double) attrFloat(tag, "z"));
			} else if (tris != null && tag.startsWith("triangle") && !tag.startsWith("triangles")) {
				tris.add((long) attrInt(tag, "v1"));
				tris.add((long) attrInt(tag, "v2"));
				tris.add((long) attrInt(tag, "v3"));
			}
		}

		if (result.isEmpty())
			throw new IllegalArgumentException("3MF file contains no <object> elements with geometry");
		return result;
	}

	// =========================================================================
	// 3MF – write
	// =========================================================================

	private static void write3MFInternal(List<MeshData64> meshes, File file) throws IOException {
		writeZip3MF(file, build3MFModelXml(meshes));
	}

	private static byte[] build3MFModelXml(List<MeshData64> meshes) {
		int cap = 512;
		for (MeshData64 m : meshes)
			cap += 200 + m.vertCount() * 60 + m.triCount() * 55;
		StringBuilder sb = new StringBuilder(cap);

		sb.append("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
		sb.append("<model unit=\"millimeter\" xml:lang=\"en-US\"\n");
		sb.append("  xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\">\n");
		sb.append("  <resources>\n");

		for (int objIdx = 0; objIdx < meshes.size(); objIdx++) {
			MeshData64 mesh = meshes.get(objIdx);
			int objectId = objIdx + 1;
			double[] v = mesh.vertices();
			long[] t = mesh.triangles();

			sb.append("    <object id=\"").append(objectId).append("\" type=\"model\">\n");
			sb.append("      <mesh>\n");
			sb.append("        <vertices>\n");
			for (int i = 0; i < mesh.vertCount(); i++) {
				sb.append("          <vertex x=\"").append(v[i * 3]).append("\" y=\"").append(v[i * 3 + 1])
						.append("\" z=\"").append(v[i * 3 + 2]).append("\"/>\n");
			}
			sb.append("        </vertices>\n");
			sb.append("        <triangles>\n");
			for (int i = 0; i < mesh.triCount(); i++) {
				sb.append("          <triangle v1=\"").append(t[i * 3]).append("\" v2=\"").append(t[i * 3 + 1])
						.append("\" v3=\"").append(t[i * 3 + 2]).append("\"/>\n");
			}
			sb.append("        </triangles>\n");
			sb.append("      </mesh>\n");
			sb.append("    </object>\n");
		}

		sb.append("  </resources>\n");
		sb.append("  <build>\n");
		for (int objIdx = 0; objIdx < meshes.size(); objIdx++)
			sb.append("    <item objectid=\"").append(objIdx + 1).append("\"/>\n");
		sb.append("  </build>\n");
		sb.append("</model>\n");

		return sb.toString().getBytes(StandardCharsets.UTF_8);
	}

	// =========================================================================
	// 3MF – ZIP container writer
	// =========================================================================

	private static void writeZip3MF(File file, byte[] modelBytes) throws IOException {
		try (ZipOutputStream zos = new ZipOutputStream(new FileOutputStream(file))) {
			zos.setMethod(ZipOutputStream.DEFLATED);
			zos.setLevel(Deflater.BEST_SPEED);

			putZipEntry(zos, "[Content_Types].xml",
					("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
							+ "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">\n"
							+ "  <Default Extension=\"rels\""
							+ " ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>\n"
							+ "  <Default Extension=\"model\""
							+ " ContentType=\"application/vnd.ms-package.3dmanufacturing-3dmodel+xml\"/>\n"
							+ "</Types>\n").getBytes(StandardCharsets.UTF_8));

			putZipEntry(zos, "_rels/.rels", ("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" + "<Relationships"
					+ " xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\n" + "  <Relationship"
					+ " Type=\"http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel\""
					+ " Target=\"/3D/3dmodel.model\" Id=\"rel0\"/>\n" + "</Relationships>\n")
					.getBytes(StandardCharsets.UTF_8));

			putZipEntry(zos, "3D/3dmodel.model", modelBytes);
		}
	}

	private static void putZipEntry(ZipOutputStream zos, String name, byte[] data) throws IOException {
		zos.putNextEntry(new ZipEntry(name));
		zos.write(data);
		zos.closeEntry();
	}

	// =========================================================================
	// XML attribute helpers
	// =========================================================================

	private static float attrFloat(String tag, String attr) {
		return Float.parseFloat(attrString(tag, attr));
	}

	private static int attrInt(String tag, String attr) {
		return Integer.parseInt(attrString(tag, attr));
	}

	private static String attrString(String tag, String attr) {
		int idx = tag.indexOf(attr + "=");
		if (idx < 0)
			throw new IllegalArgumentException("Attribute '" + attr + "' not found in tag: " + tag);

		int valStart = idx + attr.length() + 1;
		char quote = tag.charAt(valStart);
		if (quote != '"' && quote != '\'') {
			int end = valStart;
			while (end < tag.length()) {
				char c = tag.charAt(end);
				if (c == ' ' || c == '/' || c == '>')
					break;
				end++;
			}
			return tag.substring(valStart, end);
		}
		int valEnd = tag.indexOf(quote, valStart + 1);
		if (valEnd < 0)
			throw new IllegalArgumentException("Unterminated attribute for '" + attr + "' in: " + tag);
		return tag.substring(valStart + 1, valEnd);
	}

	// =========================================================================
	// Internal mesh data holder (used only during file parsing)
	// =========================================================================

	/** Flat double[]/long[] mesh arrays used only during STL/3MF parsing. */
	private record RawMesh(double[] vertices, long[] triangles) {
	}
}
