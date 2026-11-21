package com.cadoodlecad.manifold;

import java.io.File;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
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

@SuppressWarnings("preview")
public class ManifoldBindings {
	public enum ManifoldError {
		NO_ERROR, NON_FINITE_VERTEX, NOT_MANIFOLD, VERTEX_INDEX_OUT_OF_BOUNDS, PROPERTIES_WRONG_LENGTH,
		MISSING_POSITION_PROPERTIES, MERGE_VECTORS_DIFFERENT_LENGTHS, MERGE_INDEX_OUT_OF_BOUNDS, TRANSFORM_WRONG_LENGTH,
		RUN_INDEX_WRONG_LENGTH, FACE_ID_WRONG_LENGTH, INVALID_CONSTRUCTION, RESULT_TOO_LARGE;

		public static ManifoldError fromInt(int code) {
			ManifoldError[] values = values();
			if (code < 0 || code >= values.length)
				throw new IllegalArgumentException("Unknown ManifoldError code: " + code);
			return values[code];
		}
	}
	// ===== Constants to add alongside the existing OPTYPE_* constants =====

	// ManifoldFillRule values
	// ===== Enums to add alongside ManifoldError =====

	public enum FillRule {
		EVEN_ODD, NON_ZERO, POSITIVE, NEGATIVE;

		public int toInt() {
			return ordinal();
		}
	}

	public enum JoinType {
		SQUARE, ROUND, MITER;

		public int toInt() {
			return ordinal();
		}
	}

	private static boolean loaded = false;

	private static final Linker LINKER = Linker.nativeLinker();

	private final SymbolLookup library;
	private final Map<String, MethodHandle> functions = new HashMap<>();

	private static final StructLayout VEC3_LAYOUT = MemoryLayout.structLayout(ValueLayout.JAVA_DOUBLE,
			ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE);

	private static final StructLayout PAIR_LAYOUT = MemoryLayout.structLayout(ValueLayout.ADDRESS.withName("first"),
			ValueLayout.ADDRESS.withName("second"));

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
			System.out.println("Loading Library for " + os + " on " + arch);
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

			// Copy libraries to lib directory
			java.io.File libsDir = cacheDirectory == null ? Files.createTempDirectory("manifold3d").toFile()
					: cacheDirectory;
			if (cacheDirectory == null)
				libsDir.deleteOnExit();
			if (!libsDir.exists())
				libsDir.mkdirs();

			java.io.File libFile = new java.io.File(libsDir, fullName);

			// Check timestamp, copy only newer
			boolean needsCopy = !libFile.exists();
			if (needsCopy) {

				try (java.io.InputStream in = ManifoldBindings.class
						.getResourceAsStream("/manifold3d/natives/" + platform + "/" + fullName)) {
					if (in == null)
						throw new RuntimeException("Library not found: " + fullName + " for platform " + platform);

					java.nio.file.Files.copy(in, libFile.toPath(), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
					System.out.println("Extracted to libs/: " + fullName);
					if (cacheDirectory == null)
						libFile.deleteOnExit();
				}
			} else {
				System.out.println("Copy not performed, already in cache");
			}
			System.out.println("Loading library " + libFile.getAbsolutePath());
			System.load(libFile.getAbsolutePath());
			loaded = true;
		} catch (Exception e) {
			throw new RuntimeException("Failed to load: " + libName, e);
		}
	}

	public static void loadNativeLibrarys(File cacheDirectory) throws Exception {
		loadNativeLibrary("libmanifold", cacheDirectory);
		loadNativeLibrary("libmanifoldc", cacheDirectory);
	}

	public ManifoldBindings() throws Exception {
		this((File) null);
	}

	public ManifoldBindings(File cacheDirectory) throws Exception {
		if (!isNativeLibraryLoaded()) {
			loadNativeLibrarys(cacheDirectory);
		}

		this.library = SymbolLookup.loaderLookup();

		// ===== Allocation/construction =====

		// ManifoldManifold* manifold_of_meshgl64(void* mem, ManifoldMeshGL64* mesh);
		load("manifold_of_meshgl64", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldManifold* manifold_alloc_manifold();
		load("manifold_alloc_manifold", ValueLayout.ADDRESS);

		// ManifoldMeshGL64* manifold_alloc_meshgl64();
		load("manifold_alloc_meshgl64", ValueLayout.ADDRESS);

		// ManifoldManifoldVec* manifold_alloc_manifold_vec();
		load("manifold_alloc_manifold_vec", ValueLayout.ADDRESS);

		// ManifoldManifoldVec* manifold_manifold_empty_vec(void* mem);
		load("manifold_manifold_empty_vec", ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldBox* manifold_alloc_box();
		load("manifold_alloc_box", ValueLayout.ADDRESS);

		// ===== Primitives =====

		// ManifoldManifold* manifold_copy(void* mem, ManifoldManifold* m);
		load("manifold_copy", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldManifold* manifold_empty(void* mem);
		load("manifold_empty", ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldManifold* manifold_tetrahedron(void* mem);
		load("manifold_tetrahedron", ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldManifold* manifold_cube(void* mem, double x, double y, double z, int
		// center);
		load("manifold_cube", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE,
				ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_INT);

		// ManifoldManifold* manifold_sphere(void* mem, double radius, int
		// circular_segments);
		load("manifold_sphere", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE,
				ValueLayout.JAVA_INT);

		// ManifoldManifold* manifold_cylinder(void* mem, double height, double
		// radius_low, double radius_high, int circular_segments, int center);
		load("manifold_cylinder", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE,
				ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT);

		// ===== Boolean operations =====

		// ManifoldManifold* manifold_boolean(void* mem, ManifoldManifold* a,
		// ManifoldManifold* b, ManifoldOpType op);
		load("manifold_boolean", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
				ValueLayout.JAVA_INT);

		// ManifoldManifold* manifold_union(void* mem, ManifoldManifold* a,
		// ManifoldManifold* b);
		load("manifold_union", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldManifold* manifold_difference(void* mem, ManifoldManifold* a,
		// ManifoldManifold* b);
		load("manifold_difference", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldManifold* manifold_minkowski_sum(void* mem, ManifoldManifold* a,
		// ManifoldManifold* b);
		load("manifold_minkowski_sum", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
				ValueLayout.ADDRESS);

		// ManifoldManifold* manifold_minkowski_difference(void* mem, ManifoldManifold*
		// a, ManifoldManifold* b);
		load("manifold_minkowski_difference", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
				ValueLayout.ADDRESS);

		// ManifoldManifold* manifold_intersection(void* mem, ManifoldManifold* a,
		// ManifoldManifold* b);
		load("manifold_intersection", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
				ValueLayout.ADDRESS);

		// ManifoldManifold* manifold_batch_boolean(void* mem, ManifoldManifoldVec* ms,
		// ManifoldOpType op);
		load("manifold_batch_boolean", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
				ValueLayout.JAVA_INT);

		// ===== Vector operations =====

		// void manifold_manifold_vec_push_back(ManifoldManifoldVec* ms,
		// ManifoldManifold* m);
		loadVoid("manifold_manifold_vec_push_back", ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ===== Transforms =====

		// ManifoldManifold* manifold_transform(void* mem, ManifoldManifold* m, double
		// x1, double y1, double z1, double x2, double y2, double z2, double x3, double
		// y3, double z3, double x4, double y4, double z4);
		load("manifold_transform", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
				ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE,
				ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE,
				ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE);

		// ManifoldManifold* manifold_translate(void* mem, ManifoldManifold* m, double
		// x, double y, double z);
		load("manifold_translate", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
				ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE);

		// ManifoldManifold* manifold_scale(void* mem, ManifoldManifold* m, double x,
		// double y, double z);
		load("manifold_scale", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE,
				ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE);

		// ManifoldManifold* manifold_mirror(void* mem, ManifoldManifold* m, double nx,
		// double ny, double nz);
		load("manifold_mirror", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE,
				ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE);

		// ManifoldManifold* manifold_rotate(void* mem, ManifoldManifold* m, double x,
		// double y, double z);
		load("manifold_rotate", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE,
				ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE);

		// ===== Refinement =====

		// ManifoldManifold* manifold_refine(void* mem, ManifoldManifold* m, int
		// refine);
		load("manifold_refine", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT);

		// ManifoldManifold* manifold_refine_to_length(void* mem, ManifoldManifold* m,
		// double length);
		load("manifold_refine_to_length", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
				ValueLayout.JAVA_DOUBLE);

		// ManifoldManifold* manifold_refine_to_tolerance(void* mem, ManifoldManifold*
		// m, double tolerance);
		load("manifold_refine_to_tolerance", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
				ValueLayout.JAVA_DOUBLE);
		load("manifold_simplify", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
				ValueLayout.JAVA_DOUBLE);
		// ManifoldManifold* manifold_smooth_by_normals(void* mem, ManifoldManifold* m,
		// int normal_idx);
		load("manifold_smooth_by_normals", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
				ValueLayout.JAVA_INT);

		// ManifoldManifold* manifold_calculate_normals(void* mem, ManifoldManifold* m,
		// int normal_idx, double min_sharp_angle);
		load("manifold_calculate_normals", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
				ValueLayout.JAVA_INT, ValueLayout.JAVA_DOUBLE);

		// ===== Split/Trim operations =====

		// ManifoldManifoldPair manifold_split(void* mem_first, void* mem_second,
		// ManifoldManifold* a, ManifoldManifold* b);
		load("manifold_split", PAIR_LAYOUT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
				ValueLayout.ADDRESS);

		// ManifoldManifold* manifold_trim_by_plane(void* mem, ManifoldManifold* m,
		// double normal_x, double normal_y, double normal_z, double offset);
		load("manifold_trim_by_plane", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
				ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE);

		// ManifoldManifoldPair manifold_split_by_plane(void* mem_first, void*
		// mem_second,
		// ManifoldManifold* m, double normal_x, double normal_y, double normal_z,
		// double offset);
		load("manifold_split_by_plane", PAIR_LAYOUT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
				ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE);

		// ManifoldPolygons* manifold_slice(void* mem, ManifoldManifold* m, double
		// height);
		load("manifold_slice", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE);

		load("manifold_alloc_polygons", ValueLayout.ADDRESS);
		// size_t manifold_polygons_length(ManifoldPolygons* ps);
		load("manifold_polygons_length", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);
		// size_t manifold_polygons_simple_length(ManifoldPolygons* ps, size_t idx);
		load("manifold_polygons_simple_length", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);
		// ManifoldVec2 manifold_polygons_get_point(ManifoldPolygons* ps, size_t
		// simple_idx, size_t pt_idx);
		load("manifold_polygons_get_point", MemoryLayout.structLayout(ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE),
				ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG);

		// ===== Analysis =====

		// double manifold_volume(ManifoldManifold* m);
		load("manifold_volume", ValueLayout.JAVA_DOUBLE, ValueLayout.ADDRESS);

		// double manifold_surface_area(ManifoldManifold* m);
		load("manifold_surface_area", ValueLayout.JAVA_DOUBLE, ValueLayout.ADDRESS);

		// ManifoldBox* manifold_bounding_box(void* mem, ManifoldManifold* m);
		load("manifold_bounding_box", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldVec3 manifold_box_min(ManifoldBox* b);
		load("manifold_box_min", VEC3_LAYOUT, ValueLayout.ADDRESS);

		// ManifoldVec3 manifold_box_max(ManifoldBox* b);
		load("manifold_box_max", VEC3_LAYOUT, ValueLayout.ADDRESS);

		// ManifoldVec3 manifold_box_center(ManifoldBox* b);
		load("manifold_box_center", VEC3_LAYOUT, ValueLayout.ADDRESS);

		// ManifoldVec3 manifold_box_dimensions(ManifoldBox* b);
		load("manifold_box_dimensions", VEC3_LAYOUT, ValueLayout.ADDRESS);

		// double manifold_epsilon(ManifoldManifold* m);
		load("manifold_epsilon", ValueLayout.JAVA_DOUBLE, ValueLayout.ADDRESS);

		// int manifold_genus(ManifoldManifold* m);
		load("manifold_genus", ValueLayout.JAVA_INT, ValueLayout.ADDRESS);

		// ===== Composition/Decomposition =====

		// ManifoldManifold* manifold_compose(void* mem, ManifoldManifoldVec* ms);
		load("manifold_compose", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldManifoldVec* manifold_decompose(void* mem, ManifoldManifold* m);
		load("manifold_decompose", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldManifold* manifold_as_original(void* mem, ManifoldManifold* m);
		load("manifold_as_original", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ===== Hull =====

		// ManifoldManifold* manifold_hull(void* mem, ManifoldManifold* m);
		load("manifold_hull", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldManifold* manifold_batch_hull(void* mem, ManifoldManifoldVec* ms);
		load("manifold_batch_hull", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ===== Info =====

		// size_t manifold_manifold_size();
		load("manifold_manifold_size", ValueLayout.JAVA_LONG);

		// size_t manifold_meshgl64_size();
		load("manifold_meshgl64_size", ValueLayout.JAVA_LONG);

		// size_t manifold_num_edge(ManifoldManifold* m);
		load("manifold_num_edge", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);

		// size_t manifold_num_prop(ManifoldManifold* m);
		load("manifold_num_prop", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);

		// size_t manifold_num_vert(ManifoldManifold* m);
		load("manifold_num_vert", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);

		// size_t manifold_num_tri(ManifoldManifold* m);
		load("manifold_num_tri", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);

		// ManifoldError manifold_status(ManifoldManifold* m);
		load("manifold_status", ValueLayout.JAVA_INT, ValueLayout.ADDRESS);

		// int manifold_is_empty(ManifoldManifold* m);
		load("manifold_is_empty", ValueLayout.JAVA_INT, ValueLayout.ADDRESS);

		// size_t manifold_manifold_vec_size();
		load("manifold_manifold_vec_size", ValueLayout.JAVA_LONG);
		// size_t manifold_manifold_vec_length(ManifoldManifoldVec* ms);
		load("manifold_manifold_vec_length", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);
		// ManifoldManifold* manifold_manifold_vec_get(void* mem, ManifoldManifoldVec*
		// ms, size_t idx);
		load("manifold_manifold_vec_get", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
				ValueLayout.JAVA_LONG);

		// ===== Mesh export (64-bit only) =====

		// ManifoldMeshGL64* manifold_get_meshgl64(void* mem, ManifoldManifold* m);
		load("manifold_get_meshgl64", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// size_t manifold_meshgl64_num_vert(ManifoldMeshGL64* m);
		load("manifold_meshgl64_num_vert", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);

		// size_t manifold_meshgl64_num_tri(ManifoldMeshGL64* m);
		load("manifold_meshgl64_num_tri", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);

		// size_t manifold_meshgl64_num_prop(ManifoldMeshGL64* m);
		load("manifold_meshgl64_num_prop", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);

		// size_t manifold_meshgl64_vert_properties_length(ManifoldMeshGL64* m);
		load("manifold_meshgl64_vert_properties_length", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);

		// size_t manifold_meshgl64_tri_length(ManifoldMeshGL64* m);
		load("manifold_meshgl64_tri_length", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);

		// double* manifold_meshgl64_vert_properties(void* mem, ManifoldMeshGL64* m);
		load("manifold_meshgl64_vert_properties", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// uint64_t* manifold_meshgl64_tri_verts(void* mem, ManifoldMeshGL64* m);
		load("manifold_meshgl64_tri_verts", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldMeshGL64* manifold_meshgl64_merge(void* mem, ManifoldMeshGL64* m);
		load("manifold_meshgl64_merge", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldMeshGL64* manifold_meshgl64(void* mem, double* vert_props, size_t
		// n_verts, size_t n_props, uint64_t* tri_verts, size_t n_tris);
		load("manifold_meshgl64", ValueLayout.ADDRESS, ValueLayout.ADDRESS, // void* mem
				ValueLayout.ADDRESS, // double* vert_props
				ValueLayout.JAVA_LONG, // size_t n_verts
				ValueLayout.JAVA_LONG, // size_t n_props
				ValueLayout.ADDRESS, // uint64_t* tri_verts
				ValueLayout.JAVA_LONG // size_t n_tris
		);

		// ===== Cleanup =====

		// void manifold_delete_manifold(ManifoldManifold* m);
		loadVoid("manifold_delete_manifold", ValueLayout.ADDRESS);

		// void manifold_delete_meshgl64(ManifoldMeshGL64* m);
		loadVoid("manifold_delete_meshgl64", ValueLayout.ADDRESS);

		// void manifold_delete_manifold_vec(ManifoldManifoldVec* ms);
		loadVoid("manifold_delete_manifold_vec", ValueLayout.ADDRESS);

		// void manifold_delete_box(ManifoldBox* b);
		loadVoid("manifold_delete_box", ValueLayout.ADDRESS);

		// void manifold_delete_polygons(ManifoldPolygons* p);
		loadVoid("manifold_delete_polygons", ValueLayout.ADDRESS);

		// ManifoldManifold* manifold_hull_pts(void* mem, ManifoldVec3* pts, size_t
		// count);
		load("manifold_hull_pts", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);
		// ===== Static Quality Globals =====

		// void manifold_set_circular_segments(int number);
		loadVoid("manifold_set_circular_segments", ValueLayout.JAVA_INT);

		// void manifold_set_min_circular_angle(double degrees);
		loadVoid("manifold_set_min_circular_angle", ValueLayout.JAVA_DOUBLE);

		// int manifold_get_circular_segments(double radius);
		load("manifold_get_circular_segments", ValueLayout.JAVA_INT, ValueLayout.JAVA_DOUBLE);

		// ManifoldManifold* manifold_smooth_out(void* mem, ManifoldManifold* m,
		// double minSharpAngle, double minSmoothness);
		load("manifold_smooth_out", ValueLayout.ADDRESS, ValueLayout.ADDRESS, // void* mem
				ValueLayout.ADDRESS, // ManifoldManifold* m
				ValueLayout.JAVA_DOUBLE, // minSharpAngle
				ValueLayout.JAVA_DOUBLE // minSmoothness
		);
		// ===== New CrossSection bindings to add in the constructor =====

		// ManifoldCrossSection* manifold_alloc_cross_section();
		load("manifold_alloc_cross_section", ValueLayout.ADDRESS);

		// size_t manifold_cross_section_size();
		load("manifold_cross_section_size", ValueLayout.JAVA_LONG);

		// ManifoldCrossSection* manifold_cross_section_of_polygons(void* mem,
		// ManifoldPolygons* ps, ManifoldFillRule fr);
		load("manifold_cross_section_of_polygons", ValueLayout.ADDRESS, ValueLayout.ADDRESS, // void* mem
				ValueLayout.ADDRESS, // ManifoldPolygons* ps
				ValueLayout.JAVA_INT); // ManifoldFillRule fr

		// ManifoldCrossSection* manifold_cross_section_offset(void* mem,
		// ManifoldCrossSection* cs,
//		     double delta, ManifoldJoinType jt, double miter_limit, int circular_segments);
		load("manifold_cross_section_offset", ValueLayout.ADDRESS, ValueLayout.ADDRESS, // void* mem
				ValueLayout.ADDRESS, // ManifoldCrossSection* cs
				ValueLayout.JAVA_DOUBLE, // delta
				ValueLayout.JAVA_INT, // ManifoldJoinType jt
				ValueLayout.JAVA_DOUBLE, // miter_limit
				ValueLayout.JAVA_INT); // circular_segments

		// ManifoldPolygons* manifold_cross_section_to_polygons(void* mem,
		// ManifoldCrossSection* cs);
		load("manifold_cross_section_to_polygons", ValueLayout.ADDRESS, ValueLayout.ADDRESS, // void* mem
				ValueLayout.ADDRESS); // ManifoldCrossSection* cs

		// void manifold_delete_cross_section(ManifoldCrossSection* cs);
		loadVoid("manifold_delete_cross_section", ValueLayout.ADDRESS);

		// ===== Ray casting =====

		// ManifoldRayHitVec* manifold_ray_cast(void* mem, ManifoldManifold* m,
		// double ox, oy, oz, ex, ey, ez);
		load("manifold_ray_cast", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
				ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE,
				ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE);

		// size_t manifold_ray_hit_vec_length(ManifoldRayHitVec* v);
		load("manifold_ray_hit_vec_length", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);

		// ManifoldRayHit manifold_ray_hit_vec_get(ManifoldRayHitVec* v, size_t idx);
		// struct: uint64 face_id, double distance, vec3 position, vec3 normal
		load("manifold_ray_hit_vec_get",
				MemoryLayout.structLayout(ValueLayout.JAVA_LONG.withName("face_id"),
						ValueLayout.JAVA_DOUBLE.withName("distance"),
						MemoryLayout.structLayout(ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE,
								ValueLayout.JAVA_DOUBLE).withName("position"),
						MemoryLayout.structLayout(ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE,
								ValueLayout.JAVA_DOUBLE).withName("normal")),
				ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);

		// double manifold_min_gap(ManifoldManifold* m, ManifoldManifold* other,
		// double searchLength);
		load("manifold_min_gap", ValueLayout.JAVA_DOUBLE, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
				ValueLayout.JAVA_DOUBLE);

		// ManifoldManifold* manifold_calculate_curvature(void* mem,
		// ManifoldManifold* m, int gaussian_idx, int mean_idx);
		load("manifold_calculate_curvature", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
				ValueLayout.JAVA_INT, ValueLayout.JAVA_INT);

		// ManifoldManifold* manifold_set_properties(void* mem, ManifoldManifold* m,
		// int num_prop, callback, ctx) — callback not exposed yet

		// ManifoldManifold* manifold_extrude(void* mem, ManifoldPolygons* cs,
		// double height, int slices, double twist_degrees, double scale_x,
		// double scale_y);
		load("manifold_extrude", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
				ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_INT, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE,
				ValueLayout.JAVA_DOUBLE);

		// ManifoldManifold* manifold_revolve(void* mem, ManifoldPolygons* cs, int
		// circular_segments, double revolve_degrees);
		load("manifold_revolve", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
				ValueLayout.JAVA_INT, ValueLayout.JAVA_DOUBLE);

		// ManifoldTriangulation* manifold_triangulate(void* mem,
		// ManifoldPolygons* ps, double epsilon);
		load("manifold_triangulate", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
				ValueLayout.JAVA_DOUBLE);

		// size_t manifold_triangulation_num_tri(ManifoldTriangulation* m);
		load("manifold_triangulation_num_tri", ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);

		// int* manifold_triangulation_tri_verts(void* mem, ManifoldTriangulation*
		// m); copies into caller storage sized num_tri * 3 ints
		load("manifold_triangulation_tri_verts", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// void manifold_delete_triangulation(ManifoldTriangulation* m);
		loadVoid("manifold_delete_triangulation", ValueLayout.ADDRESS);

		// ManifoldPolygons* manifold_project(void* mem, ManifoldManifold* m);
		load("manifold_project", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

		// ManifoldPolygons* manifold_polygons(void* mem, ManifoldSimplePolygon**
		// ps, size_t length);
		load("manifold_polygons", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
				ValueLayout.JAVA_LONG);

		// ManifoldTriangulation* manifold_alloc_triangulation();
		load("manifold_alloc_triangulation", ValueLayout.ADDRESS);

		// int* manifold_triangulation_tri_verts(void* mem,
		// ManifoldTriangulation* m); copies into caller storage

		// ManifoldSimplePolygon* manifold_simple_polygon(void* mem, ManifoldVec2*
		// ps, size_t length);
		load("manifold_simple_polygon", ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
				ValueLayout.JAVA_LONG);

		// ManifoldSimplePolygon* manifold_alloc_simple_polygon();
		load("manifold_alloc_simple_polygon", ValueLayout.ADDRESS);

		// void manifold_delete_simple_polygon(ManifoldSimplePolygon* p);
		loadVoid("manifold_delete_simple_polygon", ValueLayout.ADDRESS);

		// ManifoldRayHitVec* manifold_alloc_ray_hit_vec();
		load("manifold_alloc_ray_hit_vec", ValueLayout.ADDRESS);

		// void manifold_delete_ray_hit_vec(ManifoldRayHitVec* v);
		loadVoid("manifold_delete_ray_hit_vec", ValueLayout.ADDRESS);

		System.out.println("Available Manifold functions: " + functions.keySet());
	}

	// ===== Native loader helpers =====

	private void load(String name, MemoryLayout returnLayout, MemoryLayout... argLayouts) {
		try {
			MemorySegment symbol = library.find(name).orElseThrow();
			FunctionDescriptor desc = FunctionDescriptor.of(returnLayout, argLayouts);
			functions.put(name, LINKER.downcallHandle(symbol, desc));
		} catch (Exception e) {
			System.err.println("ERROR: Missing Manifold function " + name);
		}
	}

	private void loadVoid(String name, ValueLayout... argLayouts) {
		try {
			MemorySegment symbol = library.find(name).orElseThrow();
			FunctionDescriptor desc = FunctionDescriptor.ofVoid(argLayouts);
			functions.put(name, LINKER.downcallHandle(symbol, desc));
		} catch (Exception e) {
			System.err.println("ERROR: Missing Manifold function " + name);
		}
	}

	/**
	 * Smooths the manifold by eliminating edges whose dihedral angle is below
	 * {@code minSharpAngle} degrees. Edges above that threshold are kept sharp.
	 *
	 * {@code minSmoothness} controls how smooth the transition is at preserved
	 * sharp edges — 0.0 keeps them perfectly sharp, 1.0 blends them fully.
	 *
	 * Useful post-difference to dissolve thin sliver faces left by near-coincident
	 * surfaces. A {@code minSharpAngle} of 30–60 degrees is a reasonable starting
	 * point; increase it to dissolve more aggressive artifacts.
	 *
	 * @param m             the input manifold
	 * @param minSharpAngle edges sharper than this (degrees) are preserved
	 * @param minSmoothness smoothness at preserved edges [0.0 = sharp, 1.0 =
	 *                      smooth]
	 */
	public MemorySegment smoothOut(MemorySegment m, double minSharpAngle, double minSmoothness) throws Throwable {
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_smooth_out").invoke(mem, m, minSharpAngle, minSmoothness);
	}

	// ===== Static Quality Globals =====

	/**
	 * Sets the global number of circular segments used when generating cylinders,
	 * spheres, and other round primitives. Call this BEFORE creating primitives.
	 * Higher values produce smoother geometry but more triangles. A value of 0
	 * resets to the angle/length defaults.
	 */
	public void setCircularSegments(int segments) throws Throwable {
		functions.get("manifold_set_circular_segments").invoke(segments);
	}

	/**
	 * Sets the minimum angle (in degrees) between circular segments. The actual
	 * segment count is the maximum implied by both this angle limit and the
	 * edge-length limit. Smaller angles = more segments = smoother curves. Default
	 * is ~1 degree.
	 */
	public void setMinCircularAngle(double degrees) throws Throwable {
		functions.get("manifold_set_min_circular_angle").invoke(degrees);
	}

	/**
	 * Returns the number of circular segments that would be used for a circle of
	 * the given radius, given the current quality settings. Useful for debugging or
	 * verifying your quality settings before building geometry.
	 */
	public int getCircularSegments(double radius) throws Throwable {
		return (int) functions.get("manifold_get_circular_segments").invoke(radius);
	}
	// ===== Status =====

	public ManifoldError status(MemorySegment m) throws Throwable {
		int code = (int) functions.get("manifold_status").invoke(m);
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
			MemorySegment sliceMem = (MemorySegment) functions.get("manifold_alloc_polygons").invoke();
			slicePolys = (MemorySegment) functions.get("manifold_slice").invoke(sliceMem, m, height);

			// 2. Polygons → CrossSection (positive fill rule handles concave shapes
			// correctly)
			MemorySegment csMem = (MemorySegment) functions.get("manifold_alloc_cross_section").invoke();
			section = (MemorySegment) functions.get("manifold_cross_section_of_polygons").invoke(csMem, slicePolys,
					FillRule.POSITIVE.toInt());

			// 3. Offset
			MemorySegment offsetMem = (MemorySegment) functions.get("manifold_alloc_cross_section").invoke();
			offsetSec = (MemorySegment) functions.get("manifold_cross_section_offset").invoke(offsetMem, section, delta,
					joinType.toInt(), miterLimit, circularSegs);

			// 4. CrossSection → Polygons → Java
			MemorySegment polyMem = (MemorySegment) functions.get("manifold_alloc_polygons").invoke();
			offsetPolys = (MemorySegment) functions.get("manifold_cross_section_to_polygons").invoke(polyMem,
					offsetSec);

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
			MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_polygons").invoke();
			polygons = (MemorySegment) functions.get("manifold_slice").invoke(mem, m, height);

			long numContours = (long) functions.get("manifold_polygons_length").invoke(polygons);
			ArrayList<double[][]> result = new ArrayList<>((int) numContours);

			for (int c = 0; c < numContours; c++) {
				long len = (long) functions.get("manifold_polygons_simple_length").invoke(polygons, (long) c);
				double[][] contour = new double[(int) len][2];

				for (int i = 0; i < len; i++) {
					MemorySegment pt = (MemorySegment) functions.get("manifold_polygons_get_point").invoke(arena,
							polygons, (long) c, (long) i);
					contour[i][0] = pt.get(ValueLayout.JAVA_DOUBLE, 0); // x
					contour[i][1] = pt.get(ValueLayout.JAVA_DOUBLE, 8); // y
				}
				result.add(contour);
			}
			return result;
		} finally {
			if (polygons != null) {
				try {
					functions.get("manifold_delete_polygons").invoke(polygons);
				} catch (Throwable ignored) {
				}
			}
		}
	}

	/** Reads a native ManifoldPolygons into a Java ArrayList of contours. */
	private ArrayList<double[][]> polygonsToJava(MemorySegment polygons, Arena arena) throws Throwable {
		long numContours = (long) functions.get("manifold_polygons_length").invoke(polygons);
		ArrayList<double[][]> result = new ArrayList<>((int) numContours);
		for (int c = 0; c < numContours; c++) {
			long len = (long) functions.get("manifold_polygons_simple_length").invoke(polygons, (long) c);
			double[][] contour = new double[(int) len][2];
			for (int i = 0; i < len; i++) {
				MemorySegment pt = (MemorySegment) functions.get("manifold_polygons_get_point").invoke(arena, polygons,
						(long) c, (long) i);
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
			functions.get("manifold_delete_cross_section").invoke(cs);
		} catch (Throwable ignored) {
		}
	}

	private void safeDeletePolygons(MemorySegment p) {
		if (p == null)
			return;
		try {
			functions.get("manifold_delete_polygons").invoke(p);
		} catch (Throwable ignored) {
		}
	}

	// ===== New upstream capability wrappers =====

	/**
	 * Version of the manifold core this binding targets, as {@code {major,
	 * minor, patch}}. The 3.5.5 native library exposes no version API, so this
	 * is a compile-time constant matching the bundled natives; it lets callers
	 * verify the expected core generation.
	 */
	public int[] version() throws Throwable {
		return new int[] { 3, 5, 5 };
	}

	/**
	 * Point-containment query: returns the winding number of the manifold at the
	 * given point — {@code 0} outside, {@code 1} inside (cavities of nested
	 * shells also return 0). Implemented by casting a ray to an endpoint far
	 * outside the bounding box and counting surface crossings: an odd count
	 * means the point is enclosed. Unreliable exactly ON the surface.
	 */
	public int windingNumber(MemorySegment m, double x, double y, double z) throws Throwable {
		MemorySegment box = (MemorySegment) functions.get("manifold_alloc_box").invoke();
		MemorySegment boxSeg = null;
		try {
			boxSeg = (MemorySegment) functions.get("manifold_bounding_box").invoke(box, m);
			double[] dims = boxDimensions(boxSeg);
			double far = 2.0 * (Math.abs(dims[0]) + Math.abs(dims[1]) + Math.abs(dims[2])) + 1.0;
			ArrayList<RayHit> hits = rayCast(m, new double[] { x, y, z },
					new double[] { x + far, y, z });
			// Each hit flips in/out state; crossing count parity decides.
			int crossings = hits.size();
			return (crossings % 2 == 0) ? 0 : 1;
		} finally {
			functions.get("manifold_delete_box").invoke(box);
		}
	}

	/** One ray-cast hit: triangle index, parametric distance, position, normal. */
	public record RayHit(long faceID, double distance, double[] position, double[] normal) {
	}

	/**
	 * Casts a ray segment from {@code origin} to {@code end} and returns every
	 * surface hit in order. An empty list means the segment misses the solid.
	 */
	public ArrayList<RayHit> rayCast(MemorySegment m, double[] origin, double[] end) throws Throwable {
		if (origin == null || origin.length != 3 || end == null || end.length != 3)
			throw new IllegalArgumentException("origin and end must each be a double[3] of {x, y, z}");
		MemorySegment hits = null;
		try (Arena arena = Arena.ofConfined()) {
			MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_ray_hit_vec").invoke();
			hits = (MemorySegment) functions.get("manifold_ray_cast").invoke(mem, m, origin[0], origin[1],
					origin[2], end[0], end[1], end[2]);

			long n = (long) functions.get("manifold_ray_hit_vec_length").invoke(hits);
			MethodHandle get = functions.get("manifold_ray_hit_vec_get");
			ArrayList<RayHit> out = new ArrayList<>((int) n);
			for (long i = 0; i < n; i++) {
				MemorySegment hit = (MemorySegment) get.invoke(arena, hits, i);
				long faceID = hit.get(ValueLayout.JAVA_LONG, 0);
				double distance = hit.get(ValueLayout.JAVA_DOUBLE, 8);
				MemorySegment pos = hit.asSlice(16, 24);
				MemorySegment nrm = hit.asSlice(40, 24);
				out.add(new RayHit(faceID, distance,
						new double[] { pos.get(ValueLayout.JAVA_DOUBLE, 0), pos.get(ValueLayout.JAVA_DOUBLE, 8),
								pos.get(ValueLayout.JAVA_DOUBLE, 16) },
						new double[] { nrm.get(ValueLayout.JAVA_DOUBLE, 0), nrm.get(ValueLayout.JAVA_DOUBLE, 8),
								nrm.get(ValueLayout.JAVA_DOUBLE, 16) }));
			}
			return out;
		} finally {
			if (hits != null) {
				try {
					functions.get("manifold_delete_ray_hit_vec").invoke(hits);
				} catch (Throwable ignored) {
				}
			}
		}
	}

	/**
	 * Minimum gap between this manifold and another, searching along a bounding
	 * sphere hierarchy up to {@code searchLength}. Returns
	 * {@code java.lang.Double.MAX_VALUE} when the solids are farther apart than
	 * the search length.
	 */
	public double minGap(MemorySegment a, MemorySegment b, double searchLength) throws Throwable {
		return (double) functions.get("manifold_min_gap").invoke(a, b, searchLength);
	}

	/**
	 * Appends Gaussian ({@code gaussianIdx}) and mean ({@code meanIdx})
	 * curvature channels to the vertex properties. Indices -1 skip a channel;
	 * the output keeps everything else unchanged.
	 */
	public MemorySegment calculateCurvature(MemorySegment m, int gaussianIdx, int meanIdx) throws Throwable {
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_calculate_curvature").invoke(mem, m, gaussianIdx, meanIdx);
	}

	/**
	 * Projects the manifold's silhouette onto the XY plane. Exactly
	 * {@link #slice} at any height for a prismatic solid, but defined for any
	 * manifold: every triangle is extruded to an infinite prism and unioned.
	 */
	public ArrayList<double[][]> project(MemorySegment m) throws Throwable {
		MemorySegment polygons = null;
		try (Arena arena = Arena.ofConfined()) {
			MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_polygons").invoke();
			polygons = (MemorySegment) functions.get("manifold_project").invoke(mem, m);
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
	 * {@code crossSectionOfPolygons(...)} / {@code crossSectionToPolygons(...)}
	 * with an explicit {@link FillRule} when even-odd semantics are needed.
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
			MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
			return (MemorySegment) functions.get("manifold_extrude").invoke(mem, polys, height, slices,
					twistDegrees, scaleTop, scaleTop);
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
			MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
			return (MemorySegment) functions.get("manifold_revolve").invoke(mem, polys, circularSegments,
					revolveDegrees);
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
			MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_triangulation").invoke();
			tri = (MemorySegment) functions.get("manifold_triangulate").invoke(mem, polys, epsilon);

			long nTri = (long) functions.get("manifold_triangulation_num_tri").invoke(tri);
			MemorySegment buf = arena.allocate(nTri * 3L * Integer.BYTES);
			functions.get("manifold_triangulation_tri_verts").invoke(buf, tri);
			MemorySegment verts = buf;
			int[][] out = new int[(int) nTri][3];
			for (long t = 0; t < nTri; t++) {
				for (int j = 0; j < 3; j++)
					out[(int) t][j] = (int) verts.getAtIndex(ValueLayout.JAVA_INT, t * 3 + j);
			}
			return out;
		} finally {
			if (tri != null) {
				try {
					functions.get("manifold_delete_triangulation").invoke(tri);
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
		MemorySegment polygons = null;
		try {
			MemorySegment contourPtrs = arena.allocate(ValueLayout.ADDRESS, contours.size());
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
				MemorySegment spMem = (MemorySegment) functions.get("manifold_alloc_simple_polygon").invoke();
				MemorySegment sp = (MemorySegment) functions.get("manifold_simple_polygon").invoke(spMem, pts,
						(long) contour.length);
				owned.add(sp);
				contourPtrs.setAtIndex(ValueLayout.ADDRESS, c, sp);
			}
			MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_polygons").invoke();
			polygons = (MemorySegment) functions.get("manifold_polygons").invoke(mem, contourPtrs,
					(long) contours.size());
		} finally {
			for (MemorySegment sp : owned) {
				try {
					functions.get("manifold_delete_simple_polygon").invoke(sp);
				} catch (Throwable ignored) {
				}
			}
		}
		return polygons;
	}

	// ===== Primitives =====

	public MemorySegment empty() throws Throwable {
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_empty").invoke(mem);
	}

	public MemorySegment tetrahedron() throws Throwable {
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_tetrahedron").invoke(mem);
	}

	public MemorySegment cube(double x, double y, double z, boolean center) throws Throwable {
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_cube").invoke(mem, x, y, z, center ? 1 : 0);
	}

	public MemorySegment sphere(double radius, int segments) throws Throwable {
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_sphere").invoke(mem, radius, segments);
	}

	public MemorySegment cylinder(double height, double radiusLow, double radiusHigh, int segments, int center)
			throws Throwable {
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_cylinder").invoke(mem, height, radiusLow, radiusHigh, segments,
				center);
	}

	// ===== Transformations =====

	public MemorySegment transform(MemorySegment m, double x1, double y1, double z1, double x2, double y2, double z2,
			double x3, double y3, double z3, double x4, double y4, double z4) throws Throwable {
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_transform").invoke(mem, m, x1, y1, z1, x2, y2, z2, x3, y3, z3,
				x4, y4, z4);
	}

	public MemorySegment translate(MemorySegment m, double x, double y, double z) throws Throwable {
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_translate").invoke(mem, m, x, y, z);
	}

	public MemorySegment scale(MemorySegment m, double x, double y, double z) throws Throwable {
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_scale").invoke(mem, m, x, y, z);
	}

	public MemorySegment rotate(MemorySegment m, double x, double y, double z) throws Throwable {
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_rotate").invoke(mem, m, x, y, z);
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
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_mirror").invoke(mem, m, nx, ny, nz);
	}

	// ===== Boolean operations =====

	public MemorySegment union(MemorySegment a, MemorySegment b) throws Throwable {
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_union").invoke(mem, a, b);
	}

	public MemorySegment difference(MemorySegment a, MemorySegment b) throws Throwable {
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_difference").invoke(mem, a, b);
	}

	public MemorySegment minkowski_difference(MemorySegment a, MemorySegment b) throws Throwable {
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_minkowski_difference").invoke(mem, a, b);
	}

	public MemorySegment minkowski_sum(MemorySegment a, MemorySegment b) throws Throwable {
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_minkowski_sum").invoke(mem, a, b);
	}

	public MemorySegment intersection(MemorySegment a, MemorySegment b) throws Throwable {
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_intersection").invoke(mem, a, b);
	}

	public MemorySegment booleanOp(MemorySegment a, MemorySegment b, int opType) throws Throwable {
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_boolean").invoke(mem, a, b, opType);
	}

	public MemorySegment batchUnion(MemorySegment[] shapes) throws Throwable {
		MemorySegment vecMem = (MemorySegment) functions.get("manifold_alloc_manifold_vec").invoke();
		MemorySegment vec = (MemorySegment) functions.get("manifold_manifold_empty_vec").invoke(vecMem);
		try {
			for (MemorySegment shape : shapes)
				functions.get("manifold_manifold_vec_push_back").invoke(vec, shape);

			MemorySegment resultMem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
			return (MemorySegment) functions.get("manifold_batch_boolean").invoke(resultMem, vec, OPTYPE_UNION);
		} finally {
			if (vec != null) {
				try {
					functions.get("manifold_delete_manifold_vec").invoke(vec);
				} catch (Throwable ignored) {
				}
			}
		}
	}

	// ===== Info =====

	public long manifoldSize() throws Throwable {
		return (long) functions.get("manifold_manifold_size").invoke();
	}

	public long meshGL64Size() throws Throwable {
		return (long) functions.get("manifold_meshgl64_size").invoke();
	}

	public long vecSize() throws Throwable {
		return (long) functions.get("manifold_manifold_vec_size").invoke();
	}

	public double volume(MemorySegment m) throws Throwable {
		return (double) functions.get("manifold_volume").invoke(m);
	}

	public double surfaceArea(MemorySegment m) throws Throwable {
		return (double) functions.get("manifold_surface_area").invoke(m);
	}

	public double epsilon(MemorySegment m) throws Throwable {
		return (double) functions.get("manifold_epsilon").invoke(m);
	}

	public int genus(MemorySegment m) throws Throwable {
		return (int) functions.get("manifold_genus").invoke(m);
	}

	public long numVert(MemorySegment m) throws Throwable {
		return (long) functions.get("manifold_num_vert").invoke(m);
	}

	public long numTri(MemorySegment m) throws Throwable {
		return (long) functions.get("manifold_num_tri").invoke(m);
	}

	public long numEdge(MemorySegment m) throws Throwable {
		return (long) functions.get("manifold_num_edge").invoke(m);
	}

	public long numProp(MemorySegment m) throws Throwable {
		return (long) functions.get("manifold_num_prop").invoke(m);
	}

	// ===== Refinement =====

	public MemorySegment refine(MemorySegment m, int level) throws Throwable {
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_refine").invoke(mem, m, level);
	}

	public MemorySegment refineToLength(MemorySegment m, double length) throws Throwable {
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_refine_to_length").invoke(mem, m, length);
	}

	public MemorySegment refineToTolerance(MemorySegment m, double tolerance) throws Throwable {
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_refine_to_tolerance").invoke(mem, m, tolerance);
	}

	public MemorySegment simplify(MemorySegment m, double tolerance) throws Throwable {
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_simplify").invoke(mem, m, tolerance);
	}

	public MemorySegment smoothByNormals(MemorySegment m, int normalIdx) throws Throwable {
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_smooth_by_normals").invoke(mem, m, normalIdx);
	}

	public MemorySegment calculateNormals(MemorySegment m, int normalIdx, double minSharpAngle) throws Throwable {
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_calculate_normals").invoke(mem, m, normalIdx, minSharpAngle);
	}

	// ===== Split/Trim =====

	public MemorySegment trimByPlane(MemorySegment m, double nx, double ny, double nz, double offset) throws Throwable {
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_trim_by_plane").invoke(mem, m, nx, ny, nz, offset);
	}

	public MemorySegment[] split(MemorySegment a, MemorySegment b) throws Throwable {
		MemorySegment firstMem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		MemorySegment secondMem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();

		try (Arena arena = Arena.ofConfined()) {
			functions.get("manifold_split").invoke(arena, firstMem, secondMem, a, b);
			return new MemorySegment[] { firstMem, secondMem };
		} catch (Throwable e) {
			try {
				delete(firstMem);
			} catch (Throwable ignored) {
				ignored.printStackTrace();
			}
			try {
				delete(secondMem);
			} catch (Throwable ignored) {
				ignored.printStackTrace();
			}
			throw e;
		}
	}

	// ===== Hull =====

	public MemorySegment hull(MemorySegment m) throws Throwable {
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_hull").invoke(mem, m);
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

			MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
			return (MemorySegment) functions.get("manifold_hull_pts").invoke(mem, ptsBuffer, count);
		}
	}

	public MemorySegment batchHull(MemorySegment[] shapes) throws Throwable {
		MemorySegment vecMem = (MemorySegment) functions.get("manifold_alloc_manifold_vec").invoke();
		MemorySegment vec = (MemorySegment) functions.get("manifold_manifold_empty_vec").invoke(vecMem);
		try {
			for (MemorySegment shape : shapes)
				functions.get("manifold_manifold_vec_push_back").invoke(vec, shape);

			MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
			return (MemorySegment) functions.get("manifold_batch_hull").invoke(mem, vec);
		} finally {
			if (vec != null) {
				
				functions.get("manifold_delete_manifold_vec").invoke(vec);
	
			}
		}
	}

	// ===== Compose/Decompose =====

	public MemorySegment compose(MemorySegment[] parts) throws Throwable {
		MemorySegment vecMem = (MemorySegment) functions.get("manifold_alloc_manifold_vec").invoke();
		MemorySegment vec = (MemorySegment) functions.get("manifold_manifold_empty_vec").invoke(vecMem);
		try {
			for (MemorySegment part : parts)
				functions.get("manifold_manifold_vec_push_back").invoke(vec, part);

			MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
			return (MemorySegment) functions.get("manifold_compose").invoke(mem, vec);
		} finally {
			if (vec != null) {
				try {
					functions.get("manifold_delete_manifold_vec").invoke(vec);
				} catch (Throwable ignored) {
				}
			}
		}
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
			MemorySegment vecMem = (MemorySegment) functions.get("manifold_alloc_manifold_vec").invoke();
			vec = (MemorySegment) functions.get("manifold_decompose").invoke(vecMem, m);
			long count = (long) functions.get("manifold_manifold_vec_length").invoke(vec);
			MemorySegment[] parts = new MemorySegment[(int) count];
			try {
				MethodHandle vecGet = functions.get("manifold_manifold_vec_get");
				MethodHandle allocManifold = functions.get("manifold_alloc_manifold");
				if (vecGet == null || allocManifold == null)
					throw new IllegalStateException(
							"manifold_manifold_vec_get / manifold_alloc_manifold were not loaded");
				for (int i = 0; i < count; i++) {
					// heap-backed storage so each part outlives the vector it came from
					MemorySegment elemMem = (MemorySegment) allocManifold.invoke();
					parts[i] = elemMem;
					parts[i] = (MemorySegment) vecGet.invoke(elemMem, vec, (long) i);
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
			if (vec != null) {
				try {
					functions.get("manifold_delete_manifold_vec").invoke(vec);
				} catch (Throwable ignored) {
				}
			}
		}
	}

	// ===== Copy / AsOriginal =====

	public MemorySegment copy(MemorySegment m) throws Throwable {
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_copy").invoke(mem, m);
	}

	public MemorySegment asOriginal(MemorySegment m) throws Throwable {
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
		return (MemorySegment) functions.get("manifold_as_original").invoke(mem, m);
	}

	// ===== Bounding box =====

	public MemorySegment boundingBox(MemorySegment m) throws Throwable {
		MemorySegment box = (MemorySegment) functions.get("manifold_alloc_box").invoke();
		return (MemorySegment) functions.get("manifold_bounding_box").invoke(box, m);
	}

	public double[] boxDimensions(MemorySegment box) throws Throwable {
		try (Arena arena = Arena.ofConfined()) {
			MemorySegment dims = (MemorySegment) functions.get("manifold_box_dimensions").invoke(arena, box);
			return new double[] { dims.get(ValueLayout.JAVA_DOUBLE, 0), dims.get(ValueLayout.JAVA_DOUBLE, 8),
					dims.get(ValueLayout.JAVA_DOUBLE, 16) };
		}
	}

	public javafx.geometry.BoundingBox getJavaFXBounds(MemorySegment m) {
		MemorySegment box = null;
		try (Arena arena = Arena.ofConfined()) {
			box = (MemorySegment) functions.get("manifold_alloc_box").invoke();
			functions.get("manifold_bounding_box").invoke(box, m);

			MemorySegment min = (MemorySegment) functions.get("manifold_box_min").invoke(arena, box);
			MemorySegment max = (MemorySegment) functions.get("manifold_box_max").invoke(arena, box);

			double minX = min.get(ValueLayout.JAVA_DOUBLE, 0);
			double minY = min.get(ValueLayout.JAVA_DOUBLE, 8);
			double minZ = min.get(ValueLayout.JAVA_DOUBLE, 16);
			double maxX = max.get(ValueLayout.JAVA_DOUBLE, 0);
			double maxY = max.get(ValueLayout.JAVA_DOUBLE, 8);
			double maxZ = max.get(ValueLayout.JAVA_DOUBLE, 16);

			return new javafx.geometry.BoundingBox(minX, minY, minZ, maxX - minX, maxY - minY, maxZ - minZ);
		} catch (Throwable e) {
			e.printStackTrace();
			return new javafx.geometry.BoundingBox(0, 0, 0, 0, 0, 0);
		} finally {
			if (box != null) {
				try {
					functions.get("manifold_delete_box").invoke(box);
				} catch (Throwable ignored) {
				}
			}
		}
	}

	public javafx.geometry.BoundingBox getBounds(MemorySegment m) throws Throwable {
		MemorySegment box = (MemorySegment) functions.get("manifold_alloc_box").invoke();
		try {
			functions.get("manifold_bounding_box").invoke(box, m);
			try (Arena arena = Arena.ofConfined()) {
				MemorySegment dims = (MemorySegment) functions.get("manifold_box_dimensions").invoke(arena, box);
				double w = dims.get(ValueLayout.JAVA_DOUBLE, 0);
				double h = dims.get(ValueLayout.JAVA_DOUBLE, 8);
				double d = dims.get(ValueLayout.JAVA_DOUBLE, 16);
				return new javafx.geometry.BoundingBox(0, 0, 0, w, h, d);
			}
		} finally {
			functions.get("manifold_delete_box").invoke(box);
		}
	}

	public MemorySegment centerObject(MemorySegment m) throws Throwable {
		MemorySegment box = null;
		try {
			box = (MemorySegment) functions.get("manifold_alloc_box").invoke();
			functions.get("manifold_bounding_box").invoke(box, m);
			try (Arena arena = Arena.ofConfined()) {
				MemorySegment center = (MemorySegment) functions.get("manifold_box_center").invoke(arena, box);
				double cx = center.get(ValueLayout.JAVA_DOUBLE, 0);
				double cy = center.get(ValueLayout.JAVA_DOUBLE, 8);
				double cz = center.get(ValueLayout.JAVA_DOUBLE, 16);
				return translate(m, -cx, -cy, -cz);
			}
		} finally {
			if (box != null) {
				try {
					functions.get("manifold_delete_box").invoke(box);
				} catch (Throwable ignored) {
				}
			}
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
		functions.get("manifold_delete_manifold").invoke(manifold);
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
		MethodHandle mh = functions.get("manifold_meshgl64");
		if (mh == null)
			throw new RuntimeException("manifold_meshgl64 not found");

		try (Arena arena = Arena.ofConfined()) {
			MemorySegment vertPtr = arena.allocate(3 * nVerts * Double.BYTES);
			vertPtr.copyFrom(MemorySegment.ofArray(vertices));

			MemorySegment triPtr = arena.allocate(3 * nTris * Long.BYTES);
			triPtr.copyFrom(MemorySegment.ofArray(triangles));

			MemorySegment meshGLMem = (MemorySegment) functions.get("manifold_alloc_meshgl64").invoke();
			MemorySegment meshGL = null;

			try {
				meshGL = (MemorySegment) mh.invoke(meshGLMem, vertPtr, nVerts, 3L, triPtr, nTris);

				MemorySegment mergedMem = (MemorySegment) functions.get("manifold_alloc_meshgl64").invoke();
				MemorySegment merged = null;

				try {
					merged = (MemorySegment) functions.get("manifold_meshgl64_merge").invoke(mergedMem, meshGL);

					MemorySegment manMem = (MemorySegment) functions.get("manifold_alloc_manifold").invoke();
					MemorySegment result = (MemorySegment) functions.get("manifold_of_meshgl64").invoke(manMem, merged);

					// Clean up intermediates — skip if they alias the result
					Set<Long> freed = new HashSet<>();
					for (MemorySegment seg : new MemorySegment[] { merged, meshGL }) {
						if (seg == null || seg.address() == result.address())
							continue;
						if (freed.add(seg.address()))
							functions.get("manifold_delete_meshgl64").invoke(seg);
					}

					return result;

				} catch (Throwable e) {
					if (merged != null)
						try {
							functions.get("manifold_delete_meshgl64").invoke(merged);
						} catch (Throwable ignored) {
						}
					throw e;
				}
			} catch (Throwable e) {
				if (meshGL != null)
					try {
						functions.get("manifold_delete_meshgl64").invoke(meshGL);
					} catch (Throwable ignored) {
					}
				throw e;
			}
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
		MemorySegment mem = (MemorySegment) functions.get("manifold_alloc_meshgl64").invoke();
		MemorySegment meshGL = (MemorySegment) functions.get("manifold_get_meshgl64").invoke(mem, manifold);

		try {
			long numVert = (long) functions.get("manifold_meshgl64_num_vert").invoke(meshGL);
			long numTri = (long) functions.get("manifold_meshgl64_num_tri").invoke(meshGL);
			long numProp = (long) functions.get("manifold_meshgl64_num_prop").invoke(meshGL);

			double[] vertices = new double[(int) (numVert * 3)];
			long[] triangles = new long[(int) (numTri * 3)];

			try (Arena temp = Arena.ofConfined()) {
				if (numVert > 0) {
					long vertLen = (long) functions.get("manifold_meshgl64_vert_properties_length").invoke(meshGL);
					MemorySegment tempMem = temp.allocate(vertLen * Double.BYTES);
					functions.get("manifold_meshgl64_vert_properties").invoke(tempMem, meshGL);

					for (int i = 0; i < numVert; i++)
						for (int j = 0; (j < 3) && (j < numProp); j++)
							vertices[i * 3 + j] = tempMem.getAtIndex(ValueLayout.JAVA_DOUBLE, i * numProp + j);
				}

				if (numTri > 0) {
					long triLen = (long) functions.get("manifold_meshgl64_tri_length").invoke(meshGL);
					MemorySegment tempMem = temp.allocate(triLen * Long.BYTES);
					functions.get("manifold_meshgl64_tri_verts").invoke(tempMem, meshGL);

					for (int i = 0; i < triLen; i++)
						triangles[i] = tempMem.getAtIndex(ValueLayout.JAVA_LONG, i);
				}
			}

			return new MeshData64(vertices, triangles, (int) numVert, (int) numTri);

		} finally {
			try {
				functions.get("manifold_delete_meshgl64").invoke(meshGL);
			} catch (Throwable ignored) {
			}
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