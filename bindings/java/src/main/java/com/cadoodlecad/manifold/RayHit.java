package com.cadoodlecad.manifold;

/**
 * One surface intersection reported by
 * {@link ManifoldBindings#rayCast(java.lang.foreign.MemorySegment, double[], double[])}.
 *
 * <p>
 * This is a plain Java value: it holds no native memory, so it stays valid
 * after the query returns and after the manifold it came from is deleted. The
 * native {@code ManifoldRayHit} and its container are read and released inside
 * {@code rayCast} and never escape it.
 *
 * <p>
 * Values follow the C API's {@code RayHit}: {@code distance} is parametric
 * along the queried segment in [0, 1], where 0 is the origin and 1 the
 * endpoint, and {@code crossing} is +1 where the ray enters the solid and -1
 * where it exits. Summing {@code crossing} along a ray gives the change in
 * winding number from origin to endpoint; prefer it over the sign of
 * {@code dot(normal, direction)}, which can disagree with the mesh's internal
 * crossing in degenerate cases. A ray grazing a vertex or edge can report a
 * +1/-1 pair at the same distance; those ties are ordered by {@code faceID}.
 *
 * @param faceID   index of the triangle that was hit
 * @param distance parametric distance along the segment in [0, 1]
 * @param position hit point as {x, y, z}
 * @param normal   geometric face normal at the hit as {x, y, z}
 * @param crossing +1 where the ray enters the solid, -1 where it exits
 */
public record RayHit(long faceID, double distance, double[] position, double[] normal, int crossing) {

	public RayHit {
		position = copyOf3(position, "position");
		normal = copyOf3(normal, "normal");
	}

	private static double[] copyOf3(double[] value, String name) {
		if (value == null || value.length != 3)
			throw new IllegalArgumentException(name + " must be a double[3] of {x, y, z}");
		return value.clone();
	}

	@Override
	public double[] position() {
		return position.clone();
	}

	@Override
	public double[] normal() {
		return normal.clone();
	}
}
