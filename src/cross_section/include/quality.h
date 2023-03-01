#pragma once

namespace manifold {
class Quality {
 private:
  static int circularSegments_;
  static float circularAngle_;
  static float circularEdgeLength_;

 public:
  /** @name Defaults
   * These static properties control how circular shapes are quantized by
   * default on construction. If circularSegments is specified, it takes
   * precedence. If it is zero, then instead the minimum is used of the segments
   * calculated based on edge length and angle, rounded up to the nearest
   * multiple of four. To get numbers not divisible by four, circularSegments
   * must be specified.
   */
  ///@{
  /**
   * Sets an angle constraint the default number of circular segments for the
   * Cylinder(), Sphere(), and Revolve() constructors. The number of segments
   * will be rounded up to the nearest factor of four.
   *
   * @param angle The minimum angle in degrees between consecutive segments. The
   * angle will increase if the the segments hit the minimum edge length.
   * Default is 10 degrees.
   */
  static void SetMinCircularAngle(float angle);

  /**
   * Sets a length constraint the default number of circular segments for the
   * Cylinder(), Sphere(), and Revolve() constructors. The number of segments
   * will be rounded up to the nearest factor of four.
   *
   * @param length The minimum length of segments. The length will
   * increase if the the segments hit the minimum angle. Default is 1.0.
   */
  static void SetMinCircularEdgeLength(float length);

  /**
   * Sets the default number of circular segments for the
   * Cylinder(), Sphere(), and Revolve() constructors. Overrides the edge length
   * and angle constraints and sets the number of segments to exactly this
   * value.
   *
   * @param number Number of circular segments. Default is 0, meaning no
   * constraint is applied.
   */
  static void SetCircularSegments(int number);

  /**
   * Determine the result of the SetMinCircularAngle(),
   * SetMinCircularEdgeLength(), and SetCircularSegments() defaults.
   *
   * @param radius For a given radius of circle, determine how many default
   * segments there will be.
   */
  static int GetCircularSegments(float radius);
  ///@}
};
}  // namespace manifold
