/**
 * @packageDocumentation
 * @mergeModuleWith manifoldCAD
 */

/**
 * @hidden
 * @inline
 */
export declare type AnimationMode = 'loop' | 'ping-pong';

/**
 * Reset the circular construction parameters to their defaults if
 * `setMinCircularAngle()`, `setMinCircularEdgeLength()`, or
 * `setCircularSegments()` have been called.
 */
export declare function resetToCircularDefaults(): void;

/**
 * Set the duration of the animation, in seconds.
 *
 * @param duration in seconds.
 */
export declare function setAnimationDuration(duration: number): void;

/**
 * Set the animation frame rate.
 *
 * @param fps in frames per second.
 */
export declare function setAnimationFPS(fps: number): void;

/**
 * Set the animation repeat mode.
 *
 * @param mode 'loop' or 'ping-pong'
 */
export declare function setAnimationMode(mode: AnimationMode): void;

/**
 * Set the default number of segments in a circle.
 * Overrides the edge length and angle
 * constraints and sets the number of segments to exactly this value.
 *
 * @param segments Number of circular segments. Default is 0, meaning no
 * constraint is applied.
 */
export declare function setCircularSegments(segments: number): void | undefined;

/**
 * Set an angle constraint when calculating the number of segments in a circle.
 * The number of segments will be rounded
 * up to the nearest factor of four.
 *
 * @param angle The minimum angle in degrees between consecutive segments. The
 * angle will increase if the the segments hit the minimum edge length.
 * Default is 10 degrees.
 */
export declare function setMinCircularAngle(angle: number): void;

/**
 * Set a length constraint when calculating the number segments in a circle.
 * The number of segments will be rounded
 * up to the nearest factor of four.
 *
 * @param length The minimum length of segments. The length will
 * increase if the the segments hit the minimum angle. Default is 1.0.
 */
export declare function setMinCircularEdgeLength(length: number): void;

export { }
