/**
 * @packageDocumentation
 * @mergeModuleWith manifoldCAD
 */

/**
 * @inline
 */
export declare type AnimationMode = 'loop' | 'ping-pong';

/**
 * Reset the circular construction parameters to their defaults if
 * {@link setMinCircularAngle}, {@link setMinCircularEdgeLength}, or {@link
 * setCircularSegments} have been called.
 * @group Global Settings
 */
export declare const resetToCircularDefaults: () => void;

/**
 * Set the duration of the animation, in seconds.
 *
 * @param duration in seconds.
 * @group Global Settings
 */
export declare const setAnimationDuration: (duration: number) => void;

/**
 * Set the animation frame rate.
 *
 * @param fps in frames per second.
 * @group Global Settings
 */
export declare const setAnimationFPS: (fps: number) => void;

/**
 * Set the animation repeat mode.
 *
 * @param mode 'loop' or 'ping-pong'
 * @group Global Settings
 */
export declare const setAnimationMode: (mode: AnimationMode) => void;

/**
 * Set the default number of segments in a circle.
 * Overrides the edge length and angle
 * constraints and sets the number of segments to exactly this value.
 *
 * @param segments Number of circular segments. Default is 0, meaning no
 * constraint is applied.
 * @group Global Settings
 */
export declare const setCircularSegments: (segments: number) => void | undefined;

/**
 * Set an angle constraint when calculating the number of segments in a circle.
 * The number of segments will be rounded
 * up to the nearest factor of four.
 *
 * @param angle The minimum angle in degrees between consecutive segments. The
 * angle will increase if the the segments hit the minimum edge length.
 * Default is 10 degrees.
 * @group Global Settings
 */
export declare const setMinCircularAngle: (angle: number) => void;

/**
 * Set a length constraint when calculating the number segments in a circle.
 * The number of segments will be rounded
 * up to the nearest factor of four.
 *
 * @param length The minimum length of segments. The length will
 * increase if the the segments hit the minimum angle. Default is 1.0.
 * @group Global Settings
 */
export declare const setMinCircularEdgeLength: (length: number) => void;

export { }
