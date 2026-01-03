/**
 * These objects and functions are specific to top-level scripts
 * running within manifoldCAD.
 *
 * They are only accessible as global objects by a top level script evaluated by
 * the worker.  Libraries will not have access to them.
 *
 * These functions will not be present at all when a model is imported as an ES
 * module. They can be imported through the {@link lib/scene-builder} or
 * directly from {@link lib/animation} and {@link lib/level-of-detail}.
 *
 * @packageDocumentation
 * @module manifold-3d/manifoldCAD Globals
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
