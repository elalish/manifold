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

export { }
