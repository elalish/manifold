// Copyright 2023 The Manifold Authors.
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

/**
 * Wrap any object with this method to display it and any copies in transparent
 * red. This is particularly useful for debugging subtract() as it will allow
 * you find the object even if it doesn't currently intersect the result.
 *
 * @param manifold The object to show - returned for chaining.
 */
declare function show(manifold: Manifold): Manifold;

/**
 * Wrap any object with this method to display it and any copies as the result,
 * while ghosting out the final result in transparent gray. Helpful for
 * debugging as it allows you to see objects that may be hidden in the interior
 * of the result. Multiple objects marked only() will all be shown.
 *
 * @param manifold The object to show - returned for chaining.
 */
declare function only(manifold: Manifold): Manifold;

// Type definitions for gl-matrix 3.4.3 Project:
// https://github.com/toji/gl-matrix
//
// Definitions by: Mattijs Kneppers <https://github.com/mattijskneppers>,
//
// based on definitions by Tat <https://github.com/tatchx>
//
// Definitions by: Nikolay Babanov <https://github.com/nbabanov>

declare const glMatrix: GLMatrix;

interface GLMatrix {
  glMatrix: TopAPI;
  vec2: Vec2API;
  vec3: Vec3API;
  vec4: Vec4API;
  mat2: Mat2API;
  mat2d: Mat2dAPI;
  mat3: Mat3API;
  mat4: Mat4API;
  quat: QuatAPI;
}

interface TopAPI {
  // Configuration constants
  EPSILON: number;
  ARRAY_TYPE: any;
  RANDOM(): number;
  ENABLE_SIMD: boolean;

  // Compatibility detection
  SIMD_AVAILABLE: boolean;
  USE_SIMD: boolean;

  /**
   * Sets the type of array used when creating new vectors and matrices
   *
   * @param {any} type - Array type, such as Float32Array or Array
   */
  setMatrixArrayType(type: any): void;

  /**
   * Convert Degree To Radian
   *
   * @param {number} a - Angle in Degrees
   */
  toRadian(a: number): number;

  /**
   * Tests whether or not the arguments have approximately the same value,
   * within an absolute or relative tolerance of glMatrix.EPSILON (an absolute
   * tolerance is used for values less than or equal to 1.0, and a relative
   * tolerance is used for larger values)
   *
   * @param {number} a - The first number to test.
   * @param {number} b - The second number to test.
   * @returns {boolean} True if the numbers are approximately equal, false
   *     otherwise.
   */
  equals(a: number, b: number): boolean;
}

type Vec4 = [number, number, number, number];
type Mat2 = [
  number,
  number,
  number,
  number,
];
type Mat2d = [
  number,
  number,
  number,
  number,
  number,
  number,
];
type Mat3 = [
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
];
type Quat = [number, number, number, number];

interface Vec2API {
  /**
   * Creates a new, empty Vec2
   *
   * @returns a new 2D vector
   */
  create(): Vec2;

  /**
   * Creates a new Vec2 initialized with values from an existing vector
   *
   * @param a a vector to clone
   * @returns a new 2D vector
   */
  clone(a: Vec2): Vec2;

  /**
   * Creates a new Vec2 initialized with the given values
   *
   * @param x X component
   * @param y Y component
   * @returns a new 2D vector
   */
  fromValues(x: number, y: number): Vec2;

  /**
   * Copy the values from one Vec2 to another
   *
   * @param out the receiving vector
   * @param a the source vector
   * @returns out
   */
  copy(out: Vec2, a: Vec2): Vec2;

  /**
   * Set the components of a Vec2 to the given values
   *
   * @param out the receiving vector
   * @param x X component
   * @param y Y component
   * @returns out
   */
  set(out: Vec2, x: number, y: number): Vec2;

  /**
   * Adds two Vec2's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  add(out: Vec2, a: Vec2, b: Vec2): Vec2;

  /**
   * Subtracts vector b from vector a
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  subtract(out: Vec2, a: Vec2, b: Vec2): Vec2;

  /**
   * Subtracts vector b from vector a
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  sub(out: Vec2, a: Vec2, b: Vec2): Vec2;

  /**
   * Multiplies two Vec2's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  multiply(out: Vec2, a: Vec2, b: Vec2): Vec2;

  /**
   * Multiplies two Vec2's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  mul(out: Vec2, a: Vec2, b: Vec2): Vec2;

  /**
   * Divides two Vec2's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  divide(out: Vec2, a: Vec2, b: Vec2): Vec2;

  /**
   * Divides two Vec2's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  div(out: Vec2, a: Vec2, b: Vec2): Vec2;

  /**
   * Math.ceil the components of a Vec2
   *
   * @param {Vec2} out the receiving vector
   * @param {Vec2} a vector to ceil
   * @returns {Vec2} out
   */
  ceil(out: Vec2, a: Vec2): Vec2;

  /**
   * Math.floor the components of a Vec2
   *
   * @param {Vec2} out the receiving vector
   * @param {Vec2} a vector to floor
   * @returns {Vec2} out
   */
  floor(out: Vec2, a: Vec2): Vec2;

  /**
   * Returns the minimum of two Vec2's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  min(out: Vec2, a: Vec2, b: Vec2): Vec2;

  /**
   * Returns the maximum of two Vec2's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  max(out: Vec2, a: Vec2, b: Vec2): Vec2;

  /**
   * Rotates the 2D vector.
   *
   * @param {Vec2} out the receiving vector
   * @param {Vec2} a vector to rotate
   * @param {Vec2} b origin of the rotation
   * @param {number} rad angle of rotation in radians
   * @returns {Vec2} out
   */
  rotate(out: Vec2, a: Vec2, b: Vec2, rad: number): Vec2;

  /**
   * Math.round the components of a Vec2
   *
   * @param {Vec2} out the receiving vector
   * @param {Vec2} a vector to round
   * @returns {Vec2} out
   */
  round(out: Vec2, a: Vec2): Vec2;

  /**
   * Scales a Vec2 by a scalar number
   *
   * @param out the receiving vector
   * @param a the vector to scale
   * @param b amount to scale the vector by
   * @returns out
   */
  scale(out: Vec2, a: Vec2, b: number): Vec2;

  /**
   * Adds two Vec2's after scaling the second operand by a scalar value
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @param scale the amount to scale b by before adding
   * @returns out
   */
  scaleAndAdd(out: Vec2, a: Vec2, b: Vec2, scale: number): Vec2;

  /**
   * Calculates the euclidian distance between two Vec2's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns distance between a and b
   */
  distance(a: Vec2, b: Vec2): number;

  /**
   * Calculates the euclidian distance between two Vec2's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns distance between a and b
   */
  dist(a: Vec2, b: Vec2): number;

  /**
   * Calculates the squared euclidian distance between two Vec2's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns squared distance between a and b
   */
  squaredDistance(a: Vec2, b: Vec2): number;

  /**
   * Calculates the squared euclidian distance between two Vec2's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns squared distance between a and b
   */
  sqrDist(a: Vec2, b: Vec2): number;

  /**
   * Calculates the length of a Vec2
   *
   * @param a vector to calculate length of
   * @returns length of a
   */
  length(a: Vec2): number;

  /**
   * Calculates the length of a Vec2
   *
   * @param a vector to calculate length of
   * @returns length of a
   */
  len(a: Vec2): number;

  /**
   * Calculates the squared length of a Vec2
   *
   * @param a vector to calculate squared length of
   * @returns squared length of a
   */
  squaredLength(a: Vec2): number;

  /**
   * Calculates the squared length of a Vec2
   *
   * @param a vector to calculate squared length of
   * @returns squared length of a
   */
  sqrLen(a: Vec2): number;

  /**
   * Negates the components of a Vec2
   *
   * @param out the receiving vector
   * @param a vector to negate
   * @returns out
   */
  negate(out: Vec2, a: Vec2): Vec2;

  /**
   * Returns the inverse of the components of a Vec2
   *
   * @param out the receiving vector
   * @param a vector to invert
   * @returns out
   */
  inverse(out: Vec2, a: Vec2): Vec2;

  /**
   * Normalize a Vec2
   *
   * @param out the receiving vector
   * @param a vector to normalize
   * @returns out
   */
  normalize(out: Vec2, a: Vec2): Vec2;

  /**
   * Calculates the dot product of two Vec2's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns dot product of a and b
   */
  dot(a: Vec2, b: Vec2): number;

  /**
   * Computes the cross product of two Vec2's
   * Note that the cross product must by definition produce a 3D vector
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  cross(out: Vec2, a: Vec2, b: Vec2): Vec2;

  /**
   * Performs a linear interpolation between two Vec2's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @param t interpolation amount between the two inputs
   * @returns out
   */
  lerp(out: Vec2, a: Vec2, b: Vec2, t: number): Vec2;

  /**
   * Generates a random unit vector
   *
   * @param out the receiving vector
   * @returns out
   */
  random(out: Vec2): Vec2;

  /**
   * Generates a random vector with the given scale
   *
   * @param out the receiving vector
   * @param scale Length of the resulting vector. If ommitted, a unit vector
   *     will be returned
   * @returns out
   */
  random(out: Vec2, scale: number): Vec2;

  /**
   * Transforms the Vec2 with a Mat2
   *
   * @param out the receiving vector
   * @param a the vector to transform
   * @param m matrix to transform with
   * @returns out
   */
  transformMat2(out: Vec2, a: Vec2, m: Mat2): Vec2;

  /**
   * Transforms the Vec2 with a Mat2d
   *
   * @param out the receiving vector
   * @param a the vector to transform
   * @param m matrix to transform with
   * @returns out
   */
  transformMat2d(out: Vec2, a: Vec2, m: Mat2d): Vec2;

  /**
   * Transforms the Vec2 with a Mat3
   * 3rd vector component is implicitly '1'
   *
   * @param out the receiving vector
   * @param a the vector to transform
   * @param m matrix to transform with
   * @returns out
   */
  transformMat3(out: Vec2, a: Vec2, m: Mat3): Vec2;

  /**
   * Transforms the Vec2 with a Mat4
   * 3rd vector component is implicitly '0'
   * 4th vector component is implicitly '1'
   *
   * @param out the receiving vector
   * @param a the vector to transform
   * @param m matrix to transform with
   * @returns out
   */
  transformMat4(out: Vec2, a: Vec2, m: Mat4): Vec2;

  /**
   * Perform some operation over an array of vec2Types.
   *
   * @param a the array of vectors to iterate over
   * @param stride Number of elements between the start of each Vec2. If 0
   *     assumes tightly packed
   * @param offset Number of elements to skip at the beginning of the array
   * @param count Number of vec2Types to iterate over. If 0 iterates over entire
   *     array
   * @param fn Function to call for each vector in the array
   * @param arg additional argument to pass to fn
   * @returns a
   */
  forEach(
      a: Float32Array, stride: number, offset: number, count: number,
      fn: (a: Vec2, b: Vec2, arg: any) => void, arg: any): Float32Array;

  /**
   * Perform some operation over an array of vec2Types.
   *
   * @param a the array of vectors to iterate over
   * @param stride Number of elements between the start of each Vec2. If 0
   *     assumes tightly packed
   * @param offset Number of elements to skip at the beginning of the array
   * @param count Number of vec2Types to iterate over. If 0 iterates over entire
   *     array
   * @param fn Function to call for each vector in the array
   * @returns a
   */
  forEach(
      a: Float32Array, stride: number, offset: number, count: number,
      fn: (a: Vec2, b: Vec2) => void): Float32Array;

  /**
   * Returns a string representation of a vector
   *
   * @param a vector to represent as a string
   * @returns string representation of the vector
   */
  str(a: Vec2): string;

  /**
   * Returns whether or not the vectors exactly have the same elements in the
   * same position (when compared with ===)
   *
   * @param {Vec2} a The first vector.
   * @param {Vec2} b The second vector.
   * @returns {boolean} True if the vectors are equal, false otherwise.
   */
  exactEquals(a: Vec2, b: Vec2): boolean;

  /**
   * Returns whether or not the vectors have approximately the same elements
   * in the same position.
   *
   * @param {Vec2} a The first vector.
   * @param {Vec2} b The second vector.
   * @returns {boolean} True if the vectors are equal, false otherwise.
   */
  equals(a: Vec2, b: Vec2): boolean;
}

// Vec3
interface Vec3API {
  /**
   * Creates a new, empty Vec3
   *
   * @returns a new 3D vector
   */
  create(): Vec3;

  /**
   * Creates a new Vec3 initialized with values from an existing vector
   *
   * @param a vector to clone
   * @returns a new 3D vector
   */
  clone(a: Vec3): Vec3;

  /**
   * Creates a new Vec3 initialized with the given values
   *
   * @param x X component
   * @param y Y component
   * @param z Z component
   * @returns a new 3D vector
   */
  fromValues(x: number, y: number, z: number): Vec3;

  /**
   * Copy the values from one Vec3 to another
   *
   * @param out the receiving vector
   * @param a the source vector
   * @returns out
   */
  copy(out: Vec3, a: Vec3): Vec3;

  /**
   * Set the components of a Vec3 to the given values
   *
   * @param out the receiving vector
   * @param x X component
   * @param y Y component
   * @param z Z component
   * @returns out
   */
  set(out: Vec3, x: number, y: number, z: number): Vec3;

  /**
   * Adds two Vec3's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  add(out: Vec3, a: Vec3, b: Vec3): Vec3;

  /**
   * Subtracts vector b from vector a
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  subtract(out: Vec3, a: Vec3, b: Vec3): Vec3;

  /**
   * Subtracts vector b from vector a
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  sub(out: Vec3, a: Vec3, b: Vec3): Vec3

  /**
   * Multiplies two Vec3's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  multiply(out: Vec3, a: Vec3, b: Vec3): Vec3;

  /**
   * Multiplies two Vec3's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  mul(out: Vec3, a: Vec3, b: Vec3): Vec3;

  /**
   * Divides two Vec3's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  divide(out: Vec3, a: Vec3, b: Vec3): Vec3;

  /**
   * Divides two Vec3's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  div(out: Vec3, a: Vec3, b: Vec3): Vec3;

  /**
   * Math.ceil the components of a Vec3
   *
   * @param {Vec3} out the receiving vector
   * @param {Vec3} a vector to ceil
   * @returns {Vec3} out
   */
  ceil(out: Vec3, a: Vec3): Vec3;

  /**
   * Math.floor the components of a Vec3
   *
   * @param {Vec3} out the receiving vector
   * @param {Vec3} a vector to floor
   * @returns {Vec3} out
   */
  floor(out: Vec3, a: Vec3): Vec3;

  /**
   * Returns the minimum of two Vec3's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  min(out: Vec3, a: Vec3, b: Vec3): Vec3;

  /**
   * Returns the maximum of two Vec3's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  max(out: Vec3, a: Vec3, b: Vec3): Vec3;

  /**
   * Math.round the components of a Vec3
   *
   * @param {Vec3} out the receiving vector
   * @param {Vec3} a vector to round
   * @returns {Vec3} out
   */
  round(out: Vec3, a: Vec3): Vec3

  /**
   * Scales a Vec3 by a scalar number
   *
   * @param out the receiving vector
   * @param a the vector to scale
   * @param b amount to scale the vector by
   * @returns out
   */
  scale(out: Vec3, a: Vec3, b: number): Vec3;

  /**
   * Adds two Vec3's after scaling the second operand by a scalar value
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @param scale the amount to scale b by before adding
   * @returns out
   */
  scaleAndAdd(out: Vec3, a: Vec3, b: Vec3, scale: number): Vec3;

  /**
   * Calculates the euclidian distance between two Vec3's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns distance between a and b
   */
  distance(a: Vec3, b: Vec3): number;

  /**
   * Calculates the euclidian distance between two Vec3's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns distance between a and b
   */
  dist(a: Vec3, b: Vec3): number;

  /**
   * Calculates the squared euclidian distance between two Vec3's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns squared distance between a and b
   */
  squaredDistance(a: Vec3, b: Vec3): number;

  /**
   * Calculates the squared euclidian distance between two Vec3's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns squared distance between a and b
   */
  sqrDist(a: Vec3, b: Vec3): number;

  /**
   * Calculates the length of a Vec3
   *
   * @param a vector to calculate length of
   * @returns length of a
   */
  length(a: Vec3): number;

  /**
   * Calculates the length of a Vec3
   *
   * @param a vector to calculate length of
   * @returns length of a
   */
  len(a: Vec3): number;

  /**
   * Calculates the squared length of a Vec3
   *
   * @param a vector to calculate squared length of
   * @returns squared length of a
   */
  squaredLength(a: Vec3): number;

  /**
   * Calculates the squared length of a Vec3
   *
   * @param a vector to calculate squared length of
   * @returns squared length of a
   */
  sqrLen(a: Vec3): number;

  /**
   * Negates the components of a Vec3
   *
   * @param out the receiving vector
   * @param a vector to negate
   * @returns out
   */
  negate(out: Vec3, a: Vec3): Vec3;

  /**
   * Returns the inverse of the components of a Vec3
   *
   * @param out the receiving vector
   * @param a vector to invert
   * @returns out
   */
  inverse(out: Vec3, a: Vec3): Vec3;

  /**
   * Normalize a Vec3
   *
   * @param out the receiving vector
   * @param a vector to normalize
   * @returns out
   */
  normalize(out: Vec3, a: Vec3): Vec3;

  /**
   * Calculates the dot product of two Vec3's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns dot product of a and b
   */
  dot(a: Vec3, b: Vec3): number;

  /**
   * Computes the cross product of two Vec3's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  cross(out: Vec3, a: Vec3, b: Vec3): Vec3;

  /**
   * Performs a linear interpolation between two Vec3's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @param t interpolation amount between the two inputs
   * @returns out
   */
  lerp(out: Vec3, a: Vec3, b: Vec3, t: number): Vec3;

  /**
   * Performs a hermite interpolation with two control points
   *
   * @param {Vec3} out the receiving vector
   * @param {Vec3} a the first operand
   * @param {Vec3} b the second operand
   * @param {Vec3} c the third operand
   * @param {Vec3} d the fourth operand
   * @param {number} t interpolation amount between the two inputs
   * @returns {Vec3} out
   */
  hermite(out: Vec3, a: Vec3, b: Vec3, c: Vec3, d: Vec3, t: number): Vec3;

  /**
   * Performs a bezier interpolation with two control points
   *
   * @param {Vec3} out the receiving vector
   * @param {Vec3} a the first operand
   * @param {Vec3} b the second operand
   * @param {Vec3} c the third operand
   * @param {Vec3} d the fourth operand
   * @param {number} t interpolation amount between the two inputs
   * @returns {Vec3} out
   */
  bezier(out: Vec3, a: Vec3, b: Vec3, c: Vec3, d: Vec3, t: number): Vec3;

  /**
   * Generates a random unit vector
   *
   * @param out the receiving vector
   * @returns out
   */
  random(out: Vec3): Vec3;

  /**
   * Generates a random vector with the given scale
   *
   * @param out the receiving vector
   * @param [scale] Length of the resulting vector. If omitted, a unit vector
   *     will be returned
   * @returns out
   */
  random(out: Vec3, scale: number): Vec3;

  /**
   * Transforms the Vec3 with a Mat3.
   *
   * @param out the receiving vector
   * @param a the vector to transform
   * @param m the 3x3 matrix to transform with
   * @returns out
   */
  transformMat3(out: Vec3, a: Vec3, m: Mat3): Vec3;

  /**
   * Transforms the Vec3 with a Mat4.
   * 4th vector component is implicitly '1'
   *
   * @param out the receiving vector
   * @param a the vector to transform
   * @param m matrix to transform with
   * @returns out
   */
  transformMat4(out: Vec3, a: Vec3, m: Mat4): Vec3;

  /**
   * Transforms the Vec3 with a Quat
   *
   * @param out the receiving vector
   * @param a the vector to transform
   * @param q Quaternion to transform with
   * @returns out
   */
  transformQuat(out: Vec3, a: Vec3, q: Quat): Vec3;


  /**
   * Rotate a 3D vector around the x-axis
   * @param out The receiving Vec3
   * @param a The Vec3 point to rotate
   * @param b The origin of the rotation
   * @param c The angle of rotation
   * @returns out
   */
  rotateX(out: Vec3, a: Vec3, b: Vec3, c: number): Vec3;

  /**
   * Rotate a 3D vector around the y-axis
   * @param out The receiving Vec3
   * @param a The Vec3 point to rotate
   * @param b The origin of the rotation
   * @param c The angle of rotation
   * @returns out
   */
  rotateY(out: Vec3, a: Vec3, b: Vec3, c: number): Vec3;

  /**
   * Rotate a 3D vector around the z-axis
   * @param out The receiving Vec3
   * @param a The Vec3 point to rotate
   * @param b The origin of the rotation
   * @param c The angle of rotation
   * @returns out
   */
  rotateZ(out: Vec3, a: Vec3, b: Vec3, c: number): Vec3;

  /**
   * Perform some operation over an array of vec3s.
   *
   * @param a the array of vectors to iterate over
   * @param stride Number of elements between the start of each Vec3. If 0
   *     assumes tightly packed
   * @param offset Number of elements to skip at the beginning of the array
   * @param count Number of vec3s to iterate over. If 0 iterates over entire
   *     array
   * @param fn Function to call for each vector in the array
   * @param arg additional argument to pass to fn
   * @returns a
   * @function
   */
  forEach(
      a: Float32Array, stride: number, offset: number, count: number,
      fn: (a: Vec3, b: Vec3, arg: any) => void, arg: any): Float32Array;

  /**
   * Perform some operation over an array of vec3s.
   *
   * @param a the array of vectors to iterate over
   * @param stride Number of elements between the start of each Vec3. If 0
   *     assumes tightly packed
   * @param offset Number of elements to skip at the beginning of the array
   * @param count Number of vec3s to iterate over. If 0 iterates over entire
   *     array
   * @param fn Function to call for each vector in the array
   * @returns a
   * @function
   */
  forEach(
      a: Float32Array, stride: number, offset: number, count: number,
      fn: (a: Vec3, b: Vec3) => void): Float32Array;

  /**
   * Get the angle between two 3D vectors
   * @param a The first operand
   * @param b The second operand
   * @returns The angle in radians
   */
  angle(a: Vec3, b: Vec3): number;

  /**
   * Returns a string representation of a vector
   *
   * @param a vector to represent as a string
   * @returns string representation of the vector
   */
  str(a: Vec3): string;

  /**
   * Returns whether or not the vectors have exactly the same elements in the
   * same position (when compared with ===)
   *
   * @param {Vec3} a The first vector.
   * @param {Vec3} b The second vector.
   * @returns {boolean} True if the vectors are equal, false otherwise.
   */
  exactEquals(a: Vec3, b: Vec3): boolean

  /**
   * Returns whether or not the vectors have approximately the same
   * elements in the same position.
   *
   * @param {Vec3} a The first vector.
   * @param {Vec3} b The second vector.
   * @returns {boolean} True if the vectors are equal, false otherwise.
   */
  equals(a: Vec3, b: Vec3): boolean
}

// Vec4
interface Vec4API {
  /**
   * Creates a new, empty Vec4
   *
   * @returns a new 4D vector
   */
  create(): Vec4;

  /**
   * Creates a new Vec4 initialized with values from an existing vector
   *
   * @param a vector to clone
   * @returns a new 4D vector
   */
  clone(a: Vec4): Vec4;

  /**
   * Creates a new Vec4 initialized with the given values
   *
   * @param x X component
   * @param y Y component
   * @param z Z component
   * @param w W component
   * @returns a new 4D vector
   */
  fromValues(x: number, y: number, z: number, w: number): Vec4;

  /**
   * Copy the values from one Vec4 to another
   *
   * @param out the receiving vector
   * @param a the source vector
   * @returns out
   */
  copy(out: Vec4, a: Vec4): Vec4;

  /**
   * Set the components of a Vec4 to the given values
   *
   * @param out the receiving vector
   * @param x X component
   * @param y Y component
   * @param z Z component
   * @param w W component
   * @returns out
   */
  set(out: Vec4, x: number, y: number, z: number, w: number): Vec4;

  /**
   * Adds two Vec4's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  add(out: Vec4, a: Vec4, b: Vec4): Vec4;

  /**
   * Subtracts vector b from vector a
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  subtract(out: Vec4, a: Vec4, b: Vec4): Vec4;

  /**
   * Subtracts vector b from vector a
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  sub(out: Vec4, a: Vec4, b: Vec4): Vec4;

  /**
   * Multiplies two Vec4's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  multiply(out: Vec4, a: Vec4, b: Vec4): Vec4;

  /**
   * Multiplies two Vec4's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  mul(out: Vec4, a: Vec4, b: Vec4): Vec4;

  /**
   * Divides two Vec4's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  divide(out: Vec4, a: Vec4, b: Vec4): Vec4;

  /**
   * Divides two Vec4's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  div(out: Vec4, a: Vec4, b: Vec4): Vec4;

  /**
   * Math.ceil the components of a Vec4
   *
   * @param {Vec4} out the receiving vector
   * @param {Vec4} a vector to ceil
   * @returns {Vec4} out
   */
  ceil(out: Vec4, a: Vec4): Vec4;

  /**
   * Math.floor the components of a Vec4
   *
   * @param {Vec4} out the receiving vector
   * @param {Vec4} a vector to floor
   * @returns {Vec4} out
   */
  floor(out: Vec4, a: Vec4): Vec4;

  /**
   * Returns the minimum of two Vec4's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  min(out: Vec4, a: Vec4, b: Vec4): Vec4;

  /**
   * Returns the maximum of two Vec4's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  max(out: Vec4, a: Vec4, b: Vec4): Vec4;

  /**
   * Math.round the components of a Vec4
   *
   * @param {Vec4} out the receiving vector
   * @param {Vec4} a vector to round
   * @returns {Vec4} out
   */
  round(out: Vec4, a: Vec4): Vec4;

  /**
   * Scales a Vec4 by a scalar number
   *
   * @param out the receiving vector
   * @param a the vector to scale
   * @param b amount to scale the vector by
   * @returns out
   */
  scale(out: Vec4, a: Vec4, b: number): Vec4;

  /**
   * Adds two Vec4's after scaling the second operand by a scalar value
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @param scale the amount to scale b by before adding
   * @returns out
   */
  scaleAndAdd(out: Vec4, a: Vec4, b: Vec4, scale: number): Vec4;

  /**
   * Calculates the euclidian distance between two Vec4's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns distance between a and b
   */
  distance(a: Vec4, b: Vec4): number;

  /**
   * Calculates the euclidian distance between two Vec4's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns distance between a and b
   */
  dist(a: Vec4, b: Vec4): number;

  /**
   * Calculates the squared euclidian distance between two Vec4's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns squared distance between a and b
   */
  squaredDistance(a: Vec4, b: Vec4): number;

  /**
   * Calculates the squared euclidian distance between two Vec4's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns squared distance between a and b
   */
  sqrDist(a: Vec4, b: Vec4): number;

  /**
   * Calculates the length of a Vec4
   *
   * @param a vector to calculate length of
   * @returns length of a
   */
  length(a: Vec4): number;

  /**
   * Calculates the length of a Vec4
   *
   * @param a vector to calculate length of
   * @returns length of a
   */
  len(a: Vec4): number;

  /**
   * Calculates the squared length of a Vec4
   *
   * @param a vector to calculate squared length of
   * @returns squared length of a
   */
  squaredLength(a: Vec4): number;

  /**
   * Calculates the squared length of a Vec4
   *
   * @param a vector to calculate squared length of
   * @returns squared length of a
   */
  sqrLen(a: Vec4): number;

  /**
   * Negates the components of a Vec4
   *
   * @param out the receiving vector
   * @param a vector to negate
   * @returns out
   */
  negate(out: Vec4, a: Vec4): Vec4;

  /**
   * Returns the inverse of the components of a Vec4
   *
   * @param out the receiving vector
   * @param a vector to invert
   * @returns out
   */
  inverse(out: Vec4, a: Vec4): Vec4;

  /**
   * Normalize a Vec4
   *
   * @param out the receiving vector
   * @param a vector to normalize
   * @returns out
   */
  normalize(out: Vec4, a: Vec4): Vec4;

  /**
   * Calculates the dot product of two Vec4's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns dot product of a and b
   */
  dot(a: Vec4, b: Vec4): number;

  /**
   * Performs a linear interpolation between two Vec4's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @param t interpolation amount between the two inputs
   * @returns out
   */
  lerp(out: Vec4, a: Vec4, b: Vec4, t: number): Vec4;

  /**
   * Generates a random unit vector
   *
   * @param out the receiving vector
   * @returns out
   */
  random(out: Vec4): Vec4;

  /**
   * Generates a random vector with the given scale
   *
   * @param out the receiving vector
   * @param scale length of the resulting vector. If ommitted, a unit vector
   *     will be returned
   * @returns out
   */
  random(out: Vec4, scale: number): Vec4;

  /**
   * Transforms the Vec4 with a Mat4.
   *
   * @param out the receiving vector
   * @param a the vector to transform
   * @param m matrix to transform with
   * @returns out
   */
  transformMat4(out: Vec4, a: Vec4, m: Mat4): Vec4;

  /**
   * Transforms the Vec4 with a Quat
   *
   * @param out the receiving vector
   * @param a the vector to transform
   * @param q Quaternion to transform with
   * @returns out
   */

  transformQuat(out: Vec4, a: Vec4, q: Quat): Vec4;

  /**
   * Perform some operation over an array of Vec4s.
   *
   * @param a the array of vectors to iterate over
   * @param stride Number of elements between the start of each Vec4. If 0
   *     assumes tightly packed
   * @param offset Number of elements to skip at the beginning of the array
   * @param count Number of Vec4s to iterate over. If 0 iterates over entire
   *     array
   * @param fn Function to call for each vector in the array
   * @param arg additional argument to pass to fn
   * @returns a
   * @function
   */
  forEach(
      a: Float32Array, stride: number, offset: number, count: number,
      fn: (a: Vec4, b: Vec4, arg: any) => void, arg: any): Float32Array;

  /**
   * Perform some operation over an array of Vec4s.
   *
   * @param a the array of vectors to iterate over
   * @param stride Number of elements between the start of each Vec4. If 0
   *     assumes tightly packed
   * @param offset Number of elements to skip at the beginning of the array
   * @param count Number of Vec4s to iterate over. If 0 iterates over entire
   *     array
   * @param fn Function to call for each vector in the array
   * @returns a
   * @function
   */
  forEach(
      a: Float32Array, stride: number, offset: number, count: number,
      fn: (a: Vec4, b: Vec4) => void): Float32Array;

  /**
   * Returns a string representation of a vector
   *
   * @param a vector to represent as a string
   * @returns string representation of the vector
   */
  str(a: Vec4): string;

  /**
   * Returns whether or not the vectors have exactly the same elements in the
   * same position (when compared with ===)
   *
   * @param {Vec4} a The first vector.
   * @param {Vec4} b The second vector.
   * @returns {boolean} True if the vectors are equal, false otherwise.
   */
  exactEquals(a: Vec4, b: Vec4): boolean;

  /**
   * Returns whether or not the vectors have approximately the same elements
   * in the same position.
   *
   * @param {Vec4} a The first vector.
   * @param {Vec4} b The second vector.
   * @returns {boolean} True if the vectors are equal, false otherwise.
   */
  equals(a: Vec4, b: Vec4): boolean;
}

// Mat2
interface Mat2API {
  /**
   * Creates a new identity Mat2
   *
   * @returns a new 2x2 matrix
   */
  create(): Mat2;

  /**
   * Creates a new Mat2 initialized with values from an existing matrix
   *
   * @param a matrix to clone
   * @returns a new 2x2 matrix
   */
  clone(a: Mat2): Mat2;

  /**
   * Copy the values from one Mat2 to another
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  copy(out: Mat2, a: Mat2): Mat2;

  /**
   * Set a Mat2 to the identity matrix
   *
   * @param out the receiving matrix
   * @returns out
   */
  identity(out: Mat2): Mat2;

  /**
   * Create a new Mat2 with the given values
   *
   * @param {number} m00 Component in column 0, row 0 position (index 0)
   * @param {number} m01 Component in column 0, row 1 position (index 1)
   * @param {number} m10 Component in column 1, row 0 position (index 2)
   * @param {number} m11 Component in column 1, row 1 position (index 3)
   * @returns {Mat2} out A new 2x2 matrix
   */
  fromValues(m00: number, m01: number, m10: number, m11: number): Mat2;

  /**
   * Set the components of a Mat2 to the given values
   *
   * @param {Mat2} out the receiving matrix
   * @param {number} m00 Component in column 0, row 0 position (index 0)
   * @param {number} m01 Component in column 0, row 1 position (index 1)
   * @param {number} m10 Component in column 1, row 0 position (index 2)
   * @param {number} m11 Component in column 1, row 1 position (index 3)
   * @returns {Mat2} out
   */
  set(out: Mat2, m00: number, m01: number, m10: number, m11: number): Mat2;

  /**
   * Transpose the values of a Mat2
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  transpose(out: Mat2, a: Mat2): Mat2;

  /**
   * Inverts a Mat2
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  invert(out: Mat2, a: Mat2): Mat2|null;

  /**
   * Calculates the adjugate of a Mat2
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  adjoint(out: Mat2, a: Mat2): Mat2;

  /**
   * Calculates the determinant of a Mat2
   *
   * @param a the source matrix
   * @returns determinant of a
   */
  determinant(a: Mat2): number;

  /**
   * Multiplies two Mat2's
   *
   * @param out the receiving matrix
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  multiply(out: Mat2, a: Mat2, b: Mat2): Mat2;

  /**
   * Multiplies two Mat2's
   *
   * @param out the receiving matrix
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  mul(out: Mat2, a: Mat2, b: Mat2): Mat2;

  /**
   * Rotates a Mat2 by the given angle
   *
   * @param out the receiving matrix
   * @param a the matrix to rotate
   * @param rad the angle to rotate the matrix by
   * @returns out
   */
  rotate(out: Mat2, a: Mat2, rad: number): Mat2;

  /**
   * Scales the Mat2 by the dimensions in the given Vec2
   *
   * @param out the receiving matrix
   * @param a the matrix to rotate
   * @param v the Vec2 to scale the matrix by
   * @returns out
   **/
  scale(out: Mat2, a: Mat2, v: Vec2): Mat2;

  /**
   * Creates a matrix from a given angle
   * This is equivalent to (but much faster than):
   *
   *     Mat2.identity(dest);
   *     Mat2.rotate(dest, dest, rad);
   *
   * @param {Mat2} out Mat2 receiving operation result
   * @param {number} rad the angle to rotate the matrix by
   * @returns {Mat2} out
   */
  fromRotation(out: Mat2, rad: number): Mat2;

  /**
   * Creates a matrix from a vector scaling
   * This is equivalent to (but much faster than):
   *
   *     Mat2.identity(dest);
   *     Mat2.scale(dest, dest, vec);
   *
   * @param {Mat2} out Mat2 receiving operation result
   * @param {Vec2} v Scaling vector
   * @returns {Mat2} out
   */
  fromScaling(out: Mat2, v: Vec2): Mat2;

  /**
   * Returns a string representation of a Mat2
   *
   * @param a matrix to represent as a string
   * @returns string representation of the matrix
   */
  str(a: Mat2): string;

  /**
   * Returns Frobenius norm of a Mat2
   *
   * @param a the matrix to calculate Frobenius norm of
   * @returns Frobenius norm
   */
  frob(a: Mat2): number;

  /**
   * Returns L, D and U matrices (Lower triangular, Diagonal and Upper
   * triangular) by factorizing the input matrix
   * @param L the lower triangular matrix
   * @param D the diagonal matrix
   * @param U the upper triangular matrix
   * @param a the input matrix to factorize
   */
  LDU(L: Mat2, D: Mat2, U: Mat2, a: Mat2): Mat2;

  /**
   * Adds two Mat2's
   *
   * @param {Mat2} out the receiving matrix
   * @param {Mat2} a the first operand
   * @param {Mat2} b the second operand
   * @returns {Mat2} out
   */
  add(out: Mat2, a: Mat2, b: Mat2): Mat2;

  /**
   * Subtracts matrix b from matrix a
   *
   * @param {Mat2} out the receiving matrix
   * @param {Mat2} a the first operand
   * @param {Mat2} b the second operand
   * @returns {Mat2} out
   */
  subtract(out: Mat2, a: Mat2, b: Mat2): Mat2;

  /**
   * Subtracts matrix b from matrix a
   *
   * @param {Mat2} out the receiving matrix
   * @param {Mat2} a the first operand
   * @param {Mat2} b the second operand
   * @returns {Mat2} out
   */
  sub(out: Mat2, a: Mat2, b: Mat2): Mat2;

  /**
   * Returns whether or not the matrices have exactly the same elements in the
   * same position (when compared with ===)
   *
   * @param {Mat2} a The first matrix.
   * @param {Mat2} b The second matrix.
   * @returns {boolean} True if the matrices are equal, false otherwise.
   */
  exactEquals(a: Mat2, b: Mat2): boolean;

  /**
   * Returns whether or not the matrices have approximately the same elements
   * in the same position.
   *
   * @param {Mat2} a The first matrix.
   * @param {Mat2} b The second matrix.
   * @returns {boolean} True if the matrices are equal, false otherwise.
   */
  equals(a: Mat2, b: Mat2): boolean;

  /**
   * Multiply each element of the matrix by a scalar.
   *
   * @param {Mat2} out the receiving matrix
   * @param {Mat2} a the matrix to scale
   * @param {number} b amount to scale the matrix's elements by
   * @returns {Mat2} out
   */
  multiplyScalar(out: Mat2, a: Mat2, b: number): Mat2

  /**
   * Adds two Mat2's after multiplying each element of the second operand
   * by a scalar value.
   *
   * @param {Mat2} out the receiving vector
   * @param {Mat2} a the first operand
   * @param {Mat2} b the second operand
   * @param {number} scale the amount to scale b's elements by before
   *     adding
   * @returns {Mat2} out
   */
  multiplyScalarAndAdd(out: Mat2, a: Mat2, b: Mat2, scale: number): Mat2
}

// Mat2d
interface Mat2dAPI {
  /**
   * Creates a new identity Mat2d
   *
   * @returns a new 2x3 matrix
   */
  create(): Mat2d;

  /**
   * Creates a new Mat2d initialized with values from an existing matrix
   *
   * @param a matrix to clone
   * @returns a new 2x3 matrix
   */
  clone(a: Mat2d): Mat2d;

  /**
   * Copy the values from one Mat2d to another
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  copy(out: Mat2d, a: Mat2d): Mat2d;

  /**
   * Set a Mat2d to the identity matrix
   *
   * @param out the receiving matrix
   * @returns out
   */
  identity(out: Mat2d): Mat2d;

  /**
   * Create a new Mat2d with the given values
   *
   * @param {number} a Component A (index 0)
   * @param {number} b Component B (index 1)
   * @param {number} c Component C (index 2)
   * @param {number} d Component D (index 3)
   * @param {number} tx Component TX (index 4)
   * @param {number} ty Component TY (index 5)
   * @returns {Mat2d} A new Mat2d
   */
  fromValues(
      a: number, b: number, c: number, d: number, tx: number, ty: number): Mat2d


  /**
   * Set the components of a Mat2d to the given values
   *
   * @param {Mat2d} out the receiving matrix
   * @param {number} a Component A (index 0)
   * @param {number} b Component B (index 1)
   * @param {number} c Component C (index 2)
   * @param {number} d Component D (index 3)
   * @param {number} tx Component TX (index 4)
   * @param {number} ty Component TY (index 5)
   * @returns {Mat2d} out
   */
  set(out: Mat2d, a: number, b: number, c: number, d: number, tx: number,
      ty: number): Mat2d

  /**
   * Inverts a Mat2d
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  invert(out: Mat2d, a: Mat2d): Mat2d|null;

  /**
   * Calculates the determinant of a Mat2d
   *
   * @param a the source matrix
   * @returns determinant of a
   */
  determinant(a: Mat2d): number;

  /**
   * Multiplies two Mat2d's
   *
   * @param out the receiving matrix
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  multiply(out: Mat2d, a: Mat2d, b: Mat2d): Mat2d;

  /**
   * Multiplies two Mat2d's
   *
   * @param out the receiving matrix
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  mul(out: Mat2d, a: Mat2d, b: Mat2d): Mat2d;

  /**
   * Rotates a Mat2d by the given angle
   *
   * @param out the receiving matrix
   * @param a the matrix to rotate
   * @param rad the angle to rotate the matrix by
   * @returns out
   */
  rotate(out: Mat2d, a: Mat2d, rad: number): Mat2d;

  /**
   * Scales the Mat2d by the dimensions in the given Vec2
   *
   * @param out the receiving matrix
   * @param a the matrix to translate
   * @param v the Vec2 to scale the matrix by
   * @returns out
   **/
  scale(out: Mat2d, a: Mat2d, v: Vec2): Mat2d;

  /**
   * Translates the Mat2d by the dimensions in the given Vec2
   *
   * @param out the receiving matrix
   * @param a the matrix to translate
   * @param v the Vec2 to translate the matrix by
   * @returns out
   **/
  translate(out: Mat2d, a: Mat2d, v: Vec2): Mat2d;

  /**
   * Creates a matrix from a given angle
   * This is equivalent to (but much faster than):
   *
   *     Mat2d.identity(dest);
   *     Mat2d.rotate(dest, dest, rad);
   *
   * @param {Mat2d} out Mat2d receiving operation result
   * @param {number} rad the angle to rotate the matrix by
   * @returns {Mat2d} out
   */
  fromRotation(out: Mat2d, rad: number): Mat2d;

  /**
   * Creates a matrix from a vector scaling
   * This is equivalent to (but much faster than):
   *
   *     Mat2d.identity(dest);
   *     Mat2d.scale(dest, dest, vec);
   *
   * @param {Mat2d} out Mat2d receiving operation result
   * @param {Vec2} v Scaling vector
   * @returns {Mat2d} out
   */
  fromScaling(out: Mat2d, v: Vec2): Mat2d;

  /**
   * Creates a matrix from a vector translation
   * This is equivalent to (but much faster than):
   *
   *     Mat2d.identity(dest);
   *     Mat2d.translate(dest, dest, vec);
   *
   * @param {Mat2d} out Mat2d receiving operation result
   * @param {Vec2} v Translation vector
   * @returns {Mat2d} out
   */
  fromTranslation(out: Mat2d, v: Vec2): Mat2d

  /**
   * Returns a string representation of a Mat2d
   *
   * @param a matrix to represent as a string
   * @returns string representation of the matrix
   */
  str(a: Mat2d): string;

  /**
   * Returns Frobenius norm of a Mat2d
   *
   * @param a the matrix to calculate Frobenius norm of
   * @returns Frobenius norm
   */
  frob(a: Mat2d): number;

  /**
   * Adds two Mat2d's
   *
   * @param {Mat2d} out the receiving matrix
   * @param {Mat2d} a the first operand
   * @param {Mat2d} b the second operand
   * @returns {Mat2d} out
   */
  add(out: Mat2d, a: Mat2d, b: Mat2d): Mat2d

  /**
   * Subtracts matrix b from matrix a
   *
   * @param {Mat2d} out the receiving matrix
   * @param {Mat2d} a the first operand
   * @param {Mat2d} b the second operand
   * @returns {Mat2d} out
   */
  subtract(out: Mat2d, a: Mat2d, b: Mat2d): Mat2d

  /**
   * Subtracts matrix b from matrix a
   *
   * @param {Mat2d} out the receiving matrix
   * @param {Mat2d} a the first operand
   * @param {Mat2d} b the second operand
   * @returns {Mat2d} out
   */
  sub(out: Mat2d, a: Mat2d, b: Mat2d): Mat2d

  /**
   * Multiply each element of the matrix by a scalar.
   *
   * @param {Mat2d} out the receiving matrix
   * @param {Mat2d} a the matrix to scale
   * @param {number} b amount to scale the matrix's elements by
   * @returns {Mat2d} out
   */
  multiplyScalar(out: Mat2d, a: Mat2d, b: number): Mat2d;

  /**
   * Adds two Mat2d's after multiplying each element of the second operand
   * by a scalar value.
   *
   * @param {Mat2d} out the receiving vector
   * @param {Mat2d} a the first operand
   * @param {Mat2d} b the second operand
   * @param {number} scale the amount to scale b's elements by before adding
   * @returns {Mat2d} out
   */
  multiplyScalarAndAdd(out: Mat2d, a: Mat2d, b: Mat2d, scale: number): Mat2d

  /**
   * Returns whether or not the matrices have exactly the same elements in
   * the same position (when compared with ===)
   *
   * @param {Mat2d} a The first matrix.
   * @param {Mat2d} b The second matrix.
   * @returns {boolean} True if the matrices are equal, false otherwise.
   */
  exactEquals(a: Mat2d, b: Mat2d): boolean;

  /**
   * Returns whether or not the matrices have approximately the same elements
   * in the same position.
   *
   * @param {Mat2d} a The first matrix.
   * @param {Mat2d} b The second matrix.
   * @returns {boolean} True if the matrices are equal, false otherwise.
   */
  equals(a: Mat2d, b: Mat2d): boolean
}

// Mat3
interface Mat3API {
  /**
   * Creates a new identity Mat3
   *
   * @returns a new 3x3 matrix
   */
  create(): Mat3;

  /**
   * Copies the upper-left 3x3 values into the given Mat3.
   *
   * @param {Mat3} out the receiving 3x3 matrix
   * @param {Mat4} a   the source 4x4 matrix
   * @returns {Mat3} out
   */
  fromMat4(out: Mat3, a: Mat4): Mat3

  /**
   * Creates a new Mat3 initialized with values from an existing matrix
   *
   * @param a matrix to clone
   * @returns a new 3x3 matrix
   */
  clone(a: Mat3): Mat3;

  /**
   * Copy the values from one Mat3 to another
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  copy(out: Mat3, a: Mat3): Mat3;

  /**
   * Create a new Mat3 with the given values
   *
   * @param {number} m00 Component in column 0, row 0 position (index 0)
   * @param {number} m01 Component in column 0, row 1 position (index 1)
   * @param {number} m02 Component in column 0, row 2 position (index 2)
   * @param {number} m10 Component in column 1, row 0 position (index 3)
   * @param {number} m11 Component in column 1, row 1 position (index 4)
   * @param {number} m12 Component in column 1, row 2 position (index 5)
   * @param {number} m20 Component in column 2, row 0 position (index 6)
   * @param {number} m21 Component in column 2, row 1 position (index 7)
   * @param {number} m22 Component in column 2, row 2 position (index 8)
   * @returns {Mat3} A new Mat3
   */
  fromValues(
      m00: number, m01: number, m02: number, m10: number, m11: number,
      m12: number, m20: number, m21: number, m22: number): Mat3;


  /**
   * Set the components of a Mat3 to the given values
   *
   * @param {Mat3} out the receiving matrix
   * @param {number} m00 Component in column 0, row 0 position (index 0)
   * @param {number} m01 Component in column 0, row 1 position (index 1)
   * @param {number} m02 Component in column 0, row 2 position (index 2)
   * @param {number} m10 Component in column 1, row 0 position (index 3)
   * @param {number} m11 Component in column 1, row 1 position (index 4)
   * @param {number} m12 Component in column 1, row 2 position (index 5)
   * @param {number} m20 Component in column 2, row 0 position (index 6)
   * @param {number} m21 Component in column 2, row 1 position (index 7)
   * @param {number} m22 Component in column 2, row 2 position (index 8)
   * @returns {Mat3} out
   */
  set(out: Mat3, m00: number, m01: number, m02: number, m10: number,
      m11: number, m12: number, m20: number, m21: number, m22: number): Mat3

  /**
   * Set a Mat3 to the identity matrix
   *
   * @param out the receiving matrix
   * @returns out
   */
  identity(out: Mat3): Mat3;

  /**
   * Transpose the values of a Mat3
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  transpose(out: Mat3, a: Mat3): Mat3;

  /**
   * Inverts a Mat3
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  invert(out: Mat3, a: Mat3): Mat3|null;

  /**
   * Calculates the adjugate of a Mat3
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  adjoint(out: Mat3, a: Mat3): Mat3;

  /**
   * Calculates the determinant of a Mat3
   *
   * @param a the source matrix
   * @returns determinant of a
   */
  determinant(a: Mat3): number;

  /**
   * Multiplies two Mat3's
   *
   * @param out the receiving matrix
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  multiply(out: Mat3, a: Mat3, b: Mat3): Mat3;

  /**
   * Multiplies two Mat3's
   *
   * @param out the receiving matrix
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  mul(out: Mat3, a: Mat3, b: Mat3): Mat3;


  /**
   * Translate a Mat3 by the given vector
   *
   * @param out the receiving matrix
   * @param a the matrix to translate
   * @param v vector to translate by
   * @returns out
   */
  translate(out: Mat3, a: Mat3, v: Vec3): Mat3;

  /**
   * Rotates a Mat3 by the given angle
   *
   * @param out the receiving matrix
   * @param a the matrix to rotate
   * @param rad the angle to rotate the matrix by
   * @returns out
   */
  rotate(out: Mat3, a: Mat3, rad: number): Mat3;

  /**
   * Scales the Mat3 by the dimensions in the given Vec2
   *
   * @param out the receiving matrix
   * @param a the matrix to rotate
   * @param v the Vec2 to scale the matrix by
   * @returns out
   **/
  scale(out: Mat3, a: Mat3, v: Vec2): Mat3;

  /**
   * Creates a matrix from a vector translation
   * This is equivalent to (but much faster than):
   *
   *     Mat3.identity(dest);
   *     Mat3.translate(dest, dest, vec);
   *
   * @param {Mat3} out Mat3 receiving operation result
   * @param {Vec2} v Translation vector
   * @returns {Mat3} out
   */
  fromTranslation(out: Mat3, v: Vec2): Mat3

  /**
   * Creates a matrix from a given angle
   * This is equivalent to (but much faster than):
   *
   *     Mat3.identity(dest);
   *     Mat3.rotate(dest, dest, rad);
   *
   * @param {Mat3} out Mat3 receiving operation result
   * @param {number} rad the angle to rotate the matrix by
   * @returns {Mat3} out
   */
  fromRotation(out: Mat3, rad: number): Mat3

  /**
   * Creates a matrix from a vector scaling
   * This is equivalent to (but much faster than):
   *
   *     Mat3.identity(dest);
   *     Mat3.scale(dest, dest, vec);
   *
   * @param {Mat3} out Mat3 receiving operation result
   * @param {Vec2} v Scaling vector
   * @returns {Mat3} out
   */
  fromScaling(out: Mat3, v: Vec2): Mat3

  /**
   * Copies the values from a Mat2d into a Mat3
   *
   * @param out the receiving matrix
   * @param {Mat2d} a the matrix to copy
   * @returns out
   **/
  fromMat2d(out: Mat3, a: Mat2d): Mat3;

  /**
   * Calculates a 3x3 matrix from the given Quaternion
   *
   * @param out Mat3 receiving operation result
   * @param q Quaternion to create matrix from
   *
   * @returns out
   */
  fromQuat(out: Mat3, q: Quat): Mat3;

  /**
   * Calculates a 3x3 normal matrix (transpose inverse) from the 4x4 matrix
   *
   * @param out Mat3 receiving operation result
   * @param a Mat4 to derive the normal matrix from
   *
   * @returns out
   */
  normalFromMat4(out: Mat3, a: Mat4): Mat3|null;

  /**
   * Returns a string representation of a Mat3
   *
   * @param mat matrix to represent as a string
   * @returns string representation of the matrix
   */
  str(mat: Mat3): string;

  /**
   * Returns Frobenius norm of a Mat3
   *
   * @param a the matrix to calculate Frobenius norm of
   * @returns Frobenius norm
   */
  frob(a: Mat3): number;

  /**
   * Adds two Mat3's
   *
   * @param {Mat3} out the receiving matrix
   * @param {Mat3} a the first operand
   * @param {Mat3} b the second operand
   * @returns {Mat3} out
   */
  add(out: Mat3, a: Mat3, b: Mat3): Mat3

  /**
   * Subtracts matrix b from matrix a
   *
   * @param {Mat3} out the receiving matrix
   * @param {Mat3} a the first operand
   * @param {Mat3} b the second operand
   * @returns {Mat3} out
   */
  subtract(out: Mat3, a: Mat3, b: Mat3): Mat3

  /**
   * Subtracts matrix b from matrix a
   *
   * @param {Mat3} out the receiving matrix
   * @param {Mat3} a the first operand
   * @param {Mat3} b the second operand
   * @returns {Mat3} out
   */
  sub(out: Mat3, a: Mat3, b: Mat3): Mat3

  /**
   * Multiply each element of the matrix by a scalar.
   *
   * @param {Mat3} out the receiving matrix
   * @param {Mat3} a the matrix to scale
   * @param {number} b amount to scale the matrix's elements by
   * @returns {Mat3} out
   */
  multiplyScalar(out: Mat3, a: Mat3, b: number): Mat3

  /**
   * Adds two Mat3's after multiplying each element of the second operand
   * by a scalar value.
   *
   * @param {Mat3} out the receiving vector
   * @param {Mat3} a the first operand
   * @param {Mat3} b the second operand
   * @param {number} scale the amount to scale b's elements by before
   *     adding
   * @returns {Mat3} out
   */
  multiplyScalarAndAdd(out: Mat3, a: Mat3, b: Mat3, scale: number): Mat3

  /**
   * Returns whether or not the matrices have exactly the same elements in
   * the same position (when compared with ===)
   *
   * @param {Mat3} a The first matrix.
   * @param {Mat3} b The second matrix.
   * @returns {boolean} True if the matrices are equal, false otherwise.
   */
  exactEquals(a: Mat3, b: Mat3): boolean;

  /**
   * Returns whether or not the matrices have approximately the same elements
   * in the same position.
   *
   * @param {Mat3} a The first matrix.
   * @param {Mat3} b The second matrix.
   * @returns {boolean} True if the matrices are equal, false otherwise.
   */
  equals(a: Mat3, b: Mat3): boolean
}

// Mat4
interface Mat4API {
  /**
   * Creates a new identity Mat4
   *
   * @returns a new 4x4 matrix
   */
  create(): Mat4;

  /**
   * Creates a new Mat4 initialized with values from an existing matrix
   *
   * @param a matrix to clone
   * @returns a new 4x4 matrix
   */
  clone(a: Mat4): Mat4;

  /**
   * Copy the values from one Mat4 to another
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  copy(out: Mat4, a: Mat4): Mat4;


  /**
   * Create a new Mat4 with the given values
   *
   * @param {number} m00 Component in column 0, row 0 position (index 0)
   * @param {number} m01 Component in column 0, row 1 position (index 1)
   * @param {number} m02 Component in column 0, row 2 position (index 2)
   * @param {number} m03 Component in column 0, row 3 position (index 3)
   * @param {number} m10 Component in column 1, row 0 position (index 4)
   * @param {number} m11 Component in column 1, row 1 position (index 5)
   * @param {number} m12 Component in column 1, row 2 position (index 6)
   * @param {number} m13 Component in column 1, row 3 position (index 7)
   * @param {number} m20 Component in column 2, row 0 position (index 8)
   * @param {number} m21 Component in column 2, row 1 position (index 9)
   * @param {number} m22 Component in column 2, row 2 position (index 10)
   * @param {number} m23 Component in column 2, row 3 position (index 11)
   * @param {number} m30 Component in column 3, row 0 position (index 12)
   * @param {number} m31 Component in column 3, row 1 position (index 13)
   * @param {number} m32 Component in column 3, row 2 position (index 14)
   * @param {number} m33 Component in column 3, row 3 position (index 15)
   * @returns {Mat4} A new Mat4
   */
  fromValues(
      m00: number, m01: number, m02: number, m03: number, m10: number,
      m11: number, m12: number, m13: number, m20: number, m21: number,
      m22: number, m23: number, m30: number, m31: number, m32: number,
      m33: number): Mat4;

  /**
   * Set the components of a Mat4 to the given values
   *
   * @param {Mat4} out the receiving matrix
   * @param {number} m00 Component in column 0, row 0 position (index 0)
   * @param {number} m01 Component in column 0, row 1 position (index 1)
   * @param {number} m02 Component in column 0, row 2 position (index 2)
   * @param {number} m03 Component in column 0, row 3 position (index 3)
   * @param {number} m10 Component in column 1, row 0 position (index 4)
   * @param {number} m11 Component in column 1, row 1 position (index 5)
   * @param {number} m12 Component in column 1, row 2 position (index 6)
   * @param {number} m13 Component in column 1, row 3 position (index 7)
   * @param {number} m20 Component in column 2, row 0 position (index 8)
   * @param {number} m21 Component in column 2, row 1 position (index 9)
   * @param {number} m22 Component in column 2, row 2 position (index 10)
   * @param {number} m23 Component in column 2, row 3 position (index 11)
   * @param {number} m30 Component in column 3, row 0 position (index 12)
   * @param {number} m31 Component in column 3, row 1 position (index 13)
   * @param {number} m32 Component in column 3, row 2 position (index 14)
   * @param {number} m33 Component in column 3, row 3 position (index 15)
   * @returns {Mat4} out
   */
  set(out: Mat4, m00: number, m01: number, m02: number, m03: number,
      m10: number, m11: number, m12: number, m13: number, m20: number,
      m21: number, m22: number, m23: number, m30: number, m31: number,
      m32: number, m33: number): Mat4;

  /**
   * Set a Mat4 to the identity matrix
   *
   * @param out the receiving matrix
   * @returns out
   */
  identity(out: Mat4): Mat4;

  /**
   * Transpose the values of a Mat4
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  transpose(out: Mat4, a: Mat4): Mat4;

  /**
   * Inverts a Mat4
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  invert(out: Mat4, a: Mat4): Mat4|null;

  /**
   * Calculates the adjugate of a Mat4
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  adjoint(out: Mat4, a: Mat4): Mat4;

  /**
   * Calculates the determinant of a Mat4
   *
   * @param a the source matrix
   * @returns determinant of a
   */
  determinant(a: Mat4): number;

  /**
   * Multiplies two Mat4's
   *
   * @param out the receiving matrix
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  multiply(out: Mat4, a: Mat4, b: Mat4): Mat4;

  /**
   * Multiplies two Mat4's
   *
   * @param out the receiving matrix
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  mul(out: Mat4, a: Mat4, b: Mat4): Mat4;

  /**
   * Translate a Mat4 by the given vector
   *
   * @param out the receiving matrix
   * @param a the matrix to translate
   * @param v vector to translate by
   * @returns out
   */
  translate(out: Mat4, a: Mat4, v: Vec3): Mat4;

  /**
   * Scales the Mat4 by the dimensions in the given Vec3
   *
   * @param out the receiving matrix
   * @param a the matrix to scale
   * @param v the Vec3 to scale the matrix by
   * @returns out
   **/
  scale(out: Mat4, a: Mat4, v: Vec3): Mat4;

  /**
   * Rotates a Mat4 by the given angle
   *
   * @param out the receiving matrix
   * @param a the matrix to rotate
   * @param rad the angle to rotate the matrix by
   * @param axis the axis to rotate around
   * @returns out
   */
  rotate(out: Mat4, a: Mat4, rad: number, axis: Vec3): Mat4;

  /**
   * Rotates a matrix by the given angle around the X axis
   *
   * @param out the receiving matrix
   * @param a the matrix to rotate
   * @param rad the angle to rotate the matrix by
   * @returns out
   */
  rotateX(out: Mat4, a: Mat4, rad: number): Mat4;

  /**
   * Rotates a matrix by the given angle around the Y axis
   *
   * @param out the receiving matrix
   * @param a the matrix to rotate
   * @param rad the angle to rotate the matrix by
   * @returns out
   */
  rotateY(out: Mat4, a: Mat4, rad: number): Mat4;

  /**
   * Rotates a matrix by the given angle around the Z axis
   *
   * @param out the receiving matrix
   * @param a the matrix to rotate
   * @param rad the angle to rotate the matrix by
   * @returns out
   */
  rotateZ(out: Mat4, a: Mat4, rad: number): Mat4;

  /**
   * Creates a matrix from a vector translation
   * This is equivalent to (but much faster than):
   *
   *     Mat4.identity(dest);
   *     Mat4.translate(dest, dest, vec);
   *
   * @param {Mat4} out Mat4 receiving operation result
   * @param {Vec3} v Translation vector
   * @returns {Mat4} out
   */
  fromTranslation(out: Mat4, v: Vec3): Mat4

  /**
   * Creates a matrix from a vector scaling
   * This is equivalent to (but much faster than):
   *
   *     Mat4.identity(dest);
   *     Mat4.scale(dest, dest, vec);
   *
   * @param {Mat4} out Mat4 receiving operation result
   * @param {Vec3} v Scaling vector
   * @returns {Mat4} out
   */
  fromScaling(out: Mat4, v: Vec3): Mat4

  /**
   * Creates a matrix from a given angle around a given axis
   * This is equivalent to (but much faster than):
   *
   *     Mat4.identity(dest);
   *     Mat4.rotate(dest, dest, rad, axis);
   *
   * @param {Mat4} out Mat4 receiving operation result
   * @param {number} rad the angle to rotate the matrix by
   * @param {Vec3} axis the axis to rotate around
   * @returns {Mat4} out
   */
  fromRotation(out: Mat4, rad: number, axis: Vec3): Mat4

  /**
   * Creates a matrix from the given angle around the X axis
   * This is equivalent to (but much faster than):
   *
   *     Mat4.identity(dest);
   *     Mat4.rotateX(dest, dest, rad);
   *
   * @param {Mat4} out Mat4 receiving operation result
   * @param {number} rad the angle to rotate the matrix by
   * @returns {Mat4} out
   */
  fromXRotation(out: Mat4, rad: number): Mat4

  /**
   * Creates a matrix from the given angle around the Y axis
   * This is equivalent to (but much faster than):
   *
   *     Mat4.identity(dest);
   *     Mat4.rotateY(dest, dest, rad);
   *
   * @param {Mat4} out Mat4 receiving operation result
   * @param {number} rad the angle to rotate the matrix by
   * @returns {Mat4} out
   */
  fromYRotation(out: Mat4, rad: number): Mat4


  /**
   * Creates a matrix from the given angle around the Z axis
   * This is equivalent to (but much faster than):
   *
   *     Mat4.identity(dest);
   *     Mat4.rotateZ(dest, dest, rad);
   *
   * @param {Mat4} out Mat4 receiving operation result
   * @param {number} rad the angle to rotate the matrix by
   * @returns {Mat4} out
   */
  fromZRotation(out: Mat4, rad: number): Mat4

  /**
   * Creates a matrix from a Quaternion rotation and vector translation
   * This is equivalent to (but much faster than):
   *
   *     Mat4.identity(dest);
   *     Mat4.translate(dest, vec);
   *     var QuatMat = Mat4.create();
   *     Quat4.toMat4(Quat, QuatMat);
   *     Mat4.multiply(dest, QuatMat);
   *
   * @param out Mat4 receiving operation result
   * @param q Rotation Quaternion
   * @param v Translation vector
   * @returns out
   */
  fromRotationTranslation(out: Mat4, q: Quat, v: Vec3): Mat4;

  /**
   * Returns the translation vector component of a transformation
   *  matrix. If a matrix is built with fromRotationTranslation,
   *  the returned vector will be the same as the translation vector
   *  originally supplied.
   * @param  {Vec3} out Vector to receive translation component
   * @param  {Mat4} mat Matrix to be decomposed (input)
   * @return {Vec3} out
   */
  getTranslation(out: Vec3, mat: Mat4): Vec3;

  /**
   * Returns a Quaternion representing the rotational component
   *  of a transformation matrix. If a matrix is built with
   *  fromRotationTranslation, the returned Quaternion will be the
   *  same as the Quaternion originally supplied.
   * @param {Quat} out Quaternion to receive the rotation component
   * @param {Mat4} mat Matrix to be decomposed (input)
   * @return {Quat} out
   */
  getRotation(out: Quat, mat: Mat4): Quat;

  /**
   * Creates a matrix from a Quaternion rotation, vector translation and
   * vector scale This is equivalent to (but much faster than):
   *
   *     Mat4.identity(dest);
   *     Mat4.translate(dest, vec);
   *     var QuatMat = Mat4.create();
   *     Quat4.toMat4(Quat, QuatMat);
   *     Mat4.multiply(dest, QuatMat);
   *     Mat4.scale(dest, scale)
   *
   * @param out Mat4 receiving operation result
   * @param q Rotation Quaternion
   * @param v Translation vector
   * @param s Scaling vector
   * @returns out
   */
  fromRotationTranslationScale(out: Mat4, q: Quat, v: Vec3, s: Vec3): Mat4;

  /**
   * Creates a matrix from a Quaternion rotation, vector translation and
   * vector scale, rotating and scaling around the given origin This is
   * equivalent to (but much faster than):
   *
   *     Mat4.identity(dest);
   *     Mat4.translate(dest, vec);
   *     Mat4.translate(dest, origin);
   *     var QuatMat = Mat4.create();
   *     Quat4.toMat4(Quat, QuatMat);
   *     Mat4.multiply(dest, QuatMat);
   *     Mat4.scale(dest, scale)
   *     Mat4.translate(dest, negativeOrigin);
   *
   * @param {Mat4} out Mat4 receiving operation result
   * @param {Quat} q Rotation Quaternion
   * @param {Vec3} v Translation vector
   * @param {Vec3} s Scaling vector
   * @param {Vec3} o The origin vector around which to scale and rotate
   * @returns {Mat4} out
   */
  fromRotationTranslationScaleOrigin(
      out: Mat4, q: Quat, v: Vec3, s: Vec3, o: Vec3): Mat4

  /**
   * Calculates a 4x4 matrix from the given Quaternion
   *
   * @param {Mat4} out Mat4 receiving operation result
   * @param {Quat} q Quaternion to create matrix from
   *
   * @returns {Mat4} out
   */
  fromQuat(out: Mat4, q: Quat): Mat4

  /**
   * Generates a frustum matrix with the given bounds
   *
   * @param out Mat4 frustum matrix will be written into
   * @param left Left bound of the frustum
   * @param right Right bound of the frustum
   * @param bottom Bottom bound of the frustum
   * @param top Top bound of the frustum
   * @param near Near bound of the frustum
   * @param far Far bound of the frustum
   * @returns out
   */
  frustum(
      out: Mat4, left: number, right: number, bottom: number, top: number,
      near: number, far: number): Mat4;

  /**
   * Generates a perspective projection matrix with the given bounds
   *
   * @param out Mat4 frustum matrix will be written into
   * @param fovy Vertical field of view in radians
   * @param aspect Aspect ratio. typically viewport width/height
   * @param near Near bound of the frustum
   * @param far Far bound of the frustum
   * @returns out
   */
  perspective(
      out: Mat4, fovy: number, aspect: number, near: number, far: number): Mat4;

  /**
   * Generates a perspective projection matrix with the given field of view.
   * This is primarily useful for generating projection matrices to be used
   * with the still experimental WebVR API.
   *
   * @param {Mat4} out Mat4 frustum matrix will be written into
   * @param {Object} fov Object containing the following values: upDegrees,
   *     downDegrees, leftDegrees, rightDegrees
   * @param {number} near Near bound of the frustum
   * @param {number} far Far bound of the frustum
   * @returns {Mat4} out
   */
  perspectiveFromFieldOfView(
      out: Mat4, fov: {
        upDegrees: number,
        downDegrees: number,
        leftDegrees: number,
        rightDegrees: number
      },
      near: number, far: number): Mat4

  /**
   * Generates a orthogonal projection matrix with the given bounds
   *
   * @param out Mat4 frustum matrix will be written into
   * @param left Left bound of the frustum
   * @param right Right bound of the frustum
   * @param bottom Bottom bound of the frustum
   * @param top Top bound of the frustum
   * @param near Near bound of the frustum
   * @param far Far bound of the frustum
   * @returns out
   */
  ortho(
      out: Mat4, left: number, right: number, bottom: number, top: number,
      near: number, far: number): Mat4;

  /**
   * Generates a look-at matrix with the given eye position, focal point, and
   * up axis
   *
   * @param out Mat4 frustum matrix will be written into
   * @param eye Position of the viewer
   * @param center Point the viewer is looking at
   * @param up Vec3 pointing up
   * @returns out
   */
  lookAt(out: Mat4, eye: Vec3, center: Vec3, up: Vec3): Mat4;

  /**
   * Returns a string representation of a Mat4
   *
   * @param mat matrix to represent as a string
   * @returns string representation of the matrix
   */
  str(mat: Mat4): string;

  /**
   * Returns Frobenius norm of a Mat4
   *
   * @param a the matrix to calculate Frobenius norm of
   * @returns Frobenius norm
   */
  frob(a: Mat4): number;

  /**
   * Adds two Mat4's
   *
   * @param {Mat4} out the receiving matrix
   * @param {Mat4} a the first operand
   * @param {Mat4} b the second operand
   * @returns {Mat4} out
   */
  add(out: Mat4, a: Mat4, b: Mat4): Mat4

  /**
   * Subtracts matrix b from matrix a
   *
   * @param {Mat4} out the receiving matrix
   * @param {Mat4} a the first operand
   * @param {Mat4} b the second operand
   * @returns {Mat4} out
   */
  subtract(out: Mat4, a: Mat4, b: Mat4): Mat4

  /**
   * Subtracts matrix b from matrix a
   *
   * @param {Mat4} out the receiving matrix
   * @param {Mat4} a the first operand
   * @param {Mat4} b the second operand
   * @returns {Mat4} out
   */
  sub(out: Mat4, a: Mat4, b: Mat4): Mat4

  /**
   * Multiply each element of the matrix by a scalar.
   *
   * @param {Mat4} out the receiving matrix
   * @param {Mat4} a the matrix to scale
   * @param {number} b amount to scale the matrix's elements by
   * @returns {Mat4} out
   */
  multiplyScalar(out: Mat4, a: Mat4, b: number): Mat4

  /**
   * Adds two Mat4's after multiplying each element of the second operand
   * by a scalar value.
   *
   * @param {Mat4} out the receiving vector
   * @param {Mat4} a the first operand
   * @param {Mat4} b the second operand
   * @param {number} scale the amount to scale b's elements by before
   *     adding
   * @returns {Mat4} out
   */
  multiplyScalarAndAdd(out: Mat4, a: Mat4, b: Mat4, scale: number): Mat4

  /**
   * Returns whether or not the matrices have exactly the same elements in
   * the same position (when compared with ===)
   *
   * @param {Mat4} a The first matrix.
   * @param {Mat4} b The second matrix.
   * @returns {boolean} True if the matrices are equal, false otherwise.
   */
  exactEquals(a: Mat4, b: Mat4): boolean

  /**
   * Returns whether or not the matrices have approximately the same
   * elements in the same position.
   *
   * @param {Mat4} a The first matrix.
   * @param {Mat4} b The second matrix.
   * @returns {boolean} True if the matrices are equal, false otherwise.
   */
  equals(a: Mat4, b: Mat4): boolean
}

// Quat
interface QuatAPI {
  /**
   * Creates a new identity Quat
   *
   * @returns a new Quaternion
   */
  create(): Quat;

  /**
   * Creates a new Quat initialized with values from an existing Quaternion
   *
   * @param a Quaternion to clone
   * @returns a new Quaternion
   * @function
   */
  clone(a: Quat): Quat;

  /**
   * Creates a new Quat initialized with the given values
   *
   * @param x X component
   * @param y Y component
   * @param z Z component
   * @param w W component
   * @returns a new Quaternion
   * @function
   */
  fromValues(x: number, y: number, z: number, w: number): Quat;

  /**
   * Copy the values from one Quat to another
   *
   * @param out the receiving Quaternion
   * @param a the source Quaternion
   * @returns out
   * @function
   */
  copy(out: Quat, a: Quat): Quat;

  /**
   * Set the components of a Quat to the given values
   *
   * @param out the receiving Quaternion
   * @param x X component
   * @param y Y component
   * @param z Z component
   * @param w W component
   * @returns out
   * @function
   */
  set(out: Quat, x: number, y: number, z: number, w: number): Quat;

  /**
   * Set a Quat to the identity Quaternion
   *
   * @param out the receiving Quaternion
   * @returns out
   */
  identity(out: Quat): Quat;

  /**
   * Sets a Quaternion to represent the shortest rotation from one
   * vector to another.
   *
   * Both vectors are assumed to be unit length.
   *
   * @param {Quat} out the receiving Quaternion.
   * @param {Vec3} a the initial vector
   * @param {Vec3} b the destination vector
   * @returns {Quat} out
   */
  rotationTo(out: Quat, a: Vec3, b: Vec3): Quat;

  /**
   * Sets the specified Quaternion with values corresponding to the given
   * axes. Each axis is a Vec3 and is expected to be unit length and
   * perpendicular to all other specified axes.
   *
   * @param {Vec3} view  the vector representing the viewing direction
   * @param {Vec3} right the vector representing the local "right" direction
   * @param {Vec3} up    the vector representing the local "up" direction
   * @returns {Quat} out
   */
  setAxes(out: Quat, view: Vec3, right: Vec3, up: Vec3): Quat



  /**
   * Sets a Quat from the given angle and rotation axis,
   * then returns it.
   *
   * @param out the receiving Quaternion
   * @param axis the axis around which to rotate
   * @param rad the angle in radians
   * @returns out
   **/
  setAxisAngle(out: Quat, axis: Vec3, rad: number): Quat;

  /**
   * Gets the rotation axis and angle for a given
   *  Quaternion. If a Quaternion is created with
   *  setAxisAngle, this method will return the same
   *  values as providied in the original parameter list
   *  OR functionally equivalent values.
   * Example: The Quaternion formed by axis [0, 0, 1] and
   *  angle -90 is the same as the Quaternion formed by
   *  [0, 0, 1] and 270. This method favors the latter.
   * @param  {Vec3} out_axis  Vector receiving the axis of rotation
   * @param  {Quat} q     Quaternion to be decomposed
   * @return {number}     Angle, in radians, of the rotation
   */
  getAxisAngle(out_axis: Vec3, q: Quat): number

  /**
   * Adds two Quat's
   *
   * @param out the receiving Quaternion
   * @param a the first operand
   * @param b the second operand
   * @returns out
   * @function
   */
  add(out: Quat, a: Quat, b: Quat): Quat;

  /**
   * Multiplies two Quat's
   *
   * @param out the receiving Quaternion
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  multiply(out: Quat, a: Quat, b: Quat): Quat;

  /**
   * Multiplies two Quat's
   *
   * @param out the receiving Quaternion
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  mul(out: Quat, a: Quat, b: Quat): Quat;

  /**
   * Scales a Quat by a scalar number
   *
   * @param out the receiving vector
   * @param a the vector to scale
   * @param b amount to scale the vector by
   * @returns out
   * @function
   */
  scale(out: Quat, a: Quat, b: number): Quat;

  /**
   * Calculates the length of a Quat
   *
   * @param a vector to calculate length of
   * @returns length of a
   * @function
   */
  length(a: Quat): number;

  /**
   * Calculates the length of a Quat
   *
   * @param a vector to calculate length of
   * @returns length of a
   * @function
   */
  len(a: Quat): number;

  /**
   * Calculates the squared length of a Quat
   *
   * @param a vector to calculate squared length of
   * @returns squared length of a
   * @function
   */
  squaredLength(a: Quat): number;

  /**
   * Calculates the squared length of a Quat
   *
   * @param a vector to calculate squared length of
   * @returns squared length of a
   * @function
   */
  sqrLen(a: Quat): number;

  /**
   * Normalize a Quat
   *
   * @param out the receiving Quaternion
   * @param a Quaternion to normalize
   * @returns out
   * @function
   */
  normalize(out: Quat, a: Quat): Quat;

  /**
   * Calculates the dot product of two Quat's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns dot product of a and b
   * @function
   */
  dot(a: Quat, b: Quat): number;

  /**
   * Performs a linear interpolation between two Quat's
   *
   * @param out the receiving Quaternion
   * @param a the first operand
   * @param b the second operand
   * @param t interpolation amount between the two inputs
   * @returns out
   * @function
   */
  lerp(out: Quat, a: Quat, b: Quat, t: number): Quat;

  /**
   * Performs a spherical linear interpolation between two Quat
   *
   * @param out the receiving Quaternion
   * @param a the first operand
   * @param b the second operand
   * @param t interpolation amount between the two inputs
   * @returns out
   */
  slerp(out: Quat, a: Quat, b: Quat, t: number): Quat;

  /**
   * Performs a spherical linear interpolation with two control points
   *
   * @param {Quat} out the receiving Quaternion
   * @param {Quat} a the first operand
   * @param {Quat} b the second operand
   * @param {Quat} c the third operand
   * @param {Quat} d the fourth operand
   * @param {number} t interpolation amount
   * @returns {Quat} out
   */
  sqlerp(out: Quat, a: Quat, b: Quat, c: Quat, d: Quat, t: number): Quat;

  /**
   * Calculates the inverse of a Quat
   *
   * @param out the receiving Quaternion
   * @param a Quat to calculate inverse of
   * @returns out
   */
  invert(out: Quat, a: Quat): Quat;

  /**
   * Calculates the conjugate of a Quat
   * If the Quaternion is normalized, this function is faster than
   * Quat.inverse and produces the same result.
   *
   * @param out the receiving Quaternion
   * @param a Quat to calculate conjugate of
   * @returns out
   */
  conjugate(out: Quat, a: Quat): Quat;

  /**
   * Returns a string representation of a Quaternion
   *
   * @param a Quat to represent as a string
   * @returns string representation of the Quat
   */
  str(a: Quat): string;

  /**
   * Rotates a Quaternion by the given angle about the X axis
   *
   * @param out Quat receiving operation result
   * @param a Quat to rotate
   * @param rad angle (in radians) to rotate
   * @returns out
   */
  rotateX(out: Quat, a: Quat, rad: number): Quat;

  /**
   * Rotates a Quaternion by the given angle about the Y axis
   *
   * @param out Quat receiving operation result
   * @param a Quat to rotate
   * @param rad angle (in radians) to rotate
   * @returns out
   */
  rotateY(out: Quat, a: Quat, rad: number): Quat;

  /**
   * Rotates a Quaternion by the given angle about the Z axis
   *
   * @param out Quat receiving operation result
   * @param a Quat to rotate
   * @param rad angle (in radians) to rotate
   * @returns out
   */
  rotateZ(out: Quat, a: Quat, rad: number): Quat;

  /**
   * Creates a Quaternion from the given 3x3 rotation matrix.
   *
   * NOTE: The resultant Quaternion is not normalized, so you should be sure
   * to renormalize the Quaternion yourself where necessary.
   *
   * @param out the receiving Quaternion
   * @param m rotation matrix
   * @returns out
   * @function
   */
  fromMat3(out: Quat, m: Mat3): Quat;

  /**
   * Sets the specified Quaternion with values corresponding to the given
   * axes. Each axis is a Vec3 and is expected to be unit length and
   * perpendicular to all other specified axes.
   *
   * @param out the receiving Quat
   * @param view  the vector representing the viewing direction
   * @param right the vector representing the local "right" direction
   * @param up    the vector representing the local "up" direction
   * @returns out
   */
  setAxes(out: Quat, view: Vec3, right: Vec3, up: Vec3): Quat;

  /**
   * Sets a Quaternion to represent the shortest rotation from one
   * vector to another.
   *
   * Both vectors are assumed to be unit length.
   *
   * @param out the receiving Quaternion.
   * @param a the initial vector
   * @param b the destination vector
   * @returns out
   */
  rotationTo(out: Quat, a: Vec3, b: Vec3): Quat;

  /**
   * Calculates the W component of a Quat from the X, Y, and Z components.
   * Assumes that Quaternion is 1 unit in length.
   * Any existing W component will be ignored.
   *
   * @param out the receiving Quaternion
   * @param a Quat to calculate W component of
   * @returns out
   */
  calculateW(out: Quat, a: Quat): Quat;

  /**
   * Returns whether or not the Quaternions have exactly the same elements in
   * the same position (when compared with ===)
   *
   * @param {Quat} a The first vector.
   * @param {Quat} b The second vector.
   * @returns {boolean} True if the Quaternions are equal, false otherwise.
   */
  exactEquals(a: Quat, b: Quat): boolean;

  /**
   * Returns whether or not the Quaternions have approximately the same
   * elements in the same position.
   *
   * @param {Quat} a The first vector.
   * @param {Quat} b The second vector.
   * @returns {boolean} True if the Quaternions are equal, false otherwise.
   */
  equals(a: Quat, b: Quat): boolean;
}
