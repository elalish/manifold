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

declare function show(manifold: Manifold): Manifold;

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

  vec2: Vec2Type;
  vec3: Vec3Type;
  vec4: Vec4Type;
  mat2: Mat2Type;
  mat2d: Mat2dType;
  mat3: Mat3Type;
  mat4: Mat4Type;
  quat: QuatType;
}

interface Vec2Type extends Float32Array {
  /**
   * Creates a new, empty Vec2Type
   *
   * @returns a new 2D vector
   */
  create(): Vec2Type;

  /**
   * Creates a new Vec2Type initialized with values from an existing vector
   *
   * @param a a vector to clone
   * @returns a new 2D vector
   */
  clone(a: Vec2Type|number[]): Vec2Type;

  /**
   * Creates a new Vec2Type initialized with the given values
   *
   * @param x X component
   * @param y Y component
   * @returns a new 2D vector
   */
  fromValues(x: number, y: number): Vec2Type;

  /**
   * Copy the values from one Vec2Type to another
   *
   * @param out the receiving vector
   * @param a the source vector
   * @returns out
   */
  copy(out: Vec2Type, a: Vec2Type|number[]): Vec2Type;

  /**
   * Set the components of a Vec2Type to the given values
   *
   * @param out the receiving vector
   * @param x X component
   * @param y Y component
   * @returns out
   */
  set(out: Vec2Type, x: number, y: number): Vec2Type;

  /**
   * Adds two Vec2Type's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  add(out: Vec2Type, a: Vec2Type|number[], b: Vec2Type|number[]): Vec2Type;

  /**
   * Subtracts vector b from vector a
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  subtract(out: Vec2Type, a: Vec2Type|number[], b: Vec2Type|number[]): Vec2Type;

  /**
   * Subtracts vector b from vector a
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  sub(out: Vec2Type, a: Vec2Type|number[], b: Vec2Type|number[]): Vec2Type;

  /**
   * Multiplies two Vec2Type's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  multiply(out: Vec2Type, a: Vec2Type|number[], b: Vec2Type|number[]): Vec2Type;

  /**
   * Multiplies two Vec2Type's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  mul(out: Vec2Type, a: Vec2Type|number[], b: Vec2Type|number[]): Vec2Type;

  /**
   * Divides two Vec2Type's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  divide(out: Vec2Type, a: Vec2Type|number[], b: Vec2Type|number[]): Vec2Type;

  /**
   * Divides two Vec2Type's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  div(out: Vec2Type, a: Vec2Type|number[], b: Vec2Type|number[]): Vec2Type;

  /**
   * Math.ceil the components of a Vec2Type
   *
   * @param {Vec2Type} out the receiving vector
   * @param {Vec2Type} a vector to ceil
   * @returns {Vec2Type} out
   */
  ceil(out: Vec2Type, a: Vec2Type|number[]): Vec2Type;

  /**
   * Math.floor the components of a Vec2Type
   *
   * @param {Vec2Type} out the receiving vector
   * @param {Vec2Type} a vector to floor
   * @returns {Vec2Type} out
   */
  floor(out: Vec2Type, a: Vec2Type|number[]): Vec2Type;

  /**
   * Returns the minimum of two Vec2Type's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  min(out: Vec2Type, a: Vec2Type|number[], b: Vec2Type|number[]): Vec2Type;

  /**
   * Returns the maximum of two Vec2Type's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  max(out: Vec2Type, a: Vec2Type|number[], b: Vec2Type|number[]): Vec2Type;

  /**
   * Rotates the 2D vector.
   *
   * @param {Vec2Type} out the receiving vector
   * @param {Vec2Type} a vector to rotate
   * @param {Vec2Type} b origin of the rotation
   * @param {number} rad angle of rotation in radians
   * @returns {Vec2Type} out
   */
  rotate(
      out: Vec2Type, a: Vec2Type|number[], b: Vec2Type|number[],
      rad: number): Vec2Type;

  /**
   * Math.round the components of a Vec2Type
   *
   * @param {Vec2Type} out the receiving vector
   * @param {Vec2Type} a vector to round
   * @returns {Vec2Type} out
   */
  round(out: Vec2Type, a: Vec2Type|number[]): Vec2Type;

  /**
   * Scales a Vec2Type by a scalar number
   *
   * @param out the receiving vector
   * @param a the vector to scale
   * @param b amount to scale the vector by
   * @returns out
   */
  scale(out: Vec2Type, a: Vec2Type|number[], b: number): Vec2Type;

  /**
   * Adds two Vec2Type's after scaling the second operand by a scalar value
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @param scale the amount to scale b by before adding
   * @returns out
   */
  scaleAndAdd(
      out: Vec2Type, a: Vec2Type|number[], b: Vec2Type|number[],
      scale: number): Vec2Type;

  /**
   * Calculates the euclidian distance between two Vec2Type's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns distance between a and b
   */
  distance(a: Vec2Type|number[], b: Vec2Type|number[]): number;

  /**
   * Calculates the euclidian distance between two Vec2Type's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns distance between a and b
   */
  dist(a: Vec2Type|number[], b: Vec2Type|number[]): number;

  /**
   * Calculates the squared euclidian distance between two Vec2Type's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns squared distance between a and b
   */
  squaredDistance(a: Vec2Type|number[], b: Vec2Type|number[]): number;

  /**
   * Calculates the squared euclidian distance between two Vec2Type's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns squared distance between a and b
   */
  sqrDist(a: Vec2Type|number[], b: Vec2Type|number[]): number;

  /**
   * Calculates the length of a Vec2Type
   *
   * @param a vector to calculate length of
   * @returns length of a
   */
  length(a: Vec2Type|number[]): number;

  /**
   * Calculates the length of a Vec2Type
   *
   * @param a vector to calculate length of
   * @returns length of a
   */
  len(a: Vec2Type|number[]): number;

  /**
   * Calculates the squared length of a Vec2Type
   *
   * @param a vector to calculate squared length of
   * @returns squared length of a
   */
  squaredLength(a: Vec2Type|number[]): number;

  /**
   * Calculates the squared length of a Vec2Type
   *
   * @param a vector to calculate squared length of
   * @returns squared length of a
   */
  sqrLen(a: Vec2Type|number[]): number;

  /**
   * Negates the components of a Vec2Type
   *
   * @param out the receiving vector
   * @param a vector to negate
   * @returns out
   */
  negate(out: Vec2Type, a: Vec2Type|number[]): Vec2Type;

  /**
   * Returns the inverse of the components of a Vec2Type
   *
   * @param out the receiving vector
   * @param a vector to invert
   * @returns out
   */
  inverse(out: Vec2Type, a: Vec2Type|number[]): Vec2Type;

  /**
   * Normalize a Vec2Type
   *
   * @param out the receiving vector
   * @param a vector to normalize
   * @returns out
   */
  normalize(out: Vec2Type, a: Vec2Type|number[]): Vec2Type;

  /**
   * Calculates the dot product of two Vec2Type's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns dot product of a and b
   */
  dot(a: Vec2Type|number[], b: Vec2Type|number[]): number;

  /**
   * Computes the cross product of two Vec2Type's
   * Note that the cross product must by definition produce a 3D vector
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  cross(out: Vec2Type, a: Vec2Type|number[], b: Vec2Type|number[]): Vec2Type;

  /**
   * Performs a linear interpolation between two Vec2Type's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @param t interpolation amount between the two inputs
   * @returns out
   */
  lerp(out: Vec2Type, a: Vec2Type|number[], b: Vec2Type|number[], t: number):
      Vec2Type;

  /**
   * Generates a random unit vector
   *
   * @param out the receiving vector
   * @returns out
   */
  random(out: Vec2Type): Vec2Type;

  /**
   * Generates a random vector with the given scale
   *
   * @param out the receiving vector
   * @param scale Length of the resulting vector. If ommitted, a unit vector
   *     will be returned
   * @returns out
   */
  random(out: Vec2Type, scale: number): Vec2Type;

  /**
   * Transforms the Vec2Type with a Mat2Type
   *
   * @param out the receiving vector
   * @param a the vector to transform
   * @param m matrix to transform with
   * @returns out
   */
  transformMat2(out: Vec2Type, a: Vec2Type|number[], m: Mat2Type): Vec2Type;

  /**
   * Transforms the Vec2Type with a Mat2dType
   *
   * @param out the receiving vector
   * @param a the vector to transform
   * @param m matrix to transform with
   * @returns out
   */
  transformMat2d(out: Vec2Type, a: Vec2Type|number[], m: Mat2dType): Vec2Type;

  /**
   * Transforms the Vec2Type with a Mat3Type
   * 3rd vector component is implicitly '1'
   *
   * @param out the receiving vector
   * @param a the vector to transform
   * @param m matrix to transform with
   * @returns out
   */
  transformMat3(out: Vec2Type, a: Vec2Type|number[], m: Mat3Type): Vec2Type;

  /**
   * Transforms the Vec2Type with a Mat4Type
   * 3rd vector component is implicitly '0'
   * 4th vector component is implicitly '1'
   *
   * @param out the receiving vector
   * @param a the vector to transform
   * @param m matrix to transform with
   * @returns out
   */
  transformMat4(out: Vec2Type, a: Vec2Type|number[], m: Mat4Type): Vec2Type;

  /**
   * Perform some operation over an array of vec2Types.
   *
   * @param a the array of vectors to iterate over
   * @param stride Number of elements between the start of each Vec2Type. If 0
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
      fn: (a: Vec2Type|number[], b: Vec2Type|number[], arg: any) => void,
      arg: any): Float32Array;

  /**
   * Perform some operation over an array of vec2Types.
   *
   * @param a the array of vectors to iterate over
   * @param stride Number of elements between the start of each Vec2Type. If 0
   *     assumes tightly packed
   * @param offset Number of elements to skip at the beginning of the array
   * @param count Number of vec2Types to iterate over. If 0 iterates over entire
   *     array
   * @param fn Function to call for each vector in the array
   * @returns a
   */
  forEach(
      a: Float32Array, stride: number, offset: number, count: number,
      fn: (a: Vec2Type|number[], b: Vec2Type|number[]) => void): Float32Array;

  /**
   * Returns a string representation of a vector
   *
   * @param a vector to represent as a string
   * @returns string representation of the vector
   */
  str(a: Vec2Type|number[]): string;

  /**
   * Returns whether or not the vectors exactly have the same elements in the
   * same position (when compared with ===)
   *
   * @param {Vec2Type} a The first vector.
   * @param {Vec2Type} b The second vector.
   * @returns {boolean} True if the vectors are equal, false otherwise.
   */
  exactEquals(a: Vec2Type|number[], b: Vec2Type|number[]): boolean;

  /**
   * Returns whether or not the vectors have approximately the same elements
   * in the same position.
   *
   * @param {Vec2Type} a The first vector.
   * @param {Vec2Type} b The second vector.
   * @returns {boolean} True if the vectors are equal, false otherwise.
   */
  equals(a: Vec2Type|number[], b: Vec2Type|number[]): boolean;
}

// Vec3Type
interface Vec3Type extends Float32Array {
  /**
   * Creates a new, empty Vec3Type
   *
   * @returns a new 3D vector
   */
  create(): Vec3Type;

  /**
   * Creates a new Vec3Type initialized with values from an existing vector
   *
   * @param a vector to clone
   * @returns a new 3D vector
   */
  clone(a: Vec3Type|number[]): Vec3Type;

  /**
   * Creates a new Vec3Type initialized with the given values
   *
   * @param x X component
   * @param y Y component
   * @param z Z component
   * @returns a new 3D vector
   */
  fromValues(x: number, y: number, z: number): Vec3Type;

  /**
   * Copy the values from one Vec3Type to another
   *
   * @param out the receiving vector
   * @param a the source vector
   * @returns out
   */
  copy(out: Vec3Type, a: Vec3Type|number[]): Vec3Type;

  /**
   * Set the components of a Vec3Type to the given values
   *
   * @param out the receiving vector
   * @param x X component
   * @param y Y component
   * @param z Z component
   * @returns out
   */
  set(out: Vec3Type, x: number, y: number, z: number): Vec3Type;

  /**
   * Adds two Vec3Type's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  add(out: Vec3Type, a: Vec3Type|number[], b: Vec3Type|number[]): Vec3Type;

  /**
   * Subtracts vector b from vector a
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  subtract(out: Vec3Type, a: Vec3Type|number[], b: Vec3Type|number[]): Vec3Type;

  /**
   * Subtracts vector b from vector a
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  sub(out: Vec3Type, a: Vec3Type|number[], b: Vec3Type|number[]): Vec3Type

  /**
   * Multiplies two Vec3Type's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  multiply(out: Vec3Type, a: Vec3Type|number[], b: Vec3Type|number[]): Vec3Type;

  /**
   * Multiplies two Vec3Type's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  mul(out: Vec3Type, a: Vec3Type|number[], b: Vec3Type|number[]): Vec3Type;

  /**
   * Divides two Vec3Type's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  divide(out: Vec3Type, a: Vec3Type|number[], b: Vec3Type|number[]): Vec3Type;

  /**
   * Divides two Vec3Type's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  div(out: Vec3Type, a: Vec3Type|number[], b: Vec3Type|number[]): Vec3Type;

  /**
   * Math.ceil the components of a Vec3Type
   *
   * @param {Vec3Type} out the receiving vector
   * @param {Vec3Type} a vector to ceil
   * @returns {Vec3Type} out
   */
  ceil(out: Vec3Type, a: Vec3Type|number[]): Vec3Type;

  /**
   * Math.floor the components of a Vec3Type
   *
   * @param {Vec3Type} out the receiving vector
   * @param {Vec3Type} a vector to floor
   * @returns {Vec3Type} out
   */
  floor(out: Vec3Type, a: Vec3Type|number[]): Vec3Type;

  /**
   * Returns the minimum of two Vec3Type's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  min(out: Vec3Type, a: Vec3Type|number[], b: Vec3Type|number[]): Vec3Type;

  /**
   * Returns the maximum of two Vec3Type's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  max(out: Vec3Type, a: Vec3Type|number[], b: Vec3Type|number[]): Vec3Type;

  /**
   * Math.round the components of a Vec3Type
   *
   * @param {Vec3Type} out the receiving vector
   * @param {Vec3Type} a vector to round
   * @returns {Vec3Type} out
   */
  round(out: Vec3Type, a: Vec3Type|number[]): Vec3Type

  /**
   * Scales a Vec3Type by a scalar number
   *
   * @param out the receiving vector
   * @param a the vector to scale
   * @param b amount to scale the vector by
   * @returns out
   */
  scale(out: Vec3Type, a: Vec3Type|number[], b: number): Vec3Type;

  /**
   * Adds two Vec3Type's after scaling the second operand by a scalar value
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @param scale the amount to scale b by before adding
   * @returns out
   */
  scaleAndAdd(
      out: Vec3Type, a: Vec3Type|number[], b: Vec3Type|number[],
      scale: number): Vec3Type;

  /**
   * Calculates the euclidian distance between two Vec3Type's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns distance between a and b
   */
  distance(a: Vec3Type|number[], b: Vec3Type|number[]): number;

  /**
   * Calculates the euclidian distance between two Vec3Type's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns distance between a and b
   */
  dist(a: Vec3Type|number[], b: Vec3Type|number[]): number;

  /**
   * Calculates the squared euclidian distance between two Vec3Type's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns squared distance between a and b
   */
  squaredDistance(a: Vec3Type|number[], b: Vec3Type|number[]): number;

  /**
   * Calculates the squared euclidian distance between two Vec3Type's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns squared distance between a and b
   */
  sqrDist(a: Vec3Type|number[], b: Vec3Type|number[]): number;

  /**
   * Calculates the length of a Vec3Type
   *
   * @param a vector to calculate length of
   * @returns length of a
   */
  length(a: Vec3Type|number[]): number;

  /**
   * Calculates the length of a Vec3Type
   *
   * @param a vector to calculate length of
   * @returns length of a
   */
  len(a: Vec3Type|number[]): number;

  /**
   * Calculates the squared length of a Vec3Type
   *
   * @param a vector to calculate squared length of
   * @returns squared length of a
   */
  squaredLength(a: Vec3Type|number[]): number;

  /**
   * Calculates the squared length of a Vec3Type
   *
   * @param a vector to calculate squared length of
   * @returns squared length of a
   */
  sqrLen(a: Vec3Type|number[]): number;

  /**
   * Negates the components of a Vec3Type
   *
   * @param out the receiving vector
   * @param a vector to negate
   * @returns out
   */
  negate(out: Vec3Type, a: Vec3Type|number[]): Vec3Type;

  /**
   * Returns the inverse of the components of a Vec3Type
   *
   * @param out the receiving vector
   * @param a vector to invert
   * @returns out
   */
  inverse(out: Vec3Type, a: Vec3Type|number[]): Vec3Type;

  /**
   * Normalize a Vec3Type
   *
   * @param out the receiving vector
   * @param a vector to normalize
   * @returns out
   */
  normalize(out: Vec3Type, a: Vec3Type|number[]): Vec3Type;

  /**
   * Calculates the dot product of two Vec3Type's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns dot product of a and b
   */
  dot(a: Vec3Type|number[], b: Vec3Type|number[]): number;

  /**
   * Computes the cross product of two Vec3Type's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  cross(out: Vec3Type, a: Vec3Type|number[], b: Vec3Type|number[]): Vec3Type;

  /**
   * Performs a linear interpolation between two Vec3Type's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @param t interpolation amount between the two inputs
   * @returns out
   */
  lerp(out: Vec3Type, a: Vec3Type|number[], b: Vec3Type|number[], t: number):
      Vec3Type;

  /**
   * Performs a hermite interpolation with two control points
   *
   * @param {Vec3Type} out the receiving vector
   * @param {Vec3Type} a the first operand
   * @param {Vec3Type} b the second operand
   * @param {Vec3Type} c the third operand
   * @param {Vec3Type} d the fourth operand
   * @param {number} t interpolation amount between the two inputs
   * @returns {Vec3Type} out
   */
  hermite(
      out: Vec3Type, a: Vec3Type|number[], b: Vec3Type|number[],
      c: Vec3Type|number[], d: Vec3Type|number[], t: number): Vec3Type;

  /**
   * Performs a bezier interpolation with two control points
   *
   * @param {Vec3Type} out the receiving vector
   * @param {Vec3Type} a the first operand
   * @param {Vec3Type} b the second operand
   * @param {Vec3Type} c the third operand
   * @param {Vec3Type} d the fourth operand
   * @param {number} t interpolation amount between the two inputs
   * @returns {Vec3Type} out
   */
  bezier(
      out: Vec3Type, a: Vec3Type|number[], b: Vec3Type|number[],
      c: Vec3Type|number[], d: Vec3Type|number[], t: number): Vec3Type;

  /**
   * Generates a random unit vector
   *
   * @param out the receiving vector
   * @returns out
   */
  random(out: Vec3Type): Vec3Type;

  /**
   * Generates a random vector with the given scale
   *
   * @param out the receiving vector
   * @param [scale] Length of the resulting vector. If omitted, a unit vector
   *     will be returned
   * @returns out
   */
  random(out: Vec3Type, scale: number): Vec3Type;

  /**
   * Transforms the Vec3Type with a Mat3Type.
   *
   * @param out the receiving vector
   * @param a the vector to transform
   * @param m the 3x3 matrix to transform with
   * @returns out
   */
  transformMat3(out: Vec3Type, a: Vec3Type|number[], m: Mat3Type): Vec3Type;

  /**
   * Transforms the Vec3Type with a Mat4Type.
   * 4th vector component is implicitly '1'
   *
   * @param out the receiving vector
   * @param a the vector to transform
   * @param m matrix to transform with
   * @returns out
   */
  transformMat4(out: Vec3Type, a: Vec3Type|number[], m: Mat4Type): Vec3Type;

  /**
   * Transforms the Vec3Type with a QuatType
   *
   * @param out the receiving vector
   * @param a the vector to transform
   * @param q Quaternion to transform with
   * @returns out
   */
  transformQuat(out: Vec3Type, a: Vec3Type|number[], q: QuatType): Vec3Type;


  /**
   * Rotate a 3D vector around the x-axis
   * @param out The receiving Vec3Type
   * @param a The Vec3Type point to rotate
   * @param b The origin of the rotation
   * @param c The angle of rotation
   * @returns out
   */
  rotateX(out: Vec3Type, a: Vec3Type|number[], b: Vec3Type|number[], c: number):
      Vec3Type;

  /**
   * Rotate a 3D vector around the y-axis
   * @param out The receiving Vec3Type
   * @param a The Vec3Type point to rotate
   * @param b The origin of the rotation
   * @param c The angle of rotation
   * @returns out
   */
  rotateY(out: Vec3Type, a: Vec3Type|number[], b: Vec3Type|number[], c: number):
      Vec3Type;

  /**
   * Rotate a 3D vector around the z-axis
   * @param out The receiving Vec3Type
   * @param a The Vec3Type point to rotate
   * @param b The origin of the rotation
   * @param c The angle of rotation
   * @returns out
   */
  rotateZ(out: Vec3Type, a: Vec3Type|number[], b: Vec3Type|number[], c: number):
      Vec3Type;

  /**
   * Perform some operation over an array of vec3s.
   *
   * @param a the array of vectors to iterate over
   * @param stride Number of elements between the start of each Vec3Type. If 0
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
      fn: (a: Vec3Type|number[], b: Vec3Type|number[], arg: any) => void,
      arg: any): Float32Array;

  /**
   * Perform some operation over an array of vec3s.
   *
   * @param a the array of vectors to iterate over
   * @param stride Number of elements between the start of each Vec3Type. If 0
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
      fn: (a: Vec3Type|number[], b: Vec3Type|number[]) => void): Float32Array;

  /**
   * Get the angle between two 3D vectors
   * @param a The first operand
   * @param b The second operand
   * @returns The angle in radians
   */
  angle(a: Vec3Type|number[], b: Vec3Type|number[]): number;

  /**
   * Returns a string representation of a vector
   *
   * @param a vector to represent as a string
   * @returns string representation of the vector
   */
  str(a: Vec3Type|number[]): string;

  /**
   * Returns whether or not the vectors have exactly the same elements in the
   * same position (when compared with ===)
   *
   * @param {Vec3Type} a The first vector.
   * @param {Vec3Type} b The second vector.
   * @returns {boolean} True if the vectors are equal, false otherwise.
   */
  exactEquals(a: Vec3Type|number[], b: Vec3Type|number[]): boolean

  /**
   * Returns whether or not the vectors have approximately the same
   * elements in the same position.
   *
   * @param {Vec3Type} a The first vector.
   * @param {Vec3Type} b The second vector.
   * @returns {boolean} True if the vectors are equal, false otherwise.
   */
  equals(a: Vec3Type|number[], b: Vec3Type|number[]): boolean
}

// Vec4Type
interface Vec4Type extends Float32Array {
  /**
   * Creates a new, empty Vec4Type
   *
   * @returns a new 4D vector
   */
  create(): Vec4Type;

  /**
   * Creates a new Vec4Type initialized with values from an existing vector
   *
   * @param a vector to clone
   * @returns a new 4D vector
   */
  clone(a: Vec4Type|number[]): Vec4Type;

  /**
   * Creates a new Vec4Type initialized with the given values
   *
   * @param x X component
   * @param y Y component
   * @param z Z component
   * @param w W component
   * @returns a new 4D vector
   */
  fromValues(x: number, y: number, z: number, w: number): Vec4Type;

  /**
   * Copy the values from one Vec4Type to another
   *
   * @param out the receiving vector
   * @param a the source vector
   * @returns out
   */
  copy(out: Vec4Type, a: Vec4Type|number[]): Vec4Type;

  /**
   * Set the components of a Vec4Type to the given values
   *
   * @param out the receiving vector
   * @param x X component
   * @param y Y component
   * @param z Z component
   * @param w W component
   * @returns out
   */
  set(out: Vec4Type, x: number, y: number, z: number, w: number): Vec4Type;

  /**
   * Adds two Vec4Type's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  add(out: Vec4Type, a: Vec4Type|number[], b: Vec4Type|number[]): Vec4Type;

  /**
   * Subtracts vector b from vector a
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  subtract(out: Vec4Type, a: Vec4Type|number[], b: Vec4Type|number[]): Vec4Type;

  /**
   * Subtracts vector b from vector a
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  sub(out: Vec4Type, a: Vec4Type|number[], b: Vec4Type|number[]): Vec4Type;

  /**
   * Multiplies two Vec4Type's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  multiply(out: Vec4Type, a: Vec4Type|number[], b: Vec4Type|number[]): Vec4Type;

  /**
   * Multiplies two Vec4Type's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  mul(out: Vec4Type, a: Vec4Type|number[], b: Vec4Type|number[]): Vec4Type;

  /**
   * Divides two Vec4Type's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  divide(out: Vec4Type, a: Vec4Type|number[], b: Vec4Type|number[]): Vec4Type;

  /**
   * Divides two Vec4Type's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  div(out: Vec4Type, a: Vec4Type|number[], b: Vec4Type|number[]): Vec4Type;

  /**
   * Math.ceil the components of a Vec4Type
   *
   * @param {Vec4Type} out the receiving vector
   * @param {Vec4Type} a vector to ceil
   * @returns {Vec4Type} out
   */
  ceil(out: Vec4Type, a: Vec4Type|number[]): Vec4Type;

  /**
   * Math.floor the components of a Vec4Type
   *
   * @param {Vec4Type} out the receiving vector
   * @param {Vec4Type} a vector to floor
   * @returns {Vec4Type} out
   */
  floor(out: Vec4Type, a: Vec4Type|number[]): Vec4Type;

  /**
   * Returns the minimum of two Vec4Type's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  min(out: Vec4Type, a: Vec4Type|number[], b: Vec4Type|number[]): Vec4Type;

  /**
   * Returns the maximum of two Vec4Type's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  max(out: Vec4Type, a: Vec4Type|number[], b: Vec4Type|number[]): Vec4Type;

  /**
   * Math.round the components of a Vec4Type
   *
   * @param {Vec4Type} out the receiving vector
   * @param {Vec4Type} a vector to round
   * @returns {Vec4Type} out
   */
  round(out: Vec4Type, a: Vec4Type|number[]): Vec4Type;

  /**
   * Scales a Vec4Type by a scalar number
   *
   * @param out the receiving vector
   * @param a the vector to scale
   * @param b amount to scale the vector by
   * @returns out
   */
  scale(out: Vec4Type, a: Vec4Type|number[], b: number): Vec4Type;

  /**
   * Adds two Vec4Type's after scaling the second operand by a scalar value
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @param scale the amount to scale b by before adding
   * @returns out
   */
  scaleAndAdd(
      out: Vec4Type, a: Vec4Type|number[], b: Vec4Type|number[],
      scale: number): Vec4Type;

  /**
   * Calculates the euclidian distance between two Vec4Type's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns distance between a and b
   */
  distance(a: Vec4Type|number[], b: Vec4Type|number[]): number;

  /**
   * Calculates the euclidian distance between two Vec4Type's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns distance between a and b
   */
  dist(a: Vec4Type|number[], b: Vec4Type|number[]): number;

  /**
   * Calculates the squared euclidian distance between two Vec4Type's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns squared distance between a and b
   */
  squaredDistance(a: Vec4Type|number[], b: Vec4Type|number[]): number;

  /**
   * Calculates the squared euclidian distance between two Vec4Type's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns squared distance between a and b
   */
  sqrDist(a: Vec4Type|number[], b: Vec4Type|number[]): number;

  /**
   * Calculates the length of a Vec4Type
   *
   * @param a vector to calculate length of
   * @returns length of a
   */
  length(a: Vec4Type|number[]): number;

  /**
   * Calculates the length of a Vec4Type
   *
   * @param a vector to calculate length of
   * @returns length of a
   */
  len(a: Vec4Type|number[]): number;

  /**
   * Calculates the squared length of a Vec4Type
   *
   * @param a vector to calculate squared length of
   * @returns squared length of a
   */
  squaredLength(a: Vec4Type|number[]): number;

  /**
   * Calculates the squared length of a Vec4Type
   *
   * @param a vector to calculate squared length of
   * @returns squared length of a
   */
  sqrLen(a: Vec4Type|number[]): number;

  /**
   * Negates the components of a Vec4Type
   *
   * @param out the receiving vector
   * @param a vector to negate
   * @returns out
   */
  negate(out: Vec4Type, a: Vec4Type|number[]): Vec4Type;

  /**
   * Returns the inverse of the components of a Vec4Type
   *
   * @param out the receiving vector
   * @param a vector to invert
   * @returns out
   */
  inverse(out: Vec4Type, a: Vec4Type|number[]): Vec4Type;

  /**
   * Normalize a Vec4Type
   *
   * @param out the receiving vector
   * @param a vector to normalize
   * @returns out
   */
  normalize(out: Vec4Type, a: Vec4Type|number[]): Vec4Type;

  /**
   * Calculates the dot product of two Vec4Type's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns dot product of a and b
   */
  dot(a: Vec4Type|number[], b: Vec4Type|number[]): number;

  /**
   * Performs a linear interpolation between two Vec4Type's
   *
   * @param out the receiving vector
   * @param a the first operand
   * @param b the second operand
   * @param t interpolation amount between the two inputs
   * @returns out
   */
  lerp(out: Vec4Type, a: Vec4Type|number[], b: Vec4Type|number[], t: number):
      Vec4Type;

  /**
   * Generates a random unit vector
   *
   * @param out the receiving vector
   * @returns out
   */
  random(out: Vec4Type): Vec4Type;

  /**
   * Generates a random vector with the given scale
   *
   * @param out the receiving vector
   * @param scale length of the resulting vector. If ommitted, a unit vector
   *     will be returned
   * @returns out
   */
  random(out: Vec4Type, scale: number): Vec4Type;

  /**
   * Transforms the Vec4Type with a Mat4Type.
   *
   * @param out the receiving vector
   * @param a the vector to transform
   * @param m matrix to transform with
   * @returns out
   */
  transformMat4(out: Vec4Type, a: Vec4Type|number[], m: Mat4Type): Vec4Type;

  /**
   * Transforms the Vec4Type with a QuatType
   *
   * @param out the receiving vector
   * @param a the vector to transform
   * @param q Quaternion to transform with
   * @returns out
   */

  transformQuat(out: Vec4Type, a: Vec4Type|number[], q: QuatType): Vec4Type;

  /**
   * Perform some operation over an array of Vec4s.
   *
   * @param a the array of vectors to iterate over
   * @param stride Number of elements between the start of each Vec4Type. If 0
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
      fn: (a: Vec4Type|number[], b: Vec4Type|number[], arg: any) => void,
      arg: any): Float32Array;

  /**
   * Perform some operation over an array of Vec4s.
   *
   * @param a the array of vectors to iterate over
   * @param stride Number of elements between the start of each Vec4Type. If 0
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
      fn: (a: Vec4Type|number[], b: Vec4Type|number[]) => void): Float32Array;

  /**
   * Returns a string representation of a vector
   *
   * @param a vector to represent as a string
   * @returns string representation of the vector
   */
  str(a: Vec4Type|number[]): string;

  /**
   * Returns whether or not the vectors have exactly the same elements in the
   * same position (when compared with ===)
   *
   * @param {Vec4Type} a The first vector.
   * @param {Vec4Type} b The second vector.
   * @returns {boolean} True if the vectors are equal, false otherwise.
   */
  exactEquals(a: Vec4Type|number[], b: Vec4Type|number[]): boolean;

  /**
   * Returns whether or not the vectors have approximately the same elements
   * in the same position.
   *
   * @param {Vec4Type} a The first vector.
   * @param {Vec4Type} b The second vector.
   * @returns {boolean} True if the vectors are equal, false otherwise.
   */
  equals(a: Vec4Type|number[], b: Vec4Type|number[]): boolean;
}

// Mat2Type
interface Mat2Type extends Float32Array {
  /**
   * Creates a new identity Mat2Type
   *
   * @returns a new 2x2 matrix
   */
  create(): Mat2Type;

  /**
   * Creates a new Mat2Type initialized with values from an existing matrix
   *
   * @param a matrix to clone
   * @returns a new 2x2 matrix
   */
  clone(a: Mat2Type): Mat2Type;

  /**
   * Copy the values from one Mat2Type to another
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  copy(out: Mat2Type, a: Mat2Type): Mat2Type;

  /**
   * Set a Mat2Type to the identity matrix
   *
   * @param out the receiving matrix
   * @returns out
   */
  identity(out: Mat2Type): Mat2Type;

  /**
   * Create a new Mat2Type with the given values
   *
   * @param {number} m00 Component in column 0, row 0 position (index 0)
   * @param {number} m01 Component in column 0, row 1 position (index 1)
   * @param {number} m10 Component in column 1, row 0 position (index 2)
   * @param {number} m11 Component in column 1, row 1 position (index 3)
   * @returns {Mat2Type} out A new 2x2 matrix
   */
  fromValues(m00: number, m01: number, m10: number, m11: number): Mat2Type;

  /**
   * Set the components of a Mat2Type to the given values
   *
   * @param {Mat2Type} out the receiving matrix
   * @param {number} m00 Component in column 0, row 0 position (index 0)
   * @param {number} m01 Component in column 0, row 1 position (index 1)
   * @param {number} m10 Component in column 1, row 0 position (index 2)
   * @param {number} m11 Component in column 1, row 1 position (index 3)
   * @returns {Mat2Type} out
   */
  set(out: Mat2Type, m00: number, m01: number, m10: number,
      m11: number): Mat2Type;

  /**
   * Transpose the values of a Mat2Type
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  transpose(out: Mat2Type, a: Mat2Type): Mat2Type;

  /**
   * Inverts a Mat2Type
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  invert(out: Mat2Type, a: Mat2Type): Mat2Type|null;

  /**
   * Calculates the adjugate of a Mat2Type
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  adjoint(out: Mat2Type, a: Mat2Type): Mat2Type;

  /**
   * Calculates the determinant of a Mat2Type
   *
   * @param a the source matrix
   * @returns determinant of a
   */
  determinant(a: Mat2Type): number;

  /**
   * Multiplies two Mat2Type's
   *
   * @param out the receiving matrix
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  multiply(out: Mat2Type, a: Mat2Type, b: Mat2Type): Mat2Type;

  /**
   * Multiplies two Mat2Type's
   *
   * @param out the receiving matrix
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  mul(out: Mat2Type, a: Mat2Type, b: Mat2Type): Mat2Type;

  /**
   * Rotates a Mat2Type by the given angle
   *
   * @param out the receiving matrix
   * @param a the matrix to rotate
   * @param rad the angle to rotate the matrix by
   * @returns out
   */
  rotate(out: Mat2Type, a: Mat2Type, rad: number): Mat2Type;

  /**
   * Scales the Mat2Type by the dimensions in the given Vec2Type
   *
   * @param out the receiving matrix
   * @param a the matrix to rotate
   * @param v the Vec2Type to scale the matrix by
   * @returns out
   **/
  scale(out: Mat2Type, a: Mat2Type, v: Vec2Type|number[]): Mat2Type;

  /**
   * Creates a matrix from a given angle
   * This is equivalent to (but much faster than):
   *
   *     Mat2Type.identity(dest);
   *     Mat2Type.rotate(dest, dest, rad);
   *
   * @param {Mat2Type} out Mat2Type receiving operation result
   * @param {number} rad the angle to rotate the matrix by
   * @returns {Mat2Type} out
   */
  fromRotation(out: Mat2Type, rad: number): Mat2Type;

  /**
   * Creates a matrix from a vector scaling
   * This is equivalent to (but much faster than):
   *
   *     Mat2Type.identity(dest);
   *     Mat2Type.scale(dest, dest, vec);
   *
   * @param {Mat2Type} out Mat2Type receiving operation result
   * @param {Vec2Type} v Scaling vector
   * @returns {Mat2Type} out
   */
  fromScaling(out: Mat2Type, v: Vec2Type|number[]): Mat2Type;

  /**
   * Returns a string representation of a Mat2Type
   *
   * @param a matrix to represent as a string
   * @returns string representation of the matrix
   */
  str(a: Mat2Type): string;

  /**
   * Returns Frobenius norm of a Mat2Type
   *
   * @param a the matrix to calculate Frobenius norm of
   * @returns Frobenius norm
   */
  frob(a: Mat2Type): number;

  /**
   * Returns L, D and U matrices (Lower triangular, Diagonal and Upper
   * triangular) by factorizing the input matrix
   * @param L the lower triangular matrix
   * @param D the diagonal matrix
   * @param U the upper triangular matrix
   * @param a the input matrix to factorize
   */
  LDU(L: Mat2Type, D: Mat2Type, U: Mat2Type, a: Mat2Type): Mat2Type;

  /**
   * Adds two Mat2Type's
   *
   * @param {Mat2Type} out the receiving matrix
   * @param {Mat2Type} a the first operand
   * @param {Mat2Type} b the second operand
   * @returns {Mat2Type} out
   */
  add(out: Mat2Type, a: Mat2Type, b: Mat2Type): Mat2Type;

  /**
   * Subtracts matrix b from matrix a
   *
   * @param {Mat2Type} out the receiving matrix
   * @param {Mat2Type} a the first operand
   * @param {Mat2Type} b the second operand
   * @returns {Mat2Type} out
   */
  subtract(out: Mat2Type, a: Mat2Type, b: Mat2Type): Mat2Type;

  /**
   * Subtracts matrix b from matrix a
   *
   * @param {Mat2Type} out the receiving matrix
   * @param {Mat2Type} a the first operand
   * @param {Mat2Type} b the second operand
   * @returns {Mat2Type} out
   */
  sub(out: Mat2Type, a: Mat2Type, b: Mat2Type): Mat2Type;

  /**
   * Returns whether or not the matrices have exactly the same elements in the
   * same position (when compared with ===)
   *
   * @param {Mat2Type} a The first matrix.
   * @param {Mat2Type} b The second matrix.
   * @returns {boolean} True if the matrices are equal, false otherwise.
   */
  exactEquals(a: Mat2Type, b: Mat2Type): boolean;

  /**
   * Returns whether or not the matrices have approximately the same elements
   * in the same position.
   *
   * @param {Mat2Type} a The first matrix.
   * @param {Mat2Type} b The second matrix.
   * @returns {boolean} True if the matrices are equal, false otherwise.
   */
  equals(a: Mat2Type, b: Mat2Type): boolean;

  /**
   * Multiply each element of the matrix by a scalar.
   *
   * @param {Mat2Type} out the receiving matrix
   * @param {Mat2Type} a the matrix to scale
   * @param {number} b amount to scale the matrix's elements by
   * @returns {Mat2Type} out
   */
  multiplyScalar(out: Mat2Type, a: Mat2Type, b: number): Mat2Type

  /**
   * Adds two Mat2Type's after multiplying each element of the second operand
   * by a scalar value.
   *
   * @param {Mat2Type} out the receiving vector
   * @param {Mat2Type} a the first operand
   * @param {Mat2Type} b the second operand
   * @param {number} scale the amount to scale b's elements by before
   *     adding
   * @returns {Mat2Type} out
   */
  multiplyScalarAndAdd(out: Mat2Type, a: Mat2Type, b: Mat2Type, scale: number):
      Mat2Type
}

// Mat2dType
interface Mat2dType extends Float32Array {
  /**
   * Creates a new identity Mat2dType
   *
   * @returns a new 2x3 matrix
   */
  create(): Mat2dType;

  /**
   * Creates a new Mat2dType initialized with values from an existing matrix
   *
   * @param a matrix to clone
   * @returns a new 2x3 matrix
   */
  clone(a: Mat2dType): Mat2dType;

  /**
   * Copy the values from one Mat2dType to another
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  copy(out: Mat2dType, a: Mat2dType): Mat2dType;

  /**
   * Set a Mat2dType to the identity matrix
   *
   * @param out the receiving matrix
   * @returns out
   */
  identity(out: Mat2dType): Mat2dType;

  /**
   * Create a new Mat2dType with the given values
   *
   * @param {number} a Component A (index 0)
   * @param {number} b Component B (index 1)
   * @param {number} c Component C (index 2)
   * @param {number} d Component D (index 3)
   * @param {number} tx Component TX (index 4)
   * @param {number} ty Component TY (index 5)
   * @returns {Mat2dType} A new Mat2dType
   */
  fromValues(
      a: number, b: number, c: number, d: number, tx: number,
      ty: number): Mat2dType


  /**
   * Set the components of a Mat2dType to the given values
   *
   * @param {Mat2dType} out the receiving matrix
   * @param {number} a Component A (index 0)
   * @param {number} b Component B (index 1)
   * @param {number} c Component C (index 2)
   * @param {number} d Component D (index 3)
   * @param {number} tx Component TX (index 4)
   * @param {number} ty Component TY (index 5)
   * @returns {Mat2dType} out
   */
  set(out: Mat2dType, a: number, b: number, c: number, d: number, tx: number,
      ty: number): Mat2dType

  /**
   * Inverts a Mat2dType
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  invert(out: Mat2dType, a: Mat2dType): Mat2dType|null;

  /**
   * Calculates the determinant of a Mat2dType
   *
   * @param a the source matrix
   * @returns determinant of a
   */
  determinant(a: Mat2dType): number;

  /**
   * Multiplies two Mat2dType's
   *
   * @param out the receiving matrix
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  multiply(out: Mat2dType, a: Mat2dType, b: Mat2dType): Mat2dType;

  /**
   * Multiplies two Mat2dType's
   *
   * @param out the receiving matrix
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  mul(out: Mat2dType, a: Mat2dType, b: Mat2dType): Mat2dType;

  /**
   * Rotates a Mat2dType by the given angle
   *
   * @param out the receiving matrix
   * @param a the matrix to rotate
   * @param rad the angle to rotate the matrix by
   * @returns out
   */
  rotate(out: Mat2dType, a: Mat2dType, rad: number): Mat2dType;

  /**
   * Scales the Mat2dType by the dimensions in the given Vec2Type
   *
   * @param out the receiving matrix
   * @param a the matrix to translate
   * @param v the Vec2Type to scale the matrix by
   * @returns out
   **/
  scale(out: Mat2dType, a: Mat2dType, v: Vec2Type|number[]): Mat2dType;

  /**
   * Translates the Mat2dType by the dimensions in the given Vec2Type
   *
   * @param out the receiving matrix
   * @param a the matrix to translate
   * @param v the Vec2Type to translate the matrix by
   * @returns out
   **/
  translate(out: Mat2dType, a: Mat2dType, v: Vec2Type|number[]): Mat2dType;

  /**
   * Creates a matrix from a given angle
   * This is equivalent to (but much faster than):
   *
   *     Mat2dType.identity(dest);
   *     Mat2dType.rotate(dest, dest, rad);
   *
   * @param {Mat2dType} out Mat2dType receiving operation result
   * @param {number} rad the angle to rotate the matrix by
   * @returns {Mat2dType} out
   */
  fromRotation(out: Mat2dType, rad: number): Mat2dType;

  /**
   * Creates a matrix from a vector scaling
   * This is equivalent to (but much faster than):
   *
   *     Mat2dType.identity(dest);
   *     Mat2dType.scale(dest, dest, vec);
   *
   * @param {Mat2dType} out Mat2dType receiving operation result
   * @param {Vec2Type} v Scaling vector
   * @returns {Mat2dType} out
   */
  fromScaling(out: Mat2dType, v: Vec2Type|number[]): Mat2dType;

  /**
   * Creates a matrix from a vector translation
   * This is equivalent to (but much faster than):
   *
   *     Mat2dType.identity(dest);
   *     Mat2dType.translate(dest, dest, vec);
   *
   * @param {Mat2dType} out Mat2dType receiving operation result
   * @param {Vec2Type} v Translation vector
   * @returns {Mat2dType} out
   */
  fromTranslation(out: Mat2dType, v: Vec2Type|number[]): Mat2dType

  /**
   * Returns a string representation of a Mat2dType
   *
   * @param a matrix to represent as a string
   * @returns string representation of the matrix
   */
  str(a: Mat2dType): string;

  /**
   * Returns Frobenius norm of a Mat2dType
   *
   * @param a the matrix to calculate Frobenius norm of
   * @returns Frobenius norm
   */
  frob(a: Mat2dType): number;

  /**
   * Adds two Mat2dType's
   *
   * @param {Mat2dType} out the receiving matrix
   * @param {Mat2dType} a the first operand
   * @param {Mat2dType} b the second operand
   * @returns {Mat2dType} out
   */
  add(out: Mat2dType, a: Mat2dType, b: Mat2dType): Mat2dType

  /**
   * Subtracts matrix b from matrix a
   *
   * @param {Mat2dType} out the receiving matrix
   * @param {Mat2dType} a the first operand
   * @param {Mat2dType} b the second operand
   * @returns {Mat2dType} out
   */
  subtract(out: Mat2dType, a: Mat2dType, b: Mat2dType): Mat2dType

  /**
   * Subtracts matrix b from matrix a
   *
   * @param {Mat2dType} out the receiving matrix
   * @param {Mat2dType} a the first operand
   * @param {Mat2dType} b the second operand
   * @returns {Mat2dType} out
   */
  sub(out: Mat2dType, a: Mat2dType, b: Mat2dType): Mat2dType

  /**
   * Multiply each element of the matrix by a scalar.
   *
   * @param {Mat2dType} out the receiving matrix
   * @param {Mat2dType} a the matrix to scale
   * @param {number} b amount to scale the matrix's elements by
   * @returns {Mat2dType} out
   */
  multiplyScalar(out: Mat2dType, a: Mat2dType, b: number): Mat2dType;

  /**
   * Adds two Mat2dType's after multiplying each element of the second operand
   * by a scalar value.
   *
   * @param {Mat2dType} out the receiving vector
   * @param {Mat2dType} a the first operand
   * @param {Mat2dType} b the second operand
   * @param {number} scale the amount to scale b's elements by before adding
   * @returns {Mat2dType} out
   */
  multiplyScalarAndAdd(
      out: Mat2dType, a: Mat2dType, b: Mat2dType, scale: number): Mat2dType

  /**
   * Returns whether or not the matrices have exactly the same elements in
   * the same position (when compared with ===)
   *
   * @param {Mat2dType} a The first matrix.
   * @param {Mat2dType} b The second matrix.
   * @returns {boolean} True if the matrices are equal, false otherwise.
   */
  exactEquals(a: Mat2dType, b: Mat2dType): boolean;

  /**
   * Returns whether or not the matrices have approximately the same elements
   * in the same position.
   *
   * @param {Mat2dType} a The first matrix.
   * @param {Mat2dType} b The second matrix.
   * @returns {boolean} True if the matrices are equal, false otherwise.
   */
  equals(a: Mat2dType, b: Mat2dType): boolean
}

// Mat3Type
interface Mat3Type extends Float32Array {
  /**
   * Creates a new identity Mat3Type
   *
   * @returns a new 3x3 matrix
   */
  create(): Mat3Type;

  /**
   * Copies the upper-left 3x3 values into the given Mat3Type.
   *
   * @param {Mat3Type} out the receiving 3x3 matrix
   * @param {Mat4Type} a   the source 4x4 matrix
   * @returns {Mat3Type} out
   */
  fromMat4(out: Mat3Type, a: Mat4Type): Mat3Type

  /**
   * Creates a new Mat3Type initialized with values from an existing matrix
   *
   * @param a matrix to clone
   * @returns a new 3x3 matrix
   */
  clone(a: Mat3Type): Mat3Type;

  /**
   * Copy the values from one Mat3Type to another
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  copy(out: Mat3Type, a: Mat3Type): Mat3Type;

  /**
   * Create a new Mat3Type with the given values
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
   * @returns {Mat3Type} A new Mat3Type
   */
  fromValues(
      m00: number, m01: number, m02: number, m10: number, m11: number,
      m12: number, m20: number, m21: number, m22: number): Mat3Type;


  /**
   * Set the components of a Mat3Type to the given values
   *
   * @param {Mat3Type} out the receiving matrix
   * @param {number} m00 Component in column 0, row 0 position (index 0)
   * @param {number} m01 Component in column 0, row 1 position (index 1)
   * @param {number} m02 Component in column 0, row 2 position (index 2)
   * @param {number} m10 Component in column 1, row 0 position (index 3)
   * @param {number} m11 Component in column 1, row 1 position (index 4)
   * @param {number} m12 Component in column 1, row 2 position (index 5)
   * @param {number} m20 Component in column 2, row 0 position (index 6)
   * @param {number} m21 Component in column 2, row 1 position (index 7)
   * @param {number} m22 Component in column 2, row 2 position (index 8)
   * @returns {Mat3Type} out
   */
  set(out: Mat3Type, m00: number, m01: number, m02: number, m10: number,
      m11: number, m12: number, m20: number, m21: number, m22: number): Mat3Type

  /**
   * Set a Mat3Type to the identity matrix
   *
   * @param out the receiving matrix
   * @returns out
   */
  identity(out: Mat3Type): Mat3Type;

  /**
   * Transpose the values of a Mat3Type
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  transpose(out: Mat3Type, a: Mat3Type): Mat3Type;

  /**
   * Inverts a Mat3Type
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  invert(out: Mat3Type, a: Mat3Type): Mat3Type|null;

  /**
   * Calculates the adjugate of a Mat3Type
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  adjoint(out: Mat3Type, a: Mat3Type): Mat3Type;

  /**
   * Calculates the determinant of a Mat3Type
   *
   * @param a the source matrix
   * @returns determinant of a
   */
  determinant(a: Mat3Type): number;

  /**
   * Multiplies two Mat3Type's
   *
   * @param out the receiving matrix
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  multiply(out: Mat3Type, a: Mat3Type, b: Mat3Type): Mat3Type;

  /**
   * Multiplies two Mat3Type's
   *
   * @param out the receiving matrix
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  mul(out: Mat3Type, a: Mat3Type, b: Mat3Type): Mat3Type;


  /**
   * Translate a Mat3Type by the given vector
   *
   * @param out the receiving matrix
   * @param a the matrix to translate
   * @param v vector to translate by
   * @returns out
   */
  translate(out: Mat3Type, a: Mat3Type, v: Vec3Type|number[]): Mat3Type;

  /**
   * Rotates a Mat3Type by the given angle
   *
   * @param out the receiving matrix
   * @param a the matrix to rotate
   * @param rad the angle to rotate the matrix by
   * @returns out
   */
  rotate(out: Mat3Type, a: Mat3Type, rad: number): Mat3Type;

  /**
   * Scales the Mat3Type by the dimensions in the given Vec2Type
   *
   * @param out the receiving matrix
   * @param a the matrix to rotate
   * @param v the Vec2Type to scale the matrix by
   * @returns out
   **/
  scale(out: Mat3Type, a: Mat3Type, v: Vec2Type|number[]): Mat3Type;

  /**
   * Creates a matrix from a vector translation
   * This is equivalent to (but much faster than):
   *
   *     Mat3Type.identity(dest);
   *     Mat3Type.translate(dest, dest, vec);
   *
   * @param {Mat3Type} out Mat3Type receiving operation result
   * @param {Vec2Type} v Translation vector
   * @returns {Mat3Type} out
   */
  fromTranslation(out: Mat3Type, v: Vec2Type|number[]): Mat3Type

  /**
   * Creates a matrix from a given angle
   * This is equivalent to (but much faster than):
   *
   *     Mat3Type.identity(dest);
   *     Mat3Type.rotate(dest, dest, rad);
   *
   * @param {Mat3Type} out Mat3Type receiving operation result
   * @param {number} rad the angle to rotate the matrix by
   * @returns {Mat3Type} out
   */
  fromRotation(out: Mat3Type, rad: number): Mat3Type

  /**
   * Creates a matrix from a vector scaling
   * This is equivalent to (but much faster than):
   *
   *     Mat3Type.identity(dest);
   *     Mat3Type.scale(dest, dest, vec);
   *
   * @param {Mat3Type} out Mat3Type receiving operation result
   * @param {Vec2Type} v Scaling vector
   * @returns {Mat3Type} out
   */
  fromScaling(out: Mat3Type, v: Vec2Type|number[]): Mat3Type

  /**
   * Copies the values from a Mat2dType into a Mat3Type
   *
   * @param out the receiving matrix
   * @param {Mat2dType} a the matrix to copy
   * @returns out
   **/
  fromMat2d(out: Mat3Type, a: Mat2dType): Mat3Type;

  /**
   * Calculates a 3x3 matrix from the given Quaternion
   *
   * @param out Mat3Type receiving operation result
   * @param q Quaternion to create matrix from
   *
   * @returns out
   */
  fromQuat(out: Mat3Type, q: QuatType): Mat3Type;

  /**
   * Calculates a 3x3 normal matrix (transpose inverse) from the 4x4 matrix
   *
   * @param out Mat3Type receiving operation result
   * @param a Mat4Type to derive the normal matrix from
   *
   * @returns out
   */
  normalFromMat4(out: Mat3Type, a: Mat4Type): Mat3Type|null;

  /**
   * Returns a string representation of a Mat3Type
   *
   * @param mat matrix to represent as a string
   * @returns string representation of the matrix
   */
  str(mat: Mat3Type): string;

  /**
   * Returns Frobenius norm of a Mat3Type
   *
   * @param a the matrix to calculate Frobenius norm of
   * @returns Frobenius norm
   */
  frob(a: Mat3Type): number;

  /**
   * Adds two Mat3Type's
   *
   * @param {Mat3Type} out the receiving matrix
   * @param {Mat3Type} a the first operand
   * @param {Mat3Type} b the second operand
   * @returns {Mat3Type} out
   */
  add(out: Mat3Type, a: Mat3Type, b: Mat3Type): Mat3Type

  /**
   * Subtracts matrix b from matrix a
   *
   * @param {Mat3Type} out the receiving matrix
   * @param {Mat3Type} a the first operand
   * @param {Mat3Type} b the second operand
   * @returns {Mat3Type} out
   */
  subtract(out: Mat3Type, a: Mat3Type, b: Mat3Type): Mat3Type

  /**
   * Subtracts matrix b from matrix a
   *
   * @param {Mat3Type} out the receiving matrix
   * @param {Mat3Type} a the first operand
   * @param {Mat3Type} b the second operand
   * @returns {Mat3Type} out
   */
  sub(out: Mat3Type, a: Mat3Type, b: Mat3Type): Mat3Type

  /**
   * Multiply each element of the matrix by a scalar.
   *
   * @param {Mat3Type} out the receiving matrix
   * @param {Mat3Type} a the matrix to scale
   * @param {number} b amount to scale the matrix's elements by
   * @returns {Mat3Type} out
   */
  multiplyScalar(out: Mat3Type, a: Mat3Type, b: number): Mat3Type

  /**
   * Adds two Mat3Type's after multiplying each element of the second operand
   * by a scalar value.
   *
   * @param {Mat3Type} out the receiving vector
   * @param {Mat3Type} a the first operand
   * @param {Mat3Type} b the second operand
   * @param {number} scale the amount to scale b's elements by before
   *     adding
   * @returns {Mat3Type} out
   */
  multiplyScalarAndAdd(out: Mat3Type, a: Mat3Type, b: Mat3Type, scale: number):
      Mat3Type

  /**
   * Returns whether or not the matrices have exactly the same elements in
   * the same position (when compared with ===)
   *
   * @param {Mat3Type} a The first matrix.
   * @param {Mat3Type} b The second matrix.
   * @returns {boolean} True if the matrices are equal, false otherwise.
   */
  exactEquals(a: Mat3Type, b: Mat3Type): boolean;

  /**
   * Returns whether or not the matrices have approximately the same elements
   * in the same position.
   *
   * @param {Mat3Type} a The first matrix.
   * @param {Mat3Type} b The second matrix.
   * @returns {boolean} True if the matrices are equal, false otherwise.
   */
  equals(a: Mat3Type, b: Mat3Type): boolean
}

// Mat4Type
interface Mat4Type extends Float32Array {
  /**
   * Creates a new identity Mat4Type
   *
   * @returns a new 4x4 matrix
   */
  create(): Mat4Type;

  /**
   * Creates a new Mat4Type initialized with values from an existing matrix
   *
   * @param a matrix to clone
   * @returns a new 4x4 matrix
   */
  clone(a: Mat4Type): Mat4Type;

  /**
   * Copy the values from one Mat4Type to another
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  copy(out: Mat4Type, a: Mat4Type): Mat4Type;


  /**
   * Create a new Mat4Type with the given values
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
   * @returns {Mat4Type} A new Mat4Type
   */
  fromValues(
      m00: number, m01: number, m02: number, m03: number, m10: number,
      m11: number, m12: number, m13: number, m20: number, m21: number,
      m22: number, m23: number, m30: number, m31: number, m32: number,
      m33: number): Mat4Type;

  /**
   * Set the components of a Mat4Type to the given values
   *
   * @param {Mat4Type} out the receiving matrix
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
   * @returns {Mat4Type} out
   */
  set(out: Mat4Type, m00: number, m01: number, m02: number, m03: number,
      m10: number, m11: number, m12: number, m13: number, m20: number,
      m21: number, m22: number, m23: number, m30: number, m31: number,
      m32: number, m33: number): Mat4Type;

  /**
   * Set a Mat4Type to the identity matrix
   *
   * @param out the receiving matrix
   * @returns out
   */
  identity(out: Mat4Type): Mat4Type;

  /**
   * Transpose the values of a Mat4Type
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  transpose(out: Mat4Type, a: Mat4Type): Mat4Type;

  /**
   * Inverts a Mat4Type
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  invert(out: Mat4Type, a: Mat4Type): Mat4Type|null;

  /**
   * Calculates the adjugate of a Mat4Type
   *
   * @param out the receiving matrix
   * @param a the source matrix
   * @returns out
   */
  adjoint(out: Mat4Type, a: Mat4Type): Mat4Type;

  /**
   * Calculates the determinant of a Mat4Type
   *
   * @param a the source matrix
   * @returns determinant of a
   */
  determinant(a: Mat4Type): number;

  /**
   * Multiplies two Mat4Type's
   *
   * @param out the receiving matrix
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  multiply(out: Mat4Type, a: Mat4Type, b: Mat4Type): Mat4Type;

  /**
   * Multiplies two Mat4Type's
   *
   * @param out the receiving matrix
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  mul(out: Mat4Type, a: Mat4Type, b: Mat4Type): Mat4Type;

  /**
   * Translate a Mat4Type by the given vector
   *
   * @param out the receiving matrix
   * @param a the matrix to translate
   * @param v vector to translate by
   * @returns out
   */
  translate(out: Mat4Type, a: Mat4Type, v: Vec3Type|number[]): Mat4Type;

  /**
   * Scales the Mat4Type by the dimensions in the given Vec3Type
   *
   * @param out the receiving matrix
   * @param a the matrix to scale
   * @param v the Vec3Type to scale the matrix by
   * @returns out
   **/
  scale(out: Mat4Type, a: Mat4Type, v: Vec3Type|number[]): Mat4Type;

  /**
   * Rotates a Mat4Type by the given angle
   *
   * @param out the receiving matrix
   * @param a the matrix to rotate
   * @param rad the angle to rotate the matrix by
   * @param axis the axis to rotate around
   * @returns out
   */
  rotate(out: Mat4Type, a: Mat4Type, rad: number, axis: Vec3Type|number[]):
      Mat4Type;

  /**
   * Rotates a matrix by the given angle around the X axis
   *
   * @param out the receiving matrix
   * @param a the matrix to rotate
   * @param rad the angle to rotate the matrix by
   * @returns out
   */
  rotateX(out: Mat4Type, a: Mat4Type, rad: number): Mat4Type;

  /**
   * Rotates a matrix by the given angle around the Y axis
   *
   * @param out the receiving matrix
   * @param a the matrix to rotate
   * @param rad the angle to rotate the matrix by
   * @returns out
   */
  rotateY(out: Mat4Type, a: Mat4Type, rad: number): Mat4Type;

  /**
   * Rotates a matrix by the given angle around the Z axis
   *
   * @param out the receiving matrix
   * @param a the matrix to rotate
   * @param rad the angle to rotate the matrix by
   * @returns out
   */
  rotateZ(out: Mat4Type, a: Mat4Type, rad: number): Mat4Type;

  /**
   * Creates a matrix from a vector translation
   * This is equivalent to (but much faster than):
   *
   *     Mat4Type.identity(dest);
   *     Mat4Type.translate(dest, dest, vec);
   *
   * @param {Mat4Type} out Mat4Type receiving operation result
   * @param {Vec3Type} v Translation vector
   * @returns {Mat4Type} out
   */
  fromTranslation(out: Mat4Type, v: Vec3Type|number[]): Mat4Type

  /**
   * Creates a matrix from a vector scaling
   * This is equivalent to (but much faster than):
   *
   *     Mat4Type.identity(dest);
   *     Mat4Type.scale(dest, dest, vec);
   *
   * @param {Mat4Type} out Mat4Type receiving operation result
   * @param {Vec3Type} v Scaling vector
   * @returns {Mat4Type} out
   */
  fromScaling(out: Mat4Type, v: Vec3Type|number[]): Mat4Type

  /**
   * Creates a matrix from a given angle around a given axis
   * This is equivalent to (but much faster than):
   *
   *     Mat4Type.identity(dest);
   *     Mat4Type.rotate(dest, dest, rad, axis);
   *
   * @param {Mat4Type} out Mat4Type receiving operation result
   * @param {number} rad the angle to rotate the matrix by
   * @param {Vec3Type} axis the axis to rotate around
   * @returns {Mat4Type} out
   */
  fromRotation(out: Mat4Type, rad: number, axis: Vec3Type|number[]): Mat4Type

  /**
   * Creates a matrix from the given angle around the X axis
   * This is equivalent to (but much faster than):
   *
   *     Mat4Type.identity(dest);
   *     Mat4Type.rotateX(dest, dest, rad);
   *
   * @param {Mat4Type} out Mat4Type receiving operation result
   * @param {number} rad the angle to rotate the matrix by
   * @returns {Mat4Type} out
   */
  fromXRotation(out: Mat4Type, rad: number): Mat4Type

  /**
   * Creates a matrix from the given angle around the Y axis
   * This is equivalent to (but much faster than):
   *
   *     Mat4Type.identity(dest);
   *     Mat4Type.rotateY(dest, dest, rad);
   *
   * @param {Mat4Type} out Mat4Type receiving operation result
   * @param {number} rad the angle to rotate the matrix by
   * @returns {Mat4Type} out
   */
  fromYRotation(out: Mat4Type, rad: number): Mat4Type


  /**
   * Creates a matrix from the given angle around the Z axis
   * This is equivalent to (but much faster than):
   *
   *     Mat4Type.identity(dest);
   *     Mat4Type.rotateZ(dest, dest, rad);
   *
   * @param {Mat4Type} out Mat4Type receiving operation result
   * @param {number} rad the angle to rotate the matrix by
   * @returns {Mat4Type} out
   */
  fromZRotation(out: Mat4Type, rad: number): Mat4Type

  /**
   * Creates a matrix from a Quaternion rotation and vector translation
   * This is equivalent to (but much faster than):
   *
   *     Mat4Type.identity(dest);
   *     Mat4Type.translate(dest, vec);
   *     var QuatMat = Mat4Type.create();
   *     Quat4.toMat4(QuatType, QuatMat);
   *     Mat4Type.multiply(dest, QuatMat);
   *
   * @param out Mat4Type receiving operation result
   * @param q Rotation Quaternion
   * @param v Translation vector
   * @returns out
   */
  fromRotationTranslation(out: Mat4Type, q: QuatType, v: Vec3Type|number[]):
      Mat4Type;

  /**
   * Returns the translation vector component of a transformation
   *  matrix. If a matrix is built with fromRotationTranslation,
   *  the returned vector will be the same as the translation vector
   *  originally supplied.
   * @param  {Vec3Type} out Vector to receive translation component
   * @param  {Mat4Type} mat Matrix to be decomposed (input)
   * @return {Vec3Type} out
   */
  getTranslation(out: Vec3Type, mat: Mat4Type): Vec3Type;

  /**
   * Returns a Quaternion representing the rotational component
   *  of a transformation matrix. If a matrix is built with
   *  fromRotationTranslation, the returned Quaternion will be the
   *  same as the Quaternion originally supplied.
   * @param {QuatType} out Quaternion to receive the rotation component
   * @param {Mat4Type} mat Matrix to be decomposed (input)
   * @return {QuatType} out
   */
  getRotation(out: QuatType, mat: Mat4Type): QuatType;

  /**
   * Creates a matrix from a Quaternion rotation, vector translation and
   * vector scale This is equivalent to (but much faster than):
   *
   *     Mat4Type.identity(dest);
   *     Mat4Type.translate(dest, vec);
   *     var QuatMat = Mat4Type.create();
   *     Quat4.toMat4(QuatType, QuatMat);
   *     Mat4Type.multiply(dest, QuatMat);
   *     Mat4Type.scale(dest, scale)
   *
   * @param out Mat4Type receiving operation result
   * @param q Rotation Quaternion
   * @param v Translation vector
   * @param s Scaling vector
   * @returns out
   */
  fromRotationTranslationScale(
      out: Mat4Type, q: QuatType, v: Vec3Type|number[],
      s: Vec3Type|number[]): Mat4Type;

  /**
   * Creates a matrix from a Quaternion rotation, vector translation and
   * vector scale, rotating and scaling around the given origin This is
   * equivalent to (but much faster than):
   *
   *     Mat4Type.identity(dest);
   *     Mat4Type.translate(dest, vec);
   *     Mat4Type.translate(dest, origin);
   *     var QuatMat = Mat4Type.create();
   *     Quat4.toMat4(QuatType, QuatMat);
   *     Mat4Type.multiply(dest, QuatMat);
   *     Mat4Type.scale(dest, scale)
   *     Mat4Type.translate(dest, negativeOrigin);
   *
   * @param {Mat4Type} out Mat4Type receiving operation result
   * @param {QuatType} q Rotation Quaternion
   * @param {Vec3Type} v Translation vector
   * @param {Vec3Type} s Scaling vector
   * @param {Vec3Type} o The origin vector around which to scale and rotate
   * @returns {Mat4Type} out
   */
  fromRotationTranslationScaleOrigin(
      out: Mat4Type, q: QuatType, v: Vec3Type|number[], s: Vec3Type|number[],
      o: Vec3Type|number[]): Mat4Type

  /**
   * Calculates a 4x4 matrix from the given Quaternion
   *
   * @param {Mat4Type} out Mat4Type receiving operation result
   * @param {QuatType} q Quaternion to create matrix from
   *
   * @returns {Mat4Type} out
   */
  fromQuat(out: Mat4Type, q: QuatType): Mat4Type

  /**
   * Generates a frustum matrix with the given bounds
   *
   * @param out Mat4Type frustum matrix will be written into
   * @param left Left bound of the frustum
   * @param right Right bound of the frustum
   * @param bottom Bottom bound of the frustum
   * @param top Top bound of the frustum
   * @param near Near bound of the frustum
   * @param far Far bound of the frustum
   * @returns out
   */
  frustum(
      out: Mat4Type, left: number, right: number, bottom: number, top: number,
      near: number, far: number): Mat4Type;

  /**
   * Generates a perspective projection matrix with the given bounds
   *
   * @param out Mat4Type frustum matrix will be written into
   * @param fovy Vertical field of view in radians
   * @param aspect Aspect ratio. typically viewport width/height
   * @param near Near bound of the frustum
   * @param far Far bound of the frustum
   * @returns out
   */
  perspective(
      out: Mat4Type, fovy: number, aspect: number, near: number,
      far: number): Mat4Type;

  /**
   * Generates a perspective projection matrix with the given field of view.
   * This is primarily useful for generating projection matrices to be used
   * with the still experimental WebVR API.
   *
   * @param {Mat4Type} out Mat4Type frustum matrix will be written into
   * @param {Object} fov Object containing the following values: upDegrees,
   *     downDegrees, leftDegrees, rightDegrees
   * @param {number} near Near bound of the frustum
   * @param {number} far Far bound of the frustum
   * @returns {Mat4Type} out
   */
  perspectiveFromFieldOfView(
      out: Mat4Type, fov: {
        upDegrees: number,
        downDegrees: number,
        leftDegrees: number,
        rightDegrees: number
      },
      near: number, far: number): Mat4Type

  /**
   * Generates a orthogonal projection matrix with the given bounds
   *
   * @param out Mat4Type frustum matrix will be written into
   * @param left Left bound of the frustum
   * @param right Right bound of the frustum
   * @param bottom Bottom bound of the frustum
   * @param top Top bound of the frustum
   * @param near Near bound of the frustum
   * @param far Far bound of the frustum
   * @returns out
   */
  ortho(
      out: Mat4Type, left: number, right: number, bottom: number, top: number,
      near: number, far: number): Mat4Type;

  /**
   * Generates a look-at matrix with the given eye position, focal point, and
   * up axis
   *
   * @param out Mat4Type frustum matrix will be written into
   * @param eye Position of the viewer
   * @param center Point the viewer is looking at
   * @param up Vec3Type pointing up
   * @returns out
   */
  lookAt(
      out: Mat4Type, eye: Vec3Type|number[], center: Vec3Type|number[],
      up: Vec3Type|number[]): Mat4Type;

  /**
   * Returns a string representation of a Mat4Type
   *
   * @param mat matrix to represent as a string
   * @returns string representation of the matrix
   */
  str(mat: Mat4Type): string;

  /**
   * Returns Frobenius norm of a Mat4Type
   *
   * @param a the matrix to calculate Frobenius norm of
   * @returns Frobenius norm
   */
  frob(a: Mat4Type): number;

  /**
   * Adds two Mat4Type's
   *
   * @param {Mat4Type} out the receiving matrix
   * @param {Mat4Type} a the first operand
   * @param {Mat4Type} b the second operand
   * @returns {Mat4Type} out
   */
  add(out: Mat4Type, a: Mat4Type, b: Mat4Type): Mat4Type

  /**
   * Subtracts matrix b from matrix a
   *
   * @param {Mat4Type} out the receiving matrix
   * @param {Mat4Type} a the first operand
   * @param {Mat4Type} b the second operand
   * @returns {Mat4Type} out
   */
  subtract(out: Mat4Type, a: Mat4Type, b: Mat4Type): Mat4Type

  /**
   * Subtracts matrix b from matrix a
   *
   * @param {Mat4Type} out the receiving matrix
   * @param {Mat4Type} a the first operand
   * @param {Mat4Type} b the second operand
   * @returns {Mat4Type} out
   */
  sub(out: Mat4Type, a: Mat4Type, b: Mat4Type): Mat4Type

  /**
   * Multiply each element of the matrix by a scalar.
   *
   * @param {Mat4Type} out the receiving matrix
   * @param {Mat4Type} a the matrix to scale
   * @param {number} b amount to scale the matrix's elements by
   * @returns {Mat4Type} out
   */
  multiplyScalar(out: Mat4Type, a: Mat4Type, b: number): Mat4Type

  /**
   * Adds two Mat4Type's after multiplying each element of the second operand
   * by a scalar value.
   *
   * @param {Mat4Type} out the receiving vector
   * @param {Mat4Type} a the first operand
   * @param {Mat4Type} b the second operand
   * @param {number} scale the amount to scale b's elements by before
   *     adding
   * @returns {Mat4Type} out
   */
  multiplyScalarAndAdd(out: Mat4Type, a: Mat4Type, b: Mat4Type, scale: number):
      Mat4Type

  /**
   * Returns whether or not the matrices have exactly the same elements in
   * the same position (when compared with ===)
   *
   * @param {Mat4Type} a The first matrix.
   * @param {Mat4Type} b The second matrix.
   * @returns {boolean} True if the matrices are equal, false otherwise.
   */
  exactEquals(a: Mat4Type, b: Mat4Type): boolean

  /**
   * Returns whether or not the matrices have approximately the same
   * elements in the same position.
   *
   * @param {Mat4Type} a The first matrix.
   * @param {Mat4Type} b The second matrix.
   * @returns {boolean} True if the matrices are equal, false otherwise.
   */
  equals(a: Mat4Type, b: Mat4Type): boolean
}

// QuatType
interface QuatType extends Float32Array {
  /**
   * Creates a new identity QuatType
   *
   * @returns a new Quaternion
   */
  create(): QuatType;

  /**
   * Creates a new QuatType initialized with values from an existing Quaternion
   *
   * @param a Quaternion to clone
   * @returns a new Quaternion
   * @function
   */
  clone(a: QuatType): QuatType;

  /**
   * Creates a new QuatType initialized with the given values
   *
   * @param x X component
   * @param y Y component
   * @param z Z component
   * @param w W component
   * @returns a new Quaternion
   * @function
   */
  fromValues(x: number, y: number, z: number, w: number): QuatType;

  /**
   * Copy the values from one QuatType to another
   *
   * @param out the receiving Quaternion
   * @param a the source Quaternion
   * @returns out
   * @function
   */
  copy(out: QuatType, a: QuatType): QuatType;

  /**
   * Set the components of a QuatType to the given values
   *
   * @param out the receiving Quaternion
   * @param x X component
   * @param y Y component
   * @param z Z component
   * @param w W component
   * @returns out
   * @function
   */
  set(out: QuatType, x: number, y: number, z: number, w: number): QuatType;

  /**
   * Set a QuatType to the identity Quaternion
   *
   * @param out the receiving Quaternion
   * @returns out
   */
  identity(out: QuatType): QuatType;

  /**
   * Sets a Quaternion to represent the shortest rotation from one
   * vector to another.
   *
   * Both vectors are assumed to be unit length.
   *
   * @param {QuatType} out the receiving Quaternion.
   * @param {Vec3Type} a the initial vector
   * @param {Vec3Type} b the destination vector
   * @returns {QuatType} out
   */
  rotationTo(out: QuatType, a: Vec3Type|number[], b: Vec3Type|number[]):
      QuatType;

  /**
   * Sets the specified Quaternion with values corresponding to the given
   * axes. Each axis is a Vec3Type and is expected to be unit length and
   * perpendicular to all other specified axes.
   *
   * @param {Vec3Type} view  the vector representing the viewing direction
   * @param {Vec3Type} right the vector representing the local "right" direction
   * @param {Vec3Type} up    the vector representing the local "up" direction
   * @returns {QuatType} out
   */
  setAxes(
      out: QuatType, view: Vec3Type|number[], right: Vec3Type|number[],
      up: Vec3Type|number[]): QuatType



  /**
   * Sets a QuatType from the given angle and rotation axis,
   * then returns it.
   *
   * @param out the receiving Quaternion
   * @param axis the axis around which to rotate
   * @param rad the angle in radians
   * @returns out
   **/
  setAxisAngle(out: QuatType, axis: Vec3Type|number[], rad: number): QuatType;

  /**
   * Gets the rotation axis and angle for a given
   *  Quaternion. If a Quaternion is created with
   *  setAxisAngle, this method will return the same
   *  values as providied in the original parameter list
   *  OR functionally equivalent values.
   * Example: The Quaternion formed by axis [0, 0, 1] and
   *  angle -90 is the same as the Quaternion formed by
   *  [0, 0, 1] and 270. This method favors the latter.
   * @param  {Vec3Type} out_axis  Vector receiving the axis of rotation
   * @param  {QuatType} q     Quaternion to be decomposed
   * @return {number}     Angle, in radians, of the rotation
   */
  getAxisAngle(out_axis: Vec3Type|number[], q: QuatType): number

  /**
   * Adds two QuatType's
   *
   * @param out the receiving Quaternion
   * @param a the first operand
   * @param b the second operand
   * @returns out
   * @function
   */
  add(out: QuatType, a: QuatType, b: QuatType): QuatType;

  /**
   * Multiplies two QuatType's
   *
   * @param out the receiving Quaternion
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  multiply(out: QuatType, a: QuatType, b: QuatType): QuatType;

  /**
   * Multiplies two QuatType's
   *
   * @param out the receiving Quaternion
   * @param a the first operand
   * @param b the second operand
   * @returns out
   */
  mul(out: QuatType, a: QuatType, b: QuatType): QuatType;

  /**
   * Scales a QuatType by a scalar number
   *
   * @param out the receiving vector
   * @param a the vector to scale
   * @param b amount to scale the vector by
   * @returns out
   * @function
   */
  scale(out: QuatType, a: QuatType, b: number): QuatType;

  /**
   * Calculates the length of a QuatType
   *
   * @param a vector to calculate length of
   * @returns length of a
   * @function
   */
  length(a: QuatType): number;

  /**
   * Calculates the length of a QuatType
   *
   * @param a vector to calculate length of
   * @returns length of a
   * @function
   */
  len(a: QuatType): number;

  /**
   * Calculates the squared length of a QuatType
   *
   * @param a vector to calculate squared length of
   * @returns squared length of a
   * @function
   */
  squaredLength(a: QuatType): number;

  /**
   * Calculates the squared length of a QuatType
   *
   * @param a vector to calculate squared length of
   * @returns squared length of a
   * @function
   */
  sqrLen(a: QuatType): number;

  /**
   * Normalize a QuatType
   *
   * @param out the receiving Quaternion
   * @param a Quaternion to normalize
   * @returns out
   * @function
   */
  normalize(out: QuatType, a: QuatType): QuatType;

  /**
   * Calculates the dot product of two QuatType's
   *
   * @param a the first operand
   * @param b the second operand
   * @returns dot product of a and b
   * @function
   */
  dot(a: QuatType, b: QuatType): number;

  /**
   * Performs a linear interpolation between two QuatType's
   *
   * @param out the receiving Quaternion
   * @param a the first operand
   * @param b the second operand
   * @param t interpolation amount between the two inputs
   * @returns out
   * @function
   */
  lerp(out: QuatType, a: QuatType, b: QuatType, t: number): QuatType;

  /**
   * Performs a spherical linear interpolation between two QuatType
   *
   * @param out the receiving Quaternion
   * @param a the first operand
   * @param b the second operand
   * @param t interpolation amount between the two inputs
   * @returns out
   */
  slerp(out: QuatType, a: QuatType, b: QuatType, t: number): QuatType;

  /**
   * Performs a spherical linear interpolation with two control points
   *
   * @param {QuatType} out the receiving Quaternion
   * @param {QuatType} a the first operand
   * @param {QuatType} b the second operand
   * @param {QuatType} c the third operand
   * @param {QuatType} d the fourth operand
   * @param {number} t interpolation amount
   * @returns {QuatType} out
   */
  sqlerp(
      out: QuatType, a: QuatType, b: QuatType, c: QuatType, d: QuatType,
      t: number): QuatType;

  /**
   * Calculates the inverse of a QuatType
   *
   * @param out the receiving Quaternion
   * @param a QuatType to calculate inverse of
   * @returns out
   */
  invert(out: QuatType, a: QuatType): QuatType;

  /**
   * Calculates the conjugate of a QuatType
   * If the Quaternion is normalized, this function is faster than
   * QuatType.inverse and produces the same result.
   *
   * @param out the receiving Quaternion
   * @param a QuatType to calculate conjugate of
   * @returns out
   */
  conjugate(out: QuatType, a: QuatType): QuatType;

  /**
   * Returns a string representation of a Quaternion
   *
   * @param a QuatType to represent as a string
   * @returns string representation of the QuatType
   */
  str(a: QuatType): string;

  /**
   * Rotates a Quaternion by the given angle about the X axis
   *
   * @param out QuatType receiving operation result
   * @param a QuatType to rotate
   * @param rad angle (in radians) to rotate
   * @returns out
   */
  rotateX(out: QuatType, a: QuatType, rad: number): QuatType;

  /**
   * Rotates a Quaternion by the given angle about the Y axis
   *
   * @param out QuatType receiving operation result
   * @param a QuatType to rotate
   * @param rad angle (in radians) to rotate
   * @returns out
   */
  rotateY(out: QuatType, a: QuatType, rad: number): QuatType;

  /**
   * Rotates a Quaternion by the given angle about the Z axis
   *
   * @param out QuatType receiving operation result
   * @param a QuatType to rotate
   * @param rad angle (in radians) to rotate
   * @returns out
   */
  rotateZ(out: QuatType, a: QuatType, rad: number): QuatType;

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
  fromMat3(out: QuatType, m: Mat3Type): QuatType;

  /**
   * Sets the specified Quaternion with values corresponding to the given
   * axes. Each axis is a Vec3Type and is expected to be unit length and
   * perpendicular to all other specified axes.
   *
   * @param out the receiving QuatType
   * @param view  the vector representing the viewing direction
   * @param right the vector representing the local "right" direction
   * @param up    the vector representing the local "up" direction
   * @returns out
   */
  setAxes(
      out: QuatType, view: Vec3Type|number[], right: Vec3Type|number[],
      up: Vec3Type|number[]): QuatType;

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
  rotationTo(out: QuatType, a: Vec3Type|number[], b: Vec3Type|number[]):
      QuatType;

  /**
   * Calculates the W component of a QuatType from the X, Y, and Z components.
   * Assumes that Quaternion is 1 unit in length.
   * Any existing W component will be ignored.
   *
   * @param out the receiving Quaternion
   * @param a QuatType to calculate W component of
   * @returns out
   */
  calculateW(out: QuatType, a: QuatType): QuatType;

  /**
   * Returns whether or not the Quaternions have exactly the same elements in
   * the same position (when compared with ===)
   *
   * @param {QuatType} a The first vector.
   * @param {QuatType} b The second vector.
   * @returns {boolean} True if the Quaternions are equal, false otherwise.
   */
  exactEquals(a: QuatType, b: QuatType): boolean;

  /**
   * Returns whether or not the Quaternions have approximately the same
   * elements in the same position.
   *
   * @param {QuatType} a The first vector.
   * @param {QuatType} b The second vector.
   * @returns {boolean} True if the Quaternions are equal, false otherwise.
   */
  equals(a: QuatType, b: QuatType): boolean;
}
