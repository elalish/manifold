// Copyright 2025 The Manifold Authors.
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
 * The globalDefaults object is shared between a manifoldCAD script and the
 * scene builder.
 *
 * It can be used to set parameters elsewhere in ManifoldCAD.  For example, the
 * GLTF exporter will look for animation type and framerate.
 *
 * It is only accessable as a global object, by a top level script evaluated by
 * the worker.  Libraries will not have access to it.
 *
 * See 'Tetrahedron Puzzle' for an example.
 */
export declare const globalDefaults: {
  roughness: number,
  metallic: number,
  baseColorFactor: [number, number, number],
  alpha: number,
  unlit: boolean,
  animationLength: number,
  animationMode: 'loop'|'ping-pong';
}