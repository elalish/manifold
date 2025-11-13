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

export {AnimationMode, setAnimationDuration, setAnimationFPS, setAnimationMode} from '../lib/animation.ts';
export {resetToCircularDefaults, setCircularSegments, setMinCircularAngle, setMinCircularEdgeLength} from '../lib/level-of-detail.ts';
