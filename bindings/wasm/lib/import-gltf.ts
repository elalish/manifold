// Copyright 2024-25 The Manifold Authors.
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

import {WebIO} from '@gltf-transform/core';

import {setupIO} from './gltf-io.ts';

const binaryFormat = {
  extension: 'glb',
  mimetype: 'model/gltf-binary'
};

const jsonFormat = {
  extension: 'gltf',
  mimetype: 'model/gltf+json'
};

export const supportedFormats = [binaryFormat, jsonFormat];

export const fetchModel = async (url: string) => {
  const io = setupIO(new WebIO());
  return await io.read(url);
};
