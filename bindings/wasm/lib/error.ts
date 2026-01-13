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
 * Exceptions, and where to find them.
 *
 * @packageDocumentation
 * @group manifoldCAD Runtime
 * @category Core
 */

import type {BuildFailure, Location, Message} from 'esbuild-wasm';

export class BundlerError extends Error {
  location?: Location;
  error: Message;
  manifoldStack?: string;

  constructor(failure: BuildFailure, options?: ErrorOptions) {
    super(undefined, options);
    this.cause = failure;
    this.error = failure.errors[0];

    if (this.error.location) {
      let {file, line, column} = this.error.location!;
      // FIXME Given that we insert metadata into each file, we need to run this
      // through sourcemap.
      line--;
      this.manifoldStack =
          `${this.toString()}\n    at ${file}:${line}:${column}`;
    }
  }

  get name(): string {
    return 'BundlerError';
  }

  get message(): string {
    return this.error.text;
  }
};

export class RuntimeError extends Error {
  manifoldStack?: string;
  cause: Error;

  constructor(cause: Error, message?: string, options?: ErrorOptions) {
    super(message ?? cause.message, options);
    this.cause = cause;
  }

  get name(): string {
    return this.cause.name;
  }
}

export class UnsupportedFormatError extends Error {
  constructor(
      identifier: string,
      supported: Array<{mimetype: string, extension: string}>) {
    const typeList =
        supported
            .map(entry => `\`${entry.mimetype}\` (\`.${entry.extension}\`)`)
            .reduceRight(
                (prev, cur, index, arr) => cur +
                    ((index > 0 || arr.length <= 2) ? ', or ' : ', ') + prev);
    super(
        `Unsupported format \`${identifier}\`.  ` +
        `Must be one of ${typeList}`);
  }
}

export class ImportError extends Error {}