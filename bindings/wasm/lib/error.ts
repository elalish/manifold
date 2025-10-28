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

import type {BuildFailure, Location, Message} from 'esbuild';

export class BundlerError extends Error {
  location?: Location;
  error: Message;

  constructor(failure: BuildFailure, options?: ErrorOptions) {
    super(undefined, options);
    this.cause = failure;
    this.error = failure.errors[0];
  }

  get name(): string {
    return 'BundlerError';
  }

  get message(): string {
    return this.error.text;
  }

  get stack(): string|undefined {
    if (!this.error.location) return undefined;
    const {file, line, column} = this.error.location!;
    return `${this.toString()}\n    at ${file}:${line}:${column}`;
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

  get message(): string {
    return this.cause.message;
  }

  get stack(): string|undefined {
    return this.manifoldStack;
  }
}
