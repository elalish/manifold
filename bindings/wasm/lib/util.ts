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

import {originalPositionFor, TraceMap} from '@jridgewell/trace-mapping';
import convert from 'convert-source-map';
import * as stackTraceParser from 'stacktrace-parser';

/**
 * Are we in a web worker?
 *
 * @returns A boolean.
 */
export const isWebWorker = (): boolean =>
    typeof self !== 'undefined' && typeof self.document == 'undefined';

/**
 * Are we in Node?
 *
 * @returns A boolean.
 */
export const isNode = (): boolean =>
    typeof process !== 'undefined' && !!process?.versions?.node;

export const getSourceMappedStackTrace =
    (code: string, error: Error, lineOffset: number = 0): string|undefined => {
      const converter = convert.fromSource(code);
      if (!converter || !error.stack) {
        // No inline source map.  We can't do anything.
        return error.stack
      }

      const tracer = new TraceMap(converter!.toObject());
      let stack = stackTraceParser.parse(error.stack);
      stack = stack.slice(
          0, stack.findIndex(call => call.methodName == 'evaluate'));

      stack = stack.map((frame: stackTraceParser.StackFrame) => {
        const {line: lineNumber, column, source: file} = originalPositionFor(
            tracer,
            {line: frame.lineNumber! + lineOffset, column: frame.column!});
        // Because the evaluator uses a Function constructor, the method name at
        // top level of the manifoldCAD script will be 'anonymous'.  Ignoring
        // that makes the stack more readable at the potential cost of confusion
        // if a manifoldCAD user is also constructing new Functions within their
        // model.
        const methodName =
            (frame.methodName != 'anonymous') ? frame.methodName : '';
        return {...frame, lineNumber, column, file, methodName};
      });

      return [
        error.toString(), ...stack.map(frame => {
          const location = `${frame.file}:${frame.lineNumber}:${frame.column}`;
          if (frame.methodName) {
            return `    at ${frame.methodName} (${location})`;
          } else {
            return `    at ${location}`;
          }
        })
      ].reduce((acc, cur) => `${acc}\n${cur}`);
    }