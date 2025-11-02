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

const parseV8StackTrace = (stack: string) =>
    stack.split('\n').filter(frame => frame.match(/<anonymous>/)).map(frame => {
      const matches = frame.matchAll(/:([0-9]+):([0-9]+)\)$/g).next().value;
      const [line, column] = [parseInt(matches![1]), parseInt(matches![2])];
      const methodName = frame.match(/^\s+at\s([^\s]+)/)![1];
      return {
        line,
        column,
        // In Node or Chrome, a function constructor shows as 'eval'.
        methodName: methodName === 'eval' ? null : methodName
      };
    });

const parseSpiderMonkeyStackTrace = (stack: string) =>
    stack.split('\n')
        .filter(frame => frame.match(/AsyncFunction/))
        .map(frame => {
          const matches =
              frame.matchAll(/AsyncFunction:([0-9]+):([0-9]+)/g).next().value;
          const [line, column] = [parseInt(matches![1]), parseInt(matches![2])];
          const methodName = frame.match(/^[^@]+/)![0];
          return {
            line,
            column,
            // In Firefox, a function constructor shows as 'anonymous'.
            methodName: methodName === 'anonymous' ? null : methodName
          };
        });

/**
 * Attempt to parse a stack trace from a dynamically created function.
 *
 *  This makes the stack trace more readable at the potential cost of confusion
 * if a manifoldCAD user is also constructing new Functions within their model.
 */
export const parseStackTrace =
    (stack: string) => {
      if (stack.match(/^\s+at\s/gm)) {
        // V8 -- Chrome, NodeJS
        return parseV8StackTrace(stack);
      } else if (stack.match(/^([^@]+)@/gm)) {
        // SpiderMonkey -- Firefox
        return parseSpiderMonkeyStackTrace(stack);
      } else
        return [];
    }

export const getSourceMappedStackTrace =
    (code: string, error: Error, lineOffset: number = 0): string|undefined => {
      const converter = convert.fromSource(code);
      if (!converter || !error.stack) {
        // No inline source map.  We can't do anything.
        return error.stack
      }
      const parsed = parseStackTrace(error.stack);
      if (!parsed.length) {
        // We can't parse this.  Chances are, it's someone in Safari.
        return error.stack
      }
      const tracer = new TraceMap(converter!.toObject());
      const stack = parsed.map(frame => {
        if ((frame.line! + lineOffset) < 1) {
          return frame;  // Line number is out of range.
        }
        const {line, column, source: file} = originalPositionFor(
            tracer, {line: frame.line! + lineOffset, column: frame.column!});
        const {methodName} = frame;
        // column numbers should be 1 indexed.  Results are 0 indexed.
        return {line, column: column! + 1, file, methodName};
      });

      return [
        error.toString(), ...stack.map((frame: any) => {
          const location = `${frame.file}:${frame.line}:${frame.column}`;
          if (frame.methodName) {
            return `    at ${frame.methodName} (${location})`;
          } else {
            return `    at ${location}`;
          }
        })
      ].reduce((acc, cur) => `${acc}\n${cur}`);
    }