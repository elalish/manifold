// Copyright 2026 The Manifold Authors.
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

const ATTEMPTS = 3;
const BASE_DELAY_MS = 200;
const FACTOR = 3;

function isRetryable(status: number): boolean {
  return status >= 500 || status === 429;
}

/**
 * Wraps `fetch` with retry-on-transient-failure. Retries on network errors,
 * HTTP 5xx, and HTTP 429; returns immediately on 2xx and on 4xx (other than
 * 429) so typo'd URLs fail fast. Aborts via `init.signal` are not retried.
 *
 * Exponential backoff: 200ms, 600ms, then give up (~800ms max added latency
 * before failure when the upstream is fully down).
 */
export async function fetchWithRetry(
    input: RequestInfo|URL, init?: RequestInit): Promise<Response> {
  for (let attempt = 0; attempt < ATTEMPTS; attempt++) {
    try {
      const response = await fetch(input, init);
      if (!isRetryable(response.status) || attempt === ATTEMPTS - 1) {
        return response;
      }
    } catch (err) {
      if (init?.signal?.aborted) throw err;
      if (attempt === ATTEMPTS - 1) throw err;
    }
    await new Promise(
        resolve => setTimeout(resolve, BASE_DELAY_MS * FACTOR ** attempt));
  }
  // Unreachable: the loop above either returns or throws on the last attempt.
  throw new Error('fetchWithRetry: unreachable');
}
