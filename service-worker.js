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

// Increment version when updating CDN URLs to clean up cache.
const cacheName = 'manifoldCAD-cache-v1';

self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then((keyList) => {
    return Promise.all(keyList.map((key) => {
      if (key !== cacheName) {
        return caches.delete(key);
      }
    }));
  }));
});

// Serves from cache, then updates the cache in the background from the network,
// if available. Update available on refresh.
self.addEventListener(
    'fetch',
    e => {e.respondWith(caches.match(e.request).then(cachedResponse => {
      const networkFetch = fetch(e.request).then(response => {
        const clone = response.clone();
        caches.open(cacheName).then(cache => {
          cache.put(e.request, clone);
        });
        return response;
      });
      return cachedResponse || networkFetch;
    }))});
