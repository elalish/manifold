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
const cacheName = 'manifoldCAD-cache-v4';

const appShellAssets = [
  '/',
  '/index.html',
  '/editor.css',
  '/editor.js',
  '/editor-examples.js',
  '/manifest.json',
  '/fonts/orbitron-black-webfont.ttf',
  '/icons/close.png',
  '/icons/docs.png',
  '/icons/manifoldCAD.png',
  '/icons/manifoldCADonly.png',
  '/icons/ManifoldIcon.png',
  '/icons/mengerSponge192.png',
  '/icons/mengerSponge512.png',
  '/icons/mengerSponge64.png',
  '/icons/pause.png',
  '/icons/pencil.png',
  '/icons/play.png',
  '/icons/redo.png',
  '/icons/share.png',
  '/icons/star.png',
  '/icons/trash.png',
  '/icons/undo.png'
];

const shouldHandleRequest = request => {
  if (request.method !== 'GET') return false;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return false;

  // Skip transient HTML proxy requests only.
  // Caching this can cause stale proxy-module lookup issues.
  if (url.searchParams.has('html-proxy')) return false;

  return true;
};

self.addEventListener('install', e => {
  e.waitUntil((async () => {
    const cache = await caches.open(cacheName);
    await cache.addAll(appShellAssets);
    await self.skipWaiting();
  })());
});

self.addEventListener('install', e => {
  e.waitUntil(self.skipWaiting());
});

self.addEventListener('activate', e => {
  e.waitUntil(caches.keys()
                  .then((keyList) => {
                    return Promise.all(keyList.map((key) => {
                      if (key !== cacheName) {
                        return caches.delete(key);
                      }
                    }));
                  })
                  .then(() => self.clients.claim()));
});

// Serves from cache, then updates the cache in the background from the network,
// if available. Update available on refresh.
self.addEventListener('fetch', e => {
  if (!shouldHandleRequest(e.request)) {
    return;
  }

  e.respondWith((async () => {
    const cachedResponse = await caches.match(e.request);

    try {
      const response = await fetch(e.request);

      // Only cache successful basic and opaque responses.
      if (response.ok || response.type === 'opaque') {
        const cache = await caches.open(cacheName);
        await cache.put(e.request, response.clone());
      }

      return response;
    } catch (error) {
      if (cachedResponse) {
        return cachedResponse;
      }

      console.debug('Service worker fetch failed:', e.request.url, error);
      return Response.error();
    }
  })());
});
