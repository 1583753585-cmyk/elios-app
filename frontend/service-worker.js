const CACHE_NAME = 'elios-v1';
const ASSETS = ['./index.html', './manifest.json'];

self.addEventListener('install', (e) => {
    e.waitUntil(caches.open(CACHE_NAME).then(c => c.addAll(ASSETS)));
    self.skipWaiting();
});

self.addEventListener('activate', (e) => {
    e.waitUntil(
        caches.keys().then(keys => 
            Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
        )
    );
    self.clients.claim();
});

self.addEventListener('fetch', (e) => {
    // API 请求不缓存
    if (e.request.url.includes('/api/')) return;
    
    e.respondWith(
        caches.match(e.request).then(r => r || fetch(e.request))
    );
});
