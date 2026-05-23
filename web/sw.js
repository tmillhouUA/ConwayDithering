// Service worker — injects COOP/COEP headers so SharedArrayBuffer is available
// on GitHub Pages (which cannot set custom HTTP headers directly).

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));

self.addEventListener('fetch', e => {
    e.respondWith(
        fetch(e.request).then(resp => {
            const headers = new Headers(resp.headers);
            headers.set('Cross-Origin-Opener-Policy', 'same-origin');
            headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
            return new Response(resp.body, {
                status:     resp.status,
                statusText: resp.statusText,
                headers
            });
        })
    );
});
