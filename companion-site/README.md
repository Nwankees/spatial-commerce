# Will It Fit? companion site

This is a dependency-free, static judge-facing site for a future `.tech` domain. The Android app remains the product; the site explains the end-to-end spatial-commerce flow and reports whether optional sponsor services are configured.

Preview locally from this directory with any static file server, for example:

```powershell
python -m http.server 4173
```

`site-config.js` may set a public API base URL. It must never contain credentials. In the Vultr/Caddy deployment, leave it blank so `/api/v1/sponsor/status` is requested from the same origin.
