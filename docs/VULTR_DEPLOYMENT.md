# Vultr deployment: companion site and optional services

This deployment intentionally excludes the GPU-dependent SF3D reconstruction sidecar. For the live hackathon demo:

- **Galaxy S25 → laptop backend via `adb reverse`** handles AR, local Qwen, and SF3D on the RTX GPU.
- **Vultr** hosts the public companion site and optional MongoDB/Backboard/ElevenLabs-facing API routes.

## Prepare the instance

1. Create a small Ubuntu 24.04 Vultr instance in a nearby region. A CPU instance is sufficient for these components.
2. Add an SSH key through the Vultr control panel. Do not copy a private key into this repository.
3. Point the `.tech` domain's `A` record at the instance IPv4 address. Domain registration/DNS changes are manual because availability and ownership are not assumed.
4. Allow inbound TCP 22, 80, and 443 in the Vultr firewall. Restrict SSH source addresses if practical.
5. Install Docker Engine plus the Compose plugin using Docker's official Ubuntu instructions.

## Configure and start

From a repository checkout on the instance:

```bash
cd deploy
cp vultr.env.example vultr.env
chmod 600 vultr.env
# Edit vultr.env on the server; never paste secrets into Git.
docker compose --env-file vultr.env -f docker-compose.vultr.yml config --quiet
docker compose --env-file vultr.env -f docker-compose.vultr.yml up -d --build
docker compose --env-file vultr.env -f docker-compose.vultr.yml ps
```

For a credential-free configuration check in a clone, set `VULTR_ENV_FILE=./vultr.env.example` and run the `config --quiet` command. This validates the Compose file without contacting a sponsor API.

Caddy obtains HTTPS certificates after DNS points to the instance and ports 80/443 are reachable. It serves the static site and proxies only `/api/v1/sponsor/*` to FastAPI, so the local Qwen, product-search, dimension, and SF3D routes are not exposed by this deployment.

## Verify

```bash
curl --fail https://YOUR_DOMAIN/health
curl --fail https://YOUR_DOMAIN/api/v1/sponsor/status
# For any non-status sponsor route:
curl --fail -H "X-Will-It-Fit-Token: YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text":"Sponsor API is reachable.","enabled":false}' \
  https://YOUR_DOMAIN/api/v1/sponsor/voice
```

Expected without credentials: HTTP 200 with providers reported as local fallbacks. After credentials are added, `configured` changes to `true`; it does not prove a paid API call succeeded, so perform one explicit demo test per configured provider.

## Operational cautions

- Keep `vultr.env` outside Git and rotate a credential if it ever appears in logs or shell history.
- Atlas should allow the Vultr instance address, not `0.0.0.0/0`, when the instance IP is stable.
- Set a long random `SPONSOR_DEMO_TOKEN` before exposing the host. The shared token limits casual public use, but add per-user authentication and rate limiting before any multi-user launch.
- Do not point hosted AR-preview requests at a laptop through an unprotected public tunnel.
- Back up only durable session metadata. Do not upload camera frames or reconstructed GLBs unless a later privacy/storage design explicitly requires it.

No cloud resources are created by the repository configuration alone; deployment is a manual post-merge action.
