# Milestone 9 sponsor integrations

Milestone 9 adds three optional backend adapters plus a deployable companion site. None of them can prevent the existing local Android/AR flow from starting.

## Boundaries

| Interface | Remote implementation | No-credential / failure behavior | Product role |
| --- | --- | --- | --- |
| `PersistenceProvider` | MongoDB Atlas | process-local repository | session, saved-product, and fit history |
| `MemoryProvider` | Backboard memory API | process-local session memory | durable shopping preferences, favorites, and goals |
| `VoiceProvider` | ElevenLabs TTS | original text response | voice playback for conversational answers |

The adapters live under `backend/app/integrations/`. `backend/app/sponsor_api.py` exposes a narrow HTTP surface for the later M7 orchestrator and Android client.

## Endpoints

- `GET /api/v1/sponsor/status` — provider names and configured flags; never secret values.
- `POST /api/v1/sponsor/sessions` / `GET /api/v1/sponsor/sessions/{id}`.
- `POST /api/v1/sponsor/products` / `GET /api/v1/sponsor/sessions/{id}/products`.
- `POST /api/v1/sponsor/fit-history` / `GET /api/v1/sponsor/sessions/{id}/fit-history`.
- `POST /api/v1/sponsor/memory` / `POST /api/v1/sponsor/memory/search`.
- `POST /api/v1/sponsor/voice` — always returns the original text; audio is optional base64 MP3.

`GET /status` stays public for deployment health checks. Set `SPONSOR_DEMO_TOKEN`
on a hosted demo to require `X-Will-It-Fit-Token` on every other sponsor route.
Leave it blank for the USB/local demo. This shared token is a narrow hackathon-demo
boundary, not a substitute for per-user authentication in a production launch.

This API intentionally does not persist SF3D/GLB model bytes in MongoDB. Generated models remain in the existing asset cache/object-storage boundary.

## Backboard API choice

The adapter uses Backboard's current documented, assistant-scoped memory endpoints:

- `POST /assistants/{assistant_id}/memories`
- `POST /assistants/{assistant_id}/memories/search`

Memory records include `sessionId`, `kind`, and a product tag in metadata. Search results are filtered by exact session metadata before they reach M7. Use a dedicated Backboard assistant for this project; do not reuse one containing unrelated user memories.

Official references:

- <https://docs.backboard.io/concepts/memory>
- <https://docs.backboard.io/api-reference/memories/add>

## ElevenLabs client flow

The backend calls the documented synchronous text-to-speech endpoint and returns audio to the client without exposing the API key:

`POST /v1/text-to-speech/{voice_id}?output_format=mp3_44100_128`

The default model is `eleven_flash_v2_5`, chosen for a low-latency conversational demo. Change it via an environment variable. The Android/M7 client owns voice on/off, playback, and stop controls; on any backend/provider failure it displays `fallbackText`.

Official reference: <https://elevenlabs.io/docs/api-reference/text-to-speech/convert>

## Manual configuration

Copy `backend/.env.example` to `backend/.env` for local work, then provide only the services you want:

1. Atlas: create a database user and network access rule, then set `MONGODB_URI` and `MONGODB_DATABASE`.
2. Backboard: create a dedicated assistant in the dashboard and set `BACKBOARD_API_KEY` and `BACKBOARD_ASSISTANT_ID`.
3. ElevenLabs: choose a voice and set `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID`.
4. Hosted demo: generate a long random `SPONSOR_DEMO_TOKEN`, configure the trusted
   caller to send it as `X-Will-It-Fit-Token`, and follow `docs/VULTR_DEPLOYMENT.md`.

Leaving every value blank is supported and is the expected local developer default.

## M7 integration

The conversation agent keeps its process-local state authoritative. Before Qwen plans a turn,
the optional Backboard adapter gets up to 1.25 seconds to recall a few shopping hints using the
Android app's random, install-scoped shopper ID. Recalled text is explicitly marked as untrusted
and cannot override the current message. After a typed action succeeds, session, selected-product,
fit, and explicit refinement data are saved in the background; every sponsor operation fails open.

The Android chat displays text immediately. ElevenLabs is off by default and is called only after
the user enables **Voice**; playback can be stopped independently and a voice failure never changes
the conversational result.
