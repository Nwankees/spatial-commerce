# Will it Fit?

**See it. Shop it. Be sure it fits.**

Will it Fit? is an Android spatial-commerce app that lets you point your phone at a product, find similar real listings online, check whether one will physically fit in your space, preview it at real scale in AR, and continue the shopping flow through a conversational assistant.

The project was built for HackGT 13.

## Demo

Public companion site:

https://willitfitt.tech

## What it does

The basic flow is:

1. Point the camera at something you like.
2. Analyze the object with a local vision model.
3. Find visually similar products from real retailers.
4. Select a product.
5. Retrieve and verify its real-world dimensions.
6. Measure an available area using AR.
7. Check whether the product fits.
8. Generate a 3D model from the selected product image.
9. Place it in the room at real scale.
10. Ask the shopping assistant to refine, compare, preview, or prepare a purchase.

Example assistant requests:

```text
Show me a cheaper one.
Only show black chairs under $150.
Would the second one fit here?
Show that one in my room.
Buy this chair.
```

## How it works

### Object understanding

The app captures the current ARCore camera frame and sends it to a local `Qwen3-VL 8B` model through Ollama.

The model extracts things like:

- object type
- color
- material
- visual features
- useful shopping queries

For example, a chair might become:

```text
brown faux leather office chair
wooden armrests
curved backrest
```

Those descriptions are then used to search for real products.

### Product search

Product retrieval uses SerpApi with:

- Google Shopping
- Google Lens / visual matches
- retailer listings
- prices
- ratings
- product images
- merchant links

The results are deduplicated and visually reranked using Qwen3-VL so the app is not relying on text similarity alone.

### Product dimensions

The fit system does not let the model guess physical dimensions.

For a selected product, the backend tries to extract dimensions from retailer evidence such as:

- JSON-LD
- embedded product data
- specification tables
- labeled width / depth / height fields
- visible product-page content

The model can help interpret the page, but the dimensions still have to appear in the retrieved evidence.

Verified dimensions are normalized to meters before fit checking or AR scaling.

If reliable dimensions cannot be found, the app reports that fit could not be verified.

### AR measurement

The Android app uses ARCore to:

- detect surfaces
- place anchors
- measure points in real space
- build an available footprint
- keep placed products stable while the phone moves

The user measures the area where they want to place the product.

### Fit checking

The actual fit result is deterministic.

For a product with width `W` and depth `D`, and a measured space with width `SW` and depth `SD`, the app checks both orientations:

```text
W <= SW and D <= SD
```

or:

```text
D <= SW and W <= SD
```

If either orientation fits, the product fits.

No LLM decides the final result.

### 3D generation

Most shopping listings do not provide a downloadable 3D model.

Will it Fit? uses Stable Fast 3D to generate a GLB from the selected shopping image.

The generated model provides the shape, while the verified retailer dimensions provide the physical scale.

```text
product image
    ↓
Stable Fast 3D
    ↓
GLB
    ↓
scale using verified width/depth/height
    ↓
ARCore placement
```

This lets the app preview the selected product instead of using a generic placeholder model.

### Shopping assistant

The conversational agent uses `Qwen3 4B Instruct`.

The model does not directly modify app state. It produces typed actions that the backend executes.

The flow is:

```text
user message
    ↓
Qwen planner
    ↓
typed action
    ↓
backend tool
    ↓
tool result
    ↓
grounded response
```

Actions include things like:

```text
search
refine_search
select_product
compare_products
check_fit
show_in_ar
prepare_purchase
confirm_purchase
cancel_purchase
open_checkout
```

This keeps references like `"the second one"` and `"that chair"` tied to real product state.

## Voice

ElevenLabs powers both sides of the voice interface.

For input:

```text
microphone
    ↓
ElevenLabs Scribe v2
    ↓
transcript
    ↓
existing Qwen shopping pipeline
```

For output:

```text
Qwen response
    ↓
ElevenLabs TTS
    ↓
spoken response
```

The app supports push-to-talk and keeps typed chat as a fallback.

## Purchase flow

The shopping assistant can prepare a purchase intent, but it cannot complete anything from a command like:

```text
Buy the second one.
```

Instead, it creates a review containing the exact:

- product
- merchant
- price
- quantity
- available variant information
- fit status
- original retailer URL

The user must explicitly confirm before the flow continues.

The backend then runs a Visa TAP-aligned development flow that includes:

- backend-only signing
- payload integrity checks
- nonce handling
- replay protection
- merchant-side verification

Visa Intelligent Commerce itself was not publicly available for the hackathon, so transaction completion is explicitly marked as simulated.

The app does not collect PAN, CVV, or real card credentials.

## Persistence and memory

### MongoDB Atlas

MongoDB is used for persistent application and commerce state such as:

- sessions
- selected products
- fit-check history
- purchase intents
- commerce status

### Backboard

Backboard is used for longer-term shopping preferences, for example:

- preferred colors
- budget
- preferred merchants
- style preferences

A useful way to think about the difference:

```text
MongoDB remembers what happened.
Backboard remembers what the shopper tends to prefer.
```

Backboard memory can influence recommendations, but it never authorizes a purchase.

## Public deployment

The public companion site is hosted on Vultr at:

https://willitfitt.tech

The deployment uses:

- Ubuntu 24.04
- Docker Compose
- Caddy
- automatic HTTPS with Let's Encrypt

The GPU-heavy services remain local:

- Ollama
- Qwen3-VL
- Qwen3
- Stable Fast 3D

## Tech stack

### Android

- Kotlin
- ARCore
- OpenGL / GLB rendering

### Backend

- Python
- FastAPI

### AI / ML

- Qwen3-VL 8B
- Qwen3 4B Instruct
- Ollama
- Stable Fast 3D

### Product retrieval

- SerpApi
- Google Shopping
- Google Lens

### Sponsor integrations

- ElevenLabs
- MongoDB Atlas
- Backboard
- Vultr
- .TECH
- Visa Trusted Agent Protocol-aligned development flow

### Deployment

- Docker
- Docker Compose
- Caddy
- Vultr

## Local architecture

```text
Android / ARCore
      |
      v
FastAPI backend
      |
      +--> Ollama / Qwen3-VL
      |
      +--> Ollama / Qwen3
      |
      +--> SerpApi
      |
      +--> retailer dimension pipeline
      |
      +--> Stable Fast 3D
      |
      +--> MongoDB Atlas
      |
      +--> Backboard
      |
      +--> ElevenLabs
```

## Running locally

### Requirements

You will need:

- Python
- Android Studio / Android SDK
- Ollama
- Qwen3-VL
- Qwen3
- Stable Fast 3D environment
- required API credentials

### Backend

From the backend directory:

```powershell
.venv\\Scripts\\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Android device

For a USB-connected Android device:

```powershell
adb reverse tcp:8000 tcp:8000
```

Then build/install the Android app and launch it on an ARCore-supported device.

## Environment variables

The backend expects configuration for services such as:

```env
SERPAPI_API_KEY=
BACKBOARD_API_KEY=
BACKBOARD_ASSISTANT_ID=
MONGODB_URI=
MONGODB_DATABASE=
ELEVENLABS_API_KEY=
ELEVENLABS_VOICE_ID=
ELEVENLABS_STT_MODEL_ID=
```

Do not commit real secrets.

## Notes

- Product dimensions are only used when they can be verified from retailer evidence.
- Generated 3D geometry may not perfectly reconstruct unseen sides of an object from a single image.
- Visa transaction completion is simulated because Visa Intelligent Commerce was not publicly accessible during the hackathon.
- The core Qwen and SF3D inference runs locally on an RTX 5080 laptop.

## Built at HackGT 13

Will it Fit? was built around one simple question:

**Can online shopping understand both what I want and the physical space I want to put it in?**
