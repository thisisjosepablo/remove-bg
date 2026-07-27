# Cutout Studio

Free, open-source, self-hosted background removal — automatic AI cutout plus
manual refinement (click points / draw a box), running entirely on your own
machine or server. No account, no cloud API, no usage limits.

- **Automatic cutout** powered by [BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
- **Manual refinement** powered by [SAM2](https://github.com/facebookresearch/sam2) (Meta) — click to include/exclude regions or drag a box
- Runs on CPU (works everywhere) or GPU (much faster, needs an NVIDIA card)
- Single FastAPI backend + a small vanilla-JS frontend, no build step

## Quick start (Docker)

Requires [Docker](https://docs.docker.com/get-docker/) and the
[Compose plugin](https://docs.docker.com/compose/install/) (included with
modern Docker Desktop / `docker-ce`).

**CPU (works on any machine):**

```bash
docker compose --profile cpu up --build
```

**GPU (NVIDIA GPU + [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) required):**

```bash
docker compose --profile gpu up --build
```

Then open http://localhost:8000

The first request downloads the model weights from Hugging Face (a few
hundred MB) and caches them in the `model-cache` Docker volume, so restarts
are fast.

### Plain `docker run` (CPU)

```bash
docker build -t cutout-studio .
docker run --rm -p 8000:8000 -v cutout-models:/data/hf-cache cutout-studio
```

## Running without Docker

```bash
python -m venv venv
source venv/bin/activate

# CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# or GPU (pick the CUDA version matching your driver, see pytorch.org):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Configuration

All optional, set as environment variables:

| Variable                    | Default                        | Description |
|------------------------------|---------------------------------|--------------|
| `CUTOUT_ALLOWED_ORIGINS`     | `*`                            | Comma-separated CORS origins |
| `CUTOUT_MAX_IMAGE_EDGE`      | `1280`                          | Max input image dimension before downscaling |
| `CUTOUT_MAX_SESSIONS`        | `150`                           | Max concurrent in-memory sessions |
| `CUTOUT_SESSION_TTL`         | `7200`                          | Session lifetime in seconds |
| `CUTOUT_MAX_CONCURRENT_ML`   | `1`                             | Max concurrent GPU/CPU inference calls |
| `CUTOUT_CPU_THREADS`         | all cores                      | PyTorch CPU thread count |
| `BIREFNET_MODEL`             | `ZhengPeng7/BiRefNet_lite`      | Hugging Face model id for auto cutout |
| `BIREFNET_SIZE`              | `768` (CPU) / `1024` (GPU)      | Inference resolution cap |
| `SAM2_MODEL`                 | `facebook/sam2.1-hiera-tiny`    | Hugging Face model id for refinement |
| `MASK_FEATHER`               | `0.8`                           | Edge feathering (Gaussian blur radius) |
| `HF_HOME`                    | `/data/hf-cache` (in Docker)    | Hugging Face model cache directory |

## How it works

1. Upload an image → BiRefNet produces an automatic mask and a transparent
   PNG in one request (`POST /api/upload-and-auto`).
2. Not happy with an edge? Switch to **Refine**: click points to include/exclude
   regions, or drag a box around the subject. SAM2 recomputes the mask on top
   of (or instead of) the automatic one, in real time.
3. Download the result as a transparent PNG.

Sessions are kept in memory only (no disk persistence, no database) — this is
meant for personal/self-hosted use, not as a multi-tenant SaaS backend.

## Project structure

```
backend/    FastAPI app + BiRefNet/SAM2 inference services
frontend/   Static HTML/CSS/JS editor (no build step)
Dockerfile      CPU image
Dockerfile.gpu  GPU (CUDA) image
```

## License

Code in this repo is MIT-licensed (see `LICENSE`). It uses two third-party
models at runtime (downloaded on first use, not vendored) — see `NOTICE.md`
for their licenses (BiRefNet: MIT, SAM2: Apache-2.0).

## Contributing

Issues and pull requests welcome.
