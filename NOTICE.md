# Third-party models and code

Cutout Studio's own code is MIT-licensed (see `LICENSE`). It downloads and runs
two third-party ML models at runtime — they are **not** vendored in this repo,
they're pulled from Hugging Face / GitHub on first run and cached locally.

## BiRefNet (automatic background removal)

- Repo: https://github.com/ZhengPeng7/BiRefNet
- Model used by default: `ZhengPeng7/BiRefNet_lite` (Hugging Face)
- License: MIT

## SAM2 (Segment Anything Model 2 — manual refinement)

- Repo: https://github.com/facebookresearch/sam2 (Meta AI / FAIR)
- Model used by default: `facebook/sam2.1-hiera-tiny` (Hugging Face)
- License: Apache License 2.0

Both are permissive licenses that allow commercial and non-commercial use,
modification, and redistribution. Verify the license terms on the linked
repos before relying on this for your own use case — model licenses can
change between versions, and this file may lag behind upstream.

If you swap `BIREFNET_MODEL` or `SAM2_MODEL` (env vars) for a different
checkpoint, check that checkpoint's own license — it may differ from the
defaults above.
