# Automotive Vision Enhancement 🚘🖼️

A reproducible computer-vision workspace for experimenting with low-light and low-visibility image enhancement. The initial version provides classical baselines and records runtime and image-quality metrics where suitable reference images exist.

> **Status:** Research/learning prototype. Enhanced images can alter or obscure visual details. Do not use them as evidence of vehicle damage, road safety, driver-assistance performance, or maintenance decisions.

## Initial features

- Batch-process images from a directory.
- Compare classical enhancement methods: CLAHE, gamma correction, and unsharp masking.
- Keep original files unchanged and write results to a separate output directory.
- Record per-image processing time.
- Calculate PSNR and SSIM only when a matching reference image is provided.
- Export a CSV report that makes experiments comparable.
- Run automated tests and GitHub Actions CI.

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Install and test:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Run enhancement

Put supported images in `data/input/`, then run one method:

```bash
python -m vision_enhance.cli --input data/input --output outputs/clahe --method clahe
python -m vision_enhance.cli --input data/input --output outputs/gamma --method gamma --gamma 0.8
python -m vision_enhance.cli --input data/input --output outputs/sharpen --method unsharp
```

The output directory contains enhanced images and `metrics.csv`. The CSV records processing time and paths. Quality metrics are blank unless `--reference-dir` is supplied.

## Compare to reference images

When aligned reference images exist, place them in a reference directory using the same filenames as the input images:

```bash
python -m vision_enhance.cli \
  --input data/degraded \
  --reference-dir data/reference \
  --output outputs/clahe-eval \
  --method clahe
```

PSNR and SSIM are meaningful only when the degraded/input and reference images correspond appropriately and are aligned. Higher scores do not automatically mean the output looks more useful for every downstream task.

## Methods

- **CLAHE:** applies contrast-limited adaptive histogram equalization to the luminance channel in LAB color space.
- **Gamma correction:** adjusts intensity with a configurable gamma value.
- **Unsharp masking:** adds a scaled high-frequency residual to improve perceived sharpness.

These methods are transparent baselines. This repository does not currently include a trained neural restoration model or claim that enhancement recovers details missing from the original image.

## Roadmap

- [x] Implement CLAHE, gamma, and unsharp-mask baselines.
- [x] Batch processing and a CSV run report.
- [x] Optional PSNR/SSIM evaluation against matching reference images.
- [x] Unit tests and CI workflow.
- [ ] Add a documented, permitted dataset and train/validation/test split.
- [ ] Record baseline metrics across lighting conditions.
- [ ] Add no-reference quality metrics with their limitations clearly documented.
- [ ] Compare classical methods against a learned restoration model.
- [ ] Add reproducible experiment configurations and visual comparison sheets.

## Data and safety

- Use images you own, have permission to use, or that are appropriately licensed.
- Record dataset origin and any required attribution.
- Preserve originals; never silently overwrite input files.
- Enhancement may introduce halos, noise amplification, or misleading artifacts. Always inspect outputs alongside originals.
- Do not use enhanced output alone for safety-critical interpretation.

## Project log

See [LEARNING_LOG.md](LEARNING_LOG.md) for implementation decisions and next steps.

## License

A license will be selected after dataset and dependency terms are reviewed.
