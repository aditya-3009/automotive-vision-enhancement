# Automotive Vision Enhancement Learning Log

## 2026-10-25: Classical baseline implementation

### Built
- Added CLAHE on the LAB luminance channel, gamma correction, and unsharp masking.
- Added batch processing for common image formats.
- Added runtime measurement and CSV reporting.
- Added optional PSNR/SSIM only for matching reference images.
- Added unit tests and a Python-version CI matrix.

### Experimental decisions
- Keep the source images unchanged and write enhanced outputs elsewhere.
- Do not report PSNR/SSIM without an aligned reference image.
- Preserve per-image status in the report so missing, unreadable, and mismatched inputs are visible.
- Treat enhancement as a visual transformation, not as recovery of guaranteed ground truth.

### Next tasks
1. Run the test suite and inspect the output report.
2. Add a small licensed or self-captured sample dataset with documented provenance.
3. Compare CLAHE, gamma, and unsharp masking on consistent image subsets.
4. Add no-reference quality metrics only with clear notes about their limitations.
5. Consider a learned restoration baseline after a credible classical benchmark exists.

### Commands
```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m vision_enhance.cli --input data/input --output outputs/clahe --method clahe
```
