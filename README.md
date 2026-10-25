# Team 25 M3 - Differentiable Self Calibration for Fourier Ptychography

## Question

Does estimating one global LED-array translation and rotation reduce aligned complex-object NRMSE relative to reconstruction with nominal LED geometry?

## Cold run

From the repository root in the course environment:

```bash
conda activate cse6730
python -m pytest -q
python scripts/make_figures.py
```

The tests verify the analytic pose gradient against centered finite differences, the zero-noise true-state limit, and the two-parameter pupil-extension derivative. The figure command reads `experiments/configs/m3_baseline.json` and writes `results/m3_first_result.png` and `results/m3_summary.json`.

If the course environment lacks a listed dependency, run:

```bash
python -m pip install -r requirements.txt
```

## M3 scope

The cold-run baseline uses a 64 x 64 synthetic complex object, 49 coherent intensity frames, a known circular pupil, and one fixed seed. This is the M1/M2 model at reduced grid size and run count for a reproducible M3 demonstration. 

## Repository map

- `src/fp_calibration/`: forward model, analytic gradients, optimizer, and aligned NRMSE.
- `tests/`: verification checks run by pytest.
- `experiments/configs/`: all reported parameters and the seed.
- `scripts/make_figures.py`: regenerates the first result.
- `scripts/extension_prototype.py`: checks defocus/astigmatism derivatives.
- `data/README.md`: data provenance and the later UConn slice.
- `results/`: generated figure and machine-readable summaries.
- `report/`: M3 memo.

## Reproducibility and AI use

No absolute paths, downloads, accounts, or private data are required. Generative AI helped in structuring the crude working python script into the current code/data pipeline in this repository. 

## Course-package note

This M3 baseline does not import `cse6730`; it is self-contained.