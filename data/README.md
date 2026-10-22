# Data note

M3 uses only the deterministic synthetic generator in `src/fp_calibration/model.py`; no external file is required for the cold run. The reported configuration and seed are in `experiments/configs/m3_baseline.json`.

M2 also identified the UConn Smart Imaging Lab Fourier-Ptychography repository for later real-data plausibility checks: https://github.com/SmartImagingLabUConn/Fourier-Ptychography (retrieved October 4, 2026). Its code is labeled GPL 2.0; separate dataset terms were not stated. The planned later slice is the centered 7 x 7 subset of `HE_blue.mat`. It is not redistributed or used in M3.
