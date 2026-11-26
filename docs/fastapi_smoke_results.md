# FastAPI smoke test

Five distinct local dataset images were uploaded over HTTP to Uvicorn with the real locked DOA-GAN checkpoint.
The test compared all eight decisions and decoded masks with direct pipeline outputs for the same RGB pixels.
Timings include upload, analysis, and the JSON response with all eight PNG masks. Labels describe the dataset.

| Image | Split | Label | HTTP | Flagged detectors | Positive rules | Elapsed ms |
|---|---|---|---:|---|---|---:|
| `Tp_D_CNN_M_N_nat00090_nat00090_10575.jpg` | casia2/calibration | forged | 200 | sift, doagan | F1, F3, F4, F5 | 3175 |
| `Tp_D_CNN_M_N_pla00042_pla00042_10976.jpg` | casia2/calibration | forged | 200 | sift, doagan | F1, F3, F4, F5 | 2914 |
| `Au_ani_00008.jpg` | casia2/calibration | authentic | 200 | None | None | 1928 |
| `Au_ani_00015.jpg` | casia2/calibration | authentic | 200 | None | None | 2204 |
| `001_F.png` | comofod/dev | forged | 200 | dct, sift | F1, F2, F3, F4 | 3431 |
