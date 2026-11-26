# DOA-GAN calibration smoke test

**Date** 2026-11-26

The smoke set was the first 25 forged and first 25 authentic images in manifest order from the CASIA2 calibration split. Both checkpoints ran on the same images using pretrained weights. The audit used the [DOA-GAN authors' repository](https://github.com/asrafulashiq/doagan_clean) at commit `081cc88aa1b67c250290311e78a7c4f101fc28d7`. Its README assigns the three-channel checkpoint to USC-ISI and the single-channel checkpoint to CASIA and CoMoFoD.

1. **Strict loading.** Both checkpoints had zero missing or unexpected keys. The three-channel model had 26,974,214 parameters and the single-channel model had 26,973,700.
2. **Inference state.** Both models used `eval()` and `torch.no_grad()`. The running mean and variance in each of the 20 BatchNorm modules matched the checkpoint.
3. **Preprocessing.** Following `run_on_folder.py` and `utils.CustomTransform`, the audit decoded RGB with scikit-image, resized to 320×320 with linear interpolation before converting to float, and normalized with RGB mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225]. The normalized tensor for `Tp_D_CNN_M_N_nat00090_nat00090_10575.jpg` had mean −0.117267 and standard deviation 0.715064.
4. **Raw three-channel detection logits.** Forged [+20.2890, +19.3753, +20.8338, +19.8051, +20.3666]. Authentic [+20.1113, +20.5960, +20.9169, +20.0938, +20.1710]. These large positive values explain the rounded image score of 1.0000 in the earlier audit.
5. **Upstream inference.** `run_on_folder.py` discards the detection logit and exports localization probabilities with softmax for three channels or sigmoid for one channel. It defines no image-level decision. This audit summarized those probability maps by peak, mean and fraction above 0.5.
6. **Corrected smoke test.** The three-channel foreground map sums softmax channels 0 (target) and 1 (source), following upstream `test.py`. The earlier audit summed channels 1 and 2 and divided by two, which capped the map at 0.5. The single-channel foreground map uses the sigmoid output.

## Score distributions

Class entries show mean [min, max] for 25 images per class. AUROC treats higher scores as evidence of forgery. The raw logit appears for diagnosis.

| Checkpoint | Score | Forged | Authentic | AUROC |
|---|---|---|---|---:|
| Three-channel | Peak | 0.330606 [0.026966, 0.990702] | 0.496358 [0.054847, 0.963753] | 0.3472 |
| Three-channel | Mean | 0.017973 [0.001877, 0.039440] | 0.024399 [0.003955, 0.093149] | 0.4352 |
| Three-channel | Fraction above 0.5 | 0.000496 [0, 0.008857] | 0.006078 [0, 0.054668] | 0.3640 |
| Three-channel | Raw detection logit | 20.133524 [19.149326, 20.890844] | 20.300060 [18.740009, 21.162136] | 0.3840 |
| Single-channel | Peak | 0.708856 [0.045825, 0.999776] | 0.513493 [0.085763, 0.967491] | 0.7104 |
| Single-channel | Mean | 0.040120 [0.001961, 0.165576] | 0.010630 [0.002736, 0.052427] | 0.7072 |
| Single-channel | Fraction above 0.5 | 0.028337 [0, 0.140566] | 0.002289 [0, 0.032637] | 0.7328 |
| Single-channel | Raw detection logit | 13.826196 [11.931944, 15.716460] | 13.905725 [12.435533, 15.700412] | 0.4816 |

The corrected three-channel mask summaries all ranked authentic images higher on average. The single-channel summaries showed some ranking signal with substantial overlap, including an authentic peak of 0.967491 and a forged peak of 0.045825. With 25 authentic examples, one false alarm is 4%, so this smoke set cannot resolve the protocol's primary 2% budget. Neither checkpoint showed adequate forged-versus-authentic separation for fusion in this audit.

Reproduction uses `.venv/bin/python -u scripts/natural_doagan_bugcheck.py`. The audit saved image and heatmap pairs to `results/natural/montage_bugcheck_3ch.png` and `results/natural/montage_bugcheck_1ch.png`.
