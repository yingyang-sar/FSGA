# Implementation Notes

## Source and version

This package reorganizes the supplied `FSGA.py`, whose header describes **FSGA-Lite**: two ResNet-18 branches with slim projection/fusion MLPs. It has 23,212,979 parameters. A public `FSGA` alias points to the original `FBSRLearnableHypersphericalModel` class, retaining parameter names for checkpoint compatibility. The source uses older HFS/FBSR terminology in some names; this packaging does not establish equivalence to every model variant described in the manuscript.

Original source SHA-256: `cdd1fc45468db99142edb7b24c10bfff797b008b10cb7225121a06203a8e8a59`.

The original uploaded file is left unchanged. Its machine-specific absolute data path is replaced by configurable paths in this package. Before presenting this code as the implementation of the main-result table, confirm that this is the intended model variant and supply the corresponding run records.

## Preserved experimental behavior

- Class order is preserved exactly, including `m60` before `m548`.
- Grayscale conversion, log compression, Haar decomposition, band normalization and interpolation follow the original script.
- The two ResNet branches use ImageNet initialization by default. CNN BatchNorm2d statistics are frozen during training; projection BatchNorm1d remains active.
- Source and target mini-batches are shuffled separately, use `drop_last=True`, and are paired by `zip`. An epoch ends when the shorter loader ends; the longer loader is not cycled.
- Source memories are updated before sample-wise margin construction. Pseudo labels use margin-free fused predictions. Thresholds, warmup, ramp, detachment points, loss weights, clipping and random SWD projections are preserved.
- The zero-weight high-logit classifier and target-center loss are retained because deleting branches can change random-number consumption or checkpoint structure.
- Adam, cosine scheduling, scheduler-step order, gradient clipping at 1.0 and finite-value handling follow the source.

## Checkpoint selection

The original script evaluates `test` after every epoch and saves weights when test PCC strictly improves over `best = 0.0`. This package retains that rule. Target adaptation labels are discarded by the training loop, but **test labels still influence checkpoint selection**. Thus the test split is not an untouched final holdout, and the protocol must not be described as selecting a model only on the target training set or as avoiding all use of test labels.

Switching to a separate validation set or a fixed final epoch would change the experimental protocol and require new experiments; this package does not silently make that change. If PCC never exceeds zero, the original rule produces no best checkpoint, and the CLI reports that outcome accurately.

## Random-seed policy

Training now generates a seed at startup by default. A positive `--seed` fixes the seed; zero selects automatic generation. The resolved seed is printed and saved in configuration, epoch logs and checkpoints. This is an intentional change from the supplied script’s fixed initialization seed; the model, losses and checkpoint-selection rule are unchanged. Public scenario configurations do not prescribe an experiment seed.

## Packaging additions

Configuration is passed explicitly to modules and datasets; YAML/CLI settings work with spawned data-loader workers. Added utilities validate folder names, count images, export test metrics/features and plot results. Files are closed after reading; output folders are guarded against accidental replacement. New checkpoints record the complete configuration and selection split, but preserve the original model and memory keys.

PCC is accuracy. Macro precision, recall and F1 are computed over the ten classes with zero division set to zero. These additional evaluation exports do not change the original training loss or checkpoint selection metric.

## Scope

No image dataset, trained weights, ImageNet weights or third-party ASC files are redistributed here. No new ablation, noise or few-shot results have been added. The README table is transcribed manuscript data, not a measured result of the packaging checks. No software license has been selected on the author’s behalf.

See [VALIDATION.md](VALIDATION.md) for checks performed and their limits.
