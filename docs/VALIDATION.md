# Validation

Checked on 2026-10-11 with Python 3.12.14, CPU PyTorch 2.5.1 and torchvision 0.20.1.

| Check | Result |
|:---|:---|
| Supplied-script defaults vs packaged defaults | Matched for model/loss settings; paths/device/runtime controls differ, and default seeding now uses random generation |
| Model state-dict names, initial tensors and parameter count | Exact match; 23,212,979 parameters |
| Image preprocessing on a synthetic grayscale image | Exact tensor equality |
| Every model forward output | Exact tensor equality |
| Loss values and diagnostics during warmup and with selected pseudo labels | Exact match |
| Gradients, memory updates and one Adam update in both cases | Exact tensor equality |
| Nonempty sliced Wasserstein loss for p=1 and p=2 | Exact match |
| CLI help for train/test/visualize/check-data | Passed |
| Three YAML configurations and expected split totals | Passed |
| Spawned data-loader worker with explicit configuration | Blocked by runtime IPC socket restrictions; not verified |
| Image decoding and count-mismatch detection | Passed |
| One-epoch train, checkpoint save and reload, test metric/CSV export | Passed on synthetic images |
| Confusion-matrix and t-SNE export | Passed on synthetic images |

For equivalence checks, both models used random ResNet initialization with the same seed so no external weight download was required. Warmup was checked at epoch 1. The active-pseudo-label case used epoch 10 and a zero confidence threshold in both copies solely to exercise that branch; release defaults remain 0.85–0.95. Numerical comparisons used zero absolute and relative tolerances on CPU.

The command-line smoke check used ten synthetic images per split, a batch size of ten, one epoch and no pretrained initialization. Its data, checkpoint, metrics and plots are not included in the release and do not represent SAR recognition performance.

Real SAMPLE image files, trained experiment checkpoints and five-run logs were not supplied. Therefore, GPU training, ImageNet weight downloading, full 100-epoch runs, Windows execution and manuscript accuracy reproduction have not been verified. Multi-worker loading could not be verified because this environment prohibits the IPC socket used by PyTorch. End-to-end checks used `--workers 0`; Windows was not tested.

## Random-seed update checks

The automatic-seed and explicit-seed branches were checked, including omission/zero handling, overriding a seed in a reused configuration, random draw bounds, invalid inputs, resolved-configuration serialization, all three scenario defaults and CLI help. Python syntax checks passed. These targeted checks did not rerun full model training after the seed-policy update; the earlier numerical equivalence checks describe model/loss behavior with matching explicit seeds.
