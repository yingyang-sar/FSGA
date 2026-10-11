# Installation and Usage

Use Python 3.10–3.12 in a separate environment. The package was checked with Python 3.12 and CPU PyTorch 2.5.1 / torchvision 0.20.1; this is not a record of the original paper’s software environment.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the matching PyTorch pair for your device. CPU:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

For a compatible NVIDIA setup, PyTorch provides a CUDA 12.1 wheel pair:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

Select a build compatible with your driver using the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/). Then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python tools/environment_info.py
```

## Data

Keep your existing folders. Each scenario must have the following paths, with the same ten class folders under each split:

| Folder | Role |
|:---|:---|
| `datasets2_sample_qpm/scenario_I/simulate/<class>/` | Labeled simulated training images |
| `datasets2_sample_qpm/scenario_I/train/<class>/` | Measured adaptation images; labels are not used by the training loss |
| `datasets2_sample_qpm/scenario_I/test/<class>/` | Measured images used for per-epoch evaluation and checkpoint selection |

Use the same layout for `scenario_II` and `scenario_III`. Class names and numerical label order are:

`2s1, bmp2, btr70, m1, m2, m35, m60, m548, t72, zsu23`.

Supported files: PNG, JPG/JPEG, TIF/TIFF and BMP. This implementation converts images to 8-bit grayscale, resizes to 128 × 128 and constructs the Haar-wavelet inputs. It does not directly read MAT files, complex-valued SAR data or ASC parameter files. The ASC link in the README is a credited resource mirror; ASC files are not required by this image loader.

Check the folder counts and image readability:

```bash
python tools/check_data.py --data-root datasets2_sample_qpm/scenario_I --scenario I --verify-images
python tools/check_data.py --data-root datasets2_sample_qpm/scenario_II --scenario II --verify-images
python tools/check_data.py --data-root datasets2_sample_qpm/scenario_III --scenario III --verify-images
```

Counts alone cannot verify depression angles, correct sample assignment or absence of duplicate images across splits.

## Train

All commands run from the repository root. Relative data paths are resolved from the working directory. You can use an absolute `--data-root` path instead.

```bash
python tools/train.py --config configs/scenario_I/FSGA.yaml --data-root datasets2_sample_qpm/scenario_I --device cuda
```

The three YAML files preserve the supplied script’s defaults: batch size 16, 100 epochs, Adam with learning rate 1e-4 and weight decay 1e-5. Their data/output paths differ by scenario. Training generates a random seed for each invocation unless a positive integer is supplied through `--seed`; omitting this option or passing zero selects automatic generation. The actual seed is printed and recorded in the run’s `config.yaml`, epoch log and saved checkpoint. To repeat a run, pass its recorded seed explicitly, even when reusing a saved configuration. ImageNet initialization is enabled and torchvision may download the ResNet-18 weights on first use. For an offline smoke check, `--no-pretrained` avoids that download but changes initialization and must not be described as reproducing the reported experiment.

`--device auto` is the default; it chooses CUDA if available. Use `--device cpu` for a CPU run. On systems where worker processes cause problems, use `--workers 0`.

A run writes `config.yaml`, `history.jsonl` and, when test PCC first exceeds 0 and subsequently improves, `best.pth`. Existing nonempty output folders are rejected. For a new run, specify a unique output path:

```bash
python tools/train.py --config configs/scenario_I/FSGA.yaml --data-root datasets2_sample_qpm/scenario_I --output-dir outputs/scenario_I/run_new
```

Checkpoints contain model weights, memories and configuration; they do not contain optimizer/scheduler states and do not support exact training resumption.

## Test and visualize

Prefer the resolved `config.yaml` saved with a run. When moving a checkpoint to another computer, override `--data-root` with the local scenario path.

```bash
python tools/test.py --config outputs/scenario_I/config.yaml --checkpoint outputs/scenario_I/best.pth --data-root datasets2_sample_qpm/scenario_I --output-dir outputs/scenario_I/metrics
python tools/visualize_results.py --config outputs/scenario_I/config.yaml --checkpoint outputs/scenario_I/best.pth --data-root datasets2_sample_qpm/scenario_I --output-dir outputs/scenario_I/plots --tsne
```

Testing writes PCC, macro precision/recall/F1 in `metrics.json`, per-image predictions, a labeled confusion-matrix CSV and `f_dir` features. Visualization adds a confusion matrix and optionally a t-SNE plot of target test features. These are computed from the selected checkpoint; they are not manuscript figures or a source/target alignment plot. Labels are used for evaluation and plot colors. Testing loads all weights from the checkpoint without downloading ImageNet weights.

New checkpoints store the complete configuration. Original-script checkpoints can also be loaded when their class order and architecture match; because old checkpoints omit some preprocessing settings, supply the configuration actually used to train them.
