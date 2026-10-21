# Installation

1. Install dependencies (Python 3.10–3.12). Install [PyTorch](https://pytorch.org/get-started/previous-versions/) for your device, then run:

```bash
pip install -r requirements.txt
```

2. Prepare the SAMPLE dataset. Each scenario (`scenario_I`, `scenario_II`, `scenario_III`) should contain `simulate/`, `train/`, and `test/` folders, each organized by class. See [README.md](README.md) for details.

3. Train (run from the repository root):

```bash
python tools/train.py --config configs/scenario_I/FSGA.yaml --data-root datasets2_sample_qpm/scenario_I
```

Replace `scenario_I` with `scenario_II` or `scenario_III` for other scenarios.

4. Test:

```bash
python tools/test.py --config outputs/scenario_I/config.yaml --checkpoint outputs/scenario_I/best.pth --data-root datasets2_sample_qpm/scenario_I
```

For visualization, use `tools/visualize_results.py` with the same arguments.
