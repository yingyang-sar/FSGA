"""Evaluate a saved checkpoint and optionally plot its confusion matrix/features."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def plot_results(output, matrix, features, labels, class_names, tsne, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Image count")
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted class", ylabel="True class", title="Measured test set")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=200)
    plt.close(fig)
    if tsne:
        if len(features) < 4:
            raise ValueError("t-SNE needs at least four test samples")
        from sklearn.manifold import TSNE
        embedded = TSNE(n_components=2, perplexity=min(30, len(features) - 1),
                        random_state=seed, init="pca", learning_rate="auto").fit_transform(features)
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, name in enumerate(class_names):
            points = embedded[labels == i]
            ax.scatter(points[:, 0], points[:, 1], s=14, color=plt.cm.tab10(i), label=name)
        ax.set(title="Target test features (f_dir, t-SNE)", xticks=[], yticks=[])
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(output / "tsne.png", dpi=200)
        plt.close(fig)
        np.savez_compressed(output / "tsne_coordinates.npz", coordinates=embedded, labels=labels)


def main(force_plots=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Prefer the config.yaml saved beside the checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--tsne", action="store_true", help="Also plot target test f_dir features")
    args = parser.parse_args()
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    from core.config import load_config
    from core.models.fsga import FSGA
    from core.data_loader.dataset import SARDataset
    from core.checkpoint import load_checkpoint
    from core.utils import set_seed
    from tools.check_data import inspect_data

    cfg = load_config(args.config)
    if args.data_root:
        cfg.ROOT = args.data_root
    cfg.DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    cfg.NUM_WORKERS = args.workers
    cfg.PRETRAINED = False  # All weights come from the checkpoint; no network needed.
    cfg.validate()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output}. Choose a new --output-dir.")
    _, errors = inspect_data(cfg.ROOT, splits=("test",))
    if errors:
        raise SystemExit("\n".join(errors))
    checkpoint = load_checkpoint(args.checkpoint, cfg)
    set_seed(cfg.SEED)
    dataset = SARDataset("test", cfg=cfg)
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
    model = FSGA(cfg=cfg).to(cfg.DEVICE)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    truth, predictions, features = [], [], []
    with torch.no_grad():
        for low, high, labels in loader:
            result = model(low.to(cfg.DEVICE), high.to(cfg.DEVICE))
            truth.extend(labels.tolist())
            predictions.extend(result["logits_fuse"].argmax(1).cpu().tolist())
            features.append(result["f_dir"].cpu().numpy())
    truth, predictions, features = np.array(truth), np.array(predictions), np.concatenate(features)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predictions, labels=list(range(cfg.NUM_CLASSES)), average="macro", zero_division=0)
    matrix = confusion_matrix(truth, predictions, labels=list(range(cfg.NUM_CLASSES)))
    metrics = {"PCC_percent": 100 * accuracy_score(truth, predictions),
               "precision_macro_percent": 100 * precision, "recall_macro_percent": 100 * recall,
               "F1_macro_percent": 100 * f1, "n_test": len(truth),
               "checkpoint_epoch": checkpoint.get("epoch"), "checkpoint": str(args.checkpoint),
               "selection_split": checkpoint.get("selection_split", "test (original script)"),
               "class_names": cfg.CLASS_NAMES}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "true_class", "predicted_class"])
        writer.writerows((str(Path(p).relative_to(cfg.ROOT)), cfg.CLASS_NAMES[y], cfg.CLASS_NAMES[q])
                         for p, y, q in zip(dataset.images, truth, predictions))
    with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/predicted", *cfg.CLASS_NAMES])
        writer.writerows([name, *row] for name, row in zip(cfg.CLASS_NAMES, matrix))
    np.savez_compressed(output / "features.npz", f_dir=features, labels=truth, predictions=predictions)
    if args.plots or args.tsne or force_plots:
        plot_results(output, matrix, features, truth, cfg.CLASS_NAMES, args.tsne, cfg.SEED)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
