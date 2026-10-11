"""Train the uploaded ResNet-18/slim-MLP FSGA variant."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int, default=0,
                        help="0 or omitted: generate a random seed; positive integer: fixed seed")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--no-pretrained", action="store_true", help="Offline smoke checks; changes training initialization")
    args = parser.parse_args()

    import torch
    import yaml
    from core.config import load_config
    from core.utils import resolve_seed, set_seed
    from core.data_loader.loaders import build_loaders
    from core.models.fsga import FSGA
    from core.memory import FeatureCenterMemory, ZMagnitudeMemory
    from core.engine import train_one_epoch, test, print_epoch_log
    from core.checkpoint import save_checkpoint
    from tools.check_data import inspect_data

    cfg = load_config(args.config)
    for argument, key in (("data_root", "ROOT"), ("seed", "SEED"), ("workers", "NUM_WORKERS"),
                          ("epochs", "EPOCHS"), ("batch_size", "BATCH_SIZE")):
        value = getattr(args, argument)
        if value is not None:
            setattr(cfg, key, value)
    cfg.DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.no_pretrained:
        cfg.PRETRAINED = False
    output = Path(args.output_dir) if args.output_dir else Path(cfg.SAVE_PATH).parent
    cfg.SAVE_PATH = str(output / "best.pth")
    cfg.validate()
    # Avoid replacing an earlier experiment by accident.
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output}. Choose a new --output-dir.")
    _, errors = inspect_data(cfg.ROOT)
    if errors:
        raise SystemExit("\n".join(errors))
    cfg.SEED = resolve_seed(cfg.SEED)
    print(f"Seed: {cfg.SEED}", flush=True)
    (output / "config.yaml").write_text(yaml.safe_dump(cfg.as_dict(), sort_keys=False), encoding="utf-8")
    set_seed(cfg.SEED)
    loaders = build_loaders(cfg)
    model = FSGA(cfg=cfg).to(cfg.DEVICE)
    low_memory = FeatureCenterMemory(cfg.NUM_CLASSES, model.feat_dim, momentum=0.90, device=cfg.DEVICE)
    zmag_memory = ZMagnitudeMemory(cfg.NUM_CLASSES, momentum=cfg.ZMAG_MEMORY_MOMENTUM, device=cfg.DEVICE, cfg=cfg)
    target_low_memory = FeatureCenterMemory(cfg.NUM_CLASSES, model.feat_dim, momentum=0.90, device=cfg.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.EPOCHS, eta_min=cfg.LR * 0.05)
    print(f"Device={cfg.DEVICE}, parameters={sum(p.numel() for p in model.parameters()):,}")
    print("Original protocol: best checkpoint selected by measured TEST accuracy each epoch.")
    best = 0.0
    with (output / "history.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, cfg.EPOCHS + 1):
            loss, ratio, detail = train_one_epoch(model, low_memory, zmag_memory, target_low_memory,
                                                 loaders[0], loaders[1], opt, epoch, cfg=cfg)
            scheduler.step()
            metrics = test(model, loaders[2], cfg=cfg)
            if metrics["acc_fuse"] > best:
                best = metrics["acc_fuse"]
                save_checkpoint(model, low_memory, zmag_memory, target_low_memory, epoch, best, cfg)
            print_epoch_log(epoch, loss, ratio, metrics, best, detail)
            log.write(json.dumps({"epoch": epoch, "seed": cfg.SEED, "loss": loss, "pseudo_ratio": ratio,
                                  "lr_next_epoch": scheduler.get_last_lr()[0], "best": best,
                                  **metrics, "last_batch": detail}) + "\n")
            log.flush()
    if Path(cfg.SAVE_PATH).exists():
        print(f"Best test PCC: {best * 100:.2f}%. Checkpoint: {cfg.SAVE_PATH}")
    else:
        print("No checkpoint saved: test PCC never exceeded 0 (original saving rule).")


if __name__ == "__main__":
    main()
