"""Count SAMPLE images and optionally verify counts or decode every image."""
import argparse
from pathlib import Path

CLASS_NAMES = ["2s1", "bmp2", "btr70", "m1", "m2", "m35", "m60", "m548", "t72", "zsu23"]
EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
ALL = [174, 107, 92, 129, 128, 129, 176, 128, 108, 174]
LOW = [116, 55, 43, 78, 75, 76, 116, 75, 56, 116]
ANGLE16 = [50, 55, 43, 52, 52, 52, 51, 52, 56, 50]
ANGLE17 = [58, 52, 49, 51, 53, 53, 60, 53, 52, 58]


def inspect_data(root, scenario=None, verify_images=False, splits=("simulate", "train", "test")):
    root = Path(root)
    expected = {"simulate": LOW if scenario == "III" else ALL,
                "train": LOW if scenario == "I" else ANGLE16, "test": ANGLE17}
    counts, errors = {}, []
    for split in splits:
        folder = root / split
        if not folder.is_dir():
            errors.append(f"Missing split: {folder}")
            continue
        unknown = sorted(p.name for p in folder.iterdir() if p.is_dir() and p.name not in CLASS_NAMES)
        if unknown:
            errors.append(f"Unknown class folders in {split}: {unknown}")
        counts[split] = {}
        for idx, name in enumerate(CLASS_NAMES):
            directory = folder / name
            files = sorted(p for p in directory.glob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
            counts[split][name] = len(files)
            if not files:
                errors.append(f"Missing or empty class: {directory}")
            elif scenario and len(files) != expected[split][idx]:
                errors.append(f"{split}/{name}: found {len(files)}, expected {expected[split][idx]}")
            if verify_images:
                from PIL import Image
                for path in files:
                    try:
                        with Image.open(path) as img:
                            img.load()
                    except Exception as exc:
                        errors.append(f"Unreadable image {path}: {exc}")
    return counts, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Folder containing simulate, train, test")
    parser.add_argument("--scenario", choices=("I", "II", "III"), help="Also validate published per-class counts")
    parser.add_argument("--verify-images", action="store_true")
    args = parser.parse_args()
    counts, errors = inspect_data(args.data_root, args.scenario, args.verify_images)
    print("Class\tsimulate\ttrain\ttest")
    for name in CLASS_NAMES:
        print(name + "\t" + "\t".join(str(counts.get(s, {}).get(name, 0)) for s in ("simulate", "train", "test")))
    print("Total\t" + "\t".join(str(sum(counts.get(s, {}).values())) for s in ("simulate", "train", "test")))
    if errors:
        raise SystemExit("\n".join(errors))
    print("Data check passed. Folder counts do not verify depression angles or split provenance.")


if __name__ == "__main__":
    main()
