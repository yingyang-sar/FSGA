"""Print package and device versions for an experiment log."""
import importlib.metadata
import platform


def main():
    print("Python:", platform.python_version())
    print("Platform:", platform.platform())
    for package in ("torch", "torchvision", "numpy", "Pillow", "PyWavelets", "tqdm", "PyYAML", "scikit-learn", "matplotlib"):
        try:
            print(f"{package}: {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{package}: not installed")
    try:
        import torch
        print("CUDA runtime:", torch.version.cuda)
        print("CUDA available:", torch.cuda.is_available())
        for idx in range(torch.cuda.device_count()):
            print(f"GPU {idx}: {torch.cuda.get_device_name(idx)}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
