"""Evaluate and plot the measured test confusion matrix; add --tsne for features."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.test import main

if __name__ == "__main__":
    main(force_plots=True)
