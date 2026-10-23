"""Run portable inference on synthetic events; no training or accuracy scoring.

This example can be executed from any working directory:
    python /path/to/hba1c_prediction_models/examples/load_model.py
"""
from pathlib import Path
import sys


BUNDLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE_DIR))

from predict import load_model, predict, read_events


if __name__ == "__main__":
    events = read_events(BUNDLE_DIR / "examples" / "example_events.json")
    loaded = load_model("transformer", seed=42, bundle_dir=BUNDLE_DIR)
    predictions = predict(loaded, events)
    print("Synthetic events only — not experimental results.")
    print(predictions["events"].to_string(index=False))
    print(predictions["patients"].to_string(index=False))
