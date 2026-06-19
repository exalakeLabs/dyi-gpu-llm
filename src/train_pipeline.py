#!/usr/bin/env python3
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    runpy.run_module("training.train_pipeline", run_name="__main__")
else:
    from training.train_pipeline import *  # noqa: F401,F403,E402
