#!/usr/bin/env python
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    runpy.run_module("training.continued_pretrain_partial", run_name="__main__")
else:
    from training.continued_pretrain_partial import *  # noqa: F401,F403,E402
