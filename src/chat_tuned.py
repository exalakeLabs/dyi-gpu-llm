#!/usr/bin/env python
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    runpy.run_module("inference.chat_tuned", run_name="__main__")
else:
    from inference.chat_tuned import *  # noqa: F401,F403,E402
