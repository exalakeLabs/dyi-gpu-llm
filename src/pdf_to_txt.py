#!/usr/bin/env python
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    runpy.run_module("utils.pdf_to_txt", run_name="__main__")
else:
    from utils.pdf_to_txt import *  # noqa: F401,F403,E402
