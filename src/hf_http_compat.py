from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.hf_http_compat import *  # noqa: F401,F403,E402
