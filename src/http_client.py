from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.http_client import *  # noqa: F401,F403,E402
