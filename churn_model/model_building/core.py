import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent.parent, file.parents[2]

PACKAGE_ROOT = root