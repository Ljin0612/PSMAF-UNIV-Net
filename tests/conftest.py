"""Test configuration that supports invoking the ``pytest`` console script.

Some environments do not place the current working directory on ``sys.path``
when the entry-point script is used.  Add the repository root without importing
the package (and therefore without requiring optional runtime dependencies such
as PyTorch during collection).
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
