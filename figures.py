"""Generate every paper figure.

Each figure can also be generated independently with, for example:
    python figures/figure_1.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    scripts = Path(__file__).with_name("figures")
    for number in range(1, 10):
        subprocess.run(
            [sys.executable, str(scripts / f"figure_{number}.py")], check=True
        )


if __name__ == "__main__":
    main()
