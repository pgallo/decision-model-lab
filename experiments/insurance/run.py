"""Run the insurance experiment and print a short report: python experiments/insurance/run.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from decision_lab.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["run", "insurance", *sys.argv[1:]]))
