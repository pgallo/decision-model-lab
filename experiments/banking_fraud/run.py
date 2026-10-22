"""Run the banking_fraud experiment and print a short report: python experiments/banking_fraud/run.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from decision_lab.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["run", "banking_fraud", *sys.argv[1:]]))
