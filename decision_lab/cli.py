"""Command line entry point: ``decision-lab run <domain>|all``."""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

DOMAINS = {
    "life_sciences": ("experiments.life_sciences.experiment", 7),
    "insurance": ("experiments.insurance.experiment", 11),
    "banking_fraud": ("experiments.banking_fraud.experiment", 23),
    "intelligence": ("experiments.intelligence.experiment", 5),
}


def run_domain(domain: str, out_dir: Path, seed: int | None = None) -> Path:
    module_name, default_seed = DOMAINS[domain]
    module = importlib.import_module(module_name)
    used_seed = default_seed if seed is None else seed
    started = time.perf_counter()
    report = module.run(seed=used_seed)
    elapsed = time.perf_counter() - started
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{domain}.md"
    path.write_text(report.to_markdown(used_seed))
    print(report.to_console())
    print(f"  -> {path} ({elapsed:.1f}s)\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="decision-lab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a domain experiment and write a Markdown report")
    run.add_argument("domain", choices=[*DOMAINS, "all"])
    run.add_argument("--out", type=Path, default=Path("reports"), help="report directory")
    run.add_argument("--seed", type=int, default=None, help="override the domain's default seed")
    sub.add_parser("list", help="list available domains")
    args = parser.parse_args(argv)

    if args.command == "list":
        for name in DOMAINS:
            print(name)
        return 0
    domains = list(DOMAINS) if args.domain == "all" else [args.domain]
    for domain in domains:
        run_domain(domain, args.out, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
