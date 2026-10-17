"""CLI entry so `python scripts/run_eval.py` matches the README."""

from __future__ import annotations

from fin_doc_analyst.evals.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
