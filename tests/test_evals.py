from fin_doc_analyst.evals.runner import evaluate, format_table
from fin_doc_analyst.evals.thresholds import THRESHOLDS


def test_eval_suite_meets_thresholds(service) -> None:
    report = evaluate(service)
    assert report["passed"], (
        format_table(report["metrics"])
        + "\n"
        + "\n".join(f"{row['id']}: {row.get('answer', '')[:240]}" for row in report["cases"])
    )
    scores = {row["metric"]: row["score"] for row in report["metrics"]}
    for name, floor in THRESHOLDS.items():
        assert scores[name] >= floor
