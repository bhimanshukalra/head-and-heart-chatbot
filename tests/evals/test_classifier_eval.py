"""Live classifier evaluation. Opt in with `pytest -m eval` (needs GROQ_API_KEY).

Runs the real classifier node, including the confidence threshold, against the labeled set and
prints accuracy, per-class precision/recall/F1 and a confusion matrix.
"""

import os
from collections import Counter
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from head_and_heart.config import Settings
from head_and_heart.llm import get_chat_model
from head_and_heart.nodes.classifier import make_classifier
from tests.evals.test_datasets import CLASSIFIER

pytestmark = pytest.mark.eval

LABELS = ["emotional", "logical", "mixed"]
MIN_ACCURACY = float(os.environ.get("EVAL_MIN_ACCURACY", "0.9"))


def prf(confusion: Counter[tuple[str, str]], label: str) -> tuple[float, float, float]:
    tp = confusion[(label, label)]
    fp = sum(confusion[(other, label)] for other in LABELS if other != label)
    fn = sum(confusion[(label, other)] for other in LABELS if other != label)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def test_classifier_accuracy(capsys: pytest.CaptureFixture[str]) -> None:
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY is not set")
    settings = Settings(_env_file=None)
    llm = get_chat_model(settings, model_name=settings.classifier_model_name, temperature=0)
    classify = make_classifier(llm, settings.classifier_confidence_threshold)

    confusion: Counter[tuple[str, str]] = Counter()
    errors: list[str] = []
    for row in CLASSIFIER:
        messages: list[Any] = [
            HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
            for m in row.get("context", [])
        ]
        messages.append(HumanMessage(row["text"]))
        predicted = classify({"messages": messages})["message_type"]
        confusion[(row["label"], predicted)] += 1
        if predicted != row["label"]:
            errors.append(f"[{row['label']} -> {predicted}] {row['text'][:80]}")

    total = len(CLASSIFIER)
    accuracy = sum(confusion[(label, label)] for label in LABELS) / total
    lines = [f"accuracy={accuracy:.3f} (n={total})"]
    f1s = []
    for label in LABELS:
        p, r, f1 = prf(confusion, label)
        f1s.append(f1)
        lines.append(f"{label:10s} precision={p:.2f} recall={r:.2f} f1={f1:.2f}")
    lines.append(f"macro_f1={sum(f1s) / len(f1s):.3f}")
    lines.append("confusion (rows=true, cols=predicted): " + ", ".join(LABELS))
    for true in LABELS:
        lines.append(f"  {true:10s} " + " ".join(f"{confusion[(true, p)]:3d}" for p in LABELS))
    lines.extend(errors)
    with capsys.disabled():
        print("\n" + "\n".join(lines))

    assert accuracy >= MIN_ACCURACY, f"accuracy {accuracy:.3f} below {MIN_ACCURACY}"
