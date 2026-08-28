from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

ABSTENTION_LABELS = {"MIXED", "INSUFFICIENT_EVIDENCE", "NON_VERIFIABLE"}
CANONICAL_LABELS = ("SUPPORTED", "REFUTED", "INSUFFICIENT_EVIDENCE", "MIXED")


def classification_accuracy(expected: Iterable[str], predicted: Iterable[str]) -> float:
    expected_list, predicted_list = list(expected), list(predicted)
    if not expected_list:
        return 0.0
    return sum(
        left == right for left, right in zip(expected_list, predicted_list, strict=False)
    ) / len(expected_list)


def macro_f1(expected: Iterable[str], predicted: Iterable[str]) -> float:
    expected_list, predicted_list = list(expected), list(predicted)
    labels = sorted(set(expected_list) | set(predicted_list))
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(
            e == label and p == label for e, p in zip(expected_list, predicted_list, strict=False)
        )
        fp = sum(
            e != label and p == label for e, p in zip(expected_list, predicted_list, strict=False)
        )
        fn = sum(
            e == label and p != label for e, p in zip(expected_list, predicted_list, strict=False)
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def evidence_recall(relevant: Iterable[str], retrieved: Iterable[str]) -> float:
    relevant_set, retrieved_set = set(relevant), set(retrieved)
    return len(relevant_set & retrieved_set) / len(relevant_set) if relevant_set else 0.0


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _labels(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    observed = {str(row["gold_label"]) for row in rows} | {
        str(row["mapped_prediction"]) for row in rows
    }
    return [label for label in CANONICAL_LABELS if label in observed] or list(CANONICAL_LABELS)


def per_class_metrics(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    labels = _labels(rows)
    metrics: dict[str, dict[str, float]] = {}
    for label in labels:
        true_positive = sum(
            row["gold_label"] == label and row["mapped_prediction"] == label for row in rows
        )
        false_positive = sum(
            row["gold_label"] != label and row["mapped_prediction"] == label for row in rows
        )
        false_negative = sum(
            row["gold_label"] == label and row["mapped_prediction"] != label for row in rows
        )
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(row["gold_label"] == label for row in rows),
        }
    return metrics


def classification_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    per_class = per_class_metrics(rows)
    values = list(per_class.values())
    confusion = {
        gold: {
            predicted: sum(
                row["gold_label"] == gold and row["mapped_prediction"] == predicted for row in rows
            )
            for predicted in per_class
        }
        for gold in per_class
    }
    return {
        "accuracy": _safe_divide(sum(bool(row["correct"]) for row in rows), len(rows)),
        "macro_precision": sum(value["precision"] for value in values) / len(values),
        "macro_recall": sum(value["recall"] for value in values) / len(values),
        "macro_f1": sum(value["f1"] for value in values) / len(values),
        "primary_metric": "macro_f1",
        "primary_metric_reason": (
            "Macro F1 is primary because the labeled dev split is imbalanced and the benchmark "
            "must expose performance on each mapped class."
        ),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "sample_size": len(rows),
    }


def abstention_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    answered = [row for row in rows if not row["abstained"]]
    correct_answered = sum(bool(row["correct"]) for row in answered)
    return {
        "coverage": _safe_divide(len(answered), len(rows)),
        "abstention_rate": _safe_divide(len(rows) - len(answered), len(rows)),
        "selective_accuracy": _safe_divide(correct_answered, len(answered)),
        "selective_error_rate": _safe_divide(len(answered) - correct_answered, len(answered)),
        "abstained_count": len(rows) - len(answered),
        "answered_count": len(answered),
        "abstention_labels": sorted(ABSTENTION_LABELS),
    }


def _evidence_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    gold_sentences = {
        evidence_id for row in rows for evidence_id in row.get("gold_evidence_ids", [])
    }
    selected_sentences = {
        evidence_id for row in rows for evidence_id in row.get("selected_evidence_ids", [])
    }
    true_positive = len(gold_sentences & selected_sentences)
    precision = true_positive / len(selected_sentences) if selected_sentences else 1.0
    recall = true_positive / len(gold_sentences) if gold_sentences else 1.0
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    gold_documents = {
        document_id for row in rows for document_id in row.get("gold_document_ids", [])
    }
    retrieved_documents = {
        document_id for row in rows for document_id in row.get("retrieved_document_ids", [])
    }
    document_tp = len(gold_documents & retrieved_documents)
    document_precision = document_tp / len(retrieved_documents) if retrieved_documents else 1.0
    document_recall = document_tp / len(gold_documents) if gold_documents else 1.0
    document_f1 = _safe_divide(
        2 * document_precision * document_recall, document_precision + document_recall
    )

    recall_at_k_values: list[float] = []
    sentence_recall_at_k_values: list[float] = []
    reciprocal_ranks: list[float] = []
    for row in rows:
        gold_doc_ids = set(row.get("gold_document_ids", []))
        retrieved_doc_ids = list(row.get("retrieved_document_ids", []))
        gold_sentence_ids = set(row.get("gold_evidence_ids", []))
        retrieved_sentence_ids = set(row.get("retrieved_sentence_ids", []))
        recall_at_k_values.append(
            len(gold_doc_ids & set(retrieved_doc_ids)) / len(gold_doc_ids) if gold_doc_ids else 1.0
        )
        sentence_recall_at_k_values.append(
            len(gold_sentence_ids & retrieved_sentence_ids) / len(gold_sentence_ids)
            if gold_sentence_ids
            else 1.0
        )
        rank = next(
            (
                index
                for index, document_id in enumerate(retrieved_doc_ids, 1)
                if document_id in gold_doc_ids
            ),
            None,
        )
        reciprocal_ranks.append(1 / rank if rank else 0.0)
    return {
        "evidence_precision": precision,
        "evidence_recall": recall,
        "evidence_f1": f1,
        "document_precision": document_precision,
        "document_recall": document_recall,
        "document_f1": document_f1,
        "document_recall_at_k": _safe_divide(sum(recall_at_k_values), len(recall_at_k_values)),
        "sentence_recall_at_k": _safe_divide(
            sum(sentence_recall_at_k_values), len(sentence_recall_at_k_values)
        ),
        "mrr": _safe_divide(sum(reciprocal_ranks), len(reciprocal_ranks)),
        "gold_evidence_count": len(gold_sentences),
        "selected_evidence_count": len(selected_sentences),
    }


def calibration_metrics(
    rows: list[Mapping[str, Any]], minimum_sample_size: int = 20
) -> dict[str, Any]:
    if len(rows) < minimum_sample_size:
        return {
            "status": "CALIBRATION_SAMPLE_TOO_SMALL",
            "sample_size": len(rows),
            "minimum_sample_size": minimum_sample_size,
            "brier_score": None,
            "ece": None,
            "bins": [],
        }
    brier = sum(
        (float(row["confidence"]) - float(bool(row["correct"]))) ** 2 for row in rows
    ) / len(rows)
    bins: list[dict[str, float | int]] = []
    weighted_gap = 0.0
    for bin_index in range(10):
        lower = bin_index / 10
        upper = (bin_index + 1) / 10
        members = [
            row
            for row in rows
            if lower <= float(row["confidence"]) < upper
            or (bin_index == 9 and float(row["confidence"]) == 1.0)
        ]
        if members:
            mean_confidence = sum(float(row["confidence"]) for row in members) / len(members)
            mean_accuracy = sum(float(bool(row["correct"])) for row in members) / len(members)
            gap = abs(mean_confidence - mean_accuracy)
            weighted_gap += gap * len(members) / len(rows)
        else:
            mean_confidence = 0.0
            mean_accuracy = 0.0
            gap = 0.0
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "mean_confidence": mean_confidence,
                "accuracy": mean_accuracy,
                "gap": gap,
            }
        )
    return {
        "status": "MEASURED",
        "sample_size": len(rows),
        "minimum_sample_size": minimum_sample_size,
        "brier_score": brier,
        "ece": weighted_gap,
        "bins": bins,
        "confidence_semantics": "confidence in the run verdict given the supplied evidence",
    }


def unsupported_verdict_rate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    non_abstained = [row for row in rows if not row["abstained"]]
    unsupported = [
        row
        for row in non_abstained
        if not row.get("selected_evidence_ids")
        or any(
            evidence_id not in set(row.get("retrieved_sentence_ids", []))
            for evidence_id in row.get("selected_evidence_ids", [])
        )
    ]
    return {
        "unsupported_verdict_rate": _safe_divide(len(unsupported), len(non_abstained)),
        "unsupported_count": len(unsupported),
        "non_abstained_count": len(non_abstained),
        "definition": (
            "non-abstaining verdict without a selected evidence ID from this run's retrieved "
            "sentence set"
        ),
    }


def critic_effect_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    assurance_rows = [
        row
        for row in rows
        if isinstance(row.get("assurance"), Mapping)
        and row["assurance"].get("critic_invoked") is not None
    ]
    invoked = [row for row in assurance_rows if row["assurance"].get("critic_invoked")]
    effects = Counter(str(row["assurance"].get("effect", "NOT_INVOKED")) for row in assurance_rows)
    decisions = Counter(
        str(row["assurance"].get("critic_decision"))
        for row in invoked
        if row["assurance"].get("critic_decision")
    )
    corrected = effects.get("CORRECTED", 0)
    damaged = effects.get("DAMAGED", 0)
    unchanged_correct = effects.get("UNCHANGED_CORRECT", 0)
    unchanged_wrong = effects.get("UNCHANGED_WRONG", 0)
    return {
        "claims_evaluated": len(rows),
        "critic_invocation_count": len(invoked),
        "critic_pass_count": decisions.get("PASS", 0),
        "critic_challenge_count": decisions.get("CHALLENGE", 0),
        "rejudge_count": sum(bool(row["assurance"].get("recheck_performed")) for row in invoked),
        "verdict_changed_count": sum(
            bool(row["assurance"].get("verdict_changed")) for row in invoked
        ),
        "wrong_to_right": corrected,
        "right_to_wrong": damaged,
        "right_to_right": unchanged_correct,
        "wrong_to_wrong": unchanged_wrong,
        "net_critic_corrections": corrected - damaged,
        "critic_correction_rate": _safe_divide(corrected, len(invoked)),
        "critic_damage_rate": _safe_divide(damaged, len(invoked)),
        "effect_counts": dict(sorted(effects.items())),
    }


def auditor_effect_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    auditor_rows = [row for row in rows if isinstance(row.get("auditor_effect"), Mapping)]
    effects = Counter(str(row["auditor_effect"].get("effect", "UNKNOWN")) for row in auditor_rows)
    return {
        "claims_evaluated": len(rows),
        "claims_with_auditor": len(auditor_rows),
        "items_audited": sum(
            int(row["auditor_effect"].get("items_audited", 0)) for row in auditor_rows
        ),
        "items_downgraded": sum(
            int(row["auditor_effect"].get("items_downgraded", 0)) for row in auditor_rows
        ),
        "items_rejected": sum(
            int(row["auditor_effect"].get("items_rejected", 0)) for row in auditor_rows
        ),
        "scope_mismatches": sum(
            int(row["auditor_effect"].get("scope_mismatches", 0)) for row in auditor_rows
        ),
        "quantifier_mismatches": sum(
            int(row["auditor_effect"].get("quantifier_mismatches", 0)) for row in auditor_rows
        ),
        "temporal_mismatches": sum(
            int(row["auditor_effect"].get("temporal_mismatches", 0)) for row in auditor_rows
        ),
        "helpful_filters": effects.get("HELPFUL_FILTER", 0),
        "harmful_filters": effects.get("HARMFUL_FILTER", 0),
        "no_effect": effects.get("NO_EFFECT", 0),
        "effect_counts": dict(sorted(effects.items())),
    }


def compute_metrics(rows: list[Mapping[str, Any]], *, evidence_applicable: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "claim": classification_metrics(rows),
        "abstention": abstention_metrics(rows),
        "calibration": calibration_metrics(rows),
        "unsupported_verdict": unsupported_verdict_rate(rows),
        "critic": critic_effect_metrics(rows),
        "auditor": auditor_effect_metrics(rows),
    }
    result["evidence"] = _evidence_metrics(rows) if evidence_applicable else "N/A"
    return result
