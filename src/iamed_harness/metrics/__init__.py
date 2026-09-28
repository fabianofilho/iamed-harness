"""Agregacao de predicoes em metricas."""

from __future__ import annotations

from collections import Counter

from ..schemas import Item, Prediction, RunConfig, RunResult
from .accuracy import compute_accuracy, wilson_ci
from .calibration import (
    brier_score,
    compute_calibration,
    expected_calibration_error,
    maximum_calibration_error,
)
from .subgroup import compute_subgroups
from .variance import compute_variance, resposta_modal


def _classificar_erro(mensagem: str) -> str:
    texto = mensagem.lower()
    if "rate limit" in texto or "conexao" in texto or "http" in texto:
        return "rede"
    if "juiz" in texto:
        return "juiz"
    if "vazia" in texto:
        return "resposta vazia"
    if "fallback" in texto:
        return "parse degradado (regex de fallback)"
    return "parse"


def summarize(
    config: RunConfig, predictions: list[Prediction], items: dict[str, Item]
) -> RunResult:
    erros = Counter(_classificar_erro(p.error) for p in predictions if p.error)
    return RunResult(
        config=config,
        accuracy=compute_accuracy(predictions),
        calibration=compute_calibration(
            [p.confidence for p in predictions], [p.correct for p in predictions]
        ),
        variance=compute_variance(predictions),
        subgroups=compute_subgroups(predictions, items, config.subgroup_fields),
        n_predictions=len(predictions),
        erros_por_tipo=dict(sorted(erros.items())),
    )


__all__ = [
    "brier_score",
    "compute_accuracy",
    "compute_calibration",
    "compute_subgroups",
    "compute_variance",
    "expected_calibration_error",
    "maximum_calibration_error",
    "resposta_modal",
    "summarize",
    "wilson_ci",
]
