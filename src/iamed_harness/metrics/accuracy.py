"""Acuracia com intervalo de confianca por bootstrap.

Duas acuracias sao reportadas de proposito:

    accuracy              falha conta como erro (denominador = todas)
    accuracy_parsed_only  falha e descartada  (denominador = so as parseadas)

A segunda existe unicamente para mostrar o tamanho da mentira. Se ela for bem
maior que a primeira, o modelo esta ganhando pontos por sumir do denominador.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..schemas import AccuracyMetrics, Prediction


def bootstrap_ci(
    acertos: Sequence[bool], *, n_amostras: int = 2000, seed: int = 20260811, alpha: float = 0.05
) -> tuple[float, float]:
    """IC percentil por reamostragem com reposicao. Seed fixa, resultado reproduzivel."""
    if not acertos:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    dados = np.array([1.0 if a else 0.0 for a in acertos])
    indices = rng.integers(0, len(dados), size=(n_amostras, len(dados)))
    medias = dados[indices].mean(axis=1)
    low = float(np.quantile(medias, alpha / 2))
    high = float(np.quantile(medias, 1 - alpha / 2))
    return (low, high)


def compute_accuracy(predictions: Sequence[Prediction], *, seed: int = 20260811) -> AccuracyMetrics:
    n_total = len(predictions)
    falhas = [p for p in predictions if p.failed]
    parseadas = [p for p in predictions if not p.failed]
    n_correct = sum(1 for p in parseadas if p.correct)

    acertos_honestos = [bool(p.correct) for p in predictions]
    low, high = bootstrap_ci(acertos_honestos, seed=seed)

    return AccuracyMetrics(
        n_total=n_total,
        n_correct=n_correct,
        n_failed=len(falhas),
        accuracy=n_correct / n_total if n_total else 0.0,
        accuracy_parsed_only=n_correct / len(parseadas) if parseadas else 0.0,
        ci_low=low,
        ci_high=high,
    )
