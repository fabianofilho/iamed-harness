"""Acuracia com intervalo de confianca de Wilson.

Duas acuracias sao reportadas de proposito:

    accuracy              falha conta como erro (denominador = todas)
    accuracy_parsed_only  falha e descartada  (denominador = so as parseadas)

A segunda existe unicamente para mostrar o tamanho da mentira. Se ela for bem
maior que a primeira, o modelo esta ganhando pontos por sumir do denominador.

Por que Wilson e nao bootstrap: com n pequeno e acuracia no extremo, o
bootstrap percentil colapsa. Com 25/25 toda reamostragem da 1.0 e o intervalo
vira "100% a 100%", certeza que 25 questoes nao sustentam. Wilson da 86.7% a
100% no mesmo caso, e ainda e fechado e deterministico, sem seed.

Limite conhecido: o intervalo trata cada predicao como independente. Em tarefa
com repeticoes, K execucoes do mesmo item nao sao K observacoes novas, entao o
intervalo sai mais estreito do que deveria.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..schemas import AccuracyMetrics, Prediction

Z_95 = 1.959963984540054


def wilson_ci(n_acertos: int, n_total: int, *, z: float = Z_95) -> tuple[float, float]:
    """IC de Wilson (score) para uma proporcao binomial."""
    if n_total <= 0:
        return (0.0, 0.0)
    if not 0 <= n_acertos <= n_total:
        raise ValueError(f"n_acertos={n_acertos} fora de 0..{n_total}")
    p = n_acertos / n_total
    z2 = z * z
    denominador = 1 + z2 / n_total
    centro = (p + z2 / (2 * n_total)) / denominador
    meia = z * math.sqrt(p * (1 - p) / n_total + z2 / (4 * n_total * n_total)) / denominador
    # Nos extremos o limite e exato (0 ou 1); a conta em float deixaria 1e-17.
    low = 0.0 if n_acertos == 0 else max(0.0, centro - meia)
    high = 1.0 if n_acertos == n_total else min(1.0, centro + meia)
    return (low, high)


def compute_accuracy(predictions: Sequence[Prediction]) -> AccuracyMetrics:
    n_total = len(predictions)
    falhas = [p for p in predictions if p.failed]
    parseadas = [p for p in predictions if not p.failed]
    n_correct = sum(1 for p in parseadas if p.correct)

    low, high = wilson_ci(n_correct, n_total)

    return AccuracyMetrics(
        n_total=n_total,
        n_correct=n_correct,
        n_failed=len(falhas),
        accuracy=n_correct / n_total if n_total else 0.0,
        accuracy_parsed_only=n_correct / len(parseadas) if parseadas else 0.0,
        ci_low=low,
        ci_high=high,
    )
