from __future__ import annotations

import math

import pytest

from iamed_harness.metrics.accuracy import Z_95, compute_accuracy, wilson_ci
from iamed_harness.schemas import Prediction


def _raizes_do_teste_score(x: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Limites de Wilson por outro caminho: raizes de
    (n + z^2) p^2 - (2x + z^2) p + x^2 / n = 0."""
    a = n + z * z
    b = -(2 * x + z * z)
    c = x * x / n
    delta = math.sqrt(b * b - 4 * a * c)
    return ((-b - delta) / (2 * a), (-b + delta) / (2 * a))


@pytest.mark.parametrize(("x", "n"), [(8, 10), (0, 25), (25, 25), (13, 25), (104, 125), (1, 3)])
def test_wilson_bate_com_as_raizes_da_quadratica(x, n):
    low, high = wilson_ci(x, n)
    esperado = _raizes_do_teste_score(x, n)
    assert low == pytest.approx(esperado[0], abs=1e-12)
    assert high == pytest.approx(esperado[1], abs=1e-12)


def test_wilson_bate_com_valor_de_livro():
    # 8/10: 0.490 a 0.943, exemplo classico de Newcombe (1998)
    low, high = wilson_ci(8, 10)
    assert round(low, 3) == 0.490
    assert round(high, 3) == 0.943


def test_acerto_total_nao_vira_certeza():
    # O bootstrap percentil devolvia 1.0 a 1.0 aqui
    low, high = wilson_ci(25, 25)
    assert round(low, 3) == 0.867
    assert high == 1.0


def test_erro_total_tem_limite_superior_positivo():
    low, high = wilson_ci(0, 25)
    assert low == 0.0
    assert round(high, 3) == 0.133


def test_intervalo_contem_a_proporcao_observada():
    for x in range(0, 26):
        low, high = wilson_ci(x, 25)
        assert low <= x / 25 <= high


def test_entradas_invalidas():
    assert wilson_ci(0, 0) == (0.0, 0.0)
    with pytest.raises(ValueError):
        wilson_ci(5, 4)


def _pred(i: int, correct: bool | None, error: str | None = None) -> Prediction:
    return Prediction(
        item_id=f"i{i}",
        run_id="r",
        repetition=0,
        raw_response="{}",
        parsed_choice="A" if correct is not None else None,
        confidence=None,
        correct=correct,
        latency_ms=1.0,
        error=error,
    )


def test_compute_accuracy_usa_falha_no_denominador_do_intervalo():
    predicoes = [_pred(i, True) for i in range(8)] + [
        _pred(8, None, "resposta vazia"),
        _pred(9, None, "resposta vazia"),
    ]
    m = compute_accuracy(predicoes)
    assert m.n_correct == 8 and m.n_total == 10 and m.n_failed == 2
    assert (m.ci_low, m.ci_high) == wilson_ci(8, 10)
