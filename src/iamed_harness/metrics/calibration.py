"""Calibracao implementada a mao: ECE, MCE, Brier e a tabela de bins.

Nada aqui vem de biblioteca de metricas, de proposito. A formula precisa
estar visivel na tela durante a aula, e o valor precisa ser conferivel com
papel e caneta (ver `tests/test_calibration.py`).

Definicoes usadas:

    ECE = soma_b (n_b / N) * |acc_b - conf_b|
    MCE = max_b |acc_b - conf_b|
    Brier = (1/N) * soma_i (conf_i - y_i)^2

onde b percorre apenas os bins NAO VAZIOS. Bin vazio nao entra na media
ponderada: ele tem peso n_b/N = 0 e nao tem `acc_b` nem `conf_b` definidos.
Incluir bins vazios como zero puxaria o ECE artificialmente para baixo, que e
exatamente o erro que este modulo existe para nao cometer.

Predicoes sem confianca declarada (parse falhou, ou o modelo omitiu o campo)
NAO entram no calculo, porque nao ha confianca para comparar. O numero delas e
devolvido em `n_sem_confianca` e aparece no relatorio, para que a exclusao
nunca seja silenciosa.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..schemas import BinStats, CalibrationMetrics


def _bins_uniformes(n_bins: int) -> list[float]:
    return [i / n_bins for i in range(n_bins + 1)]


def _bins_por_quantil(confiancas: np.ndarray, n_bins: int) -> list[float]:
    """Bordas por quantil, com duplicatas removidas.

    Quando a confianca e concentrada (o modelo responde 90 em quase tudo), as
    bordas colapsam e sobram menos bins que o pedido. Isso nao e um bug: e o
    sintoma de que o binning uniforme estava distribuindo mal a massa.
    """
    quantis = np.linspace(0.0, 1.0, n_bins + 1)
    bordas = np.quantile(confiancas, quantis)
    unicas = np.unique(np.round(bordas, 12))
    if len(unicas) < 2:
        centro = float(unicas[0])
        return [max(0.0, centro - 1e-9), min(1.0, centro + 1e-9)]
    return [float(x) for x in unicas]


def _tabela_de_bins(
    confiancas: np.ndarray, acertos: np.ndarray, bordas: list[float]
) -> list[BinStats]:
    tabela: list[BinStats] = []
    n_bins = len(bordas) - 1
    for i in range(n_bins):
        low, high = bordas[i], bordas[i + 1]
        if i == n_bins - 1:
            mascara = (confiancas >= low) & (confiancas <= high)
        else:
            mascara = (confiancas >= low) & (confiancas < high)
        n = int(mascara.sum())
        tabela.append(
            BinStats(
                lower=low,
                upper=high,
                n=n,
                mean_confidence=float(confiancas[mascara].mean()) if n else 0.0,
                mean_accuracy=float(acertos[mascara].mean()) if n else 0.0,
            )
        )
    return tabela


def expected_calibration_error(tabela: list[BinStats]) -> float:
    """ECE a partir da tabela de bins. Bins vazios sao ignorados."""
    total = sum(b.n for b in tabela)
    if total == 0:
        return 0.0
    return sum((b.n / total) * abs(b.mean_accuracy - b.mean_confidence) for b in tabela if b.n > 0)


def maximum_calibration_error(tabela: list[BinStats]) -> float:
    """MCE: o pior bin nao vazio. Nao e ponderado por tamanho."""
    gaps = [abs(b.mean_accuracy - b.mean_confidence) for b in tabela if b.n > 0]
    return max(gaps) if gaps else 0.0


def brier_score(confiancas: np.ndarray, acertos: np.ndarray) -> float:
    """Erro quadratico medio entre confianca e desfecho binario."""
    if len(confiancas) == 0:
        return 0.0
    return float(np.mean((confiancas - acertos) ** 2))


def compute_calibration(
    confiancas: Sequence[float | None],
    acertos: Sequence[bool | None],
    *,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Calcula todas as metricas de calibracao de um conjunto de predicoes.

    Uma predicao so e pontuada quando tem confianca E desfecho conhecidos.
    Predicoes que falharam entram em `n_sem_confianca`, nunca desaparecem.
    """
    if len(confiancas) != len(acertos):
        raise ValueError("confiancas e acertos precisam ter o mesmo tamanho")

    pares = [
        (c, a) for c, a in zip(confiancas, acertos, strict=True) if c is not None and a is not None
    ]
    n_sem = len(confiancas) - len(pares)

    if not pares:
        return CalibrationMetrics(
            n_scored=0,
            n_sem_confianca=n_sem,
            ece_uniform=0.0,
            ece_quantile=0.0,
            mce=0.0,
            brier=0.0,
            bins_uniform=[],
            bins_quantile=[],
        )

    conf = np.array([p[0] for p in pares], dtype=float)
    acc = np.array([1.0 if p[1] else 0.0 for p in pares], dtype=float)

    bins_u = _tabela_de_bins(conf, acc, _bins_uniformes(n_bins))
    bins_q = _tabela_de_bins(conf, acc, _bins_por_quantil(conf, n_bins))

    return CalibrationMetrics(
        n_scored=len(pares),
        n_sem_confianca=n_sem,
        ece_uniform=expected_calibration_error(bins_u),
        ece_quantile=expected_calibration_error(bins_q),
        mce=maximum_calibration_error(bins_u),
        brier=brier_score(conf, acc),
        bins_uniform=bins_u,
        bins_quantile=bins_q,
    )
