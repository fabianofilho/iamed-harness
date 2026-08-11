"""Calibracao conferida contra valores calculados a mao.

O caso base tem 10 predicoes distribuidas em tres bins, e todos os numeros
abaixo foram obtidos com papel e caneta antes de rodar o codigo. Se este
teste quebrar, o suspeito e a implementacao, nao o valor esperado.

    bin [0.9, 1.0]  n=4  conf=0.95  acc=0.75  gap=0.20
    bin [0.8, 0.9)  n=4  conf=0.85  acc=0.50  gap=0.35
    bin [0.5, 0.6)  n=2  conf=0.55  acc=0.50  gap=0.05

    ECE = (4/10)(0.20) + (4/10)(0.35) + (2/10)(0.05)
        = 0.080 + 0.140 + 0.010
        = 0.230

    MCE = max(0.20, 0.35, 0.05) = 0.350

    Brier = [3(0.95-1)^2 + 1(0.95-0)^2
           + 2(0.85-1)^2 + 2(0.85-0)^2
           + 1(0.55-1)^2 + 1(0.55-0)^2] / 10
          = [0.0075 + 0.9025 + 0.0450 + 1.4450 + 0.2025 + 0.3025] / 10
          = 2.9050 / 10
          = 0.2905
"""

from __future__ import annotations

import pytest

from iamed_harness.metrics.calibration import (
    brier_score,
    compute_calibration,
    expected_calibration_error,
)

CONFIANCAS = [0.95, 0.95, 0.95, 0.95, 0.85, 0.85, 0.85, 0.85, 0.55, 0.55]
ACERTOS = [True, True, True, False, True, True, False, False, True, False]


def test_ece_uniforme_bate_com_calculo_manual():
    m = compute_calibration(CONFIANCAS, ACERTOS, n_bins=10)
    assert m.ece_uniform == pytest.approx(0.230, abs=1e-9)


def test_mce_bate_com_calculo_manual():
    m = compute_calibration(CONFIANCAS, ACERTOS, n_bins=10)
    assert m.mce == pytest.approx(0.350, abs=1e-9)


def test_brier_bate_com_calculo_manual():
    m = compute_calibration(CONFIANCAS, ACERTOS, n_bins=10)
    assert m.brier == pytest.approx(0.2905, abs=1e-9)


def test_tabela_de_bins_reproduz_os_tres_bins_ocupados():
    m = compute_calibration(CONFIANCAS, ACERTOS, n_bins=10)
    ocupados = [b for b in m.bins_uniform if b.n > 0]
    assert len(ocupados) == 3
    assert [b.n for b in ocupados] == [2, 4, 4]

    bin_55, bin_85, bin_95 = ocupados
    assert bin_55.mean_confidence == pytest.approx(0.55)
    assert bin_55.mean_accuracy == pytest.approx(0.50)
    assert bin_85.mean_confidence == pytest.approx(0.85)
    assert bin_85.mean_accuracy == pytest.approx(0.50)
    assert bin_95.mean_confidence == pytest.approx(0.95)
    assert bin_95.mean_accuracy == pytest.approx(0.75)


def test_bins_vazios_nao_entram_na_media_ponderada():
    """Sete dos dez bins estao vazios e nao podem puxar o ECE para baixo."""
    m = compute_calibration(CONFIANCAS, ACERTOS, n_bins=10)
    assert sum(1 for b in m.bins_uniform if b.n == 0) == 7

    so_ocupados = [b for b in m.bins_uniform if b.n > 0]
    assert expected_calibration_error(so_ocupados) == pytest.approx(m.ece_uniform)


def test_predicoes_sem_confianca_sao_contadas_e_nao_pontuadas():
    confs = [*CONFIANCAS, None, None]
    acertos = [*ACERTOS, None, False]
    m = compute_calibration(confs, acertos, n_bins=10)

    assert m.n_scored == 10
    assert m.n_sem_confianca == 2
    assert m.ece_uniform == pytest.approx(0.230, abs=1e-9)


def test_modelo_perfeitamente_calibrado_tem_ece_zero():
    confs = [1.0, 1.0, 0.0, 0.0]
    acertos = [True, True, False, False]
    m = compute_calibration(confs, acertos, n_bins=10)
    assert m.ece_uniform == pytest.approx(0.0)
    assert m.brier == pytest.approx(0.0)


def test_confianca_maxima_e_sempre_errado_tem_ece_um():
    m = compute_calibration([1.0] * 5, [False] * 5, n_bins=10)
    assert m.ece_uniform == pytest.approx(1.0)
    assert m.mce == pytest.approx(1.0)
    assert m.brier == pytest.approx(1.0)


def test_ece_por_quantil_difere_do_uniforme_quando_a_massa_e_desbalanceada():
    confs = [0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.10, 0.20]
    acertos = [True, True, True, True, False, False, False, False, True, True]
    m = compute_calibration(confs, acertos, n_bins=5)

    assert m.ece_uniform != pytest.approx(m.ece_quantile)
    assert 0.0 <= m.ece_quantile <= 1.0


def test_conjunto_vazio_nao_explode():
    m = compute_calibration([], [], n_bins=10)
    assert m.n_scored == 0
    assert m.ece_uniform == 0.0
    assert m.bins_uniform == []


def test_tamanhos_diferentes_sao_rejeitados():
    with pytest.raises(ValueError, match="mesmo tamanho"):
        compute_calibration([0.5], [True, False])


def test_brier_direto_com_arrays():
    import numpy as np

    conf = np.array([0.8, 0.2])
    acc = np.array([1.0, 0.0])
    assert brier_score(conf, acc) == pytest.approx((0.04 + 0.04) / 2)
