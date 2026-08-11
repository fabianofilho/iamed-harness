"""Variancia entre execucoes identicas.

Mesmo prompt, mesmo modelo, mesma temperatura, K vezes. Se a resposta muda, a
acuracia de um run unico e uma amostra de uma distribuicao, nao um numero.

    flip_rate      proporcao de itens com mais de uma resposta distinta em K
    agreement      por item, fracao das repeticoes que caiu na resposta modal
    accuracy_std   desvio-padrao da acuracia entre as K repeticoes
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..schemas import Prediction, RepetitionAccuracy, VarianceMetrics

SEM_RESPOSTA = "<falha>"


def _rotulo(p: Prediction) -> str:
    return p.parsed_choice if p.parsed_choice is not None else SEM_RESPOSTA


def compute_variance(predictions: list[Prediction]) -> VarianceMetrics | None:
    """Devolve None quando ha uma unica repeticao (nao ha o que medir)."""
    repeticoes = sorted({p.repetition for p in predictions})
    if len(repeticoes) < 2:
        return None

    por_item: dict[str, list[Prediction]] = {}
    for p in predictions:
        por_item.setdefault(p.item_id, []).append(p)

    instaveis: list[str] = []
    concordancias: list[float] = []
    for item_id, preds in sorted(por_item.items()):
        rotulos = [_rotulo(p) for p in preds]
        contagem = Counter(rotulos)
        _, n_modal = contagem.most_common(1)[0]
        concordancias.append(n_modal / len(rotulos))
        if len(contagem) > 1:
            instaveis.append(item_id)

    por_repeticao: list[RepetitionAccuracy] = []
    for r in repeticoes:
        preds = [p for p in predictions if p.repetition == r]
        acertos = sum(1 for p in preds if p.correct)
        por_repeticao.append(
            RepetitionAccuracy(
                repetition=r,
                accuracy=acertos / len(preds) if preds else 0.0,
                n=len(preds),
            )
        )

    accs = np.array([r.accuracy for r in por_repeticao], dtype=float)

    return VarianceMetrics(
        repetitions=len(repeticoes),
        n_items=len(por_item),
        flip_rate=len(instaveis) / len(por_item) if por_item else 0.0,
        mean_agreement=float(np.mean(concordancias)) if concordancias else 0.0,
        accuracy_por_repeticao=por_repeticao,
        accuracy_std=float(np.std(accs, ddof=0)),
        itens_instaveis=instaveis,
    )


def resposta_modal(predictions: list[Prediction]) -> str:
    """Resposta mais frequente de um item entre as repeticoes."""
    if not predictions:
        raise ValueError("lista de predicoes vazia")
    return Counter(_rotulo(p) for p in predictions).most_common(1)[0][0]
