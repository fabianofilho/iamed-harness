"""Quebra de acuracia por qualquer campo de metadata do item.

Media global esconde subgrupo ruim. Se o dataset tem `especialidade` ou
`dificuldade` no metadata, a quebra e uma linha de config, nao um script novo.
"""

from __future__ import annotations

from ..schemas import Item, Prediction, SubgroupRow


def compute_subgroups(
    predictions: list[Prediction], items: dict[str, Item], fields: list[str]
) -> list[SubgroupRow]:
    linhas: list[SubgroupRow] = []
    for campo in fields:
        buckets: dict[str, list[Prediction]] = {}
        for p in predictions:
            item = items.get(p.item_id)
            if item is None:
                continue
            valor = item.metadata.get(campo)
            if valor is None:
                continue
            buckets.setdefault(str(valor), []).append(p)
        for valor, preds in sorted(buckets.items()):
            n_correct = sum(1 for p in preds if p.correct)
            n_failed = sum(1 for p in preds if p.failed)
            linhas.append(
                SubgroupRow(
                    field=campo,
                    value=valor,
                    n=len(preds),
                    n_correct=n_correct,
                    n_failed=n_failed,
                    accuracy=n_correct / len(preds) if preds else 0.0,
                )
            )
    return linhas
