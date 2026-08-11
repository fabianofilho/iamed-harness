"""Leitura e validacao de datasets JSONL.

Falha na primeira linha invalida, com o numero da linha. Um dataset meio
carregado e pior que nenhum: esconde de onde veio a diferenca de acuracia.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from .schemas import Item


class DatasetError(Exception):
    """Erro de leitura ou validacao de dataset, sempre com localizacao."""


def load_items(path: str | Path) -> list[Item]:
    caminho = Path(path)
    if not caminho.exists():
        raise DatasetError(f"dataset nao encontrado: {caminho}")

    items: list[Item] = []
    vistos: set[str] = set()

    with caminho.open(encoding="utf-8") as f:
        for numero, linha in enumerate(f, start=1):
            texto = linha.strip()
            if not texto or texto.startswith("//"):
                continue
            try:
                bruto = json.loads(texto)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{caminho}:{numero} JSON invalido: {exc.msg}") from exc
            try:
                item = Item.model_validate(bruto)
            except ValidationError as exc:
                detalhes = "; ".join(
                    f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
                )
                raise DatasetError(f"{caminho}:{numero} item invalido: {detalhes}") from exc
            if item.id in vistos:
                raise DatasetError(f"{caminho}:{numero} id duplicado: {item.id!r}")
            vistos.add(item.id)
            items.append(item)

    if not items:
        raise DatasetError(f"{caminho}: dataset vazio")
    return items


def write_items(items: list[Item], path: str | Path) -> Path:
    caminho = Path(path)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json(exclude_defaults=False) + "\n")
    return caminho


def sha256_arquivo(path: str | Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()
