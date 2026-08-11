"""Cache em disco de respostas do modelo.

A chave inclui a repeticao porque, com temperatura > 0, duas repeticoes do
mesmo prompt sao chamadas diferentes de proposito. Cachear as duas juntas
zeraria a variancia, que e justamente o que o harness quer medir.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

CACHE_DIR_PADRAO = Path(os.environ.get("HARNESS_CACHE_DIR", ".cache/harness"))


def chave_cache(
    *,
    provider: str,
    model: str,
    prompt: str,
    temperature: float,
    seed: int,
    repetition: int,
) -> str:
    material = json.dumps(
        {
            "provider": provider,
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "seed": seed,
            "repetition": repetition,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, diretorio: str | Path | None = None, *, enabled: bool = True) -> None:
        self.diretorio = Path(diretorio) if diretorio is not None else CACHE_DIR_PADRAO
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        if self.enabled:
            self.diretorio.mkdir(parents=True, exist_ok=True)

    def _caminho(self, chave: str) -> Path:
        return self.diretorio / f"{chave}.json"

    def get(self, chave: str) -> str | None:
        if not self.enabled:
            return None
        caminho = self._caminho(chave)
        if not caminho.exists():
            self.misses += 1
            return None
        conteudo = json.loads(caminho.read_text(encoding="utf-8"))
        self.hits += 1
        texto: str = conteudo["response"]
        return texto

    def set(self, chave: str, resposta: str) -> None:
        if not self.enabled:
            return
        self._caminho(chave).write_text(
            json.dumps({"response": resposta}, ensure_ascii=False),
            encoding="utf-8",
        )
