"""Provedor offline: le respostas gravadas de `fixtures/*.jsonl`.

Zero chamadas de rede, nem para checar credencial. Se o prompt nao estiver
gravado, o erro diz qual comando regrava os fixtures, porque durante uma
sessao ao vivo ninguem tem tempo de adivinhar.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import ProviderError

COMANDO_REGRAVAR = "uv run python scripts/record_fixtures.py --task <tarefa>"


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class FixtureProvider:
    name = "fixture"

    def __init__(self, fixtures_dir: str | Path = "fixtures") -> None:
        self.fixtures_dir = Path(fixtures_dir)
        self._respostas: dict[str, list[str]] = {}
        self._carregar()

    def _carregar(self) -> None:
        if not self.fixtures_dir.exists():
            raise ProviderError(
                f"diretorio de fixtures nao encontrado: {self.fixtures_dir}. "
                f"Regrave com: {COMANDO_REGRAVAR}"
            )
        arquivos = sorted(self.fixtures_dir.glob("*.jsonl"))
        if not arquivos:
            raise ProviderError(
                f"nenhum fixture em {self.fixtures_dir}. Regrave com: {COMANDO_REGRAVAR}"
            )
        for arquivo in arquivos:
            for numero, linha in enumerate(arquivo.read_text(encoding="utf-8").splitlines(), 1):
                texto = linha.strip()
                if not texto:
                    continue
                try:
                    registro = json.loads(texto)
                except json.JSONDecodeError as exc:
                    raise ProviderError(f"{arquivo}:{numero} fixture invalido: {exc.msg}") from exc
                chave = registro["prompt_sha256"]
                respostas = registro["responses"]
                if not isinstance(respostas, list) or not respostas:
                    raise ProviderError(
                        f"{arquivo}:{numero} campo responses precisa ser lista nao vazia"
                    )
                self._respostas[chave] = [str(r) for r in respostas]

    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        variant: int,
    ) -> str:
        chave = hash_prompt(prompt)
        gravadas = self._respostas.get(chave)
        if gravadas is None:
            raise ProviderError(
                f"prompt sem resposta gravada (sha256={chave[:12]}...). "
                f"O dataset ou o template mudou desde a ultima gravacao. "
                f"Regrave com: {COMANDO_REGRAVAR}"
            )
        if temperature == 0.0:
            return gravadas[0]
        return gravadas[variant % len(gravadas)]
