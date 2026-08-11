"""Provedor deterministico para testes.

Responde sempre a primeira letra que encontra na secao de alternativas do
prompt, com confianca fixa. Nao acerta quase nada e nao deveria: ele existe
para exercitar o pipeline sem rede e sem fixture.
"""

from __future__ import annotations

import json
import re

PADRAO_ALTERNATIVA = re.compile(r"^\(?([A-Z])\)?[\).:]\s", re.MULTILINE)


class EchoProvider:
    name = "echo"

    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        variant: int,
    ) -> str:
        letras = PADRAO_ALTERNATIVA.findall(prompt)
        escolha = letras[variant % len(letras)] if letras else "A"
        return json.dumps(
            {
                "answer": escolha,
                "confidence": 50,
                "reasoning": "provedor echo, resposta deterministica sem raciocinio clinico",
            },
            ensure_ascii=False,
        )
