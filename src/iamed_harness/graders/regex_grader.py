"""Grader por expressao regular, para tarefas cujo formato de saida e livre."""

from __future__ import annotations

import re

from ..schemas import ABSTAIN, Item
from . import GradeOutcome
from .exact_choice import _normalizar_confianca, extrair_payload


class RegexGrader:
    name = "regex"

    def __init__(self, pattern: str) -> None:
        self.pattern = re.compile(pattern)

    async def grade(self, item: Item, raw_response: str) -> GradeOutcome:
        payload = extrair_payload(raw_response)
        confianca = _normalizar_confianca(payload.get("confidence")) if payload else None

        m = self.pattern.search(raw_response)
        if m is None:
            return GradeOutcome(
                parsed_choice=None,
                confidence=confianca,
                correct=None,
                error=f"padrao {self.pattern.pattern!r} nao encontrado na resposta",
            )
        capturado = (m.group(1) if m.groups() else m.group(0)).strip().upper()
        if capturado not in item.choices and capturado != ABSTAIN:
            return GradeOutcome(
                parsed_choice=None,
                confidence=confianca,
                correct=None,
                error=f"captura {capturado!r} nao e uma alternativa valida",
            )
        correto = capturado == ABSTAIN if item.expected_abstain else capturado == item.answer
        return GradeOutcome(
            parsed_choice=capturado, confidence=confianca, correct=correto, error=None
        )
