"""Grader que usa o proprio modelo como juiz, com rubrica em Jinja2.

Serve para respostas em texto livre, onde extrair uma letra nao basta. Custa
uma chamada extra por item e herda os vieses do modelo julgado: use com a
mesma desconfianca que o harness aplica ao resto.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Template

from ..providers import Provider, ProviderError
from ..schemas import ABSTAIN, Item
from . import GradeOutcome
from .exact_choice import _normalizar_confianca, _normalizar_letra, extrair_payload


class LLMJudgeGrader:
    name = "llm_judge"

    def __init__(
        self,
        provider: Provider,
        *,
        model: str,
        template_path: str | Path = "prompts/judge_rubric.j2",
    ) -> None:
        self.provider = provider
        self.model = model
        caminho = Path(template_path)
        if not caminho.exists():
            raise FileNotFoundError(f"rubrica do juiz nao encontrada: {caminho}")
        self.template = Template(caminho.read_text(encoding="utf-8"), keep_trailing_newline=True)

    async def grade(self, item: Item, raw_response: str) -> GradeOutcome:
        prompt = self.template.render(item=item, resposta=raw_response, abstain=ABSTAIN)
        try:
            veredito = await self.provider.complete(
                prompt, model=self.model, temperature=0.0, max_tokens=256, variant=0
            )
        except ProviderError as exc:
            return GradeOutcome(
                parsed_choice=None,
                confidence=None,
                correct=None,
                error=f"juiz falhou: {exc}",
            )

        payload = extrair_payload(veredito)
        if payload is None:
            return GradeOutcome(
                parsed_choice=None,
                confidence=None,
                correct=None,
                error="juiz nao devolveu JSON interpretavel",
            )
        escolha = _normalizar_letra(payload.get("answer"), item.choices)
        confianca = _normalizar_confianca(payload.get("confidence"))
        if escolha is None:
            return GradeOutcome(
                parsed_choice=None,
                confidence=confianca,
                correct=None,
                error=f"juiz devolveu answer invalido: {payload.get('answer')!r}",
            )
        correto = escolha == ABSTAIN if item.expected_abstain else escolha == item.answer
        return GradeOutcome(
            parsed_choice=escolha, confidence=confianca, correct=correto, error=None
        )
