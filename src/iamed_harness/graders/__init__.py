"""Graders: comparam resposta bruta com gabarito.

Um grader nunca levanta excecao por resposta ruim. Resposta ruim e um
resultado: `parsed_choice=None` e `error` preenchido. O item continua no
denominador.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict

from ..schemas import Item


class GradeOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parsed_choice: str | None
    confidence: float | None
    correct: bool | None
    error: str | None


class Grader(Protocol):
    name: str

    async def grade(self, item: Item, raw_response: str) -> GradeOutcome: ...


def get_grader(name: str, options: dict[str, str] | None = None, **kwargs: object) -> Grader:
    from .exact_choice import ExactChoiceGrader
    from .llm_judge import LLMJudgeGrader
    from .regex_grader import RegexGrader

    opcoes = options or {}
    if name == "exact_choice":
        return ExactChoiceGrader()
    if name == "regex":
        return RegexGrader(opcoes.get("pattern", r"\b([A-E])\b"))
    if name == "llm_judge":
        provider = kwargs.get("provider")
        if provider is None:
            raise ValueError("grader llm_judge exige um provider")
        return LLMJudgeGrader(
            provider=provider,  # type: ignore[arg-type]
            model=str(kwargs.get("model", "claude-sonnet-5")),
            template_path=opcoes.get("template", "prompts/judge_rubric.j2"),
        )
    raise ValueError(f"grader desconhecido: {name!r}")


__all__ = ["GradeOutcome", "Grader", "get_grader"]
