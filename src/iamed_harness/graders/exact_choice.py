"""Grader padrao: extrai a letra escolhida e a confianca de uma resposta JSON.

O parser tolera cerca de codigo, texto antes e depois, e JSON aninhado em
prosa. O que ele nao faz e chutar: se nao houver letra extraivel, o resultado
e um erro registrado, nao um acerto por acaso.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..schemas import ABSTAIN, Item
from . import GradeOutcome

CERCA = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
GATILHO_RESPOSTA = re.compile(
    r"\b(?:resposta|answer|alternativa|letra|opcao|opção)\b", re.IGNORECASE
)
LETRA_ISOLADA = re.compile(r"\(?\b([A-E])\b\)?")
ABSTENCAO = re.compile(r"\b(abstain|abstenção|abstencao|nenhuma das alternativas)\b", re.IGNORECASE)

JANELA_FALLBACK = 80
"""Quantos caracteres depois do gatilho ainda contam como a mesma afirmacao.

Sem esse limite, uma letra maiuscula solta tres paragrafos abaixo viraria
resposta. O fallback existe para recuperar prosa curta, nao para garimpar.
"""


def _letra_por_fallback(texto: str) -> str | None:
    """Recupera a letra de uma resposta em prosa, sem chutar.

    Exige duas coisas: uma palavra-gatilho ("resposta", "alternativa", ...) e,
    logo depois dela, uma letra MAIUSCULA isolada. A exigencia de maiuscula e
    o que impede que o "e" de "correta e B" seja lido como alternativa E.
    """
    gatilho = GATILHO_RESPOSTA.search(texto)
    if gatilho is None:
        return None
    cauda = texto[gatilho.end() : gatilho.end() + JANELA_FALLBACK]
    m = LETRA_ISOLADA.search(cauda)
    return m.group(1) if m else None


def _candidatos_json(texto: str) -> list[str]:
    """Blocos que podem ser JSON, do mais provavel ao menos."""
    candidatos: list[str] = []
    for bloco in CERCA.findall(texto):
        candidatos.append(bloco.strip())
    profundidade = 0
    inicio = -1
    for i, ch in enumerate(texto):
        if ch == "{":
            if profundidade == 0:
                inicio = i
            profundidade += 1
        elif ch == "}" and profundidade > 0:
            profundidade -= 1
            if profundidade == 0 and inicio >= 0:
                candidatos.append(texto[inicio : i + 1])
    candidatos.append(texto.strip())
    return candidatos


def extrair_payload(texto: str) -> dict[str, Any] | None:
    """Devolve o primeiro objeto JSON com chave `answer`, ou None."""
    for candidato in _candidatos_json(texto):
        try:
            valor = json.loads(candidato)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(valor, dict) and "answer" in valor:
            return valor
    return None


def _normalizar_letra(valor: object, choices: dict[str, str]) -> str | None:
    texto = str(valor).strip()
    if ABSTENCAO.search(texto):
        return ABSTAIN
    letra = texto.strip("().: ").upper()
    if letra in choices:
        return letra
    if len(letra) > 1:
        m = re.match(r"^([A-Z])\b", letra)
        if m and m.group(1) in choices:
            return m.group(1)
    return None


def _normalizar_confianca(valor: object) -> float | None:
    if valor is None:
        return None
    try:
        numero = float(str(valor).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None
    if 0.0 <= numero <= 1.0:
        numero *= 100.0
    if not 0.0 <= numero <= 100.0:
        return None
    return numero / 100.0


class ExactChoiceGrader:
    name = "exact_choice"

    async def grade(self, item: Item, raw_response: str) -> GradeOutcome:
        if not raw_response.strip():
            return GradeOutcome(
                parsed_choice=None,
                confidence=None,
                correct=None,
                error="resposta vazia",
            )

        payload = extrair_payload(raw_response)
        confianca: float | None = None
        escolha: str | None = None
        erro: str | None = None

        if payload is not None:
            escolha = _normalizar_letra(payload.get("answer"), item.choices)
            confianca = _normalizar_confianca(payload.get("confidence"))
            if escolha is None:
                erro = f"campo answer nao mapeia para uma alternativa: {payload.get('answer')!r}"
        else:
            recuperada = _letra_por_fallback(raw_response)
            if recuperada is not None and recuperada in item.choices:
                escolha = recuperada
                erro = "JSON ausente, letra recuperada por regex de fallback"
            elif ABSTENCAO.search(raw_response):
                escolha = ABSTAIN
                erro = "JSON ausente, abstencao recuperada por regex de fallback"
            else:
                erro = "nao foi possivel extrair JSON nem letra da resposta"

        if escolha is None:
            return GradeOutcome(parsed_choice=None, confidence=confianca, correct=None, error=erro)

        correto = escolha == ABSTAIN if item.expected_abstain else escolha == item.answer
        return GradeOutcome(
            parsed_choice=escolha, confidence=confianca, correct=correto, error=erro
        )
