from __future__ import annotations

import pytest

from iamed_harness.graders import get_grader
from iamed_harness.graders.exact_choice import ExactChoiceGrader
from iamed_harness.graders.regex_grader import RegexGrader
from iamed_harness.schemas import ABSTAIN, Item

ITEM = Item(
    id="t1",
    question="Qual e a conduta?",
    choices={"A": "primeira", "B": "segunda", "C": "terceira"},
    answer="B",
)
ITEM_ABSTAIN = Item(
    id="t2",
    question="Qual e a conduta?",
    choices={"A": "primeira", "B": "segunda"},
    answer=ABSTAIN,
    expected_abstain=True,
)


async def test_json_puro_e_parseado():
    r = await ExactChoiceGrader().grade(
        ITEM, '{"answer": "B", "confidence": 90, "reasoning": "porque sim"}'
    )
    assert r.parsed_choice == "B"
    assert r.confidence == pytest.approx(0.90)
    assert r.correct is True
    assert r.error is None


async def test_json_dentro_de_cerca_de_codigo():
    bruto = '```json\n{"answer": "A", "confidence": 55, "reasoning": "x"}\n```'
    r = await ExactChoiceGrader().grade(ITEM, bruto)
    assert r.parsed_choice == "A"
    assert r.correct is False


async def test_json_com_prosa_ao_redor():
    bruto = (
        'Analisando o caso:\n{"answer": "B", "confidence": 77, "reasoning": "y"}\n'
        "Espero ter ajudado."
    )
    r = await ExactChoiceGrader().grade(ITEM, bruto)
    assert r.parsed_choice == "B"
    assert r.confidence == pytest.approx(0.77)


async def test_confianca_em_escala_zero_a_um_e_aceita():
    r = await ExactChoiceGrader().grade(ITEM, '{"answer": "B", "confidence": 0.8}')
    assert r.confidence == pytest.approx(0.80)


async def test_confianca_ausente_nao_invalida_a_escolha():
    r = await ExactChoiceGrader().grade(ITEM, '{"answer": "B"}')
    assert r.parsed_choice == "B"
    assert r.confidence is None
    assert r.correct is True


async def test_confianca_fora_de_faixa_vira_none():
    r = await ExactChoiceGrader().grade(ITEM, '{"answer": "B", "confidence": 480}')
    assert r.parsed_choice == "B"
    assert r.confidence is None


async def test_letra_inexistente_e_erro_registrado_e_nao_acerto():
    r = await ExactChoiceGrader().grade(ITEM, '{"answer": "Z", "confidence": 90}')
    assert r.parsed_choice is None
    assert r.correct is None
    assert r.error is not None


async def test_resposta_vazia_e_erro_registrado():
    r = await ExactChoiceGrader().grade(ITEM, "   ")
    assert r.parsed_choice is None
    assert r.correct is None
    assert r.error == "resposta vazia"


async def test_prosa_sem_json_nao_vira_acerto_por_acaso():
    r = await ExactChoiceGrader().grade(
        ITEM, "Nao e possivel determinar a conduta com os dados fornecidos."
    )
    assert r.parsed_choice is None
    assert r.correct is None


async def test_letra_recuperada_por_fallback_marca_o_erro_mesmo_acertando():
    r = await ExactChoiceGrader().grade(ITEM, "A resposta correta e B, sem duvida.")
    assert r.parsed_choice == "B"
    assert r.correct is True
    assert r.error is not None and "fallback" in r.error


async def test_abstencao_conta_como_acerto_em_item_expected_abstain():
    r = await ExactChoiceGrader().grade(ITEM_ABSTAIN, '{"answer": "ABSTAIN", "confidence": 60}')
    assert r.parsed_choice == ABSTAIN
    assert r.correct is True


async def test_escolher_alternativa_em_item_expected_abstain_e_erro():
    r = await ExactChoiceGrader().grade(ITEM_ABSTAIN, '{"answer": "A", "confidence": 95}')
    assert r.parsed_choice == "A"
    assert r.correct is False


async def test_abstencao_por_texto_livre_e_reconhecida():
    r = await ExactChoiceGrader().grade(
        ITEM_ABSTAIN, "Nenhuma das alternativas esta correta neste caso."
    )
    assert r.parsed_choice == ABSTAIN
    assert r.correct is True


async def test_grader_regex_extrai_a_letra():
    r = await RegexGrader(r"Resposta:\s*([A-C])").grade(ITEM, "Resposta: B")
    assert r.parsed_choice == "B"
    assert r.correct is True


async def test_grader_regex_sem_casamento_registra_erro():
    r = await RegexGrader(r"Resposta:\s*([A-C])").grade(ITEM, "sem padrao aqui")
    assert r.parsed_choice is None
    assert r.correct is None
    assert r.error is not None


def test_fabrica_devolve_o_grader_certo():
    assert get_grader("exact_choice").name == "exact_choice"
    assert get_grader("regex", {"pattern": r"([A-C])"}).name == "regex"


def test_fabrica_rejeita_grader_desconhecido():
    with pytest.raises(ValueError, match="grader desconhecido"):
        get_grader("inexistente")


def test_llm_judge_exige_provider():
    with pytest.raises(ValueError, match="exige um provider"):
        get_grader("llm_judge")
