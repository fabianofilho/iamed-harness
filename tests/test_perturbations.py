"""Cada perturbacao tem que tratar o gabarito corretamente.

Perturbacao que embaralha alternativa sem reindexar gabarito, ou que mantem
gabarito depois de invalidar o enunciado, produz queda de acuracia falsa. Esses
testes existem para que a queda medida seja do modelo.
"""

from __future__ import annotations

import pytest

from iamed_harness.perturbations import (
    PERTURBACOES,
    add_distractor,
    apply_perturbation,
    irrelevant_context,
    negate_stem,
    shuffle_choices,
    unit_swap,
)
from iamed_harness.schemas import ABSTAIN, Item

ITEM = Item(
    id="p1",
    question="Paciente com glicemia de 138 mg/dL em jejum. Qual e a conduta indicada?",
    choices={"A": "conduta alfa", "B": "conduta beta", "C": "conduta gama", "D": "conduta delta"},
    answer="B",
    metadata={"especialidade": "endocrinologia", "distractor": "conduta absurda"},
)
ITEM_SEM_UNIDADE = Item(
    id="p2",
    question="Qual estrutura anatomica e descrita?",
    choices={"A": "alfa", "B": "beta"},
    answer="A",
    metadata={},
)


def test_shuffle_reindexa_o_gabarito_para_o_mesmo_texto():
    (novo,) = shuffle_choices([ITEM], seed=7)
    assert novo.choices[novo.answer] == ITEM.choices[ITEM.answer]
    assert sorted(novo.choices.values()) == sorted(ITEM.choices.values())
    assert novo.id == "p1::shuffle_choices"
    assert novo.metadata["perturbation"] == "shuffle_choices"
    assert novo.metadata["source_id"] == "p1"


def test_shuffle_e_deterministico_com_a_mesma_seed():
    a = shuffle_choices([ITEM], seed=42)[0]
    b = shuffle_choices([ITEM], seed=42)[0]
    assert a.choices == b.choices
    assert a.answer == b.answer


def test_shuffle_realmente_muda_a_ordem_em_algum_item():
    items = [ITEM.model_copy(update={"id": f"p{i}"}) for i in range(20)]
    embaralhados = shuffle_choices(items, seed=3)
    assert any(n.choices != o.choices for n, o in zip(embaralhados, items, strict=True))


def test_add_distractor_acrescenta_letra_nova_sem_mexer_no_gabarito():
    (novo,) = add_distractor([ITEM])
    assert novo.answer == ITEM.answer
    assert len(novo.choices) == len(ITEM.choices) + 1
    assert novo.choices["E"] == "conduta absurda"
    assert novo.choices[novo.answer] == ITEM.choices[ITEM.answer]


def test_add_distractor_usa_texto_generico_quando_o_item_nao_declara():
    item = ITEM.model_copy(update={"metadata": {}})
    (novo,) = add_distractor([item])
    assert novo.choices["E"]
    assert novo.answer == item.answer


def test_negate_stem_sem_gabarito_declarado_vira_expected_abstain():
    (novo,) = negate_stem([ITEM])
    assert novo.expected_abstain is True
    assert novo.answer == ABSTAIN
    assert "NAO" in novo.question
    assert novo.choices == ITEM.choices


def test_negate_stem_com_gabarito_declarado_inverte_em_vez_de_abster():
    item = ITEM.model_copy(update={"metadata": {**ITEM.metadata, "negated_answer": "C"}})
    (novo,) = negate_stem([item])
    assert novo.expected_abstain is False
    assert novo.answer == "C"


def test_negate_stem_ignora_gabarito_declarado_invalido():
    item = ITEM.model_copy(update={"metadata": {**ITEM.metadata, "negated_answer": "Z"}})
    (novo,) = negate_stem([item])
    assert novo.expected_abstain is True
    assert novo.answer == ABSTAIN


def test_negate_stem_usa_enunciado_declarado_quando_existe():
    item = ITEM.model_copy(
        update={"metadata": {**ITEM.metadata, "negated_question": "Qual NAO se aplica?"}}
    )
    (novo,) = negate_stem([item])
    assert novo.question == "Qual NAO se aplica?"


def test_unit_swap_troca_a_unidade_sem_converter_o_valor():
    (novo,) = unit_swap([ITEM])
    assert "138 mmol/L" in novo.question
    assert "mg/dL" not in novo.question
    assert novo.expected_abstain is True
    assert novo.answer == ABSTAIN
    assert novo.metadata["unit_swap"] == "mg/dL->mmol/L"


def test_unit_swap_mantem_gabarito_quando_o_item_declara():
    item = ITEM.model_copy(update={"metadata": {**ITEM.metadata, "unit_swap_answer": "D"}})
    (novo,) = unit_swap([item])
    assert novo.expected_abstain is False
    assert novo.answer == "D"


def test_unit_swap_pula_item_sem_unidade_reconhecida():
    saida = unit_swap([ITEM, ITEM_SEM_UNIDADE])
    assert len(saida) == 1
    assert saida[0].metadata["source_id"] == "p1"


def test_irrelevant_context_preserva_gabarito_e_alternativas():
    (novo,) = irrelevant_context([ITEM], seed=5)
    assert novo.answer == ITEM.answer
    assert novo.choices == ITEM.choices
    assert novo.question.endswith(ITEM.question)
    assert len(novo.question) > len(ITEM.question)


def test_irrelevant_context_acrescenta_exatamente_duas_frases():
    (novo,) = irrelevant_context([ITEM], seed=5)
    prefixo = novo.question[: -len(ITEM.question)].strip()
    assert prefixo.count(".") == 2


def test_todas_as_perturbacoes_marcam_o_tipo_e_a_origem():
    for kind in PERTURBACOES:
        saida = apply_perturbation(kind, [ITEM], seed=11)
        assert saida, f"{kind} nao produziu saida"
        for novo in saida:
            assert novo.metadata["perturbation"] == kind
            assert novo.metadata["source_id"] == ITEM.id
            assert novo.id.endswith(f"::{kind}")


def test_todas_as_perturbacoes_produzem_itens_validos():
    """Item invalido nao chega nem a ser construido: o validator do schema barra."""
    for kind in PERTURBACOES:
        for novo in apply_perturbation(kind, [ITEM], seed=11):
            revalidado = Item.model_validate(novo.model_dump())
            assert revalidado == novo


def test_perturbacao_desconhecida_e_rejeitada():
    with pytest.raises(ValueError, match="perturbacao desconhecida"):
        apply_perturbation("inventada", [ITEM])


def test_perturbacoes_sobre_o_dataset_do_repositorio():
    from iamed_harness.loader import load_items

    items = load_items("data/seed_ptbr.jsonl")
    embaralhados = shuffle_choices(items, seed=20260811)
    assert len(embaralhados) == len(items)
    for novo, antigo in zip(embaralhados, items, strict=True):
        assert novo.choices[novo.answer] == antigo.choices[antigo.answer]
