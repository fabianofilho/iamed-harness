"""Perturbacoes semanticas do dataset.

Cada funcao recebe itens e devolve itens, preservando o `id` original com um
sufixo e registrando o tipo em `metadata.perturbation`. Nenhuma delas muda o
conteudo clinico da pergunta: elas mudam a forma, a ordem ou o contexto. Se a
acuracia cai, a queda e do modelo, nao da medicina.

Sobre gabarito: quando a perturbacao destroi a unicidade da resposta certa, o
item nao e descartado nem mantido com gabarito falso. Ele vira
`expected_abstain=True`, e a resposta correta passa a ser recusar-se a
escolher. Um modelo que responde com confianca a uma pergunta sem resposta
esta errando, e o harness precisa contar isso como erro.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np

from .schemas import ABSTAIN, Item

DISTRATOR_PADRAO = "Solicitar ressonancia magnetica de corpo inteiro antes de qualquer conduta"

FRASES_IRRELEVANTES: list[str] = [
    "O paciente relata que prefere consultas no periodo da manha.",
    "A unidade de saude fica a doze quilometros da residencia do paciente.",
    "O acompanhante informou que o transporte ate o servico levou cerca de uma hora.",
    "O prontuario eletronico da unidade foi atualizado para a versao mais recente no mes passado.",
    "A sala de espera do ambulatorio passou por reforma no semestre anterior.",
    "O paciente e canhoto e trabalha em turno administrativo.",
]

TROCAS_DE_UNIDADE: list[tuple[str, str]] = [
    ("mg/dL", "mmol/L"),
    ("mmol/L", "mg/dL"),
    ("mEq/L", "mmol/L"),
    ("mmHg", "cmH2O"),
    ("g/dL", "mg/dL"),
    ("UI/L", "mg/dL"),
    ("mcg/dL", "mg/mL"),
    ("/mm3", "/L"),
]

VERBOS_NEGAVEIS = re.compile(
    r"\b(e|sao|esta|estao|deve|devem|pode|podem|indica|indicam|caracteriza|"
    r"caracterizam|corresponde|correspondem|representa|representam|constitui)\b",
    re.IGNORECASE,
)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _derivar(
    item: Item,
    kind: str,
    *,
    question: str | None = None,
    choices: dict[str, str] | None = None,
    answer: str | None = None,
    expected_abstain: bool = False,
    extra: dict[str, str | int | float] | None = None,
) -> Item:
    metadata: dict[str, str | int | float] = dict(item.metadata)
    metadata["perturbation"] = kind
    metadata["source_id"] = item.id
    if extra:
        metadata.update(extra)
    return Item(
        id=f"{item.id}::{kind}",
        question=question if question is not None else item.question,
        choices=choices if choices is not None else dict(item.choices),
        answer=ABSTAIN if expected_abstain else (answer if answer is not None else item.answer),
        metadata=metadata,
        expected_abstain=expected_abstain,
    )


def shuffle_choices(items: list[Item], *, seed: int = 20260811) -> list[Item]:
    """Embaralha as alternativas e reindexa o gabarito.

    Mede sensibilidade a posicao: um modelo que raciocina sobre o conteudo
    clinico responde igual; um que aprendeu a preferir a alternativa C, nao.
    """
    rng = _rng(seed)
    saida: list[Item] = []
    for item in items:
        letras = sorted(item.choices)
        textos = [item.choices[letra] for letra in letras]
        ordem = rng.permutation(len(textos))
        novos = {letras[i]: textos[ordem[i]] for i in range(len(letras))}

        if item.expected_abstain:
            saida.append(_derivar(item, "shuffle_choices", choices=novos, expected_abstain=True))
            continue

        texto_correto = item.choices[item.answer]
        nova_letra = next(letra for letra, texto in novos.items() if texto == texto_correto)
        saida.append(_derivar(item, "shuffle_choices", choices=novos, answer=nova_letra))
    return saida


def add_distractor(items: list[Item], *, seed: int = 20260811) -> list[Item]:
    """Acrescenta uma alternativa plausivel e claramente errada.

    O texto vem de `metadata.distractor` quando o item define um; caso
    contrario usa um distrator generico. O gabarito nao muda.
    """
    saida: list[Item] = []
    for item in items:
        letras = sorted(item.choices)
        proxima = chr(ord(max(letras)) + 1)
        if not proxima.isalpha():
            raise ValueError(f"item {item.id}: nao ha letra livre para o distrator")
        texto = str(item.metadata.get("distractor", DISTRATOR_PADRAO))
        novos = dict(item.choices)
        novos[proxima] = texto
        saida.append(
            _derivar(
                item,
                "add_distractor",
                choices=novos,
                expected_abstain=item.expected_abstain,
                extra={"distractor_letter": proxima},
            )
        )
    return saida


def _negar_enunciado(question: str) -> str:
    """Insere uma negacao antes do primeiro verbo negavel do enunciado.

    Se nenhum verbo conhecido aparecer, anexa uma instrucao de negacao
    explicita. As duas saidas sao deterministicas e legiveis em portugues.
    """
    m = VERBOS_NEGAVEIS.search(question)
    if m is None:
        return f"{question.rstrip()} Considere qual alternativa NAO se aplica."
    inicio = m.start()
    return f"{question[:inicio]}NAO {question[inicio:]}"


def negate_stem(items: list[Item], *, seed: int = 20260811) -> list[Item]:
    """Nega o enunciado.

    Se o item declara `metadata.negated_answer`, o gabarito passa a ser essa
    alternativa. Sem essa declaracao, negar um enunciado de multipla escolha
    normalmente torna varias alternativas corretas ao mesmo tempo, entao o
    item vira `expected_abstain`: nao existe uma unica resposta certa, e
    fingir que existe seria fabricar gabarito.
    """
    saida: list[Item] = []
    for item in items:
        enunciado = str(item.metadata.get("negated_question", "")) or _negar_enunciado(
            item.question
        )
        declarado = item.metadata.get("negated_answer")
        if declarado is not None and str(declarado) in item.choices:
            saida.append(_derivar(item, "negate_stem", question=enunciado, answer=str(declarado)))
        else:
            saida.append(_derivar(item, "negate_stem", question=enunciado, expected_abstain=True))
    return saida


def _trocar_unidade(texto: str) -> tuple[str, str] | None:
    for origem, destino in TROCAS_DE_UNIDADE:
        if origem in texto:
            return texto.replace(origem, destino, 1), f"{origem}->{destino}"
    return None


def unit_swap(items: list[Item], *, seed: int = 20260811) -> list[Item]:
    """Troca a unidade de medida do enunciado sem converter o valor.

    O numero passa a ser clinicamente absurdo (glicemia de 126 mmol/L nao
    existe em paciente vivo). O gabarito so e mantido quando o item declara
    `metadata.unit_swap_answer`; caso contrario o item vira `expected_abstain`,
    porque a conduta correta diante de um valor impossivel e questionar o dado,
    nao escolher uma alternativa.

    Itens sem unidade reconhecida nao geram saida: perturbar o que nao tem
    unidade seria inventar mudanca. O chamador compara os tamanhos das listas.
    """
    saida: list[Item] = []
    for item in items:
        resultado = _trocar_unidade(item.question)
        if resultado is None:
            continue
        enunciado, troca = resultado
        declarado = item.metadata.get("unit_swap_answer")
        if declarado is not None and str(declarado) in item.choices:
            saida.append(
                _derivar(
                    item,
                    "unit_swap",
                    question=enunciado,
                    answer=str(declarado),
                    extra={"unit_swap": troca},
                )
            )
        else:
            saida.append(
                _derivar(
                    item,
                    "unit_swap",
                    question=enunciado,
                    expected_abstain=True,
                    extra={"unit_swap": troca},
                )
            )
    return saida


def irrelevant_context(items: list[Item], *, seed: int = 20260811) -> list[Item]:
    """Prefixa duas frases clinicas verdadeiras e irrelevantes.

    Nada do que e acrescentado muda a resposta. Se a acuracia cai, o modelo
    esta sendo distraido por contexto, nao derrotado por medicina.
    """
    rng = _rng(seed)
    saida: list[Item] = []
    for item in items:
        escolhidas = rng.choice(len(FRASES_IRRELEVANTES), size=2, replace=False)
        prefixo = " ".join(FRASES_IRRELEVANTES[int(i)] for i in escolhidas)
        saida.append(
            _derivar(
                item,
                "irrelevant_context",
                question=f"{prefixo} {item.question}",
                expected_abstain=item.expected_abstain,
            )
        )
    return saida


PERTURBACOES: dict[str, Callable[..., list[Item]]] = {
    "shuffle_choices": shuffle_choices,
    "add_distractor": add_distractor,
    "negate_stem": negate_stem,
    "unit_swap": unit_swap,
    "irrelevant_context": irrelevant_context,
}


def apply_perturbation(kind: str, items: list[Item], *, seed: int = 20260811) -> list[Item]:
    if kind not in PERTURBACOES:
        disponiveis = ", ".join(sorted(PERTURBACOES))
        raise ValueError(f"perturbacao desconhecida: {kind!r}. Disponiveis: {disponiveis}")
    return PERTURBACOES[kind](items, seed=seed)
