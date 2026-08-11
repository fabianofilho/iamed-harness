"""Grava os fixtures usados pelo provider offline.

Dois modos:

    --mode synthetic   (padrao) gera respostas deterministicas, sem rede.
                       E o que faz `make run` funcionar logo apos o clone.
    --mode anthropic   chama o modelo de verdade e grava o que ele respondeu.
                       Exige ANTHROPIC_API_KEY.

As respostas sinteticas NAO sao de um modelo real e nao devem ser
apresentadas como resultado de avaliacao. Elas existem para que o pipeline,
as metricas e o relatorio possam ser demonstrados sem depender de rede. Cada
linha gravada carrega o campo `source`, entao a origem nunca fica ambigua.

O perfil sintetico e calibrado de proposito para exibir os tres fenomenos que
a sessao discute: acuracia intermediaria, confianca alta demais para essa
acuracia (ECE grande) e instabilidade entre execucoes identicas.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np
from jinja2 import Template

from iamed_harness.loader import load_items
from iamed_harness.providers import get_provider
from iamed_harness.providers.fixture import hash_prompt
from iamed_harness.schemas import Item

N_RESPOSTAS = 5

PERFIS: dict[str, dict[str, float]] = {
    # p_acerto: chance de a resposta modal estar certa
    # p_flip:   chance de o item mudar de resposta entre repeticoes
    # p_lixo:   chance de a resposta sair sem JSON extraivel
    "data/seed_ptbr.jsonl": {"p_acerto": 0.72, "p_flip": 0.32, "p_lixo": 0.04},
    "data/seed_ptbr.shuffled.jsonl": {"p_acerto": 0.52, "p_flip": 0.44, "p_lixo": 0.06},
}
PERFIL_PADRAO = {"p_acerto": 0.60, "p_flip": 0.35, "p_lixo": 0.05}

RACIOCINIOS_CERTOS = [
    "O quadro clinico e o achado complementar apontam de forma direta para essa conduta.",
    "Essa e a alternativa que corresponde a recomendacao consolidada para o cenario descrito.",
    "Os dados do enunciado fecham criterio para essa opcao.",
]
RACIOCINIOS_ERRADOS = [
    "O achado descrito favorece essa alternativa em detrimento das demais.",
    "Essa conduta parece a mais coerente com a apresentacao clinica.",
    "A combinacao de sinais aponta para essa opcao.",
]
RESPOSTAS_LIXO = [
    "Analisando o caso clinico apresentado, a conduta mais adequada seria avaliar "
    "cuidadosamente cada alternativa antes de concluir.",
    "Nao e possivel determinar a resposta com as informacoes fornecidas no enunciado.",
    "```\nresposta: depende do contexto clinico completo do paciente\n```",
]


def _rng_do_item(item_id: str, seed: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{item_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _resposta_json(letra: str, confianca: int, raciocinio: str) -> str:
    return json.dumps(
        {"answer": letra, "confidence": confianca, "reasoning": raciocinio},
        ensure_ascii=False,
    )


def _gerar_respostas(item: Item, perfil: dict[str, float], seed: int) -> list[str]:
    rng = _rng_do_item(item.id, seed)
    letras = sorted(item.choices)
    erradas = [letra for letra in letras if letra != item.answer] or letras

    acerta_modal = bool(rng.random() < perfil["p_acerto"])
    letra_modal = item.answer if acerta_modal else str(rng.choice(erradas))
    letra_alternativa = str(rng.choice([letra for letra in letras if letra != letra_modal]))
    instavel = bool(rng.random() < perfil["p_flip"])

    respostas: list[str] = []
    for k in range(N_RESPOSTAS):
        # Vale para k=0 tambem: a temperatura zero so le a primeira resposta, e
        # sem lixo nela o baseline nunca exibiria a categoria de falha.
        if rng.random() < perfil["p_lixo"]:
            respostas.append(str(rng.choice(RESPOSTAS_LIXO)))
            continue

        letra = letra_alternativa if (instavel and k % 2 == 1) else letra_modal
        certa = letra == item.answer
        # Confianca alta mesmo quando erra: e assim que o ECE aparece.
        confianca = int(rng.integers(88, 98) if certa else rng.integers(79, 94))
        raciocinio = str(rng.choice(RACIOCINIOS_CERTOS if certa else RACIOCINIOS_ERRADOS))
        respostas.append(_resposta_json(letra, confianca, raciocinio))
    return respostas


async def _gravar_de_verdade(
    items: list[Item], template: Template, model: str, temperature: float
) -> dict[str, list[str]]:
    provider = get_provider("anthropic")
    saida: dict[str, list[str]] = {}
    for item in items:
        prompt = template.render(item=item)
        respostas = [
            await provider.complete(
                prompt, model=model, temperature=temperature, max_tokens=300, variant=k
            )
            for k in range(N_RESPOSTAS)
        ]
        saida[prompt] = respostas
        print(f"  gravado {item.id}")
    return saida


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/seed_ptbr.jsonl")
    parser.add_argument("--prompt", default="prompts/mcq_answer.j2")
    parser.add_argument("--out", default=None, help="padrao: fixtures/<nome>.fixture.jsonl")
    parser.add_argument("--mode", choices=["synthetic", "anthropic"], default="synthetic")
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    items = load_items(args.dataset)
    template = Template(Path(args.prompt).read_text(encoding="utf-8"), keep_trailing_newline=True)
    perfil = PERFIS.get(args.dataset, PERFIL_PADRAO)

    destino = Path(args.out or f"fixtures/{Path(args.dataset).stem}.fixture.jsonl")
    destino.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "anthropic":
        gravadas = asyncio.run(_gravar_de_verdade(items, template, args.model, args.temperature))
        linhas = [
            {
                "prompt_sha256": hash_prompt(template.render(item=item)),
                "item_id": item.id,
                "model": args.model,
                "source": "anthropic",
                "responses": gravadas[template.render(item=item)],
            }
            for item in items
        ]
    else:
        linhas = [
            {
                "prompt_sha256": hash_prompt(template.render(item=item)),
                "item_id": item.id,
                "model": args.model,
                "source": "synthetic",
                "responses": _gerar_respostas(item, perfil, args.seed),
            }
            for item in items
        ]

    with destino.open("w", encoding="utf-8") as f:
        for linha in linhas:
            f.write(json.dumps(linha, ensure_ascii=False) + "\n")

    print(f"{len(linhas)} fixtures gravados em {destino} (modo {args.mode})")
    if args.mode == "synthetic":
        print("Atencao: respostas sinteticas, nao sao saida de um modelo real.")


if __name__ == "__main__":
    main()
