"""Baixa MedQA ou PubMedQA sob demanda e converte para o schema `Item`.

Os dados de terceiros nao sao commitados: eles caem em `data/external/`, que
esta no .gitignore. Cada dataset tem licenca propria, e redistribuir por aqui
seria problema de licenciamento e de tamanho de repositorio.

Requer `datasets` instalado (nao e dependencia do harness):

    uv pip install datasets
    uv run python scripts/fetch_datasets.py --dataset medqa --split test --limit 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CONFIGS: dict[str, dict[str, str]] = {
    "medqa": {
        "hf_id": "GBaker/MedQA-USMLE-4-options",
        "campo_pergunta": "question",
        "campo_opcoes": "options",
        "campo_resposta": "answer_idx",
    },
    "pubmedqa": {
        "hf_id": "qiaojin/PubMedQA",
        "campo_pergunta": "question",
        "campo_opcoes": "",
        "campo_resposta": "final_decision",
    },
}

OPCOES_PUBMEDQA = {"A": "yes", "B": "no", "C": "maybe"}


def _converter_medqa(registro: dict[str, Any], indice: int, cfg: dict[str, str]) -> dict[str, Any]:
    opcoes = registro[cfg["campo_opcoes"]]
    choices = {str(k): str(v) for k, v in opcoes.items()}
    return {
        "id": f"medqa-{indice:05d}",
        "question": registro[cfg["campo_pergunta"]],
        "choices": choices,
        "answer": str(registro[cfg["campo_resposta"]]),
        "metadata": {"fonte": "medqa", "idioma": "en"},
    }


def _converter_pubmedqa(
    registro: dict[str, Any], indice: int, cfg: dict[str, str]
) -> dict[str, Any]:
    decisao = str(registro[cfg["campo_resposta"]]).strip().lower()
    letra = next((k for k, v in OPCOES_PUBMEDQA.items() if v == decisao), None)
    if letra is None:
        raise ValueError(f"decisao inesperada em PubMedQA: {decisao!r}")
    contexto = " ".join(registro.get("context", {}).get("contexts", []))
    pergunta = registro[cfg["campo_pergunta"]]
    return {
        "id": f"pubmedqa-{indice:05d}",
        "question": f"{contexto}\n\n{pergunta}".strip(),
        "choices": dict(OPCOES_PUBMEDQA),
        "answer": letra,
        "metadata": {"fonte": "pubmedqa", "idioma": "en"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--subset", default=None, help="config do HuggingFace, quando houver")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    try:
        # Dependencia opcional de proposito: o harness nao precisa dela para rodar.
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError:
        raise SystemExit("pacote `datasets` nao instalado. Rode: uv pip install datasets") from None

    cfg = CONFIGS[args.dataset]
    bruto = (
        load_dataset(cfg["hf_id"], args.subset, split=args.split)
        if args.subset
        else load_dataset(cfg["hf_id"], split=args.split)
    )

    conversor = _converter_medqa if args.dataset == "medqa" else _converter_pubmedqa
    destino = Path(args.out or f"data/external/{args.dataset}_{args.split}.jsonl")
    destino.parent.mkdir(parents=True, exist_ok=True)

    escritos = 0
    with destino.open("w", encoding="utf-8") as f:
        for i, registro in enumerate(bruto):
            if escritos >= args.limit:
                break
            item = conversor(dict(registro), i, cfg)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            escritos += 1

    print(f"{escritos} itens escritos em {destino}")
    print("Aponte uma tarefa em tasks/ para esse arquivo e rode: harness run <tarefa>")


if __name__ == "__main__":
    main()
