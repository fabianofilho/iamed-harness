# Datasets

## `seed_ptbr.jsonl`

25 itens autorais de multipla escolha em portugues, escritos diretamente neste
repositorio. Cobrem cardiologia, endocrinologia, nefrologia, pneumologia, neurologia,
emergencia, pediatria, ginecologia e obstetricia, infectologia, hematologia,
gastroenterologia, psiquiatria, reumatologia, dermatologia, urologia, oftalmologia,
geriatria, medicina de familia e toxicologia.

Sao questoes de nivel basico e intermediario, com gabarito unico e pouco controverso.
Isso e proposital: o dataset existe para expor variancia, calibracao e fragilidade do
modelo, nao para discutir medicina de fronteira. Se um item admitisse discussao, a queda
de acuracia deixaria de ser atribuivel ao modelo.

### Schema

```json
{
  "id": "ptbr-001",
  "question": "enunciado",
  "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "answer": "A",
  "metadata": {"especialidade": "cardiologia", "dificuldade": "facil", "distractor": "..."},
  "expected_abstain": false
}
```

`metadata` e livre e alimenta a quebra por subgrupo. Campos que as perturbacoes leem:

| Campo | Usado por | Efeito |
|---|---|---|
| `distractor` | `add_distractor` | Texto da alternativa extra. Sem ele, usa um distrator generico |
| `negated_answer` | `negate_stem` | Gabarito sob negacao. Sem ele, o item vira `expected_abstain` |
| `negated_question` | `negate_stem` | Enunciado negado escrito a mao. Sem ele, a negacao e gerada por regra |
| `unit_swap_answer` | `unit_swap` | Gabarito apos a troca de unidade. Sem ele, o item vira `expected_abstain` |

`expected_abstain: true` significa que a resposta certa e recusar-se a escolher. Nesse
caso `answer` tem que ser exatamente `"ABSTAIN"`, e o validator do schema recusa qualquer
outra combinacao.

## `seed_ptbr.shuffled.jsonl`

Gerado a partir do anterior, e commitado para que `make demo` funcione logo apos o clone.
Regenerar:

```bash
uv run harness perturb data/seed_ptbr.jsonl --kind shuffle_choices --out data/seed_ptbr.shuffled.jsonl
```

A seed padrao e 20260811, entao a saida e identica a cada execucao. Se voce mudar a seed,
regrave o fixture correspondente (`make fixtures`), senao o provider offline nao vai
encontrar os prompts.

## MedQA e PubMedQA

Nao sao commitados aqui: tem licenca propria e tamanho incompativel com um repositorio de
aula. Baixe sob demanda para `data/external/`, que esta no `.gitignore`.

```bash
uv pip install datasets
uv run python scripts/fetch_datasets.py --dataset medqa --split test --limit 200
uv run python scripts/fetch_datasets.py --dataset pubmedqa --subset pqa_labeled --split train --limit 200
```

O script converte para o schema `Item` acima. A tarefa `tasks/medqa_baseline.yaml` ja
aponta para `data/external/medqa_test.jsonl`; com `--limit 5000` o split de teste vem
inteiro (1273 itens). Para PubMedQA, crie uma tarefa no mesmo molde apontando para o
arquivo baixado.

As questoes do MedQA estao em ingles e o prompt padrao em portugues. Leve isso em conta ao
comparar com o seed pt-BR, e lembre que um benchmark publico desse porte provavelmente
esteve no treino do modelo avaliado.

Datasets externos nao tem fixture gravado. Rode com `--provider anthropic` ou `openai`, ou grave os
seus com `scripts/record_fixtures.py`.
