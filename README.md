# iamed-harness

Harness mínimo e auditável para avaliar LLMs em tarefas clínicas de múltipla escolha.

O objetivo não é maximizar acurácia. É tornar visível, em poucos comandos, o que a
acurácia esconde: variância entre execuções idênticas, excesso de confiança e
fragilidade a perturbações semânticas.

## Setup

```bash
make install
```

## Uso

O primeiro comando roda offline, sem API key e sem internet, logo após o clone:

```bash
make run
```

Fluxo completo (baseline, dataset perturbado, comparação e os dois relatórios):

```bash
make demo
```

Comandos individuais:

```bash
uv run harness tasks
uv run harness run mcq_baseline --provider fixture --limit 25
uv run harness run mcq_selfconsistency --provider fixture
uv run harness perturb data/seed_ptbr.jsonl --kind shuffle_choices --out data/seed_ptbr.shuffled.jsonl
uv run harness compare <run_id_a> <run_id_b>
uv run harness report <run_id> --baseline <run_id_a>
```

Para rodar contra o modelo de verdade, preencha `ANTHROPIC_API_KEY` no `.env` e troque
o provider:

```bash
uv run harness run mcq_baseline --provider anthropic --model claude-sonnet-5
```

## O que o harness mede

| Métrica | Onde | O que revela |
|---|---|---|
| Acurácia com IC 95% | `metrics/accuracy.py` | O intervalo costuma ser mais largo do que a diferença entre dois modelos |
| ECE, MCE, Brier | `metrics/calibration.py` | Confiança declarada alta demais para a acurácia observada |
| Flip rate, concordância | `metrics/variance.py` | Quanto a resposta muda entre execuções idênticas |
| Quebra por subgrupo | `metrics/subgroup.py` | Média global escondendo especialidade ou dificuldade ruim |
| Baseline versus perturbado | `perturbations.py` | Quanto do acerto vinha da forma, não do conteúdo |

## Perturbações

| Tipo | O que faz | Gabarito |
|---|---|---|
| `shuffle_choices` | Embaralha as alternativas | Reindexado para a nova letra |
| `add_distractor` | Acrescenta alternativa plausível e errada | Inalterado |
| `negate_stem` | Nega o enunciado | Invertido se o item declarar `negated_answer`, senão vira abstenção |
| `unit_swap` | Troca a unidade sem converter o valor | Mantido se o item declarar `unit_swap_answer`, senão vira abstenção |
| `irrelevant_context` | Prefixa duas frases verdadeiras e irrelevantes | Inalterado |

Quando a perturbação destrói a unicidade da resposta certa, o item não é descartado nem
mantido com gabarito falso: ele passa a cobrar abstenção (`expected_abstain`), e um
modelo que responde com confiança a uma pergunta sem resposta conta como erro.

## Três regras do projeto

**1. Nenhuma falha silenciosa.** Resposta não parseada, item inválido e chamada com erro
aparecem no relatório como categoria própria e continuam no denominador. O relatório
mostra lado a lado a acurácia honesta e a acurácia que sairia se as falhas fossem
descartadas, para que o tamanho da diferença fique explícito.

**2. Todo run é um artefato reprodutível.** Cada execução grava em `runs/<run_id>/` a
config resolvida (modelo, temperatura, seed, sha256 do dataset e do prompt) e as
predições brutas em JSONL append-only. `harness report <run_id>` regenera qualquer
análise sem tocar no modelo.

**3. Calibração é implementada à mão.** ECE, MCE e Brier não vêm de biblioteca, e os
testes conferem os valores contra um caso calculado com papel e caneta
(`tests/test_calibration.py`). Bins vazios não entram na média ponderada; predições sem
confiança declarada não entram no cálculo e são contadas à parte.

## Sobre os fixtures

O provider `fixture` lê respostas gravadas em `fixtures/*.jsonl` e não faz nenhuma
chamada de rede. As respostas commitadas no repositório são **sintéticas**, geradas por
`scripts/record_fixtures.py --mode synthetic`, e cada linha carrega `"source":
"synthetic"`.

Elas existem para que o pipeline, as métricas e o relatório possam ser demonstrados sem
depender de rede ou credencial. **Não são saída de um modelo real e não devem ser
apresentadas como resultado de avaliação.** Para gravar respostas reais:

```bash
uv run python scripts/record_fixtures.py --dataset data/seed_ptbr.jsonl --mode anthropic
```

## Estrutura

```
src/iamed_harness/
  cli.py            Typer: tasks, run, perturb, compare, report
  schemas.py        Pydantic: Item, Prediction, RunConfig, RunResult, métricas
  loader.py         JSONL validado linha a linha, falha com o número da linha
  registry.py       tarefas em tasks/*.yaml, nunca em código
  runner.py         execução assíncrona, cache, retry com backoff, run_id
  cache.py          cache em disco por hash de provider, modelo, prompt, temp, seed, repetição
  providers/        anthropic, fixture (offline), echo (determinístico)
  graders/          exact_choice, regex, llm_judge
  metrics/          accuracy, calibration, variance, subgroup
  perturbations.py  cinco perturbações, cada uma com teste de gabarito
  report/           HTML autocontido, sem CDN
prompts/            templates Jinja2 do prompt e da rubrica do juiz
tasks/              mcq_baseline, mcq_selfconsistency, mcq_perturbed
data/               25 itens autorais em pt-BR (ver data/README.md)
fixtures/           respostas gravadas para o modo offline
tests/              pytest
```

## Desenvolvimento

```bash
make test
make lint
make fixtures   # só depois de mexer no dataset ou no template de prompt
```

Se o template de prompt ou o dataset mudarem, o hash do prompt muda e o provider
`fixture` passa a não encontrar as respostas gravadas. O erro diz exatamente qual comando
regrava.

## Ecossistema IA.med

Este repositório é material de uma aula ao vivo sobre avaliação de LLMs em medicina,
dentro do ecossistema IA.med (`med-ia-app`, a landing `medicina-ia.github.io`,
`iamed-analytics`, `iamed-analytics-app` e `iamed-leads`). Hoje ainda não é citado nem
linkado no app nem na landing, e nenhum outro repositório do ecossistema referencia este
harness.

## Licença

MIT.
