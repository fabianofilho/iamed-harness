# iamed-harness

Harness minimo e auditavel para avaliar LLMs em tarefas clinicas de multipla escolha.

O objetivo nao e maximizar acuracia. E tornar visivel, em poucos comandos, o que a
acuracia esconde: variancia entre execucoes identicas, excesso de confianca e
fragilidade a perturbacoes semanticas.

## Setup

```bash
make install
```

## Uso

O primeiro comando roda offline, sem API key e sem internet, logo apos o clone:

```bash
make run
```

Fluxo completo (baseline, dataset perturbado, comparacao e os dois relatorios):

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

Para um modelo local ou qualquer endpoint compativel com a API de chat da OpenAI
(llama.cpp, vLLM, Ollama em `/v1`), use o provider `openai` e aponte `OPENAI_BASE_URL`.
A chave e opcional; servidor local costuma ignorar:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8080/v1 uv run harness run mcq_baseline --provider openai --model local-model
```

Servidor local com um slot so enfileira as chamadas concorrentes. O timeout padrao e
de 300 s por chamada (`HARNESS_TIMEOUT_S`); `--concurrency 1` deixa a latencia por item
mais legivel no relatorio.

## O que o harness mede

| Metrica | Onde | O que revela |
|---|---|---|
| Acuracia com IC 95% (Wilson) | `metrics/accuracy.py` | O intervalo costuma ser mais largo do que a diferenca entre dois modelos |
| ECE, MCE, Brier | `metrics/calibration.py` | Confianca declarada alta demais para a acuracia observada |
| Flip rate, concordancia | `metrics/variance.py` | Quanto a resposta muda entre execucoes identicas |
| Quebra por subgrupo | `metrics/subgroup.py` | Media global escondendo especialidade ou dificuldade ruim |
| Baseline versus perturbado | `perturbations.py` | Quanto do acerto vinha da forma, nao do conteudo |

## Perturbacoes

| Tipo | O que faz | Gabarito |
|---|---|---|
| `shuffle_choices` | Embaralha as alternativas | Reindexado para a nova letra |
| `add_distractor` | Acrescenta alternativa plausivel e errada | Inalterado |
| `negate_stem` | Nega o enunciado | Invertido se o item declarar `negated_answer`, senao vira abstencao |
| `unit_swap` | Troca a unidade sem converter o valor | Mantido se o item declarar `unit_swap_answer`, senao vira abstencao |
| `irrelevant_context` | Prefixa duas frases verdadeiras e irrelevantes | Inalterado |

Quando a perturbacao destroi a unicidade da resposta certa, o item nao e descartado nem
mantido com gabarito falso: ele passa a cobrar abstencao (`expected_abstain`), e um
modelo que responde com confianca a uma pergunta sem resposta conta como erro.

## Tres regras do projeto

**1. Nenhuma falha silenciosa.** Resposta nao parseada, item invalido e chamada com erro
aparecem no relatorio como categoria propria e continuam no denominador. O relatorio
mostra lado a lado a acuracia honesta e a acuracia que sairia se as falhas fossem
descartadas, para que o tamanho da diferenca fique explicito.

**2. Todo run e um artefato reproduzivel.** Cada execucao grava em `runs/<run_id>/` a
config resolvida (modelo, temperatura, seed, sha256 do dataset e do prompt) e as
predicoes brutas em JSONL append-only. `harness report <run_id>` regenera qualquer
analise sem tocar no modelo.

**3. Calibracao e implementada a mao.** ECE, MCE e Brier nao vem de biblioteca, e os
testes conferem os valores contra um caso calculado com papel e caneta
(`tests/test_calibration.py`). Bins vazios nao entram na media ponderada; predicoes sem
confianca declarada nao entram no calculo e sao contadas a parte.

## Sobre os fixtures

O provider `fixture` le respostas gravadas em `fixtures/*.jsonl` e nao faz nenhuma
chamada de rede. As respostas commitadas no repositorio sao **sinteticas**, geradas por
`scripts/record_fixtures.py --mode synthetic`, e cada linha carrega `"source":
"synthetic"`.

Elas existem para que o pipeline, as metricas e o relatorio possam ser demonstrados sem
depender de rede ou credencial. **Nao sao saida de um modelo real e nao devem ser
apresentadas como resultado de avaliacao.** Para gravar respostas reais:

```bash
uv run python scripts/record_fixtures.py --dataset data/seed_ptbr.jsonl --mode anthropic
```

## Estrutura

```
src/iamed_harness/
  cli.py            Typer: tasks, run, perturb, compare, report
  schemas.py        Pydantic: Item, Prediction, RunConfig, RunResult, metricas
  loader.py         JSONL validado linha a linha, falha com o numero da linha
  registry.py       tarefas em tasks/*.yaml, nunca em codigo
  runner.py         execucao assincrona, cache, retry com backoff, run_id
  cache.py          cache em disco por hash de provider, modelo, prompt, temp, seed, repeticao
  providers/        anthropic, openai (endpoint compativel), fixture (offline), echo
  graders/          exact_choice, regex, llm_judge
  metrics/          accuracy, calibration, variance, subgroup
  perturbations.py  cinco perturbacoes, cada uma com teste de gabarito
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
make fixtures   # so depois de mexer no dataset ou no template de prompt
```

Se o template de prompt ou o dataset mudarem, o hash do prompt muda e o provider
`fixture` passa a nao encontrar as respostas gravadas. O erro diz exatamente qual comando
regrava.

## Licenca

MIT.
