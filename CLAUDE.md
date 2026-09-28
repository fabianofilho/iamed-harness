## Antes de Escrever Código
- Leia todos os arquivos relevantes primeiro. Nunca edite no escuro.
- Entenda o requisito completo antes de escrever qualquer coisa.

## Enquanto Escreve Código
- Teste após escrever. Nunca deixe código sem testar.
- Corrija erros antes de seguir em frente. Nunca ignore falhas.
- Prefira editar a reescrever arquivos inteiros.
- Solução mais simples que funcione. Nada de engenharia excessiva.

## Antes de Declarar Concluído
- Execute o código uma última vez para confirmar que funciona.
- Nunca declare concluído sem um teste passando.

## Saída
- Sem aberturas bajuladoras ou enrolação no final.
- Seja conciso. Se não tiver certeza, diga. Nunca chute.

## Português
- Escreva português correto, com acentos e cedilhas normalmente (ã, á, ç, é, í, ó, ú, etc).
- Acentuação Unicode é permitida e esperada. A restrição abaixo vale só para pontuação decorativa.

## Humanização (vale para TUDO: texto, código, arquivos, MCPs, Notion, emails)
- NUNCA usar travessões (em dash, en dash). Usar vírgula, ponto ou reestruturar a frase.
- NUNCA usar hífens decorativos como separador em títulos ou nomes. Usar dois pontos ou vírgula.
- NUNCA usar aspas tipográficas (curly quotes). Apenas aspas retas " e '.
- NUNCA usar reticências Unicode. Usar três pontos normais ...
- Esta regra se aplica a TODO output: respostas, arquivos, títulos, conteúdo enviado a MCPs, emails, Notion, qualquer destino.

## Sobre este projeto

Harness mínimo e auditável para avaliar LLMs em tarefas clínicas de múltipla
escolha. O objetivo não é maximizar acurácia: é tornar visível, em poucos
comandos, o que a acurácia esconde, ou seja, variância entre execuções
idênticas, excesso de confiança (calibração) e fragilidade a perturbações
semânticas.

Será usado ao vivo numa sessão de uma hora. Cada comando precisa rodar em
segundos e falhar de forma legível.

### Três regras invioláveis

1. **`make run` funciona sem API key e sem internet, logo após o clone.** É o
   requisito mais importante do repositório. Se ele falhar, a sessão morre nos
   primeiros cinco minutos. O provider `fixture` não pode fazer nenhuma chamada
   de rede, nem para checar credencial.
2. **Nenhuma falha silenciosa.** Resposta não parseada, item inválido e chamada
   com erro aparecem no relatório como categoria própria e nunca saem do
   denominador. Se a acurácia sobe porque erros sumiram, o harness está mentindo.
3. **Métricas de calibração são implementadas à mão e testadas contra valores
   calculados manualmente.** Não importar de biblioteca: a fórmula precisa estar
   visível na tela durante a aula.

### Convenções

- Tipagem estática obrigatória em todos os módulos (`mypy strict`).
- Tarefas em `tasks/*.yaml`, nunca em código.
- Todo run grava `config.json` (config resolvida, com sha256 do dataset e do
  prompt) e `predictions.jsonl` append-only. `harness report <run_id>` regenera
  qualquer análise sem tocar no modelo.
- Fixtures commitados são sintéticos e carregam `"source": "synthetic"`. Não
  apresentar como resultado de avaliação de modelo real.
- Commits pequenos e nomeados por camada (loader, runner, graders, metrics,
  perturbations, report), para que o histórico possa ser percorrido ao vivo como
  roteiro da aula.

## Stack
- Python 3.11+, uv (lockfile commitado)
- Typer + Rich (CLI), Pydantic v2 (contratos), Jinja2 (prompts e relatório)
- numpy + pandas, matplotlib. Sem scikit-learn: as métricas de calibração são à mão
- Anthropic SDK e httpx atrás da interface `Provider` (anthropic, openai, fixture, echo)
- pytest, ruff, mypy

## Comandos úteis
- `make install` - uv sync com extras de dev
- `make run` - baseline offline, comando de abertura da sessão
- `make demo` - baseline, perturbado, compare e os dois relatórios
- `make test` - pytest
- `make lint` - ruff check + mypy
- `make fixtures` - regrava os fixtures offline (só após mexer no dataset ou no prompt)
