"""Execucao de uma tarefa: assincrona, com cache, retry e artefato reproduzivel.

Todo run grava tres coisas em `runs/<run_id>/`:

    config.json        a config resolvida inteira, incluindo hashes
    predictions.jsonl  uma linha por chamada, append-only, erros incluidos
    report.html        gerado depois, a partir das duas de cima

Com esses arquivos, `harness report <run_id>` reconstroi qualquer analise sem
tocar no modelo.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Template
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .cache import DiskCache, chave_cache
from .graders import get_grader
from .loader import load_items, sha256_arquivo
from .providers import ProviderError, get_provider
from .schemas import Item, Prediction, RunConfig, TaskSpec

RUNS_DIR = Path("runs")
BACKOFF_BASE_S = 1.0
MAX_TENTATIVAS = 3


@dataclass
class RunnerOptions:
    provider: str = "fixture"
    model: str = "claude-sonnet-5"
    temperature: float | None = None
    repetitions: int | None = None
    limit: int | None = None
    concurrency: int = 8
    seed: int = 20260811
    use_cache: bool = True
    cache_dir: str | None = None
    runs_dir: Path = field(default_factory=lambda: RUNS_DIR)
    fixtures_dir: str = "fixtures"
    backoff_base_s: float = BACKOFF_BASE_S
    mostrar_progresso: bool = True


def _hash_texto(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def _montar_config(task: TaskSpec, opts: RunnerOptions, n_items: int) -> RunConfig:
    prompt_path = Path(task.prompt)
    if not prompt_path.exists():
        raise FileNotFoundError(f"template de prompt nao encontrado: {prompt_path}")

    temperatura = opts.temperature if opts.temperature is not None else task.sampling.temperature
    repeticoes = opts.repetitions if opts.repetitions is not None else task.sampling.repetitions

    determinantes = {
        "task": task.name,
        "dataset_sha256": sha256_arquivo(task.dataset),
        "prompt_sha256": _hash_texto(prompt_path.read_text(encoding="utf-8")),
        "grader": task.grader,
        "grader_options": task.grader_options,
        "provider": opts.provider,
        "model": opts.model,
        "temperature": temperatura,
        "max_tokens": task.sampling.max_tokens,
        "repetitions": repeticoes,
        "limit": opts.limit,
        "seed": opts.seed,
    }
    curto = _hash_texto(json.dumps(determinantes, sort_keys=True, ensure_ascii=False))[:8]
    agora = datetime.now(UTC)

    return RunConfig(
        run_id=f"{agora.strftime('%Y%m%dT%H%M%SZ')}-{curto}",
        created_at=agora.isoformat(),
        task_name=task.name,
        dataset_path=task.dataset,
        dataset_sha256=str(determinantes["dataset_sha256"]),
        dataset_n_items=n_items,
        prompt_path=str(prompt_path),
        prompt_sha256=str(determinantes["prompt_sha256"]),
        grader=task.grader,
        grader_options=task.grader_options,
        provider=opts.provider,
        model=opts.model,
        temperature=temperatura,
        max_tokens=task.sampling.max_tokens,
        repetitions=repeticoes,
        limit=opts.limit,
        concurrency=opts.concurrency,
        seed=opts.seed,
        subgroup_fields=task.subgroup_fields,
    )


async def _com_retry(
    coro_factory: object,
    *,
    tentativas: int,
    backoff_base_s: float,
) -> str:
    """Chama o provedor com backoff exponencial. So erros retryable sao repetidos."""
    ultimo: ProviderError | None = None
    for tentativa in range(1, tentativas + 1):
        try:
            return await coro_factory()  # type: ignore[operator,no-any-return]
        except ProviderError as exc:
            ultimo = exc
            if not exc.retryable or tentativa == tentativas:
                raise
            await asyncio.sleep(backoff_base_s * (2 ** (tentativa - 1)))
    raise ultimo if ultimo else ProviderError("falha desconhecida no provedor")


async def execute_run(
    task: TaskSpec, opts: RunnerOptions, *, console: Console | None = None
) -> tuple[RunConfig, list[Prediction], dict[str, Item]]:
    console = console or Console()

    todos = load_items(task.dataset)
    items = todos[: opts.limit] if opts.limit is not None else todos
    por_id = {i.id: i for i in items}

    config = _montar_config(task, opts, len(items))
    diretorio = opts.runs_dir / config.run_id
    diretorio.mkdir(parents=True, exist_ok=True)
    (diretorio / "config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")

    template = Template(Path(task.prompt).read_text(encoding="utf-8"), keep_trailing_newline=True)
    provider = get_provider(opts.provider, fixtures_dir=opts.fixtures_dir)
    grader = get_grader(task.grader, task.grader_options, provider=provider, model=opts.model)
    cache = DiskCache(opts.cache_dir, enabled=opts.use_cache)

    trabalhos = [(item, r) for r in range(config.repetitions) for item in items]
    semaforo = asyncio.Semaphore(opts.concurrency)
    lock_escrita = asyncio.Lock()
    arquivo_predicoes = diretorio / "predictions.jsonl"
    arquivo_predicoes.touch()
    predicoes: list[Prediction] = []
    acertos = 0

    async def processar(item: Item, repeticao: int) -> Prediction:
        nonlocal acertos
        prompt = template.render(item=item)
        chave = chave_cache(
            provider=opts.provider,
            model=opts.model,
            prompt=prompt,
            temperature=config.temperature,
            seed=opts.seed,
            repetition=repeticao,
        )

        async with semaforo:
            inicio = time.perf_counter()
            cacheada = cache.get(chave)
            veio_do_cache = cacheada is not None
            erro_provedor: str | None = None
            bruto = cacheada or ""

            if not veio_do_cache:
                try:
                    bruto = await _com_retry(
                        lambda: provider.complete(
                            prompt,
                            model=opts.model,
                            temperature=config.temperature,
                            max_tokens=config.max_tokens,
                            variant=repeticao,
                        ),
                        tentativas=MAX_TENTATIVAS,
                        backoff_base_s=opts.backoff_base_s,
                    )
                    cache.set(chave, bruto)
                except ProviderError as exc:
                    erro_provedor = str(exc)

            if erro_provedor is None:
                resultado = await grader.grade(item, bruto)
                predicao = Prediction(
                    item_id=item.id,
                    run_id=config.run_id,
                    repetition=repeticao,
                    raw_response=bruto,
                    parsed_choice=resultado.parsed_choice,
                    confidence=resultado.confidence,
                    correct=resultado.correct,
                    latency_ms=(time.perf_counter() - inicio) * 1000,
                    error=resultado.error,
                    from_cache=veio_do_cache,
                )
            else:
                predicao = Prediction(
                    item_id=item.id,
                    run_id=config.run_id,
                    repetition=repeticao,
                    raw_response="",
                    parsed_choice=None,
                    confidence=None,
                    correct=None,
                    latency_ms=(time.perf_counter() - inicio) * 1000,
                    error=erro_provedor,
                    from_cache=False,
                )

        async with lock_escrita:
            with arquivo_predicoes.open("a", encoding="utf-8") as f:
                f.write(predicao.model_dump_json() + "\n")
            predicoes.append(predicao)
            if predicao.correct:
                acertos += 1
        return predicao

    if opts.mostrar_progresso:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(complete_style="green", finished_style="green"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progresso:
            barra = progresso.add_task(f"{task.name}  acertos 0", total=len(trabalhos))
            pendentes = [asyncio.create_task(processar(i, r)) for i, r in trabalhos]
            for concluida in asyncio.as_completed(pendentes):
                await concluida
                feitos = len(predicoes)
                pct = 100 * acertos / feitos if feitos else 0.0
                progresso.update(
                    barra,
                    advance=1,
                    description=f"{task.name}  acertos {acertos}/{feitos} ({pct:.0f}%)",
                )
    else:
        await asyncio.gather(*(processar(i, r) for i, r in trabalhos))

    predicoes.sort(key=lambda p: (p.item_id, p.repetition))
    return config, predicoes, por_id


def load_predictions(run_dir: Path) -> list[Prediction]:
    caminho = run_dir / "predictions.jsonl"
    if not caminho.exists():
        raise FileNotFoundError(f"predicoes nao encontradas: {caminho}")
    predicoes = [
        Prediction.model_validate_json(linha)
        for linha in caminho.read_text(encoding="utf-8").splitlines()
        if linha.strip()
    ]
    predicoes.sort(key=lambda p: (p.item_id, p.repetition))
    return predicoes


def load_run_config(run_dir: Path) -> RunConfig:
    caminho = run_dir / "config.json"
    if not caminho.exists():
        raise FileNotFoundError(f"config do run nao encontrada: {caminho}")
    return RunConfig.model_validate_json(caminho.read_text(encoding="utf-8"))


def resolve_run_dir(run_id: str, runs_dir: Path = RUNS_DIR) -> Path:
    direto = runs_dir / run_id
    if direto.exists():
        return direto
    candidatos = sorted(runs_dir.glob(f"{run_id}*"))
    if len(candidatos) == 1:
        return candidatos[0]
    if not candidatos:
        raise FileNotFoundError(f"run {run_id!r} nao encontrado em {runs_dir}")
    nomes = ", ".join(c.name for c in candidatos)
    raise ValueError(f"prefixo {run_id!r} e ambiguo: {nomes}")
