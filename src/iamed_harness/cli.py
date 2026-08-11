"""Interface de linha de comando. O repositorio inteiro e operado por aqui."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .loader import DatasetError, load_items, write_items
from .metrics import summarize
from .perturbations import PERTURBACOES, apply_perturbation
from .providers import ProviderError
from .registry import TaskError, get_task, list_tasks
from .report import build_report
from .runner import (
    RUNS_DIR,
    RunnerOptions,
    execute_run,
    load_predictions,
    load_run_config,
    resolve_run_dir,
)
from .schemas import Prediction, RunResult

load_dotenv()

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Harness minimo e auditavel para avaliar LLMs em tarefas clinicas.",
)
console = Console()

VERDE = "bold green"
VERMELHO = "bold red"


def _erro(mensagem: str) -> None:
    console.print(Panel(mensagem, title="erro", border_style="red"))
    raise typer.Exit(code=1)


def _resumo_terminal(result: RunResult) -> None:
    a = result.accuracy
    c = result.calibration

    tabela = Table(title=f"{result.config.task_name} :: {result.config.run_id}", box=None)
    tabela.add_column("metrica", style="green")
    tabela.add_column("valor", justify="right")
    tabela.add_column("nota", style="dim")

    tabela.add_row(
        "acuracia",
        f"{a.accuracy:.1%}",
        f"IC 95% {a.ci_low:.1%} a {a.ci_high:.1%}  ({a.n_correct}/{a.n_total})",
    )
    tabela.add_row(
        "acuracia sem as falhas",
        f"{a.accuracy_parsed_only:.1%}",
        "numero inflado, so para contraste",
    )
    tabela.add_row(
        "falhas",
        str(a.n_failed),
        "seguem no denominador" if a.n_failed else "nenhuma",
        style=VERMELHO if a.n_failed else "",
    )
    tabela.add_row("ECE uniforme", f"{c.ece_uniform:.3f}", "10 bins de largura fixa")
    tabela.add_row(
        "ECE por quantil", f"{c.ece_quantile:.3f}", f"{len(c.bins_quantile)} bins efetivos"
    )
    tabela.add_row("MCE", f"{c.mce:.3f}", "pior bin nao vazio")
    tabela.add_row("Brier", f"{c.brier:.3f}", f"{c.n_scored} predicoes pontuadas")

    if result.variance:
        v = result.variance
        tabela.add_row(
            "flip rate",
            f"{v.flip_rate:.1%}",
            f"{len(v.itens_instaveis)}/{v.n_items} itens mudaram entre {v.repetitions} execucoes",
            style=VERMELHO if v.flip_rate > 0 else "",
        )
        tabela.add_row("desvio da acuracia", f"{v.accuracy_std:.3f}", "entre repeticoes identicas")

    console.print()
    console.print(tabela)

    if result.erros_por_tipo:
        console.print()
        console.print(
            "[dim]erros por tipo:[/dim] "
            + ", ".join(f"{k}={v}" for k, v in result.erros_por_tipo.items())
        )


@app.command("tasks")
def cmd_tasks() -> None:
    """Lista as tarefas registradas em tasks/."""
    try:
        tarefas = list_tasks()
    except TaskError as exc:
        _erro(str(exc))
        return

    tabela = Table(title="tarefas registradas", box=None)
    tabela.add_column("nome", style=VERDE)
    tabela.add_column("dataset")
    tabela.add_column("grader")
    tabela.add_column("rep", justify="right")
    tabela.add_column("temp", justify="right")
    tabela.add_column("descricao", style="dim")
    for t in tarefas:
        tabela.add_row(
            t.name,
            t.dataset,
            t.grader,
            str(t.sampling.repetitions),
            f"{t.sampling.temperature:g}",
            t.description,
        )
    console.print(tabela)


@app.command("run")
def cmd_run(
    task: Annotated[str, typer.Argument(help="nome da tarefa (ver: harness tasks)")],
    provider: Annotated[
        str, typer.Option("--provider", help="anthropic, fixture ou echo")
    ] = os.environ.get("HARNESS_PROVIDER", "fixture"),
    model: Annotated[str, typer.Option("--model")] = os.environ.get(
        "HARNESS_MODEL", "claude-sonnet-5"
    ),
    limit: Annotated[
        int | None, typer.Option("--limit", help="roda apenas os N primeiros itens")
    ] = None,
    repetitions: Annotated[
        int | None, typer.Option("--repetitions", help="K execucoes identicas por item")
    ] = None,
    temperature: Annotated[float | None, typer.Option("--temperature")] = None,
    concurrency: Annotated[int, typer.Option("--concurrency")] = int(
        os.environ.get("HARNESS_CONCURRENCY", "8")
    ),
    seed: Annotated[int, typer.Option("--seed")] = 20260811,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="ignora o cache em disco")] = False,
    no_report: Annotated[bool, typer.Option("--no-report", help="nao gera o HTML")] = False,
) -> None:
    """Executa uma tarefa e grava o run completo em runs/<run_id>/."""
    try:
        spec = get_task(task)
    except TaskError as exc:
        _erro(str(exc))
        return

    opts = RunnerOptions(
        provider=provider,
        model=model,
        temperature=temperature,
        repetitions=repetitions,
        limit=limit,
        concurrency=concurrency,
        seed=seed,
        use_cache=not no_cache,
    )

    try:
        config, predicoes, items = asyncio.run(execute_run(spec, opts, console=console))
    except (DatasetError, ProviderError, FileNotFoundError) as exc:
        _erro(str(exc))
        return

    resultado = summarize(config, predicoes, items)
    _resumo_terminal(resultado)

    diretorio = RUNS_DIR / config.run_id
    (diretorio / "result.json").write_text(resultado.model_dump_json(indent=2), encoding="utf-8")

    if not no_report:
        caminho = build_report(resultado, predicoes, diretorio / "report.html")
        console.print(f"\n[green]relatorio:[/green] {caminho}")
    console.print(f"[green]run_id:[/green] {config.run_id}")


@app.command("perturb")
def cmd_perturb(
    dataset: Annotated[str, typer.Argument(help="caminho do JSONL de entrada")],
    kind: Annotated[
        str, typer.Option("--kind", help=f"uma de: {', '.join(sorted(PERTURBACOES))}")
    ] = "shuffle_choices",
    out: Annotated[Path, typer.Option("--out", help="caminho do JSONL de saida")] = Path(
        "data/perturbed.jsonl"
    ),
    seed: Annotated[int, typer.Option("--seed")] = 20260811,
) -> None:
    """Gera uma variante perturbada de um dataset."""
    try:
        items = load_items(dataset)
        perturbados = apply_perturbation(kind, items, seed=seed)
    except (DatasetError, ValueError) as exc:
        _erro(str(exc))
        return

    write_items(perturbados, out)
    n_abstain = sum(1 for i in perturbados if i.expected_abstain)
    pulados = len(items) - len(perturbados)

    console.print(
        f"[green]{len(perturbados)}[/green] itens escritos em {out}  (perturbacao: {kind})"
    )
    if n_abstain:
        console.print(
            f"[yellow]{n_abstain}[/yellow] item(ns) viraram expected_abstain: a perturbacao "
            "destruiu a unicidade do gabarito e a resposta certa passou a ser recusar."
        )
    if pulados:
        console.print(
            f"[yellow]{pulados}[/yellow] item(ns) nao foram perturbados por nao terem "
            "alvo aplicavel."
        )


def _carregar_resultado(run_id: str) -> tuple[RunResult, list[Prediction]]:
    diretorio = resolve_run_dir(run_id)
    config = load_run_config(diretorio)
    predicoes = load_predictions(diretorio)
    items = {i.id: i for i in load_items(config.dataset_path)}
    return summarize(config, predicoes, items), predicoes


@app.command("report")
def cmd_report(
    run_id: Annotated[str, typer.Argument(help="run_id ou prefixo")],
    baseline: Annotated[str | None, typer.Option("--baseline", help="run_id de comparacao")] = None,
) -> None:
    """Regenera o HTML a partir das predicoes gravadas, sem chamar o modelo."""
    try:
        resultado, predicoes = _carregar_resultado(run_id)
        comparacao = _carregar_resultado(baseline)[0] if baseline else None
    except (FileNotFoundError, ValueError, DatasetError) as exc:
        _erro(str(exc))
        return

    _resumo_terminal(resultado)
    diretorio = resolve_run_dir(run_id)
    caminho = build_report(resultado, predicoes, diretorio / "report.html", comparacao=comparacao)
    console.print(f"\n[green]relatorio:[/green] {caminho}")


def _chave_item(item_id: str) -> str:
    return item_id.split("::", 1)[0]


@app.command("compare")
def cmd_compare(
    run_a: Annotated[str, typer.Argument(help="run_id de referencia")],
    run_b: Annotated[str, typer.Argument(help="run_id a comparar")],
) -> None:
    """Diff de metricas lado a lado, com destaque para itens que mudaram de status."""
    try:
        res_a, pred_a = _carregar_resultado(run_a)
        res_b, pred_b = _carregar_resultado(run_b)
    except (FileNotFoundError, ValueError, DatasetError) as exc:
        _erro(str(exc))
        return

    tabela = Table(
        title=f"{res_a.config.task_name} ({res_a.config.run_id})  versus  "
        f"{res_b.config.task_name} ({res_b.config.run_id})",
        box=None,
    )
    tabela.add_column("metrica", style="green")
    tabela.add_column("A", justify="right")
    tabela.add_column("B", justify="right")
    tabela.add_column("delta", justify="right")

    def linha(rotulo: str, va: float, vb: float, fmt: str, inverter: bool = False) -> None:
        delta = vb - va
        melhor = delta < 0 if inverter else delta > 0
        estilo = "" if abs(delta) < 1e-9 else (VERDE if melhor else VERMELHO)
        tabela.add_row(rotulo, format(va, fmt), format(vb, fmt), f"{delta:+{fmt}}", style=estilo)

    linha("acuracia", res_a.accuracy.accuracy, res_b.accuracy.accuracy, ".1%")
    linha(
        "acuracia sem falhas",
        res_a.accuracy.accuracy_parsed_only,
        res_b.accuracy.accuracy_parsed_only,
        ".1%",
    )
    linha("ECE uniforme", res_a.calibration.ece_uniform, res_b.calibration.ece_uniform, ".3f", True)
    linha("MCE", res_a.calibration.mce, res_b.calibration.mce, ".3f", True)
    linha("Brier", res_a.calibration.brier, res_b.calibration.brier, ".3f", True)
    linha("falhas", float(res_a.accuracy.n_failed), float(res_b.accuracy.n_failed), ".0f", True)

    console.print()
    console.print(tabela)

    status_a = {_chave_item(p.item_id): bool(p.correct) for p in pred_a if p.repetition == 0}
    status_b = {_chave_item(p.item_id): bool(p.correct) for p in pred_b if p.repetition == 0}
    comuns = sorted(set(status_a) & set(status_b))

    quebrou = [k for k in comuns if status_a[k] and not status_b[k]]
    consertou = [k for k in comuns if not status_a[k] and status_b[k]]

    if not comuns:
        console.print(
            "\n[yellow]os dois runs nao compartilham itens comparaveis "
            "(ids diferentes apos remover o sufixo de perturbacao).[/yellow]"
        )
        return

    console.print(
        f"\n[dim]{len(comuns)} itens comparaveis  "
        f"({len(quebrou)} quebraram, {len(consertou)} passaram a acertar)[/dim]"
    )
    if quebrou:
        console.print(f"[red]acertava em A e errou em B:[/red] {', '.join(quebrou)}")
    if consertou:
        console.print(f"[green]errava em A e acertou em B:[/green] {', '.join(consertou)}")
    if not quebrou and not consertou:
        console.print("[green]nenhum item mudou de status.[/green]")


if __name__ == "__main__":
    app()
