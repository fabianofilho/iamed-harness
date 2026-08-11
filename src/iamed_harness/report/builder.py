"""Geracao do relatorio HTML.

Autocontido de proposito: CSS embutido, imagens em base64, nenhum CDN. O
arquivo precisa abrir num notebook sem internet, no meio de uma sessao, sem
depender de nada externo.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from ..schemas import CalibrationMetrics, Prediction, RunResult  # noqa: E402

TEMPLATES_DIR = Path(__file__).parent / "templates"

FUNDO = "#050807"
PAINEL = "#0d1512"
ESMERALDA = "#10b981"
ESMERALDA_CLARA = "#6ee7b7"
ALERTA = "#f87171"
TEXTO = "#d1fae5"
GRADE = "#1f2f29"


def _figura_para_base64(fig: Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, facecolor=FUNDO, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _estilizar(ax: Axes) -> None:
    ax.set_facecolor(PAINEL)
    ax.tick_params(colors=TEXTO, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRADE)
    ax.xaxis.label.set_color(TEXTO)
    ax.yaxis.label.set_color(TEXTO)
    ax.title.set_color(TEXTO)
    ax.grid(True, color=GRADE, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def reliability_diagram(calibration: CalibrationMetrics) -> str:
    fig, ax = plt.subplots(figsize=(6.2, 4.4), facecolor=FUNDO)
    _estilizar(ax)

    bins = [b for b in calibration.bins_uniform if b.n > 0]
    if bins:
        centros = [(b.lower + b.upper) / 2 for b in bins]
        largura = min(b.upper - b.lower for b in calibration.bins_uniform) * 0.85
        ax.bar(
            centros,
            [b.mean_accuracy for b in bins],
            width=largura,
            color=ESMERALDA,
            edgecolor=ESMERALDA_CLARA,
            linewidth=0.8,
            label="acuracia observada",
        )
        ax.plot(
            centros,
            [b.mean_confidence for b in bins],
            "o--",
            color=ALERTA,
            markersize=5,
            linewidth=1.4,
            label="confianca declarada",
        )
    ax.plot([0, 1], [0, 1], color=TEXTO, linewidth=1.0, alpha=0.45, label="calibracao perfeita")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confianca")
    ax.set_ylabel("acuracia")
    ax.set_title(f"Reliability diagram  (ECE uniforme = {calibration.ece_uniform:.3f})")
    legenda = ax.legend(facecolor=PAINEL, edgecolor=GRADE, fontsize=8)
    for texto in legenda.get_texts():
        texto.set_color(TEXTO)
    return _figura_para_base64(fig)


def variance_curve(result: RunResult) -> str | None:
    if result.variance is None:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 3.6), facecolor=FUNDO)
    _estilizar(ax)

    repeticoes = [r.repetition + 1 for r in result.variance.accuracy_por_repeticao]
    acuracias = [r.accuracy for r in result.variance.accuracy_por_repeticao]
    media = sum(acuracias) / len(acuracias)

    ax.plot(repeticoes, acuracias, "o-", color=ESMERALDA, linewidth=1.8, markersize=7)
    ax.axhline(media, color=ALERTA, linestyle="--", linewidth=1.2, label=f"media {media:.3f}")
    ax.set_xticks(repeticoes)
    ax.set_xlabel("repeticao (mesmo prompt, mesma config)")
    ax.set_ylabel("acuracia")
    ax.set_title(
        f"Variancia entre execucoes identicas  "
        f"(desvio {result.variance.accuracy_std:.3f}, flip rate {result.variance.flip_rate:.1%})"
    )
    legenda = ax.legend(facecolor=PAINEL, edgecolor=GRADE, fontsize=8)
    for texto in legenda.get_texts():
        texto.set_color(TEXTO)
    return _figura_para_base64(fig)


def comparison_chart(baseline: RunResult, outro: RunResult) -> str:
    fig, ax = plt.subplots(figsize=(6.2, 3.6), facecolor=FUNDO)
    _estilizar(ax)

    rotulos = ["acuracia", "ECE uniforme", "taxa de falha"]
    valores_a = [
        baseline.accuracy.accuracy,
        baseline.calibration.ece_uniform,
        baseline.accuracy.n_failed / max(1, baseline.accuracy.n_total),
    ]
    valores_b = [
        outro.accuracy.accuracy,
        outro.calibration.ece_uniform,
        outro.accuracy.n_failed / max(1, outro.accuracy.n_total),
    ]
    x = range(len(rotulos))
    largura = 0.36
    ax.bar(
        [i - largura / 2 for i in x],
        valores_a,
        width=largura,
        color=ESMERALDA,
        label=baseline.config.task_name,
    )
    ax.bar(
        [i + largura / 2 for i in x],
        valores_b,
        width=largura,
        color=ALERTA,
        label=outro.config.task_name,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(rotulos)
    ax.set_ylabel("valor")
    ax.set_title("Comparacao entre runs")
    legenda = ax.legend(facecolor=PAINEL, edgecolor=GRADE, fontsize=8)
    for texto in legenda.get_texts():
        texto.set_color(TEXTO)
    return _figura_para_base64(fig)


def build_report(
    result: RunResult,
    predictions: list[Prediction],
    saida: Path,
    *,
    comparacao: RunResult | None = None,
) -> Path:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html.j2")

    falhas = [p for p in predictions if p.failed][:20]

    html = template.render(
        r=result,
        config=result.config,
        reliability_png=reliability_diagram(result.calibration),
        variance_png=variance_curve(result),
        comparison_png=comparison_chart(comparacao, result) if comparacao else None,
        comparacao=comparacao,
        falhas=falhas,
        n_falhas=result.accuracy.n_failed,
    )
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(html, encoding="utf-8")
    return saida
