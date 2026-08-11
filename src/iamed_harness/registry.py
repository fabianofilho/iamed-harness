"""Registro de tarefas: YAML em `tasks/`, nunca codigo.

Trocar de dataset, de prompt ou de politica de amostragem nao deveria exigir
deploy. Exige editar um arquivo de 10 linhas.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from .schemas import TaskSpec

TASKS_DIR = Path("tasks")


class TaskError(Exception):
    """Erro de definicao de tarefa, sempre com o arquivo que causou."""


def _parse_task(caminho: Path) -> TaskSpec:
    try:
        bruto = yaml.safe_load(caminho.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TaskError(f"{caminho}: YAML invalido: {exc}") from exc
    if not isinstance(bruto, dict):
        raise TaskError(f"{caminho}: esperado um mapeamento no topo do arquivo")
    try:
        spec = TaskSpec.model_validate(bruto)
    except ValidationError as exc:
        detalhes = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        raise TaskError(f"{caminho}: tarefa invalida: {detalhes}") from exc
    if spec.name != caminho.stem:
        raise TaskError(
            f"{caminho}: campo name={spec.name!r} nao bate com o nome do arquivo ({caminho.stem!r})"
        )
    return spec


def list_tasks(tasks_dir: str | Path = TASKS_DIR) -> list[TaskSpec]:
    diretorio = Path(tasks_dir)
    if not diretorio.exists():
        raise TaskError(f"diretorio de tarefas nao encontrado: {diretorio}")
    arquivos = sorted(diretorio.glob("*.yaml"))
    if not arquivos:
        raise TaskError(f"nenhuma tarefa em {diretorio}")
    return [_parse_task(a) for a in arquivos]


def get_task(name: str, tasks_dir: str | Path = TASKS_DIR) -> TaskSpec:
    caminho = Path(tasks_dir) / f"{name}.yaml"
    if not caminho.exists():
        disponiveis = ", ".join(t.name for t in list_tasks(tasks_dir))
        raise TaskError(f"tarefa {name!r} nao existe. Disponiveis: {disponiveis}")
    return _parse_task(caminho)
