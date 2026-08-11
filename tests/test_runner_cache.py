from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from iamed_harness.cache import DiskCache, chave_cache
from iamed_harness.providers import ProviderError
from iamed_harness.runner import RunnerOptions, execute_run, load_predictions, load_run_config
from iamed_harness.schemas import SamplingPolicy, TaskSpec

DATASET = (
    '{"id": "r1", "question": "Pergunta um?", "choices": {"A": "a", "B": "b"}, "answer": "A"}\n'
    '{"id": "r2", "question": "Pergunta dois?", "choices": {"A": "a", "B": "b"}, "answer": "B"}\n'
)


@pytest.fixture
def ambiente(tmp_path: Path):
    dataset = tmp_path / "d.jsonl"
    dataset.write_text(DATASET, encoding="utf-8")
    prompt = tmp_path / "p.j2"
    prompt.write_text(
        "{{ item.question }}\n"
        "{% for l, t in item.choices | dictsort %}{{ l }}) {{ t }}\n{% endfor %}",
        encoding="utf-8",
    )
    spec = TaskSpec(
        name="t",
        description="teste",
        dataset=str(dataset),
        prompt=str(prompt),
        grader="exact_choice",
        sampling=SamplingPolicy(repetitions=1, temperature=0.0, max_tokens=64),
    )
    opts = RunnerOptions(
        provider="echo",
        model="modelo-de-teste",
        concurrency=2,
        runs_dir=tmp_path / "runs",
        cache_dir=str(tmp_path / "cache"),
        mostrar_progresso=False,
        backoff_base_s=0.0,
    )
    return spec, opts, tmp_path


def test_chave_de_cache_muda_com_a_repeticao():
    base: dict[str, Any] = dict(provider="p", model="m", prompt="x", temperature=1.0, seed=1)
    assert chave_cache(**base, repetition=0) != chave_cache(**base, repetition=1)


def test_chave_de_cache_muda_com_cada_componente():
    base: dict[str, Any] = dict(
        provider="p", model="m", prompt="x", temperature=0.0, seed=1, repetition=0
    )
    referencia = chave_cache(**base)
    assert chave_cache(**{**base, "model": "outro"}) != referencia
    assert chave_cache(**{**base, "prompt": "y"}) != referencia
    assert chave_cache(**{**base, "temperature": 1.0}) != referencia
    assert chave_cache(**{**base, "seed": 2}) != referencia


def test_cache_grava_e_le(tmp_path: Path):
    cache = DiskCache(tmp_path / "c")
    assert cache.get("k") is None
    cache.set("k", "resposta gravada")
    assert cache.get("k") == "resposta gravada"
    assert cache.hits == 1
    assert cache.misses == 1


def test_cache_desligado_nunca_devolve_nada(tmp_path: Path):
    cache = DiskCache(tmp_path / "c", enabled=False)
    cache.set("k", "v")
    assert cache.get("k") is None


def test_run_grava_config_e_predicoes(ambiente):
    spec, opts, tmp_path = ambiente
    config, predicoes, items = asyncio.run(execute_run(spec, opts))

    diretorio = tmp_path / "runs" / config.run_id
    assert (diretorio / "config.json").exists()
    assert (diretorio / "predictions.jsonl").exists()

    linhas = (diretorio / "predictions.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(linhas) == 2 == len(predicoes)
    assert len(items) == 2

    gravada = load_run_config(diretorio)
    assert gravada.model_dump() == config.model_dump()
    assert gravada.dataset_sha256 and gravada.prompt_sha256
    assert gravada.dataset_n_items == 2


def test_segunda_execucao_aproveita_o_cache(ambiente):
    spec, opts, _ = ambiente
    asyncio.run(execute_run(spec, opts))
    _, predicoes, _ = asyncio.run(execute_run(spec, opts))
    assert all(p.from_cache for p in predicoes)


def test_no_cache_forca_nova_chamada(ambiente):
    spec, opts, _ = ambiente
    asyncio.run(execute_run(spec, opts))
    sem_cache = RunnerOptions(**{**opts.__dict__, "use_cache": False})
    _, predicoes, _ = asyncio.run(execute_run(spec, sem_cache))
    assert not any(p.from_cache for p in predicoes)


def test_limit_reduz_o_numero_de_chamadas(ambiente):
    spec, opts, _ = ambiente
    limitado = RunnerOptions(**{**opts.__dict__, "limit": 1})
    config, predicoes, _ = asyncio.run(execute_run(spec, limitado))
    assert len(predicoes) == 1
    assert config.limit == 1
    assert config.dataset_n_items == 1


def test_repetitions_multiplica_as_chamadas(ambiente):
    spec, opts, _ = ambiente
    repetido = RunnerOptions(**{**opts.__dict__, "repetitions": 3, "temperature": 1.0})
    config, predicoes, _ = asyncio.run(execute_run(spec, repetido))
    assert len(predicoes) == 6
    assert sorted({p.repetition for p in predicoes}) == [0, 1, 2]
    assert config.repetitions == 3


def test_run_ids_diferem_quando_a_config_determinante_muda(ambiente):
    spec, opts, _ = ambiente
    a, _, _ = asyncio.run(execute_run(spec, opts))
    outro = RunnerOptions(**{**opts.__dict__, "model": "outro-modelo"})
    b, _, _ = asyncio.run(execute_run(spec, outro))
    assert a.run_id.split("-")[-1] != b.run_id.split("-")[-1]


def test_falha_do_provedor_vira_predicao_registrada_e_nao_some(ambiente, monkeypatch):
    spec, opts, tmp_path = ambiente

    async def sempre_falha(self, prompt, *, model, temperature, max_tokens, variant):
        raise ProviderError("falha de conexao simulada", retryable=False)

    from iamed_harness.providers.echo import EchoProvider

    monkeypatch.setattr(EchoProvider, "complete", sempre_falha)

    _, predicoes, _ = asyncio.run(execute_run(spec, opts))
    assert len(predicoes) == 2
    assert all(p.correct is None for p in predicoes)
    assert all(p.error and "simulada" in p.error for p in predicoes)


def test_erro_retryable_e_repetido_ate_o_limite(ambiente, monkeypatch):
    spec, opts, _ = ambiente
    tentativas = {"n": 0}

    async def falha_transitoria(self, prompt, *, model, temperature, max_tokens, variant):
        tentativas["n"] += 1
        raise ProviderError("rate limit simulado", retryable=True)

    from iamed_harness.providers.echo import EchoProvider

    monkeypatch.setattr(EchoProvider, "complete", falha_transitoria)

    _, predicoes, _ = asyncio.run(execute_run(spec, opts))
    assert tentativas["n"] == 6  # 2 itens x 3 tentativas
    assert all(p.error for p in predicoes)


def test_erro_de_parsing_nao_e_retentado(ambiente, monkeypatch):
    spec, opts, _ = ambiente
    chamadas = {"n": 0}

    async def responde_lixo(self, prompt, *, model, temperature, max_tokens, variant):
        chamadas["n"] += 1
        return "isso nao e json e nao tem letra de alternativa"

    from iamed_harness.providers.echo import EchoProvider

    monkeypatch.setattr(EchoProvider, "complete", responde_lixo)

    _, predicoes, _ = asyncio.run(execute_run(spec, opts))
    assert chamadas["n"] == 2
    assert all(p.correct is None and p.error for p in predicoes)


def test_predicoes_gravadas_sao_relidas_identicas(ambiente):
    spec, opts, tmp_path = ambiente
    config, predicoes, _ = asyncio.run(execute_run(spec, opts))
    relidas = load_predictions(tmp_path / "runs" / config.run_id)
    assert [p.model_dump() for p in relidas] == [p.model_dump() for p in predicoes]


def test_predictions_jsonl_e_uma_linha_por_chamada(ambiente):
    spec, opts, tmp_path = ambiente
    config, _, _ = asyncio.run(execute_run(spec, opts))
    caminho = tmp_path / "runs" / config.run_id / "predictions.jsonl"
    for linha in caminho.read_text(encoding="utf-8").strip().split("\n"):
        registro = json.loads(linha)
        assert registro["run_id"] == config.run_id
        assert "raw_response" in registro
