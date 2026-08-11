from __future__ import annotations

import pytest

from iamed_harness.loader import DatasetError, load_items, sha256_arquivo, write_items

LINHA_VALIDA = (
    '{"id": "x1", "question": "Pergunta?", '
    '"choices": {"A": "primeira", "B": "segunda"}, "answer": "A"}'
)


def _escrever(tmp_path, *linhas):
    caminho = tmp_path / "d.jsonl"
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return caminho


def test_carrega_itens_validos(tmp_path):
    caminho = _escrever(tmp_path, LINHA_VALIDA)
    items = load_items(caminho)
    assert len(items) == 1
    assert items[0].id == "x1"
    assert items[0].expected_abstain is False


def test_json_malformado_reporta_o_numero_da_linha(tmp_path):
    caminho = _escrever(tmp_path, LINHA_VALIDA, "{isso nao e json}")
    with pytest.raises(DatasetError, match=r":2 JSON invalido"):
        load_items(caminho)


def test_gabarito_fora_das_alternativas_reporta_a_linha(tmp_path):
    ruim = '{"id": "x2", "question": "P?", "choices": {"A": "a", "B": "b"}, "answer": "Z"}'
    caminho = _escrever(tmp_path, LINHA_VALIDA, ruim)
    with pytest.raises(DatasetError, match=r":2 item invalido"):
        load_items(caminho)


def test_id_duplicado_e_rejeitado(tmp_path):
    caminho = _escrever(tmp_path, LINHA_VALIDA, LINHA_VALIDA)
    with pytest.raises(DatasetError, match=r":2 id duplicado"):
        load_items(caminho)


def test_uma_unica_alternativa_e_rejeitada(tmp_path):
    ruim = '{"id": "x3", "question": "P?", "choices": {"A": "a"}, "answer": "A"}'
    caminho = _escrever(tmp_path, ruim)
    with pytest.raises(DatasetError, match="pelo menos 2 alternativas"):
        load_items(caminho)


def test_dataset_vazio_e_rejeitado(tmp_path):
    caminho = tmp_path / "vazio.jsonl"
    caminho.write_text("\n\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="dataset vazio"):
        load_items(caminho)


def test_arquivo_inexistente_e_rejeitado(tmp_path):
    with pytest.raises(DatasetError, match="nao encontrado"):
        load_items(tmp_path / "nao-existe.jsonl")


def test_expected_abstain_exige_answer_abstain(tmp_path):
    ruim = (
        '{"id": "x4", "question": "P?", "choices": {"A": "a", "B": "b"}, '
        '"answer": "A", "expected_abstain": true}'
    )
    caminho = _escrever(tmp_path, ruim)
    with pytest.raises(DatasetError, match="expected_abstain"):
        load_items(caminho)


def test_ida_e_volta_preserva_os_itens(tmp_path):
    caminho = _escrever(tmp_path, LINHA_VALIDA)
    items = load_items(caminho)
    destino = write_items(items, tmp_path / "saida.jsonl")
    assert load_items(destino) == items


def test_dataset_do_repositorio_e_valido():
    items = load_items("data/seed_ptbr.jsonl")
    assert len(items) == 25
    assert all(i.answer in i.choices for i in items)
    assert all("especialidade" in i.metadata for i in items)


def test_sha256_muda_quando_o_conteudo_muda(tmp_path):
    a = _escrever(tmp_path, LINHA_VALIDA)
    antes = sha256_arquivo(a)
    a.write_text(a.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert sha256_arquivo(a) != antes
