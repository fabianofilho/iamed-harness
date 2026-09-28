"""Provedor OpenAI-compativel testado sem rede, via httpx.MockTransport."""

from __future__ import annotations

import json

import httpx
import pytest

from iamed_harness.providers import ProviderError, get_provider
from iamed_harness.providers.openai_compat import OpenAICompatProvider

RESPOSTA_OK = '{"answer": "B", "confidence": 90, "reasoning": "uma frase"}'


def _provider(handler, **kwargs) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        base_url="http://modelo.local/v1/", transport=httpx.MockTransport(handler), **kwargs
    )


async def _chamar(provider: OpenAICompatProvider) -> str:
    return await provider.complete("pergunta", model="m", temperature=0.0, max_tokens=64, variant=0)


async def test_devolve_content_e_manda_o_corpo_certo():
    visto: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        visto["url"] = str(req.url)
        visto["auth"] = req.headers.get("authorization")
        visto["corpo"] = json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": RESPOSTA_OK}}]})

    assert await _chamar(_provider(handler, api_key="k")) == RESPOSTA_OK
    assert visto["url"] == "http://modelo.local/v1/chat/completions"
    assert visto["auth"] == "Bearer k"
    corpo = visto["corpo"]
    assert isinstance(corpo, dict)
    assert corpo["model"] == "m" and corpo["max_tokens"] == 64 and corpo["temperature"] == 0.0
    assert corpo["messages"] == [{"role": "user", "content": "pergunta"}]


async def test_sem_chave_nao_manda_authorization(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def handler(req: httpx.Request) -> httpx.Response:
        assert "authorization" not in req.headers
        return httpx.Response(200, json={"choices": [{"message": {"content": RESPOSTA_OK}}]})

    assert await _chamar(_provider(handler)) == RESPOSTA_OK


@pytest.mark.parametrize(("status", "retryable"), [(429, True), (503, True), (400, False)])
async def test_erro_http_marca_retryable_pelo_status(status, retryable):
    provider = _provider(lambda req: httpx.Response(status, text="falhou"))
    with pytest.raises(ProviderError) as exc:
        await _chamar(provider)
    assert exc.value.retryable is retryable


async def test_content_vazio_com_raciocinio_vira_erro_explicito():
    def handler(req: httpx.Request) -> httpx.Response:
        mensagem = {"content": "", "reasoning_content": "pensando..."}
        return httpx.Response(200, json={"choices": [{"message": mensagem}]})

    with pytest.raises(ProviderError, match="reasoning_content"):
        await _chamar(_provider(handler))


async def test_falha_de_conexao_e_retryable():
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("recusada", request=req)

    with pytest.raises(ProviderError) as exc:
        await _chamar(_provider(handler))
    assert exc.value.retryable is True


def test_variaveis_vazias_do_env_caem_no_padrao(monkeypatch):
    # .env copiado do .env.example chega com as variaveis definidas e vazias
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    monkeypatch.setenv("HARNESS_TIMEOUT_S", "")
    provider = OpenAICompatProvider()
    assert provider._url == "http://127.0.0.1:8080/v1/chat/completions"
    assert provider._timeout.read == 300.0


def test_fabrica_le_base_url_do_ambiente(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://outro:9999/v1")
    provider = get_provider("openai")
    assert isinstance(provider, OpenAICompatProvider)
    assert provider._url == "http://outro:9999/v1/chat/completions"
