"""Provedor para qualquer endpoint compativel com a API de chat da OpenAI.

Serve para modelo local (llama.cpp, vLLM, Ollama em /v1) e para APIs hospedadas
que seguem o mesmo formato. Fala HTTP direto com httpx, sem SDK extra.

O padrao aponta para 127.0.0.1:8080/v1, porta padrao do llama-server. A chave
e opcional: servidor local costuma ignorar o cabecalho.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from . import ProviderError

BASE_URL_PADRAO = "http://127.0.0.1:8080/v1"
TIMEOUT_PADRAO_S = 300.0
"""Folgado de proposito: servidor local com um slot so enfileira as chamadas
concorrentes, e a ultima da fila espera todas as anteriores."""


class OpenAICompatProvider:
    name = "openai"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        timeout_s: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        url = base_url or os.environ.get("OPENAI_BASE_URL") or BASE_URL_PADRAO
        self._url = url.rstrip("/") + "/chat/completions"
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        timeout = timeout_s or float(os.environ.get("HARNESS_TIMEOUT_S") or TIMEOUT_PADRAO_S)
        self._timeout = httpx.Timeout(timeout)
        self._transport = transport

    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        variant: int,
    ) -> str:
        cabecalhos = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        corpo = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as cliente:
                resposta = await cliente.post(self._url, json=corpo, headers=cabecalhos)
        except httpx.TransportError as exc:
            raise ProviderError(
                f"falha de conexao com {self._url}: {exc!r}", retryable=True
            ) from exc

        if resposta.status_code != 200:
            retryable = resposta.status_code == 429 or resposta.status_code >= 500
            raise ProviderError(
                f"erro HTTP {resposta.status_code}: {resposta.text[:300]}", retryable=retryable
            )

        try:
            dados: dict[str, Any] = resposta.json()
            mensagem = dados["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"resposta fora do formato OpenAI: {resposta.text[:300]}") from exc

        conteudo = mensagem.get("content") or ""
        if not conteudo.strip():
            # Modelo com raciocinio pode gastar o max_tokens inteiro pensando e
            # devolver content vazio. Isso e falha do run, nao resposta em branco.
            if mensagem.get("reasoning_content"):
                raise ProviderError(
                    "resposta so com reasoning_content: o modelo esgotou max_tokens "
                    "pensando. Aumente max_tokens na tarefa ou desligue o raciocinio."
                )
            raise ProviderError("resposta sem conteudo de texto")
        return str(conteudo)
