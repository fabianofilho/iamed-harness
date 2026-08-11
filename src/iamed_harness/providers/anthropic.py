"""Provedor Anthropic. Unico caminho do harness que toca a rede."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from . import ProviderError

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str | None = None) -> None:
        chave = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not chave:
            raise ProviderError(
                "ANTHROPIC_API_KEY nao definida. Copie .env.example para .env, "
                "ou rode com --provider fixture para o modo offline."
            )
        self._api_key = chave
        self._cliente: AsyncAnthropic | None = None

    def _get_cliente(self) -> AsyncAnthropic:
        if self._cliente is None:
            from anthropic import AsyncAnthropic

            self._cliente = AsyncAnthropic(api_key=self._api_key)
        return self._cliente

    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        variant: int,
    ) -> str:
        from anthropic import APIConnectionError, APIStatusError, RateLimitError

        cliente = self._get_cliente()
        try:
            resposta = await cliente.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        except RateLimitError as exc:
            raise ProviderError(f"rate limit: {exc}", retryable=True) from exc
        except APIConnectionError as exc:
            raise ProviderError(f"falha de conexao: {exc}", retryable=True) from exc
        except APIStatusError as exc:
            retryable = exc.status_code >= 500
            raise ProviderError(f"erro HTTP {exc.status_code}: {exc}", retryable=retryable) from exc

        partes = [bloco.text for bloco in resposta.content if bloco.type == "text"]
        if not partes:
            raise ProviderError("resposta sem bloco de texto", retryable=False)
        return "".join(partes)
