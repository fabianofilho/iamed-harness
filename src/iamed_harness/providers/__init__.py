"""Interface de provedor e fabrica.

Tres implementacoes: `anthropic` (rede), `fixture` (respostas gravadas, zero
rede) e `echo` (deterministico, para teste). A sessao ao vivo roda em
`fixture`, entao nada no caminho critico pode depender de credencial.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class ProviderError(Exception):
    """Falha de provedor. Erros de rede sao retentaveis, os demais nao."""

    def __init__(self, mensagem: str, *, retryable: bool = False) -> None:
        super().__init__(mensagem)
        self.retryable = retryable


@runtime_checkable
class Provider(Protocol):
    name: str

    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        variant: int,
    ) -> str:
        """Devolve o texto bruto da resposta.

        `variant` e o indice da repeticao. Provedores deterministicos usam ele
        para escolher entre respostas gravadas distintas do mesmo prompt.
        """
        ...


def get_provider(name: str, **kwargs: object) -> Provider:
    from .anthropic import AnthropicProvider
    from .echo import EchoProvider
    from .fixture import FixtureProvider

    if name == "anthropic":
        return AnthropicProvider()
    if name == "fixture":
        caminho = kwargs.get("fixtures_dir")
        return FixtureProvider(str(caminho) if caminho is not None else "fixtures")
    if name == "echo":
        return EchoProvider()
    raise ProviderError(f"provider desconhecido: {name!r}. Use anthropic, fixture ou echo.")


__all__ = ["Provider", "ProviderError", "get_provider"]
