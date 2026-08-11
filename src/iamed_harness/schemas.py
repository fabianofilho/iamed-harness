"""Contratos de dados do harness.

Regra que atravessa o modulo inteiro: nenhum campo opcional silencioso.
Se algo falhou, o campo correspondente fica `None` e `error` explica o motivo.
A predicao entra no arquivo de qualquer jeito, nunca e descartada.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MetadataValue = str | int | float
ABSTAIN = "ABSTAIN"


class Item(BaseModel):
    """Uma questao de multipla escolha com gabarito conhecido."""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    choices: dict[str, str]
    answer: str
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)
    expected_abstain: bool = False
    """Quando True, a resposta certa e recusar-se a escolher.

    Perturbacoes como `negate_stem` e `unit_swap` podem destruir a validade do
    gabarito. Nesses casos o item nao e jogado fora: ele passa a cobrar
    abstencao, que e o comportamento clinicamente correto.
    """

    @field_validator("choices")
    @classmethod
    def _choices_nao_vazias(cls, v: dict[str, str]) -> dict[str, str]:
        if len(v) < 2:
            raise ValueError("um item precisa de pelo menos 2 alternativas")
        for chave in v:
            if not chave.isalpha() or len(chave) != 1:
                raise ValueError(f"chave de alternativa invalida: {chave!r} (esperado uma letra)")
        return v

    @model_validator(mode="after")
    def _gabarito_existe(self) -> Item:
        if self.expected_abstain:
            if self.answer != ABSTAIN:
                raise ValueError(
                    f"item {self.id}: expected_abstain=True exige answer={ABSTAIN!r}, "
                    f"recebido {self.answer!r}"
                )
            return self
        if self.answer not in self.choices:
            raise ValueError(
                f"item {self.id}: gabarito {self.answer!r} nao esta entre as alternativas "
                f"{sorted(self.choices)}"
            )
        return self


class Prediction(BaseModel):
    """Uma chamada ao modelo, com ou sem sucesso."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    run_id: str
    repetition: int
    raw_response: str
    parsed_choice: str | None
    confidence: float | None
    correct: bool | None
    latency_ms: float
    error: str | None
    from_cache: bool = False

    @property
    def failed(self) -> bool:
        """True quando nao ha resposta utilizavel, seja por rede ou por parsing."""
        return self.correct is None


class SamplingPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repetitions: int = 1
    temperature: float = 0.0
    max_tokens: int = 512

    @field_validator("repetitions")
    @classmethod
    def _pelo_menos_uma(cls, v: int) -> int:
        if v < 1:
            raise ValueError("repetitions precisa ser >= 1")
        return v


class TaskSpec(BaseModel):
    """Uma tarefa registrada em `tasks/*.yaml`."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    dataset: str
    prompt: str
    grader: Literal["exact_choice", "regex", "llm_judge"]
    grader_options: dict[str, str] = Field(default_factory=dict)
    sampling: SamplingPolicy = Field(default_factory=SamplingPolicy)
    subgroup_fields: list[str] = Field(default_factory=list)


class RunConfig(BaseModel):
    """Config resolvida de uma execucao. E o que torna o run reproduzivel."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    created_at: str
    task_name: str
    dataset_path: str
    dataset_sha256: str
    dataset_n_items: int
    prompt_path: str
    prompt_sha256: str
    grader: str
    grader_options: dict[str, str]
    provider: str
    model: str
    temperature: float
    max_tokens: int
    repetitions: int
    limit: int | None
    concurrency: int
    seed: int
    subgroup_fields: list[str]


class BinStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lower: float
    upper: float
    n: int
    mean_confidence: float
    mean_accuracy: float


class CalibrationMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_scored: int
    n_sem_confianca: int
    ece_uniform: float
    ece_quantile: float
    mce: float
    brier: float
    bins_uniform: list[BinStats]
    bins_quantile: list[BinStats]


class AccuracyMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_total: int
    n_correct: int
    n_failed: int
    accuracy: float
    """Acuracia com falhas contando como erro. E o numero honesto."""
    accuracy_parsed_only: float
    """Acuracia ignorando falhas. So existe para mostrar o quanto ela mente."""
    ci_low: float
    ci_high: float


class RepetitionAccuracy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repetition: int
    accuracy: float
    n: int


class VarianceMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repetitions: int
    n_items: int
    flip_rate: float
    mean_agreement: float
    accuracy_por_repeticao: list[RepetitionAccuracy]
    accuracy_std: float
    itens_instaveis: list[str]


class SubgroupRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    value: str
    n: int
    n_correct: int
    n_failed: int
    accuracy: float


class RunResult(BaseModel):
    """Agregado completo de um run, o que alimenta o relatorio."""

    model_config = ConfigDict(extra="forbid")

    config: RunConfig
    accuracy: AccuracyMetrics
    calibration: CalibrationMetrics
    variance: VarianceMetrics | None
    subgroups: list[SubgroupRow]
    n_predictions: int
    erros_por_tipo: dict[str, int]
