"""Project-wide configuration objects."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the local activation encoder."""

    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_length: int = 512
    device: str = "auto"
    layer_offset: int = -2
    use_chat_template: bool = True


@dataclass(frozen=True)
class SLAWeights:
    """User-adjustable utility weights."""

    accuracy: float = 1.0
    cost: float = 0.3
    latency: float = 0.2
