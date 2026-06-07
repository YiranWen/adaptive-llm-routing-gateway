"""Mechanistic activation feature extraction for local causal LMs.

The extractor runs only the prompt prefill forward pass. It does not call
``generate`` and therefore does not decode any new tokens.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from llm_router.config import ModelConfig


@dataclass(frozen=True)
class FeatureExtractionResult:
    """Activation features plus lightweight runtime metadata."""

    features: Any
    feature_dim: int
    num_tokens: int
    layer_index: int
    elapsed_ms: float


class MechanisticFeatureExtractor:
    """Extract mean-pooled hidden-state activations from a local causal LM.

    Parameters
    ----------
    config:
        Model loading and tokenization settings. ``layer_offset=-2`` means the
        second-to-last transformer block.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.device: str | None = None
        self.layer_index: int | None = None
        self.feature_dim: int | None = None

    def load(self) -> None:
        """Load tokenizer and model lazily."""

        if self.model is not None and self.tokenizer is not None:
            return

        torch = _import_torch()
        transformers = _import_transformers()

        self.device = resolve_device(self.config.device)
        dtype = choose_torch_dtype(torch, self.device)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = load_causal_lm(
            transformers=transformers,
            model_name=self.config.model_name,
            dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()

        layers = get_transformer_layers(self.model)
        self.layer_index = resolve_layer_index(len(layers), self.config.layer_offset)
        self.feature_dim = int(getattr(self.model.config, "hidden_size"))

    def extract(self, prompt: str, *, return_torch: bool = False) -> FeatureExtractionResult:
        """Extract a single prompt feature vector."""

        batch_result = self.extract_batch([prompt], return_torch=return_torch)
        return FeatureExtractionResult(
            features=batch_result.features[0],
            feature_dim=batch_result.feature_dim,
            num_tokens=batch_result.num_tokens,
            layer_index=batch_result.layer_index,
            elapsed_ms=batch_result.elapsed_ms,
        )

    def extract_batch(
        self,
        prompts: list[str],
        *,
        return_torch: bool = False,
    ) -> FeatureExtractionResult:
        """Extract one activation vector per prompt.

        Returns
        -------
        FeatureExtractionResult
            ``features`` has shape ``[batch_size, hidden_size]``.
        """

        if not prompts:
            raise ValueError("prompts must contain at least one prompt")

        self.load()
        assert self.model is not None
        assert self.tokenizer is not None
        assert self.device is not None
        assert self.layer_index is not None

        torch = _import_torch()

        encoded_prompts = [self._format_prompt(prompt) for prompt in prompts]
        inputs = self.tokenizer(
            encoded_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        captured: dict[str, Any] = {}
        layers = get_transformer_layers(self.model)

        def capture_hidden_states(_module: Any, _inputs: Any, output: Any) -> None:
            captured["hidden_states"] = first_tensor(output)

        handle = layers[self.layer_index].register_forward_hook(capture_hidden_states)

        start = time.perf_counter()
        try:
            with torch.inference_mode():
                self.model(**inputs, use_cache=False)
        finally:
            handle.remove()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if "hidden_states" not in captured:
            raise RuntimeError("forward hook did not capture hidden states")

        pooled = mean_pool_hidden_states(
            captured["hidden_states"],
            inputs["attention_mask"],
        )
        pooled = pooled.detach().float().cpu()
        features = pooled if return_torch else pooled.numpy()

        return FeatureExtractionResult(
            features=features,
            feature_dim=int(pooled.shape[-1]),
            num_tokens=int(inputs["attention_mask"].sum().item()),
            layer_index=self.layer_index,
            elapsed_ms=elapsed_ms,
        )

    def _format_prompt(self, prompt: str) -> str:
        if not self.config.use_chat_template:
            return prompt

        assert self.tokenizer is not None
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return prompt

        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt


def resolve_device(device: str) -> str:
    """Resolve ``auto`` to the fastest locally available PyTorch device."""

    if device != "auto":
        return device

    torch = _import_torch()
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_torch_dtype(torch: Any, device: str) -> Any:
    """Choose a conservative dtype for local inference."""

    if device == "cuda":
        return torch.float16
    return torch.float32


def load_causal_lm(transformers: Any, model_name: str, dtype: Any) -> Any:
    """Load a causal LM across Transformers versions.

    Newer Transformers versions prefer ``dtype``. Older versions used
    ``torch_dtype``.
    """

    try:
        return transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=True,
        )
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )


def get_transformer_layers(model: Any) -> Any:
    """Return the decoder block list for common HuggingFace causal LMs."""

    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for parent_name, child_name in candidates:
        parent = getattr(model, parent_name, None)
        layers = getattr(parent, child_name, None)
        if layers is not None:
            return layers

    raise AttributeError(
        "Could not find transformer layers on the model. "
        "Expected one of model.layers, transformer.h, or gpt_neox.layers."
    )


def resolve_layer_index(num_layers: int, layer_offset: int) -> int:
    """Convert a positive index or negative offset into a valid layer index."""

    if num_layers <= 0:
        raise ValueError("num_layers must be positive")

    index = layer_offset if layer_offset >= 0 else num_layers + layer_offset
    if index < 0 or index >= num_layers:
        raise IndexError(
            f"layer_offset={layer_offset} resolves to {index}, "
            f"but model has {num_layers} layers"
        )
    return index


def first_tensor(output: Any) -> Any:
    """Extract the hidden-state tensor from a transformer block output."""

    if isinstance(output, tuple):
        return output[0]
    return output


def mean_pool_hidden_states(hidden_states: Any, attention_mask: Any) -> Any:
    """Mean-pool hidden states across non-padding tokens."""

    mask = attention_mask.to(
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    ).unsqueeze(-1)
    masked_hidden_states = hidden_states * mask
    token_counts = mask.sum(dim=1).clamp(min=1)
    return masked_hidden_states.sum(dim=1) / token_counts


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for feature extraction. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc
    return torch


def _import_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise ImportError(
            "transformers is required for feature extraction. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc
    try:
        import transformers.utils.import_utils as import_utils

        # Some conda Mac environments have a mismatched torchvision install.
        # This project is text-only, so disabling torchvision avoids unrelated
        # vision import failures when loading Qwen through AutoModelForCausalLM.
        import_utils._torchvision_available = False
    except Exception:
        pass
    return transformers
