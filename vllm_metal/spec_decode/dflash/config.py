# SPDX-License-Identifier: Apache-2.0
"""Validated configuration view for the first Metal DFlash backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

_MISSING = object()


def _read(config: Any, *names: str, default: Any = _MISSING) -> Any:
    """Read the first present name from a mapping or config-like object."""
    for name in names:
        if isinstance(config, Mapping) and name in config:
            return config[name]
        if hasattr(config, name):
            return getattr(config, name)
    return default


@dataclass(frozen=True, slots=True)
class DFlashConfig:
    """Normalized DFlash draft-model configuration.

    Attributes:
        target_hidden_layer_ids:
            Zero-based target decoder-layer indices whose output hidden
            states are supplied to the draft model.
        target_hidden_size:
            Hidden size of each target-model hidden state supplied to the
            draft model. Defaults to the draft model hidden size.
        block_size:
            Number of positions in one parallel draft block, including the
            anchor position. Standard DFlash produces ``block_size - 1``
            speculative tokens.
        mask_token_id: Vocabulary token used for masked draft-query positions.

        DFlahs draft-model parameters:
            hidden_size: Hidden size of the DFlash draft model.
            intermediate_size: Intermediate size of the draft model MLP.
            num_hidden_layers: Number of DFlash draft decoder layers.
            num_attention_heads: Number of draft query-attention heads.
            num_key_value_heads: Number of draft key/value heads.
            head_dim: Dimension of each draft attention head.
            vocab_size: Vocabulary size expected by the draft model.
            rms_norm_eps: Epsilon used by draft RMSNorm layers.
            hidden_act: Activation function used by the draft model MLP.
            max_position_embeddings:
                Maximum sequence length supported by draft rotary embeddings.
            attention_bias:
                Whether draft attention projection layers include bias terms.


        rope_theta: Base frequency used by draft rotary embeddings.
        sample_from_anchor:
            Whether the anchor position also predicts a draft token. False
            for standard DFlash and normally true for DSpark.
        causal: Whether draft query attention is causal.
        rope_type: Rotary embedding variant used by the draft model.
        layer_types: Per-draft-layer attention types.
        use_swa: Whether DFlash-specific sliding-window attention is enabled.
        use_sliding_window: Legacy/top-level sliding-window attention flag.
        num_target_layers:
            Optional target decoder-layer count used to validate target layer
            indices.

        DSpark-specific fields:
            markov_rank:
                Rank of the optional DSpark Markov head. Zero means disabled.
            enable_confidence_head:
                Whether the DSpark adaptive-verification confidence head exists.
    """

    target_hidden_layer_ids: tuple[int, ...]
    target_hidden_size: int
    block_size: int
    mask_token_id: int

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    hidden_act: str
    max_position_embeddings: int
    attention_bias: bool

    rope_theta: float
    sample_from_anchor: bool
    causal: bool
    markov_rank: int
    enable_confidence_head: bool

    # TODO If scaled/linear RoPE is supported in the future,
    # it will also be necessary to save and parse the complete rope_parameters,
    # not just retain rope_theta.
    rope_type: Literal["default"] = "default"
    layer_types: tuple[str, ...] | None = None
    use_swa: bool = False
    use_sliding_window: bool = False
    num_target_layers: int | None = None

    @property
    def max_draft_tokens(self) -> int:
        return self.block_size if self.sample_from_anchor else self.block_size - 1

    def validate(self) -> None:
        """Validate invariants shared by official DFlash implementation."""
        if not self.target_hidden_layer_ids:
            raise ValueError(
                "DFlash requires at least one target auxiliary hidden layer."
            )
        if any(layer_id < 0 for layer_id in self.target_hidden_layer_ids):
            raise ValueError("Target hidden layer IDs must be non-negative.")
        if len(set(self.target_hidden_layer_ids)) != len(self.target_hidden_layer_ids):
            raise ValueError("Target hidden layer IDs must be unique.")

        if self.block_size <= 0:
            raise ValueError("DFlash config field 'block_size' must be > 0.")
        if not self.sample_from_anchor and self.block_size < 2:
            raise ValueError("Standard DFlash requires block_size >= 2.")
        for field_name in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "target_hidden_size",
            "max_position_embeddings",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"DFlash config field '{field_name}' must be > 0.")

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads."
            )
        if not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError("mask_token_id must be within the draft vocabulary.")
        if self.rms_norm_eps <= 0:
            raise ValueError("DFlash config field 'rms_norm_eps' must be > 0.")
        if self.rope_theta <= 0:
            raise ValueError("DFlash config field 'rope_theta' must be > 0.")
        if self.markov_rank < 0:
            raise ValueError("DFlash config field 'markov_rank' must be >= 0.")

        if self.layer_types is not None:
            if len(self.layer_types) != self.num_hidden_layers:
                raise ValueError(
                    "DFlash layer_types length must equal num_hidden_layers."
                )
            supported_layer_types = {"full_attention", "sliding_attention"}
            if any(
                layer_type not in supported_layer_types
                for layer_type in self.layer_types
            ):
                raise ValueError(
                    "DFlash layer_types entries must be 'full_attention' or 'sliding_attention'."
                )

        if self.num_target_layers is not None:
            if self.num_target_layers <= 0:
                raise ValueError("DFlash config field 'num_target_layers' must be > 0.")
            if any(
                layer_id >= self.num_target_layers
                for layer_id in self.target_hidden_layer_ids
            ):
                raise ValueError(
                    "Target hidden layer IDs must be smaller than num_target_layers."
                )

    def validate_v1(self) -> None:
        """Validate the feature subset supported by the first Metal backend."""
        self.validate()

        if self.markov_rank != 0:
            raise ValueError(
                "The first vLLM-Metal DFlash implementation only supports markov_rank=0."
            )
        if self.enable_confidence_head:
            raise ValueError(
                "Adaptive verification/confidence heads are not supported yet."
            )
        if self.sample_from_anchor:
            raise ValueError(
                "sample_from_anchor=True belongs to the DSpark-style path and "
                "is not supported by the initial DFlash implementation."
            )
        if self.causal:
            raise ValueError(
                "The initial DFlash implementation requires non-causal block attention."
            )
        if self.rope_type != "default":
            raise ValueError(
                "The initial vLLM-Metal DFlash implementation only supports "
                f"default RoPE, got {self.rope_type!r}."
            )
        if self.layer_types is not None and any(
            layer_type != "full_attention" for layer_type in self.layer_types
        ):
            raise ValueError(
                "The initial DFlash implementation only supports full_attention layers."
            )
        if self.use_swa or self.use_sliding_window:
            raise ValueError(
                "The initial DFlash implementation does not support sliding-window attention."
            )

    @classmethod
    def from_hf_config(cls, config: Any) -> DFlashConfig:
        """Normalize a Transformers config, plain mapping, or namespace."""

        dflash_config = _read(config, "dflash_config", default={})
        if dflash_config is None:
            dflash_config = {}
        if not isinstance(dflash_config, Mapping) and not hasattr(
            dflash_config, "__dict__"
        ):
            raise ValueError("DFlash config field 'dflash_config' must be an object.")

        rope_parameters = _read(config, "rope_parameters", default={})
        if rope_parameters is None:
            rope_parameters = {}
        if not isinstance(rope_parameters, Mapping) and not hasattr(
            rope_parameters, "__dict__"
        ):
            raise ValueError("DFlash config field 'rope_parameters' must be an object.")

        rope_theta = _read(
            config, "rope_theta", default=_read(rope_parameters, "rope_theta")
        )
        if rope_theta is _MISSING or rope_theta is None:
            raise ValueError(
                "DFlash checkpoint config is missing required field 'rope_theta'."
            )

        rope_type = _read(rope_parameters, "rope_type", "type", default="default")
        if not isinstance(rope_type, str):
            raise ValueError("DFlash RoPE type must be a string.")

        target_layer_ids = _read(
            dflash_config,
            "target_hidden_layer_ids",
            "target_layer_ids",
            default=_read(
                config,
                "target_hidden_layer_ids",
                "target_layer_ids",
            ),
        )
        if target_layer_ids is _MISSING or target_layer_ids is None:
            raise ValueError(
                "DFlash checkpoint config is missing required field "
                "'target_hidden_layer_ids' or 'target_layer_ids'."
            )
        if isinstance(target_layer_ids, (str, bytes)) or not isinstance(
            target_layer_ids, Sequence
        ):
            raise ValueError(
                "DFlash config field 'target_layer_ids' must be a sequence of integers."
            )

        is_causal = _read(config, "is_causal")
        # _read(...) will return None directly if 'is_causal' is None in top-level config,
        # so we explicit fallback to 'causal' in dflash_config and config
        if is_causal is _MISSING or is_causal is None:
            is_causal = _read(
                dflash_config,
                "causal",
                default=_read(config, "causal", default=False),
            )

        layer_types = _read(
            dflash_config,
            "layer_types",
            default=_read(config, "layer_types", default=None),
        )
        if layer_types is not None:
            if isinstance(layer_types, (str, bytes)) or not isinstance(
                layer_types, Sequence
            ):
                raise ValueError(
                    "DFlash config field 'layer_types' must be a sequence of strings."
                )
            layer_types = tuple(layer_types)

        use_swa = _read(
            dflash_config,
            "use_swa",
            default=_read(config, "use_swa", default=False),
        )

        use_sliding_window = _read(
            dflash_config,
            "use_sliding_window",
            default=_read(config, "use_sliding_window", default=False),
        )

        num_target_layers = _read(config, "num_target_layers", default=None)

        attention_bias = _read(config, "attention_bias", default=False)

        sample_from_anchor = _read(
            dflash_config,
            "sample_from_anchor",
            default=_read(config, "sample_from_anchor", default=False),
        )

        markov_rank = _read(
            dflash_config,
            "markov_rank",
            default=_read(config, "markov_rank", default=0),
        )

        enable_confidence_head = _read(
            dflash_config,
            "enable_confidence_head",
            default=_read(config, "enable_confidence_head", default=False),
        )

        required_kwargs: dict[str, Any] = {
            "block_size": _read(
                dflash_config,
                "block_size",
                default=_read(config, "block_size"),
            ),
            "mask_token_id": _read(
                dflash_config,
                "mask_token_id",
                default=_read(config, "mask_token_id"),
            ),
            "hidden_size": _read(config, "hidden_size"),
            "intermediate_size": _read(config, "intermediate_size"),
            "num_hidden_layers": _read(config, "num_hidden_layers"),
            "num_attention_heads": _read(config, "num_attention_heads"),
            "num_key_value_heads": _read(config, "num_key_value_heads"),
            "head_dim": _read(config, "head_dim"),
            "vocab_size": _read(config, "vocab_size"),
            "rms_norm_eps": _read(config, "rms_norm_eps"),
            "hidden_act": _read(config, "hidden_act"),
            "max_position_embeddings": _read(config, "max_position_embeddings"),
        }

        for field_name, value in required_kwargs.items():
            if value is _MISSING or value is None:
                raise ValueError(
                    f"DFlash checkpoint config is missing required field '{field_name}'."
                )

        target_hidden_size = _read(
            config,
            "target_hidden_size",
            default=required_kwargs["hidden_size"],
        )

        result = cls(
            target_hidden_layer_ids=tuple(target_layer_ids),
            target_hidden_size=target_hidden_size,
            rope_theta=rope_theta,
            sample_from_anchor=sample_from_anchor,
            causal=is_causal,
            markov_rank=markov_rank,
            enable_confidence_head=enable_confidence_head,
            attention_bias=attention_bias,
            rope_type=rope_type,
            layer_types=layer_types,
            use_swa=use_swa,
            use_sliding_window=use_sliding_window,
            num_target_layers=num_target_layers,
            **required_kwargs,
        )
        result.validate_v1()
        return result
