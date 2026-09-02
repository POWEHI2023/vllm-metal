# SPDX-License-Identifier: Apache-2.0
"""Tests for the normalized DFlash checkpoint configuration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_metal.spec_decode.dflash.config import DFlashConfig


def _deepseek_dflash_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "architectures": ["Qwen3DSparkModel"],
        "attention_bias": False,
        "block_size": 7,
        "enable_confidence_head": False,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "layer_types": ["full_attention"] * 5,
        "markov_rank": 0,
        "mask_token_id": 151669,
        "max_position_embeddings": 40960,
        "num_attention_heads": 32,
        "num_hidden_layers": 5,
        "num_key_value_heads": 8,
        "num_target_layers": 36,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "rope_theta": 1_000_000,
            "rope_type": "default",
        },
        "target_layer_ids": [1, 9, 17, 25, 33],
        "target_hidden_size": 2560,
        "use_sliding_window": False,
        "vocab_size": 151936,
    }
    config.update(overrides)
    return config


def test_parses_deepseek_top_level_config() -> None:
    config = DFlashConfig.from_hf_config(_deepseek_dflash_config())

    assert config == DFlashConfig(
        target_hidden_layer_ids=(1, 9, 17, 25, 33),
        target_hidden_size=2560,
        block_size=7,
        mask_token_id=151669,
        hidden_size=2560,
        intermediate_size=9728,
        num_hidden_layers=5,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=151936,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        max_position_embeddings=40960,
        attention_bias=False,
        rope_theta=1_000_000.0,
        sample_from_anchor=False,
        causal=False,
        markov_rank=0,
        enable_confidence_head=False,
        rope_type="default",
        layer_types=("full_attention",) * 5,
        use_swa=False,
        use_sliding_window=False,
        num_target_layers=36,
    )
    assert config.max_draft_tokens == 6


def test_parses_config_like_objects_and_top_level_rope_theta() -> None:
    raw = _deepseek_dflash_config(rope_theta=500_000)
    raw.pop("rope_parameters")
    config_object = SimpleNamespace(**raw)

    config = DFlashConfig.from_hf_config(config_object)

    assert config.rope_theta == 500_000.0


def test_target_hidden_size_defaults_to_draft_hidden_size() -> None:
    raw = _deepseek_dflash_config()
    del raw["target_hidden_size"]

    config = DFlashConfig.from_hf_config(raw)

    assert config.target_hidden_size == config.hidden_size


def test_attention_bias_defaults_to_false() -> None:
    raw = _deepseek_dflash_config()
    del raw["attention_bias"]

    config = DFlashConfig.from_hf_config(raw)

    assert config.attention_bias is False


def test_nested_dflash_config_takes_precedence_over_legacy_fields() -> None:
    raw = _deepseek_dflash_config(
        block_size=99,
        mask_token_id=1,
        target_layer_ids=[0],
        dflash_config={
            "block_size": 7,
            "mask_token_id": 151669,
            "target_layer_ids": [1, 9, 17, 25, 33],
            "sample_from_anchor": False,
            "causal": False,
        },
    )

    config = DFlashConfig.from_hf_config(raw)

    assert config.block_size == 7
    assert config.mask_token_id == 151669
    assert config.target_hidden_layer_ids == (1, 9, 17, 25, 33)


def test_target_layer_ids_remain_zero_based_decoder_indices() -> None:
    config = DFlashConfig.from_hf_config(_deepseek_dflash_config())

    # vLLM's +1 conversion belongs to its one-based aux-state API. Metal's
    # adapter captures decoder layer outputs directly, so the checkpoint's
    # DeepSpec indices remain unchanged here.
    assert config.target_hidden_layer_ids == (1, 9, 17, 25, 33)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"markov_rank": 256}, "markov_rank=0"),
        ({"enable_confidence_head": True}, "confidence heads"),
        ({"sample_from_anchor": True}, "sample_from_anchor=True"),
        ({"causal": True}, "non-causal block attention"),
        ({"rope_type": "linear"}, "only supports default RoPE"),
        (
            {"layer_types": ("full_attention",) * 4 + ("sliding_attention",)},
            "only supports full_attention",
        ),
        ({"use_swa": True}, "does not support sliding-window"),
    ],
)
def test_validate_accepts_future_variants_that_validate_v1_rejects(
    changes: dict[str, Any], message: str
) -> None:
    config = replace(
        DFlashConfig.from_hf_config(_deepseek_dflash_config()),
        **changes,
    )

    config.validate()
    with pytest.raises(ValueError, match=message):
        config.validate_v1()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("markov_rank", 256, "markov_rank=0"),
        ("enable_confidence_head", True, "confidence heads"),
        ("sample_from_anchor", True, "sample_from_anchor=True"),
        ("is_causal", True, "non-causal block attention"),
    ],
)
def test_rejects_unsupported_dflash_variants(
    field: str, value: Any, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DFlashConfig.from_hf_config(_deepseek_dflash_config(**{field: value}))


def test_rejects_nested_causal_override() -> None:
    raw = _deepseek_dflash_config(dflash_config={"causal": True})

    with pytest.raises(ValueError, match="non-causal block attention"):
        DFlashConfig.from_hf_config(raw)


def test_null_is_causal_falls_back_to_nested_setting() -> None:
    raw = _deepseek_dflash_config(
        is_causal=None,
        dflash_config={"causal": False},
    )

    assert DFlashConfig.from_hf_config(raw).causal is False


@pytest.mark.parametrize(
    "override",
    [
        {"layer_types": ["full_attention"] * 4 + ["sliding_attention"]},
        {"use_sliding_window": True},
        {"dflash_config": {"use_swa": True}},
    ],
)
def test_rejects_sliding_window_attention(override: dict[str, Any]) -> None:
    raw = _deepseek_dflash_config(**override)

    with pytest.raises(ValueError, match="full_attention|sliding-window"):
        DFlashConfig.from_hf_config(raw)


def test_rejects_layer_types_with_wrong_length() -> None:
    raw = _deepseek_dflash_config(layer_types=["full_attention"] * 4)

    with pytest.raises(ValueError, match="length must equal num_hidden_layers"):
        DFlashConfig.from_hf_config(raw)


@pytest.mark.parametrize(
    "target_layer_ids",
    [[], [-1, 9], [1, 9, 9], [1, 36]],
)
def test_rejects_invalid_target_layer_ids(target_layer_ids: list[int]) -> None:
    raw = _deepseek_dflash_config(target_layer_ids=target_layer_ids)

    with pytest.raises(
        ValueError, match="target auxiliary|non-negative|unique|smaller"
    ):
        DFlashConfig.from_hf_config(raw)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("block_size", 1, "block_size >= 2"),
        ("hidden_size", 0, "hidden_size.*> 0"),
        ("target_hidden_size", 0, "target_hidden_size.*> 0"),
        ("num_hidden_layers", 0, "num_hidden_layers.*> 0"),
        ("max_position_embeddings", 0, "max_position_embeddings.*> 0"),
        ("rms_norm_eps", 0.0, "rms_norm_eps.*> 0"),
    ],
)
def test_rejects_non_positive_values(field: str, value: Any, message: str) -> None:
    raw = _deepseek_dflash_config(**{field: value})
    if field == "num_hidden_layers":
        raw["layer_types"] = []

    with pytest.raises(ValueError, match=message):
        DFlashConfig.from_hf_config(raw)


def test_rejects_invalid_attention_head_ratio() -> None:
    raw = _deepseek_dflash_config(num_attention_heads=30)

    with pytest.raises(ValueError, match="divisible by num_key_value_heads"):
        DFlashConfig.from_hf_config(raw)


def test_validate_rejects_negative_markov_rank() -> None:
    config = replace(
        DFlashConfig.from_hf_config(_deepseek_dflash_config()),
        markov_rank=-1,
    )

    with pytest.raises(ValueError, match="markov_rank.*>= 0"):
        config.validate()


@pytest.mark.parametrize("mask_token_id", [-1, 151936])
def test_rejects_mask_token_outside_vocabulary(mask_token_id: int) -> None:
    raw = _deepseek_dflash_config(mask_token_id=mask_token_id)

    with pytest.raises(ValueError, match="within the draft vocabulary"):
        DFlashConfig.from_hf_config(raw)


def test_rejects_missing_required_field_with_actionable_error() -> None:
    raw = _deepseek_dflash_config()
    del raw["mask_token_id"]

    with pytest.raises(ValueError, match="missing required field 'mask_token_id'"):
        DFlashConfig.from_hf_config(raw)


@pytest.mark.parametrize("field", ["hidden_act", "max_position_embeddings"])
def test_rejects_missing_required_model_construction_field(field: str) -> None:
    raw = _deepseek_dflash_config()
    del raw[field]

    with pytest.raises(ValueError, match=f"missing required field '{field}'"):
        DFlashConfig.from_hf_config(raw)


def test_rejects_non_sequence_target_layer_ids() -> None:
    raw = _deepseek_dflash_config(target_layer_ids="1,9,17")

    with pytest.raises(ValueError, match="sequence of integers"):
        DFlashConfig.from_hf_config(raw)


def test_rejects_non_default_rope() -> None:
    raw = _deepseek_dflash_config()
    raw["rope_parameters"] = deepcopy(raw["rope_parameters"])
    raw["rope_parameters"]["rope_type"] = "linear"

    with pytest.raises(ValueError, match="only supports default RoPE"):
        DFlashConfig.from_hf_config(raw)
