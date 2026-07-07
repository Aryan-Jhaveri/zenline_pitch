"""Tests for the Vastra taxonomy helpers and llm.py non-network paths."""

from __future__ import annotations

import pytest

from substitutes_agent import llm
from substitutes_agent.vastra_taxonomy import (
    align_article_type,
    category_for_subcategory,
    is_known_subcategory,
)

# ---------------------------------------------------------------------------
# vastra_taxonomy helpers
# ---------------------------------------------------------------------------


def test_is_known_subcategory_true_for_vastra_names() -> None:
    assert is_known_subcategory("t-shirt")
    assert is_known_subcategory("jeans")
    assert is_known_subcategory("sneakers")


def test_is_known_subcategory_false_for_unknown() -> None:
    assert not is_known_subcategory("flibbertigibbet")
    assert not is_known_subcategory("")


def test_category_for_subcategory() -> None:
    assert category_for_subcategory("t-shirt") == "tops"
    assert category_for_subcategory("jeans") == "bottoms"
    assert category_for_subcategory("sneakers") == "shoes"
    assert category_for_subcategory("unknown") is None


def test_align_article_type_passes_through_unknown() -> None:
    assert align_article_type("Perfume and Body Mist") == "Perfume and Body Mist"
    assert align_article_type("") == ""


# ---------------------------------------------------------------------------
# llm.py non-network paths
# ---------------------------------------------------------------------------


def test_provider_for_dispatch() -> None:
    assert llm.provider_for("claude-haiku-4-5-20251001") == "anthropic"
    assert llm.provider_for("gemini-2.5-flash") == "google"


def test_provider_for_openrouter_dispatch() -> None:
    # Any model id containing "/" routes through OpenRouter.
    assert llm.provider_for("anthropic/claude-haiku-4.5") == "openrouter"
    assert llm.provider_for("openai/gpt-4o-mini") == "openrouter"


def test_provider_for_unknown_raises() -> None:
    try:
        llm.provider_for("gpt-4")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown provider")


def test_pick_default_model_none_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert llm.pick_default_model() is None
    assert llm.any_key_present() is False


def test_any_key_present_true_with_fake_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    assert llm.any_key_present() is True


def test_any_key_present_true_with_only_openrouter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    assert llm.any_key_present() is True


def test_is_available_openrouter_true_with_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The openai SDK is installed via the [llm] extra, so is_available
    # should be True when only OPENROUTER_API_KEY is present.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    assert llm.is_available("anthropic/claude-haiku-4.5") is True
    assert llm.is_available("openai/gpt-4o-mini") is True


def test_is_available_openrouter_false_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert llm.is_available("anthropic/claude-haiku-4.5") is False


def test_extract_prompt_includes_observed_values() -> None:
    prompt = llm._extract_prompt("Nike Tee", ["Solid", "Striped"], ["Cotton"])
    assert "Nike Tee" in prompt
    assert "Solid" in prompt
    assert "Cotton" in prompt


def test_classify_prompt_includes_both_skus() -> None:
    a = {"code": "1", "article_type": "t-shirt"}
    b = {"code": "2", "article_type": "t-shirt"}
    prompt = llm._classify_prompt(a, b)
    assert '"code": "1"' in prompt
    assert '"code": "2"' in prompt


def test_extract_json_parses_embedded_object() -> None:
    assert llm._extract_json('noise {"verdict": "yes", "reason": "x"} tail') == {
        "verdict": "yes",
        "reason": "x",
    }


def test_extract_json_returns_empty_on_no_json() -> None:
    assert llm._extract_json("no json here") == {}
    assert llm._extract_json("{ broken") == {}


def test_extract_json_returns_empty_on_malformed() -> None:
    assert llm._extract_json("{ not valid json }") == {}
