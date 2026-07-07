"""Multi-provider LLM wrapper (Anthropic, Google Gemini, and OpenRouter), env-gated.

Uniform interface for the two judgment points in the pipeline:
  - Step 2: extract_attributes (fill pattern/material for ambiguous rows)
  - Step 3: classify_pair (tie-break borderline substitute pairs)

Provider dispatch is by model prefix: claude-* -> Anthropic, gemini-* ->
Google. The SDKs are imported lazily inside the call sites so the
deterministic pipeline runs without the optional [llm] extra installed.

Prompt shape: I reuse the pattern from my Vastra categorizer (vastra.cc) —
pass the set of values already observed in this run so the model aligns
to a stable vocabulary instead of inventing free-form labels. That is the
"product ontology" idea in code: anchor LLM judgment to a fixed vocabulary.

Live network paths are marked `# pragma: no cover` so coverage is measured
against the deterministic rules path only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from substitutes_agent.models import PairVerdict

DEFAULT_MODELS = (
    "gemini-2.5-flash",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-4o-mini",
)


def provider_for(model: str) -> str:
    """Return the provider name for a model id."""
    if "/" in model:
        return "openrouter"
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gemini"):
        return "google"
    raise ValueError(f"unknown model provider for: {model}")


def _key_for(provider: str) -> str | None:
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY")
    if provider == "google":
        return os.environ.get("GOOGLE_API_KEY")
    if provider == "openrouter":
        return os.environ.get("OPENROUTER_API_KEY")
    return None


def _sdk_available(provider: str) -> bool:
    try:
        if provider == "anthropic":
            import anthropic  # noqa: F401

            return True
        if provider == "google":
            import google.genai  # noqa: F401

            return True
        if provider == "openrouter":
            import openai  # noqa: F401

            return True
    except ImportError:
        return False
    return False


def is_available(model: str) -> bool:
    """True if the model's SDK is installed and its key is present."""
    provider = provider_for(model)
    return _sdk_available(provider) and bool(_key_for(provider))


def pick_default_model() -> str | None:
    """First available model from DEFAULT_MODELS, or None."""
    for model in DEFAULT_MODELS:
        try:
            if is_available(model):
                return model
        except ValueError:
            continue
    return None


def any_key_present() -> bool:
    """True if any provider key is set (SDK may or may not be installed)."""
    return bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )


# ---------------------------------------------------------------------------
# Prompt builders (the Vastra buildPrompt reuse pattern).
# ---------------------------------------------------------------------------


def _extract_prompt(
    product_name: str,
    observed_patterns: list[str],
    observed_materials: list[str],
) -> str:
    prompt = (
        "Extract the pattern and material of this apparel product from its "
        "display name. Reply with ONLY valid JSON: "
        '{"pattern": str|null, "material": str|null}.\n'
        f"Product name: {product_name}\n"
    )
    if observed_patterns:
        prompt += (
            "Patterns already seen in this catalog — reuse the exact spelling "
            f"if one fits, else null: {', '.join(observed_patterns)}\n"
        )
    if observed_materials:
        prompt += (
            "Materials already seen in this catalog — reuse the exact spelling "
            f"if one fits, else null: {', '.join(observed_materials)}\n"
        )
    return prompt


def _classify_prompt(sku_a: dict[str, object], sku_b: dict[str, object]) -> str:
    return (
        "Are these two apparel products realistic substitutes for each other "
        "(same purpose, interchangeable for a shopper)? Reply with ONLY valid "
        'JSON: {"verdict": "yes"|"no", "reason": str}.\n'
        f"Product A: {json.dumps(sku_a, ensure_ascii=False)}\n"
        f"Product B: {json.dumps(sku_b, ensure_ascii=False)}\n"
    )


# ---------------------------------------------------------------------------
# Provider call sites (lazy import; live network -> pragma: no cover).
# ---------------------------------------------------------------------------


def _call_anthropic(model: str, prompt: str) -> str:  # pragma: no cover
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    if not resp.content:
        return ""
    block = resp.content[0]
    return getattr(block, "text", "") or ""


def _call_google(model: str, prompt: str) -> str:  # pragma: no cover
    from google import genai

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    resp = client.models.generate_content(model=model, contents=prompt)
    return resp.text or ""


def _call_openrouter(model: str, prompt: str) -> str:  # pragma: no cover
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    resp = client.chat.completions.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    if not resp.choices:
        return ""
    return resp.choices[0].message.content or ""


def _call(model: str, prompt: str) -> str:
    provider = provider_for(model)
    if not _sdk_available(provider):
        raise RuntimeError(
            f"provider {provider} SDK not installed; install the [llm] extra"
        )
    if not _key_for(provider):
        raise RuntimeError(f"provider {provider} key not set")
    if provider == "anthropic":  # pragma: no cover
        return _call_anthropic(model, prompt)
    if provider == "google":  # pragma: no cover
        return _call_google(model, prompt)
    if provider == "openrouter":  # pragma: no cover
        return _call_openrouter(model, prompt)
    raise RuntimeError(f"unhandled provider {provider}")  # pragma: no cover


def _extract_json(text: str) -> dict[str, object]:
    """Best-effort extraction of a JSON object from an LLM response."""
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def extract_attributes(
    model: str,
    product_name: str,
    observed_patterns: list[str],
    observed_materials: list[str],
    cache_path: str | Path | None = None,
) -> dict[str, str | None]:
    """Fill pattern/material for one ambiguous SKU (Step 2 LLM path).

    Cached to `cache_path` so re-runs are deterministic and free.
    """
    if cache_path is not None:
        cp = Path(cache_path)
        if cp.exists():
            data = json.loads(cp.read_text(encoding="utf-8"))
            return {
                "pattern": data.get("pattern"),
                "material": data.get("material"),
            }

    prompt = _extract_prompt(product_name, observed_patterns, observed_materials)
    raw = _call(model, prompt)  # pragma: no cover
    parsed = _extract_json(raw)  # pragma: no cover
    pat = parsed.get("pattern")
    mat = parsed.get("material")
    result: dict[str, str | None] = {
        "pattern": str(pat) if pat else None,
        "material": str(mat) if mat else None,
    }

    if cache_path is not None:
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def classify_pair(
    model: str,
    sku_a: dict[str, object],
    sku_b: dict[str, object],
    ontology_context: dict[str, object] | None = None,
) -> PairVerdict:
    """Tie-break a borderline substitute pair (Step 3 LLM path)."""
    _ = ontology_context  # reserved for future few-shot context
    prompt = _classify_prompt(sku_a, sku_b)
    raw = _call(model, prompt)  # pragma: no cover
    parsed = _extract_json(raw)  # pragma: no cover
    verdict_raw = str(parsed.get("verdict", "")).lower().strip()
    verdict = "yes" if verdict_raw.startswith("y") else "no"
    return PairVerdict(
        verdict=verdict,  # type: ignore[arg-type]
        reason=str(parsed.get("reason", "")),
        model=model,
    )
