"""Optional Gemini-backed deep analysis.

This is strictly an add-on: the app is fully functional without a key, and
the UI panel only appears when one is configured. The key is read from
Streamlit secrets or the environment and never written into source.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import List, Optional

import requests

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"

# A "-latest" alias tracks whichever model is current, so it cannot go stale
# the way a pinned version does. This matters: ListModels returns retired
# models too, and it returned `gemini-2.5-flash` first -- a model that now
# 404s for new keys ("no longer available to new users"). Taking the first
# flash entry from that listing is therefore wrong.
FALLBACK_MODEL = "gemini-flash-latest"

# Variants that are not general text models, so never auto-select them.
_SPECIALIZED = ("tts", "image", "embedding", "vision", "omni", "aqa")

NO_KEY_MESSAGE = """**No API key configured.**

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add:

    GEMINI_API_KEY = "your-key-here"

Get a key at https://aistudio.google.com/apikey
"""

# How much of the paper to send. The body text is already stripped of
# bibliography and front matter, so this goes much further than the same
# budget spent on a raw page dump.
MAX_CHARS = 60_000


# Where a key may live, in priority order. The secrets file is parsed
# directly rather than through Streamlit, so the engine has no UI framework
# dependency at all and can be imported by any server.
SECRETS_PATH = os.path.join(".streamlit", "secrets.toml")

_KEY_LINE = re.compile(
    r"""^\s*GEMINI_API_KEY\s*=\s*["']?([^"'\s#]+)["']?""", re.M
)


def _key_from_file(path: str = SECRETS_PATH) -> str:
    """Read GEMINI_API_KEY out of a secrets.toml, if one exists."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            match = _KEY_LINE.search(fh.read())
    except (OSError, UnicodeDecodeError):
        return ""
    if not match:
        return ""
    value = match.group(1).strip()
    # Ignore the placeholder shipped in secrets.toml.example.
    return "" if value.startswith("paste-your-key") else value


def get_key() -> str:
    """The Gemini API key, from the environment or a local secrets file."""
    return (
        os.environ.get("GEMINI_API_KEY", "").strip() or _key_from_file()
    ).strip()


def has_key() -> bool:
    return bool(get_key())


def _version_key(name: str) -> tuple:
    """Sort key for a model name, so newer versions rank first.

    "gemini-3.8-flash" outranks "gemini-3.5-flash" outranks "gemini-2.5-flash".
    """
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?", name)
    if not m:
        return (0, 0)
    return (int(m.group(1)), int(m.group(2) or 0))


@lru_cache(maxsize=4)
def candidate_models(api_key: str) -> tuple:
    """Models to try, best first.

    The listing includes retired models, and in listing order, so it is
    ranked here rather than trusted: stable aliases first, then plain text
    models newest-first, then a hardcoded fallback.
    """
    try:
        r = requests.get(f"{API_ROOT}/models", params={"key": api_key}, timeout=20)
        if r.status_code != 200:
            return (FALLBACK_MODEL,)
        available = [
            m["name"].replace("models/", "")
            for m in r.json().get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]
    except (requests.RequestException, ValueError, KeyError):
        return (FALLBACK_MODEL,)

    if not available:
        return (FALLBACK_MODEL,)

    ordered: List[str] = []

    # 1. Self-updating aliases, which never go stale.
    for alias in ("gemini-flash-latest", "gemini-pro-latest"):
        if alias in available:
            ordered.append(alias)

    # 2. Plain versioned flash models, newest first.
    plain = [
        n for n in available
        if "flash" in n
        and "preview" not in n
        and "lite" not in n
        and not any(tag in n for tag in _SPECIALIZED)
    ]
    ordered.extend(sorted(set(plain), key=_version_key, reverse=True))

    # 3. Anything else general-purpose, as a last resort.
    rest = [
        n for n in available
        if "preview" not in n and not any(tag in n for tag in _SPECIALIZED)
    ]
    ordered.extend(sorted(set(rest) - set(ordered), key=_version_key, reverse=True))

    if FALLBACK_MODEL not in ordered:
        ordered.append(FALLBACK_MODEL)

    # Deduplicate, preserving order.
    seen, out = set(), []
    for n in ordered:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return tuple(out[:6])


def _resolve_model(api_key: str) -> str:
    """The single best model to use."""
    return candidate_models(api_key)[0]


def deep_analysis(body_text: str, title: str = "", tone: str = "Professional") -> str:
    """Ask Gemini for a structured report on the paper.

    Returns markdown, or a human-readable error string -- never raises, so
    the UI can always render the result directly.
    """
    api_key = get_key()
    if not api_key:
        return NO_KEY_MESSAGE

    prompt = f"""You are a careful research analyst. Analyze the paper below.

Title: {title or "(unknown)"}
Tone: {tone}

Write markdown with exactly these sections:

1. **What this paper claims** -- two or three sentences.
2. **Method** -- how it works, concretely.
3. **Evidence** -- what was measured, on what data, with what result.
4. **Weaknesses** -- what is unconvincing, missing, or overstated.
5. **Who should read this** -- and who should not.

Base every statement on the text provided. If the text does not support a
claim, write "not stated in the provided text" rather than guessing.

Paper text:
{body_text[:MAX_CHARS]}
"""

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    models = candidate_models(api_key)
    last_error = ""

    # Try candidates in turn. A 404 means that model was retired -- the
    # listing still advertises retired models -- so move on to the next
    # rather than surfacing a dead end to the user.
    for model in models:
        try:
            response = requests.post(
                f"{API_ROOT}/models/{model}:generateContent",
                headers={"Content-Type": "application/json"},
                params={"key": api_key},
                json=payload,
                timeout=150,
            )
        except requests.Timeout:
            return "**Timed out** after 150s. Try a shorter paper."
        except requests.RequestException as e:
            return f"**Connection error:** {e}"

        # 404: this model has been retired. 500/503: it is overloaded right
        # now. Both are specific to the model, so another candidate may well
        # work -- move on rather than failing the whole request.
        if response.status_code in (404, 500, 503):
            reason = {
                404: "retired",
                500: "server error",
                503: "overloaded",
            }[response.status_code]
            last_error = f"{model}: {reason}"
            continue

        # A quota limit belongs to the key, not the model, so trying other
        # models would only burn more of it.
        if response.status_code == 429:
            return (
                "**Rate limited.** The API quota for this key is exhausted "
                "-- wait a minute and retry."
            )
        if response.status_code in (401, 403):
            return (
                "**Authentication failed.** The API key is missing, invalid, "
                "or revoked. Get a new one at https://aistudio.google.com/apikey"
            )
        if response.status_code != 200:
            detail = response.text[:900]
            try:
                detail = response.json()["error"]["message"][:900]
            except (ValueError, KeyError, TypeError):
                pass
            return f"**API error {response.status_code}** ({model})\n\n    {detail}"

        try:
            data = response.json()
            entries = data.get("candidates", [])
            if not entries:
                reason = data.get("promptFeedback", {}).get("blockReason", "unknown")
                return f"**No response generated.** Reason: {reason}"
            parts = entries[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts).strip()
        except (KeyError, ValueError, IndexError, TypeError) as e:
            return f"**Unexpected API response shape:** {e}"

        if text:
            return text
        last_error = f"{model}: empty response"

    if "overloaded" in last_error or "server error" in last_error:
        return (
            "**Every model is busy right now.**\n\n"
            "Google reported high demand across all of "
            f"{len(models)} candidates. This is temporary -- try again in a "
            "minute.\n\n"
            f"Tried: {', '.join(models)}"
        )
    return (
        "**No usable model found.**\n\n"
        f"Tried: {', '.join(models)}\n\n"
        f"Last issue -- {last_error or 'unknown'}"
    )
