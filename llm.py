"""One direct LLM call (Gemini / Claude / OpenAI) + retries + a small disk cache.

Fixes vs. the old version:
- cache key now includes json_out (same messages, different mode = different entry)
- answers are only cached AFTER they parsed successfully (bad/truncated output is never cached)
- a corrupt cached JSON entry is deleted and re-fetched instead of crashing forever
- Anthropic max_tokens raised (2000 -> 8000) and truncation is detected and reported clearly
- OpenAI uses JSON mode when json_out=True
- Gemini rate-limit protection: slower requests and longer backoff
"""
import hashlib, json, os, pathlib, re, time

CACHE = pathlib.Path(".llm_cache")
CACHE.mkdir(exist_ok=True)

ANTHROPIC_MAX_TOKENS = 8000
_LAST_REQUEST_TIME = {}  # track time of last request per provider


class TruncatedOutput(RuntimeError):
    """The model stopped because it hit the output token limit."""


def _throttle(provider):
    """Space out calls so a low Gemini RPM quota is less likely to be exceeded.

    Set GEMINI_MIN_INTERVAL_SECONDS to match the active quota shown in AI Studio.
    The default is 12 seconds (about 5 requests/minute).
    """
    try:
        gemini_interval = max(0.0, float(os.getenv("GEMINI_MIN_INTERVAL_SECONDS", "12")))
    except ValueError:
        gemini_interval = 12.0
    min_interval = gemini_interval if provider == "gemini" else 0.5
    now = time.time()
    last = _LAST_REQUEST_TIME.get(provider, 0)
    wait = min_interval - (now - last)
    if wait > 0:
        time.sleep(wait)
    _LAST_REQUEST_TIME[provider] = time.time()


def _retry(fn, tries=3):
    """Retry transient errors briefly; avoid long loops on exhausted quotas."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            code = getattr(e, "code", None) or getattr(e, "status_code", None)
            is_rate_limit = code in (429, 500, 502, 503, 504) or "resource exhausted" in str(e).lower()
            if not is_rate_limit or i == tries - 1:
                raise
            wait = 2 ** (i + 1)  # 2s, then 4s; at most two manual retries
            print(f"[llm] transient API error, retrying in {wait}s ({i + 1}/{tries - 1})...")
            time.sleep(wait)


def _split(messages):
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [m for m in messages if m["role"] != "system"]
    return system, rest


GEMINI_FALLBACKS = [m.strip() for m in os.getenv(
    "GEMINI_FALLBACK_MODELS", "gemini-3.6-flash,gemini-3.7-flash,gemini-3.8-flash").split(",") if m.strip()]


def _gemini_generate(client, model, user, cfg, passes=2):
    """Try the chosen model, then the fallback chain in order (3.6 -> 3.7 -> 3.8).

    - 500/503/504 (overloaded): move on to the next model, pause briefly between models.
    - 404 (model not available to this key): skip it and try the next one.
    - Any other error (bad key, bad request, 429 quota): raised immediately.
    The whole chain is tried `passes` times before giving up.
    """
    chain = [model] + [m for m in GEMINI_FALLBACKS if m != model]
    last = None
    for p in range(passes):
        for m in chain:
            try:
                return client.models.generate_content(model=m, contents=user, config=cfg)
            except Exception as e:
                code = getattr(e, "code", None) or getattr(e, "status_code", None)
                text = str(e).lower()
                busy = code in (500, 503, 504) or "unavailable" in text
                missing = code == 404 or "not_found" in text
                if not (busy or missing):
                    raise
                last = e
                print(f"[llm] Gemini {m} {'unavailable' if busy else 'not found'} "
                      f"(pass {p + 1}/{passes}), trying next model...")
                if busy:
                    time.sleep(3)
    raise last


def _raw_call(messages, model, provider, json_out):
    """One uncached API call. Returns the raw text."""
    _throttle(provider)
    if provider == "gemini":
        from google import genai
        from google.genai import types
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        key_source = "GEMINI_API_KEY" if os.getenv("GEMINI_API_KEY") else "GOOGLE_API_KEY"
        if not api_key:
            raise RuntimeError("Gemini API key is missing; set GEMINI_API_KEY in your environment or .env file.")
        client = genai.Client(api_key=api_key)
        system, rest = _split(messages)
        user = "\n\n".join(m["content"] for m in rest) or "Go."
        cfg = types.GenerateContentConfig(
            system_instruction=system or None,
            response_mime_type="application/json" if json_out else None)
        # google-genai already retries transient API errors internally. Avoid wrapping
        # it in another long retry loop, which multiplies attempts for 429 responses.
        try:
            r = _gemini_generate(client, model, user, cfg)
        except Exception as e:
            code = getattr(e, "code", None) or getattr(e, "status_code", None)
            message = str(e).lower()
            if code == 429 or "resource_exhausted" in message or "resource exhausted" in message:
                raise RuntimeError(
                    "Gemini rate limit/quota reached (429 RESOURCE_EXHAUSTED). "
                    f"Model: {model}; key variable used: {key_source}. "
                    "Check Google AI Studio → Rate limits for this project and model. "
                    "For a per-minute limit, wait before retrying; for a daily quota, "
                    "wait for reset or use a model/tier with available quota. "
                    f"Gemini details: {str(e)[:600]}"
                ) from e
            if code == 503 or "unavailable" in message:
                raise RuntimeError(
                    f"Gemini is temporarily unavailable (503) for model {model}. "
                    f"Please try again shortly. Gemini details: {str(e)[:600]}"
                ) from e
            raise
        try:
            reason = str(r.candidates[0].finish_reason)
            if "MAX_TOKENS" in reason:
                raise TruncatedOutput("Gemini stopped at its output limit; try fewer paragraphs.")
        except (AttributeError, IndexError, TypeError):
            pass
        if not r.text:
            raise RuntimeError("Gemini returned an empty answer (possibly blocked by a safety filter).")
        return r.text

    if provider == "anthropic":
        import anthropic
        system, rest = _split(messages)
        rest = rest or [{"role": "user", "content": "Go."}]
        r = _retry(lambda: anthropic.Anthropic().messages.create(
            model=model, max_tokens=ANTHROPIC_MAX_TOKENS, system=system, messages=rest))
        if r.stop_reason == "max_tokens":
            raise TruncatedOutput(
                f"Claude stopped at {ANTHROPIC_MAX_TOKENS} tokens; try fewer paragraphs.")
        return r.content[0].text

    from openai import OpenAI
    kwargs = {"response_format": {"type": "json_object"}} if json_out else {}
    r = _retry(lambda: OpenAI().chat.completions.create(model=model, messages=messages, **kwargs))
    if r.choices[0].finish_reason == "length":
        raise TruncatedOutput("OpenAI stopped at its output limit; try fewer paragraphs.")
    return r.choices[0].message.content


# If every Gemini model is busy, fall back to another provider whose API key is set.
FALLBACK_PROVIDERS = [
    ("anthropic", "ANTHROPIC_API_KEY", os.getenv("FALLBACK_ANTHROPIC_MODEL", "claude-sonnet-5-5")),
    ("openai", "OPENAI_API_KEY", os.getenv("FALLBACK_OPENAI_MODEL", "gpt-4o")),
]


def _raw_call_with_fallback(messages, model, provider, json_out):
    try:
        return _raw_call(messages, model, provider, json_out)
    except RuntimeError as e:
        text = str(e).lower()
        gemini_busy = provider == "gemini" and ("temporarily unavailable" in text or "rate limit" in text)
        if not gemini_busy:
            raise
        for p, env, m in FALLBACK_PROVIDERS:
            if os.getenv(env):
                print(f"[llm] Gemini unavailable, falling back to {p} ({m})")
                return _raw_call(messages, m, p, json_out)
        raise


def call(messages, model="gpt-4o", provider="openai", json_out=False):
    key = hashlib.sha256(json.dumps([provider, model, json_out, messages],
                                    ensure_ascii=False).encode()).hexdigest()
    p = CACHE / key

    # 1) try the cache
    if p.exists() and p.stat().st_size > 0:
        cached = p.read_text(encoding="utf-8")
        if not json_out:
            return cached
        try:
            return parse_json(cached)
        except ValueError:
            p.unlink(missing_ok=True)      # corrupt entry: drop it and ask again

    # 2) call the API; only cache what we could actually use
    out = _raw_call_with_fallback(messages, model, provider, json_out)
    if json_out:
        parsed = parse_json(out)           # raises ValueError if the model returned broken JSON
        p.write_text(out, encoding="utf-8")
        return parsed
    p.write_text(out, encoding="utf-8")
    return out


def parse_json(s):
    """Parse JSON that may be wrapped in ``` fences or surrounded by chatter. Raises ValueError."""
    s = (s or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except ValueError:
        a, b = s.find("{"), s.rfind("}")
        if a != -1 and b > a:
            try:
                return json.loads(s[a:b + 1])
            except ValueError:
                pass
        raise ValueError(f"Model did not return valid JSON (starts with: {s[:80]!r})")
