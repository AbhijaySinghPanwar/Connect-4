from abc import ABC, abstractmethod
from groq import Groq
import google.generativeai as genai
import logging
from typing import Dict, Type, List, Optional
import os
import time
import json
import re
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# groq/compound and groq/compound-mini removed: they are agentic models that
# auto-activate web search / code execution, producing tool_call objects the
# JSON pipeline cannot parse.  Evidence: audit Step 4, confirmed by Groq docs.
# ---------------------------------------------------------------------------
MODEL_PROVIDER = {
    "openai/gpt-oss-120b": "groq",
    "openai/gpt-oss-20b": "groq",
    "qwen/qwen3-32b": "groq",
    "qwen/qwen3.6-27b": "groq",
    "llama-3.3-70b-versatile": "groq",
    "meta-llama/llama-4-scout-17b-16e-instruct": "groq",
    "llama-3.1-8b-instant": "groq",
    "gemini-2.5-flash": "gemini",
}

# ---------------------------------------------------------------------------
# Per-model capability flags
# Only set for models whose support is confirmed in provider documentation.
# ---------------------------------------------------------------------------

# Groq: models that support reasoning_format='hidden'.
# Routes thinking tokens out of message.content so they never reach the parser.
# Evidence: Groq docs — "qwen3 models support the following values: none, default"
# and reasoning_format hidden/raw/parsed.
GROQ_SUPPORTS_REASONING_FORMAT = {
    "qwen/qwen3-32b",
    "qwen/qwen3.6-27b",
}

# Groq: models that route reasoning into a separate message.reasoning field
# (not into message.content).  reasoning_format is NOT supported for these;
# reasoning_effort controls the budget.
# Evidence: Groq docs — "with gpt-oss-20b and gpt-oss-120b, the reasoning_format
# parameter is not supported. By default, these models will include reasoning
# content in the reasoning field of the assistant response."
GROQ_REASONING_IN_SEPARATE_FIELD = {
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
}

# Groq: models confirmed to support response_format={"type":"json_object"}.
# meta-llama/llama-4-scout-17b-16e-instruct excluded: vision model,
# json_object support not confirmed.
GROQ_SUPPORTS_JSON_MODE = {
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "qwen/qwen3.6-27b",
}

# Gemini: models where thinking tokens consume max_output_tokens budget.
# Fix: set thinking_budget=0 to disable thinking so the full token budget is
# available for the JSON response.
# Evidence: 15+ GitHub issues on googleapis/python-genai and langchain-google
# confirming empty response.text / finishReason=MAX_TOKENS when thinking budget
# exhausts max_output_tokens.
GEMINI_SUPPORTS_THINKING_CONFIG = {
    "gemini-2.5-flash",
}

# ---------------------------------------------------------------------------
# Max tokens — configurable via environment variable
# ---------------------------------------------------------------------------
_DEFAULT_MAX_TOKENS = 4096


def _get_max_tokens() -> int:
    """
    Return max tokens from the LLM_MAX_TOKENS environment variable.
    Falls back to 4096 if unset or invalid.
    """
    raw = os.getenv("LLM_MAX_TOKENS")
    if raw is None:
        return _DEFAULT_MAX_TOKENS
    try:
        value = int(raw)
        if value <= 0:
            raise ValueError("must be positive")
        return value
    except (ValueError, TypeError):
        logger.warning(
            f"LLM_MAX_TOKENS={raw!r} is not a valid positive integer; "
            f"using default {_DEFAULT_MAX_TOKENS}"
        )
        return _DEFAULT_MAX_TOKENS


# ---------------------------------------------------------------------------
# Response cleanup helpers
# ---------------------------------------------------------------------------

def strip_thinking(text: str) -> str:
    """
    Remove <think>...</think> blocks from model output.

    Three cases handled:

    1. No <think> tag present
       Strip any stray <think>...</think> via regex and return.

    2. Complete block: <think>...</think> present
       Return everything after the final </think>.

    3. Truncated block: <think> present but </think> never arrived
       (model hit max_tokens mid-thought and never emitted the answer).
       Scan the full text for any recoverable JSON rather than discarding
       the entire response.
       - First: text before the <think> tag (model preamble)
       - Then: any {…} span anywhere in the full text
       - Finally: return "" only when truly nothing is recoverable

    Proven failure path without this fix (audit Step 1 + Step 5, Failure 1):
       raw = '<think>\\nAnalyzing...'   # truncated, no </think>
       old strip_thinking() → ''       # before_think is '' for all reasoning models
       json.loads('') → JSONDecodeError → forfeit
    """
    if "<think>" not in text:
        # No thinking block: strip any stray tags via regex
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return cleaned.strip()

    if "</think>" in text:
        # Complete block: return everything after the last closing tag
        after = text.split("</think>")[-1].strip()
        if after:
            return after
        # Nothing follows the closing tag — fall through to scan for JSON

    # Truncated or empty-after case.
    # Preference 1: text before the <think> tag
    before_think = text.split("<think>", 1)[0].strip()
    if before_think and "{" in before_think:
        return before_think

    # Preference 2: any { … } span in the full text
    # (recovers JSON accidentally emitted inside a thinking block)
    left = text.find("{")
    right = text.rfind("}")
    if left != -1 and right != -1 and right > left:
        return text[left:right + 1]

    # Nothing recoverable
    return ""


def strip_markdown(text: str) -> str:
    """Remove markdown formatting like ```json and ```"""
    text = re.sub(r'```(?:json)?\n?(.*?)\n?```', r'\1', text, flags=re.DOTALL)
    return text.strip()


def extract_json(text: str) -> str:
    """Extract JSON from the text by finding the first { and last }"""
    left = text.find("{")
    right = text.rfind("}")
    if left > -1 and right > -1:
        return text[left:right + 1]
    return text


def clean_response(text: str) -> str:
    """Apply all cleanup methods to get raw JSON string"""
    if not text:
        return "{}"
    text = strip_thinking(text)
    text = strip_markdown(text)
    text = extract_json(text)
    # Guard: if all stages produced an empty string, return minimal valid JSON
    # so that process_move() fails cleanly on a missing move_column rather than
    # on a JSONDecodeError.
    return text.strip() or "{}"


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

def _is_rate_limit_error(exc: Exception) -> bool:
    """
    Detect a 429 rate-limit error without hard-importing both providers'
    exception hierarchies. Checks the common `status_code` attribute (Groq's
    SDK exposes this on RateLimitError) and falls back to matching the
    exception class name (covers google.api_core.exceptions.ResourceExhausted
    / TooManyRequests from the Gemini SDK).
    """
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    exc_name = type(exc).__name__
    return exc_name in ("RateLimitError", "ResourceExhausted", "TooManyRequests")


def _rate_limit_backoff_seconds(attempt: int) -> float:
    """
    Exponential backoff for rate-limit retries: 4s, 8s, 16s, ...
    A per-minute quota (e.g. Gemini free-tier's ~10 RPM) needs real wall-clock
    time to pass before it resets; a 1-second wait just fails again immediately.
    """
    return min(4 * (2 ** (attempt - 1)), 30)


class LLMParsingException(Exception):
    pass


class LLM(ABC):
    """
    An abstract superclass for interacting with LLMs - subclass for Groq and Gemini
    """

    model_names = []

    def __init__(self, model_name: str, temperature: float = 0.0, max_tokens: int = None):
        self.model_name = model_name
        self.temperature = temperature
        # max_tokens comes from the caller (backward compat) or the env var
        self.max_tokens = max_tokens if max_tokens is not None else _get_max_tokens()

    def send(self, system: str, user: str, max_tokens: int = None) -> str:
        """
        Send a message to the model.
        :param system: the context in which this message is to be taken
        :param user: the prompt
        :param max_tokens: unused; kept for backward compatibility.
                           Token limit is controlled by LLM_MAX_TOKENS env var.
        :return: the cleaned JSON response string
        """
        result = self.protected_send(system, user)
        result = clean_response(result)
        logger.debug(f"[{self.model_name}] Cleaned response (first 200): {result[:200]}")
        return result

    def protected_send(self, system: str, user: str) -> str:
        """
        Wrap _send in an exception handler, retrying on transient errors.
        Rate-limit errors (HTTP 429) get longer, exponentially-increasing waits
        since they need real time to pass before the provider's per-minute
        quota resets — a short fixed wait (e.g. 1s) will just fail again on
        the very next attempt. Other transient errors retry quickly as before.
        Returns '{}' if all attempts fail.
        """
        max_attempts = 3
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            try:
                start_time = time.time()
                raw = self._send(system, user)
                latency = time.time() - start_time
                logger.debug(f"[{self.model_name}] API call took {latency:.2f}s")
                if not raw:
                    logger.error(
                        f"[{self.model_name}] API returned empty/None content"
                    )
                    raw = "{}"
                else:
                    logger.debug(
                        f"[{self.model_name}] Raw response (first 300): {raw[:300]}"
                    )
                return raw
            except Exception as e:
                is_rate_limit = _is_rate_limit_error(e)
                logger.error(
                    f"[{self.model_name}] Exception calling LLM "
                    f"(attempt {attempt}/{max_attempts}, rate_limit={is_rate_limit}): {e}"
                )
                if attempt < max_attempts:
                    wait_s = _rate_limit_backoff_seconds(attempt) if is_rate_limit else 1
                    logger.debug(f"[{self.model_name}] Waiting {wait_s}s before retry...")
                    time.sleep(wait_s)
        return "{}"

    @abstractmethod
    def _send(self, system: str, user: str) -> str:
        """Send a message to the model — implemented by subclasses."""
        pass

    @classmethod
    def all_supported_model_names(cls) -> List[str]:
        """Return all model names registered in MODEL_PROVIDER."""
        return list(MODEL_PROVIDER.keys())

    @classmethod
    def all_model_names(cls) -> List[str]:
        """
        Return the list of models to use in the arena.
        Respects the MODELS env var to restrict which models are active.
        """
        models = cls.all_supported_model_names()
        allowed = os.getenv("MODELS")
        logger.info(f"Allowed models env: {allowed}")
        if allowed:
            allowed_models = allowed.split(",")
            return [m for m in allowed_models if m in models]
        return models

    @classmethod
    def create(cls, model_name: str, temperature: float = 0.0) -> 'LLM':
        """
        Return a provider-specific LLM subclass for the given model name.
        """
        provider = MODEL_PROVIDER.get(model_name)
        if not provider:
            raise ValueError(f"Unrecognized LLM model name specified: {model_name}")

        logger.info(f"Creating LLM client for {model_name} via {provider}")
        if provider == "groq":
            return GroqLLM(model_name, temperature)
        elif provider == "gemini":
            return GeminiLLM(model_name, temperature)
        else:
            raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Groq provider
# ---------------------------------------------------------------------------

class GroqLLM(LLM):
    """Interface to the Groq API."""

    def __init__(self, model_name: str, temperature: float = 0.0):
        super().__init__(model_name, temperature)
        api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=api_key, timeout=60.0)

    def _send(self, system: str, user: str) -> str:
        kwargs = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Provider-specific: hide thinking tokens from content for Qwen models.
        # Without this, <think>…</think> appears in content and causes the
        # strip_thinking truncation failure when max_tokens is hit mid-thought.
        # Evidence: Groq API reference — reasoning_format='hidden' supported for
        # qwen3 family; mutually exclusive with include_reasoning.
        # Note: when reasoning_format is set, only 'parsed' or 'hidden' can be
        # used alongside json_object mode — 'hidden' satisfies both constraints.
        if self.model_name in GROQ_SUPPORTS_REASONING_FORMAT:
            kwargs["reasoning_format"] = "hidden"
            logger.debug(f"[{self.model_name}] reasoning_format=hidden applied")

        # Provider-specific: enforce JSON output at the API level for supported models.
        # Eliminates parse failures from prose preambles or markdown wrapping.
        if self.model_name in GROQ_SUPPORTS_JSON_MODE:
            kwargs["response_format"] = {"type": "json_object"}
            logger.debug(f"[{self.model_name}] response_format=json_object applied")

        response = self.client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        message = choice.message
        usage = response.usage

        # ---- Full response instrumentation (DEBUG) ----
        finish_reason = choice.finish_reason
        logger.debug(f"[{self.model_name}] finish_reason={finish_reason}")
        logger.debug(
            f"[{self.model_name}] token usage — "
            f"prompt={getattr(usage, 'prompt_tokens', None)} "
            f"completion={getattr(usage, 'completion_tokens', None)} "
            f"total={getattr(usage, 'total_tokens', None)}"
        )
        logger.debug(
            f"[{self.model_name}] message.content "
            f"(len={len(message.content) if message.content else 0}): "
            f"{str(message.content)[:300]}"
        )

        # Inspect reasoning field (gpt-oss models route CoT here, not to content)
        reasoning_field = getattr(message, "reasoning", None)
        if reasoning_field:
            logger.debug(
                f"[{self.model_name}] message.reasoning present "
                f"(len={len(reasoning_field)}): {str(reasoning_field)[:200]}"
            )

        # Inspect tool_calls (compound models and unexpected agentic responses)
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            logger.debug(f"[{self.model_name}] message.tool_calls: {tool_calls}")

        # Inspect refusal field
        refusal = getattr(message, "refusal", None)
        if refusal:
            logger.debug(f"[{self.model_name}] message.refusal: {refusal}")

        if finish_reason == "length":
            logger.error(
                f"[{self.model_name}] Response TRUNCATED — hit max_tokens={self.max_tokens}. "
                f"Increase LLM_MAX_TOKENS in .env to reduce forfeit risk."
            )

        content = message.content

        if not content:
            logger.error(
                f"[{self.model_name}] message.content is empty or None. "
                f"finish_reason={finish_reason} | "
                f"reasoning_present={bool(reasoning_field)} | "
                f"tool_calls_present={bool(tool_calls)}"
            )

        return content or "{}"


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------

class GeminiLLM(LLM):
    """Interface to the Google Gemini API."""

    def __init__(self, model_name: str, temperature: float = 0.0):
        super().__init__(model_name, temperature)
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)

    def _send(self, system: str, user: str) -> str:
        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "response_mime_type": "application/json",
        }

        # Provider-specific: disable thinking for models where thinking tokens
        # are deducted from the same max_output_tokens budget as visible output.
        # Without this, Gemini 2.5 Flash consumes the entire 4096-token budget
        # on internal thinking and returns response.text = '' with
        # finishReason=MAX_TOKENS.
        # Evidence: 15+ GitHub issues on googleapis/python-genai (#782, #811),
        # langchain-google (#1020), and published developer analysis confirming
        # thinking tokens eat the same pool as output tokens.
        if self.model_name in GEMINI_SUPPORTS_THINKING_CONFIG:
            generation_config["thinking_config"] = {"thinking_budget": 0}
            logger.debug(
                f"[{self.model_name}] thinking_config.thinking_budget=0 applied"
            )

        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system,
            generation_config=generation_config,
        )
        response = model.generate_content(
            user,
            request_options={"timeout": 60.0}
        )

        # ---- Full response instrumentation (DEBUG) ----
        candidate = response.candidates[0] if response.candidates else None
        finish_reason = candidate.finish_reason if candidate else None
        usage = getattr(response, "usage_metadata", None)

        logger.debug(f"[{self.model_name}] finish_reason={finish_reason}")
        if usage:
            logger.debug(
                f"[{self.model_name}] token usage — "
                f"prompt={getattr(usage, 'prompt_token_count', None)} "
                f"candidates={getattr(usage, 'candidates_token_count', None)} "
                f"thoughts={getattr(usage, 'thoughts_token_count', None)} "
                f"total={getattr(usage, 'total_token_count', None)}"
            )

        prompt_feedback = getattr(response, "prompt_feedback", None)
        if prompt_feedback:
            logger.debug(
                f"[{self.model_name}] prompt_feedback={prompt_feedback}"
            )

        if finish_reason and str(finish_reason) in ("FinishReason.MAX_TOKENS", "MAX_TOKENS"):
            logger.error(
                f"[{self.model_name}] Response TRUNCATED — hit max_output_tokens={self.max_tokens}. "
                f"If thoughts_token_count is non-zero, thinking_config may not be taking effect "
                f"(may require migration to google.genai package)."
            )

        # response.text raises ValueError/AttributeError when candidates is empty
        # or content.parts has no text (documented Gemini 2.5 Flash failure mode).
        try:
            text = response.text
        except Exception as exc:
            logger.error(
                f"[{self.model_name}] response.text raised {type(exc).__name__}: {exc} "
                f"(finish_reason={finish_reason})"
            )
            text = None

        logger.debug(
            f"[{self.model_name}] response.text "
            f"(len={len(text) if text else 0}): {str(text)[:300]}"
        )

        return text or "{}"
