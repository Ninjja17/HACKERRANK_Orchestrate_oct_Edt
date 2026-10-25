"""Provider-agnostic LLM client with retry/backoff wired in from the start.

This module isolates the one place where an outbound model call happens
(`LLMClient._raw_complete_with_tools`). The public `complete_with_tools`
method always goes through `call_with_retry` so every model call gets
exponential backoff on transient failures (rate limits, timeouts,
connection errors, 5xx) — this is what powers the agent's tool-calling loop
in `agent.py`.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import openai

from config import MODEL_SETTINGS, RETRY_SETTINGS, ModelSettings, RetrySettings
from retry import DEFAULT_RETRYABLE_EXCEPTIONS, call_with_retry

logger = logging.getLogger(__name__)

# Provider-specific transient errors, retried with the same backoff policy
# as generic network errors.
LLM_RETRYABLE_EXCEPTIONS = DEFAULT_RETRYABLE_EXCEPTIONS + (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class ModelResponseError(RuntimeError):
    """Raised when the model response is unusable (e.g. empty content)."""


@dataclass
class ModelUsage:
    """Token accounting for a single model call, for the usage/cost report."""

    provider: str
    model_name: str
    input_tokens: int
    output_tokens: int


class LLMClient:
    """Thin wrapper around one model provider, with retry/backoff baked in.

    Safe to share across threads: one instance (and its rate limiter and
    usage log) is used by every concurrent worker in `pipeline.run`, so the
    requests-per-minute cap and token-usage accounting stay accurate
    account-wide regardless of how many cases are in flight at once.
    """

    def __init__(
        self,
        model_settings: ModelSettings = MODEL_SETTINGS,
        retry_settings: RetrySettings = RETRY_SETTINGS,
    ) -> None:
        if not model_settings.api_key:
            raise RuntimeError(
                "ORCHESTRATOR_API_KEY is not set. Copy code/.env.example to code/.env and fill it in."
            )
        self.model_settings = model_settings
        self.retry_settings = retry_settings
        self.usage_log: list[ModelUsage] = []
        self._usage_lock = threading.Lock()
        self._request_times: deque[float] = deque()
        self._throttle_lock = threading.Lock()
        self._client = openai.OpenAI(
            api_key=model_settings.api_key,
            base_url=model_settings.api_base or None,
        )

    def _throttle(self) -> None:
        """Sleep if needed to stay under `max_requests_per_minute`.

        Proactive, not reactive: avoids spending retry attempts (and quota)
        on 429s that a known rate limit would have made predictable, e.g. a
        provider free tier capped at 15 requests/minute. The check, sleep,
        and reservation happen under one lock so the global cap holds even
        when called concurrently from multiple worker threads.
        """
        limit = self.model_settings.max_requests_per_minute
        if limit <= 0:
            return
        window_s = 60.0
        with self._throttle_lock:
            now = time.monotonic()
            while self._request_times and now - self._request_times[0] > window_s:
                self._request_times.popleft()
            if len(self._request_times) >= limit:
                sleep_for = window_s - (now - self._request_times[0]) + 0.1
                if sleep_for > 0:
                    logger.info("Throttling: sleeping %.1fs to stay under %s requests/minute", sleep_for, limit)
                    time.sleep(sleep_for)
            self._request_times.append(time.monotonic())

    def complete_with_tools(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        """Call the model with function-calling tools enabled.

        Retries transient failures with exponential backoff. Returns the
        raw assistant message (may carry `.tool_calls`); the caller (the
        agent loop) is responsible for dispatching any tool calls and
        feeding results back.
        """
        return call_with_retry(
            self._raw_complete_with_tools,
            messages,
            tools,
            settings=self.retry_settings,
            retryable_exceptions=LLM_RETRYABLE_EXCEPTIONS,
        )

    def _raw_complete_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        """Single, un-retried call to the configured model provider."""
        self._throttle()
        response = self._client.chat.completions.create(
            model=self.model_settings.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=self.model_settings.temperature,
            max_tokens=self.model_settings.max_output_tokens,
            timeout=self.model_settings.request_timeout_s,
        )
        if response.usage is not None:
            with self._usage_lock:
                self.usage_log.append(
                    ModelUsage(
                        provider=self.model_settings.provider,
                        model_name=self.model_settings.model_name,
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                    )
                )
        message = response.choices[0].message
        if not message.content and not message.tool_calls:
            raise ModelResponseError("Model returned neither content nor a tool call")
        return message
