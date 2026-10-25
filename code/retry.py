"""Retry and exponential-backoff helpers for transient model/API failures.

Every outbound model or HTTP call in this project must go through
`call_with_retry` (or the `@with_retry` decorator) so transient errors
(rate limits, timeouts, 5xx responses, connection resets) are retried with
exponential backoff + jitter instead of failing the whole run.
"""
from __future__ import annotations

import functools
import logging
import random
import time
from typing import Callable, ParamSpec, TypeVar

from config import RETRY_SETTINGS, RetrySettings

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

# Exceptions treated as transient/retryable by default. Narrowed or widened
# per-call via the `retryable_exceptions` argument once real provider SDK
# exceptions (e.g. rate-limit, timeout, connection errors) are wired in.
DEFAULT_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
    OSError,
)


class RetryExhaustedError(RuntimeError):
    """Raised when all retry attempts for a call have been exhausted."""


def _compute_delay(attempt: int, settings: RetrySettings) -> float:
    """Exponential backoff with full jitter, capped at `max_delay_s`."""
    exponential = settings.base_delay_s * (2 ** (attempt - 1))
    capped = min(exponential, settings.max_delay_s)
    jitter = random.uniform(0, settings.jitter_s)
    return capped + jitter


def call_with_retry(
    func: Callable[..., T],
    *args,
    settings: RetrySettings = RETRY_SETTINGS,
    retryable_exceptions: tuple[type[BaseException], ...] = DEFAULT_RETRYABLE_EXCEPTIONS,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    **kwargs,
) -> T:
    """Invoke `func(*args, **kwargs)`, retrying on transient failures.

    Raises `RetryExhaustedError` (chained to the last exception) once
    `settings.max_attempts` have failed.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, settings.max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except retryable_exceptions as exc:  # noqa: PERF203 - clarity over micro-perf
            last_exc = exc
            if attempt == settings.max_attempts:
                break
            delay = _compute_delay(attempt, settings)
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            else:
                logger.warning(
                    "Transient failure on attempt %s/%s for %s: %s. Retrying in %.2fs.",
                    attempt,
                    settings.max_attempts,
                    getattr(func, "__name__", repr(func)),
                    exc,
                    delay,
                )
            time.sleep(delay)
    raise RetryExhaustedError(
        f"{getattr(func, '__name__', repr(func))} failed after {settings.max_attempts} attempts"
    ) from last_exc


def with_retry(
    settings: RetrySettings = RETRY_SETTINGS,
    retryable_exceptions: tuple[type[BaseException], ...] = DEFAULT_RETRYABLE_EXCEPTIONS,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator form of `call_with_retry` for wrapping API/model call sites."""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return call_with_retry(
                func,
                *args,
                settings=settings,
                retryable_exceptions=retryable_exceptions,
                **kwargs,
            )

        return wrapper

    return decorator
