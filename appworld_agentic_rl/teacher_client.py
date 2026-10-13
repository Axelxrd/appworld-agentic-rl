"""Minimal OpenAI-compatible client used by teacher rollout collection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
import time
from typing import Any, Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_USER_AGENT = "appworld-agentic-rl/0.1"


class TeacherModelError(RuntimeError):
    """A teacher endpoint did not return a usable chat-completion response."""


class TeacherProviderError(TeacherModelError):
    """The remote endpoint could not complete a teacher request."""


class TeacherQuotaError(TeacherProviderError):
    """The provider still reports HTTP 429 after request retries are exhausted."""


class TeacherProtocolError(TeacherModelError):
    """The endpoint response did not satisfy the chat/tool protocol."""


@runtime_checkable
class TeacherModelClient(Protocol):
    """Contract used by the rollout loop, allowing offline fake clients in tests."""

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return one OpenAI-style assistant message."""


def _completion_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def _safe_response_detail(response: Any) -> str:
    """Return a bounded provider error body without exposing request credentials."""

    try:
        body = response.read().decode("utf-8", errors="replace").strip()
    except Exception:
        return ""
    if not body:
        return ""
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return body[:1000]
    if isinstance(parsed, Mapping):
        error = parsed.get("error")
        if isinstance(error, Mapping):
            message = error.get("message") or error.get("code")
            if message is not None:
                return str(message)[:1000]
        message = parsed.get("message")
        if message is not None:
            return str(message)[:1000]
    return body[:1000]


def normalize_assistant_message(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the useful subset of a Chat Completions assistant message."""

    content = payload.get("content")
    if content is not None and not isinstance(content, str):
        raise TeacherProtocolError("teacher response content must be a string or null")

    # DeepSeek thinking models require this field to be echoed on the next
    # request when an assistant message contains tool calls. Keep it optional
    # for providers that do not return it, but never discard it when present.
    reasoning_content = payload.get("reasoning_content")
    if reasoning_content is not None and not isinstance(reasoning_content, str):
        raise TeacherProtocolError("teacher response reasoning_content must be a string or null")

    raw_tool_calls = payload.get("tool_calls") or []
    if not isinstance(raw_tool_calls, Sequence) or isinstance(raw_tool_calls, (str, bytes)):
        raise TeacherProtocolError("teacher response tool_calls must be a list")

    tool_calls: list[dict[str, Any]] = []
    for index, raw_call in enumerate(raw_tool_calls):
        if not isinstance(raw_call, Mapping):
            raise TeacherProtocolError(f"teacher tool call {index} must be an object")
        function = raw_call.get("function")
        if not isinstance(function, Mapping):
            raise TeacherProtocolError(f"teacher tool call {index} has no function object")
        name = function.get("name")
        arguments = function.get("arguments", "{}")
        if not isinstance(name, str) or not name.strip():
            raise TeacherProtocolError(f"teacher tool call {index} has no function name")
        if not isinstance(arguments, (str, Mapping)):
            raise TeacherProtocolError(
                f"teacher tool call {index} arguments must be JSON text or an object"
            )
        tool_calls.append(
            {
                "id": str(raw_call.get("id") or f"teacher_call_{index}"),
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )

    message: dict[str, Any] = {"role": "assistant", "content": content}
    if isinstance(reasoning_content, str):
        message["reasoning_content"] = reasoning_content
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


class OpenAICompatibleTeacherClient:
    """Small dependency-free client for an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
        user_agent: str = DEFAULT_USER_AGENT,
        disable_provider_parallel_tool_calls: bool = False,
        disable_thinking: bool = False,
    ) -> None:
        self.model = model.strip()
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.user_agent = user_agent.strip()
        self.disable_provider_parallel_tool_calls = bool(disable_provider_parallel_tool_calls)
        self.disable_thinking = bool(disable_thinking)
        self.request_count = 0
        self.total_retries = 0
        self.total_latency_seconds = 0.0
        self.usage_totals: dict[str, int] = {}
        self.last_request_metadata: dict[str, Any] = {}
        if not self.model:
            raise ValueError("model is required")
        if not self.base_url:
            raise ValueError("OPENAI_BASE_URL or --base-url is required")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds cannot be negative")
        if not self.user_agent:
            raise ValueError("user_agent is required")

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        request_payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        # Some OpenAI-compatible providers reject or ignore this optional field.
        # Serial semantics are always enforced locally by Teacher Harness.
        if self.disable_provider_parallel_tool_calls:
            request_payload["parallel_tool_calls"] = False
        if self.disable_thinking:
            # vLLM/Qwen otherwise places XML tool calls in the reasoning field
            # instead of returning OpenAI-compatible ``tool_calls``.
            request_payload["chat_template_kwargs"] = {"enable_thinking": False}
        body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
        retryable_statuses = {429, 500, 502, 503, 504}
        last_error: TeacherModelError | None = None
        started = time.perf_counter()
        retries_before = self.total_retries
        self.request_count += 1
        usage: dict[str, int] = {}
        try:
            for attempt in range(self.max_retries + 1):
                request = Request(
                    _completion_url(self.base_url),
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                        # urllib otherwise identifies itself as Python-urllib/*,
                        # which several provider WAF configurations reject.
                        "User-Agent": self.user_agent,
                    },
                    method="POST",
                )
                try:
                    with urlopen(request, timeout=self.timeout) as response:
                        response_payload = json.loads(response.read().decode("utf-8"))
                    break
                except HTTPError as exc:
                    detail = _safe_response_detail(exc)
                    suffix = f": {detail}" if detail else ""
                    error_type = TeacherQuotaError if exc.code == 429 else TeacherProviderError
                    last_error = error_type(f"teacher endpoint returned HTTP {exc.code}{suffix}")
                    if exc.code not in retryable_statuses or attempt >= self.max_retries:
                        raise last_error from exc
                except URLError as exc:
                    last_error = TeacherProviderError(
                        f"teacher endpoint is unavailable: {exc.reason}"
                    )
                    if attempt >= self.max_retries:
                        raise last_error from exc
                except TimeoutError as exc:
                    last_error = TeacherProviderError("teacher request timed out")
                    if attempt >= self.max_retries:
                        raise last_error from exc
                except json.JSONDecodeError as exc:
                    raise TeacherProtocolError("teacher endpoint returned invalid JSON") from exc

                self.total_retries += 1
                delay = self.retry_backoff_seconds * (2**attempt)
                if delay:
                    time.sleep(delay)
            else:
                raise last_error or TeacherProviderError("teacher request failed")

            if not isinstance(response_payload, Mapping):
                raise TeacherProtocolError("teacher endpoint JSON root must be an object")
            raw_usage = response_payload.get("usage")
            if isinstance(raw_usage, Mapping):
                usage = {
                    str(key): int(value)
                    for key, value in raw_usage.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
                for key, value in usage.items():
                    self.usage_totals[key] = self.usage_totals.get(key, 0) + value
            choices = response_payload.get("choices")
            if not isinstance(choices, list) or not choices:
                raise TeacherProtocolError("teacher endpoint returned no choices")
            first_choice = choices[0]
            if not isinstance(first_choice, Mapping) or not isinstance(
                first_choice.get("message"), Mapping
            ):
                raise TeacherProtocolError("teacher endpoint choice has no assistant message")
            return normalize_assistant_message(first_choice["message"])
        finally:
            latency = time.perf_counter() - started
            self.total_latency_seconds += latency
            self.last_request_metadata = {
                "latency_seconds": round(latency, 6),
                "retry_count": self.total_retries - retries_before,
                "usage": usage,
            }

    def telemetry_snapshot(self) -> dict[str, Any]:
        return {
            "request_count": self.request_count,
            "retry_count": self.total_retries,
            "latency_seconds": round(self.total_latency_seconds, 6),
            "usage": dict(sorted(self.usage_totals.items())),
        }
