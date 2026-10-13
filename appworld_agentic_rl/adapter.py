"""Agent-tool AppWorld environment adapter for SFT and GRPO rollouts."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from appworld_agentic_rl.reward import compute_terminal_reward
from appworld_agentic_rl.harness import make_violation
from appworld_agentic_rl.trajectory import TrajectoryRecord, TrajectoryStep


EVALUATOR_SUBPROCESS_TIMEOUT_SECONDS = 300.0


def _evaluate_appworld_subprocess(
    *,
    task_id: str,
    experiment_name: str,
    appworld_root: Path | None,
    suppress_errors: bool,
) -> dict[str, Any]:
    """Evaluate persisted AppWorld state in a pristine Python interpreter."""

    with tempfile.NamedTemporaryFile(
        prefix="appworld_evaluation_",
        suffix=".json",
        delete=False,
    ) as stream:
        result_path = Path(stream.name)
    command = [
        sys.executable,
        "-m",
        "appworld_agentic_rl.evaluator_worker",
        "--task-id",
        task_id,
        "--experiment-name",
        experiment_name,
        "--output",
        str(result_path),
    ]
    if appworld_root is not None:
        command.extend(["--appworld-root", str(appworld_root.resolve())])
    if suppress_errors:
        command.append("--suppress-errors")

    try:
        child_environment = os.environ.copy()
        child_environment["PYTHONIOENCODING"] = "utf-8"
        child_environment["PYTHONUTF8"] = "1"
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_environment,
            timeout=EVALUATOR_SUBPROCESS_TIMEOUT_SECONDS,
            check=False,
        )
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            stderr = completed.stderr.strip()[-2000:]
            detail = f"; stderr: {stderr}" if stderr else ""
            raise RuntimeError(
                f"evaluator subprocess returned no valid result "
                f"(exit code {completed.returncode}){detail}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise TypeError("evaluator subprocess result must be an object")
        if payload.get("error"):
            raise RuntimeError(str(payload["error"]))
        evaluation = payload.get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise TypeError("evaluator subprocess result has no evaluation object")
        return dict(evaluation)
    finally:
        result_path.unlink(missing_ok=True)


WRITE_API_PATTERN = re.compile(
    r"\bapis\.[A-Za-z_][\w]*\."
    r"(create|update|delete|send|pay|add|remove|cancel|complete|submit|book|"
    r"transfer|purchase|order|refund|set|mark)_[A-Za-z_][\w]*\s*\("
)

EXECUTION_ERROR_PREFIXES = (
    "Execution failed.",
    "No code available to execute.",
)

QUERY_EXPLORATION_PATTERN = re.compile(
    r"(?:\b(?:inspect|getsource|dir|help|globals|locals)\s*\(|"
    r"__dict__|__file__|__code__|\bimport\s+appworld\b|\bfrom\s+appworld\b|"
    r"\b(?:open|compile|eval|exec)\s*\()",
    re.IGNORECASE,
)

FORBIDDEN_RUNTIME_ROOTS = {
    "appworld",
    "httpx",
    "os",
    "pathlib",
    "requests",
    "socket",
    "subprocess",
    "sys",
    "urllib",
}
FORBIDDEN_RUNTIME_CALLS = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "globals",
    "help",
    "input",
    "locals",
    "open",
}

AGENT_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "inspect_task",
            "description": "Return task instruction, current datetime, supervisor summary, and allowed apps.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_api_docs",
            "description": "Search AppWorld API docs before unfamiliar API calls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_api_schema",
            "description": "Return the exact schema for one app API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app": {"type": "string"},
                    "api": {"type": "string"},
                },
                "required": ["app", "api"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_state",
            "description": (
                "Execute read-only AppWorld Python code. The preloaded API namespace is "
                "apis: call only apis.<app>.<api>(...). Variables persist across agent "
                "code calls. Do not import appworld, use HTTP, or guess another client namespace."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": (
                "Execute non-mutating computation or formatting code. App APIs are available "
                "only as apis.<app>.<api>(...), and variables persist across code calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_write_code",
            "description": (
                "Execute code that intentionally mutates AppWorld state. Call APIs only as "
                "apis.<app>.<api>(...); variables persist from earlier agent code calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "intended_change": {"type": "string"},
                },
                "required": ["code", "intended_change"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_task",
            "description": "Submit an optional concise answer and signal AppWorld supervisor completion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "description": "Answer for question tasks; omit for state-change tasks.",
                        "type": ["string", "number", "integer", "boolean", "null"],
                    },
                    "status": {
                        "type": "string",
                        "enum": ["success", "fail"],
                        "default": "success",
                    },
                    "summary": {
                        "description": "Optional local trace summary; not sent as the answer.",
                        "type": "string",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_without_completion",
            "description": "Terminate when no valid safe completion exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "enum": [
                            "missing_information",
                            "api_unavailable",
                            "unsafe_or_ambiguous",
                            "no_valid_solution",
                        ],
                    }
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
]


def parse_tool_call(tool_call: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Accept compact or OpenAI-style tool call objects."""

    if "function" in tool_call:
        function = tool_call.get("function") or {}
        if not isinstance(function, Mapping):
            raise TypeError("tool_call.function must be an object")
        name = str(function.get("name", "")).strip()
        raw_args = function.get("arguments", {})
    else:
        name = str(tool_call.get("name", "")).strip()
        raw_args = tool_call.get("arguments", {})

    if isinstance(raw_args, str):
        args = json.loads(raw_args or "{}")
    elif isinstance(raw_args, Mapping):
        args = dict(raw_args)
    else:
        raise TypeError("tool_call arguments must be an object or JSON string")
    if not name:
        raise ValueError("tool_call name is required")
    return name, args


def looks_state_changing(code: str) -> bool:
    return bool(WRITE_API_PATTERN.search(code))


def _attribute_path(node: ast.AST) -> tuple[str, ...] | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return tuple(reversed(parts))


_SENSITIVE_VARIABLE_PATTERN = re.compile(
    r"(?:token|password|secret|credential|authorization|auth|session)",
    re.IGNORECASE,
)


def _assigned_names(target: ast.AST) -> list[str]:
    """Return simple names introduced by an assignment target."""

    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for element in target.elts:
            names.extend(_assigned_names(element))
        return names
    return []


def _subscript_field(node: ast.AST) -> str | None:
    """Extract a literal mapping field from an expression such as result['token']."""

    if not isinstance(node, ast.Subscript):
        return None
    slice_node = node.slice
    if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
        return slice_node.value
    return None


def _variable_binding_kind(value: ast.AST | None) -> tuple[str, str | None, str | None]:
    """Return a non-secret semantic description for an assigned value."""

    if isinstance(value, ast.Call):
        path = _attribute_path(value.func)
        if path and len(path) == 3 and path[0] == "apis":
            return "api_result", path[1], None
    if isinstance(value, ast.Subscript):
        parent = value.value
        if isinstance(parent, ast.Name):
            field = _subscript_field(value)
            if field:
                return "derived_field", None, f"{parent.id}[{field}]"
    if isinstance(value, ast.Name):
        return "derived_variable", None, value.id
    return "local", None, None


def extract_variable_bindings(code: str) -> list[dict[str, Any]]:
    """Extract safe variable-binding metadata without evaluating model code.

    The returned records intentionally contain no literal values. They describe
    names and provenance only; the adapter's persistent execution namespace
    remains the source of the actual values.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    bindings: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        targets: list[ast.AST] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        elif isinstance(node, ast.NamedExpr):
            targets = [node.target]
            value = node.value
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
            value = node.value
        if not targets:
            continue

        kind, app, derived_from = _variable_binding_kind(value)
        for target in targets:
            for name in _assigned_names(target):
                semantic = "authentication" if _SENSITIVE_VARIABLE_PATTERN.search(name) else "agent_state"
                if derived_from and _SENSITIVE_VARIABLE_PATTERN.search(derived_from):
                    semantic = "authentication"
                bindings.append(
                    {
                        "name": name,
                        "kind": kind,
                        "semantic": semantic,
                        "app": app,
                        "derived_from": derived_from,
                    }
                )

    # A deterministic last assignment wins, matching normal sequential code
    # for common straight-line agent code snippets.
    deduplicated: dict[str, dict[str, Any]] = {}
    for binding in bindings:
        deduplicated[str(binding["name"])] = binding
    return [deduplicated[name] for name in sorted(deduplicated)]


def _provenance_scalar(value: Any) -> str | None:
    """Normalize a scalar for equality checks without storing it in metadata."""

    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none"
    if isinstance(value, (str, int, float)):
        return str(value).strip().casefold()
    return None


def _collect_provenance_values(value: Any, output: set[str]) -> None:
    """Collect scalar values from prior actor-visible observations."""

    if isinstance(value, Mapping):
        for child in value.values():
            _collect_provenance_values(child, output)
    elif isinstance(value, (list, tuple, set)):
        for child in list(value)[:64]:
            _collect_provenance_values(child, output)
    else:
        normalized = _provenance_scalar(value)
        if normalized:
            output.add(normalized)


def _provenance_expression(
    node: ast.AST,
    *,
    observed_values: set[str],
    initial_text: str,
    bindings: Mapping[str, Mapping[str, Any]],
) -> tuple[str, bool]:
    """Classify an API argument expression and whether its source is verified."""

    if isinstance(node, ast.Constant):
        normalized = _provenance_scalar(node.value)
        if normalized and normalized in observed_values:
            return "observed_fact", True
        if isinstance(node.value, str) and node.value.strip().casefold() in initial_text.casefold():
            return "task_instruction", True
        return "literal_unverified", False
    if isinstance(node, ast.Name):
        binding = bindings.get(node.id)
        if binding and bool(binding.get("available", False)):
            return "variable_binding", True
        return "unknown_variable", False
    if isinstance(node, (ast.Attribute, ast.Subscript)):
        root: ast.AST = node.value
        while isinstance(root, (ast.Attribute, ast.Subscript)):
            root = root.value
        if isinstance(root, ast.Name):
            binding = bindings.get(root.id)
            if binding and bool(binding.get("available", False)):
                return "derived_binding", True
        return "unknown_derived_value", False
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        parts = [
            _provenance_expression(
                child,
                observed_values=observed_values,
                initial_text=initial_text,
                bindings=bindings,
            )
            for child in node.elts
        ]
        return "derived_literal", bool(parts) and all(verified for _, verified in parts)
    if isinstance(node, ast.Dict):
        parts = [
            _provenance_expression(
                child,
                observed_values=observed_values,
                initial_text=initial_text,
                bindings=bindings,
            )
            for child in [*node.keys, *node.values]
            if child is not None
        ]
        return "derived_literal", bool(parts) and all(verified for _, verified in parts)
    if isinstance(node, ast.BinOp):
        left = _provenance_expression(
            node.left,
            observed_values=observed_values,
            initial_text=initial_text,
            bindings=bindings,
        )
        right = _provenance_expression(
            node.right,
            observed_values=observed_values,
            initial_text=initial_text,
            bindings=bindings,
        )
        return "derived_expression", left[1] and right[1]
    return "unknown_expression", False


def extract_api_call_provenance(
    code: str,
    *,
    docs: Mapping[tuple[str, str], Mapping[str, Any]],
    observed_values: set[str],
    initial_text: str = "",
    bindings: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Extract non-secret provenance metadata for API call parameters.

    This function never evaluates code and never emits literal argument values.
    A write is considered grounded only when all required parameters (or all
    parameters when the schema has no required list) have a verified source.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    known_bindings = bindings or {}
    records: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        path = _attribute_path(node.func)
        if path is None or len(path) != 3 or path[0] != "apis":
            continue
        app, api = path[1], path[2]
        doc = docs.get((app, api))
        if not isinstance(doc, Mapping) or not _documented_api_is_mutating(api, doc):
            continue
        required, _allowed = _documented_parameters(doc)
        parameters: list[dict[str, Any]] = []
        for keyword in node.keywords:
            if keyword.arg is None:
                parameters.append({"name": "**kwargs", "source": "unknown_expression", "verified": False})
                continue
            source, verified = _provenance_expression(
                keyword.value,
                observed_values=observed_values,
                initial_text=initial_text,
                bindings=known_bindings,
            )
            parameters.append({"name": str(keyword.arg), "source": source, "verified": verified})
        required_names = sorted(required)
        by_name = {str(item["name"]): item for item in parameters}
        gate_names = required_names or sorted(by_name)
        missing_required = [name for name in required_names if name not in by_name]
        verified = not missing_required and all(bool(by_name[name].get("verified")) for name in gate_names)
        records.append(
            {
                "app": app,
                "api": api,
                "parameters": parameters,
                "required_parameters": required_names,
                "missing_required_parameters": missing_required,
                "all_required_verified": verified,
            }
        )
    return records


def _runtime_safety_errors(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"Python syntax error at line {exc.lineno}: {exc.msg}"]

    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            errors.append("imports are not allowed in agent code")
            continue
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            errors.append(f"runtime introspection attribute {node.attr!r} is not allowed")
            continue
        if not isinstance(node, ast.Call):
            continue
        path = _attribute_path(node.func)
        if path is None:
            continue
        if path[0] in FORBIDDEN_RUNTIME_ROOTS:
            errors.append(f"external/runtime namespace {path[0]!r} is not allowed")
        if len(path) == 1 and path[0] in FORBIDDEN_RUNTIME_CALLS:
            errors.append(f"runtime function {path[0]!r} is not allowed")
        if path[-1] in {"dir", "getsource"}:
            errors.append(f"runtime introspection function {path[-1]!r} is not allowed")
    return list(dict.fromkeys(errors))


def _documented_parameters(doc: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    parameters = doc.get("parameters")
    required: set[str] = set()
    allowed: set[str] = set()
    if isinstance(parameters, list):
        for item in parameters:
            if not isinstance(item, Mapping) or not item.get("name"):
                continue
            name = str(item["name"])
            allowed.add(name)
            if bool(item.get("required")):
                required.add(name)
    elif isinstance(parameters, Mapping):
        for name in parameters.get("required") or []:
            required.add(str(name))
            allowed.add(str(name))
        for name in parameters.get("optional") or []:
            allowed.add(str(name))
    return required, allowed


def _documented_api_is_mutating(api: str, doc: Mapping[str, Any]) -> bool:
    method = str(doc.get("method") or "").upper()
    if not method or method == "GET":
        return False
    # Authentication and search endpoints can legitimately use POST while not
    # changing benchmark state. Other non-GET operations are treated as writes.
    non_mutating_prefixes = ("login", "search", "show", "get", "check", "verify")
    return not api.casefold().startswith(non_mutating_prefixes)


def execution_error_from_observation(observation: Any) -> str | None:
    """Return a stable error summary for AppWorld string-level execution failures."""

    if not isinstance(observation, str):
        return None
    message = observation.strip()
    if not message.startswith(EXECUTION_ERROR_PREFIXES):
        return None
    first_lines = [line.strip() for line in message.splitlines() if line.strip()]
    return "\n".join(first_lines[:6])


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict())
        except TypeError:
            pass
    if hasattr(value, "dict"):
        try:
            return _jsonable(value.dict())
        except TypeError:
            pass
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _short_doc(app: str, api: str, doc: Mapping[str, Any]) -> dict[str, Any]:
    description = (
        doc.get("description")
        or doc.get("docstring")
        or doc.get("summary")
        or doc.get("api_description")
        or ""
    )
    return {
        "app": app,
        "api": api,
        "description": str(description)[:500],
        "parameters": _jsonable(doc.get("parameters") or doc.get("request_schema") or {}),
        "returns": _jsonable(doc.get("response_schema") or doc.get("returns") or {}),
    }


class AppWorldAdapter:
    """Small AppWorld wrapper exposing agent-selectable tools instead of 457 raw APIs."""

    def __init__(
        self,
        *,
        experiment_name: str = "appworld_agentic_rl_mvp",
        appworld_root: str | Path | None = None,
        max_steps: int = 50,
        allow_state_change_in_run_code: bool = False,
        appworld_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.appworld_root = Path(appworld_root) if appworld_root else None
        self.max_steps = max_steps
        self.allow_state_change_in_run_code = allow_state_change_in_run_code
        self.appworld_kwargs = appworld_kwargs or {}
        self.world: Any | None = None
        self.trajectory: TrajectoryRecord | None = None
        self.split: str | None = None
        self._closed = True
        self._finished_without_completion = False

    def reset(self, task_id: str, *, split: str | None = None) -> dict[str, Any]:
        self.close()
        from appworld import AppWorld, load_task_ids, update_root

        if self.appworld_root is not None:
            update_root(str(self.appworld_root))
        official_splits = {"train", "dev", "test", "test_normal", "test_challenge"}
        # RL manifests use project-level split names that are not registered
        # AppWorld datasets. Load those task IDs directly from the supplied root.
        if split in official_splits and task_id not in set(load_task_ids(split)):
            raise ValueError(f"task_id {task_id!r} is not in AppWorld split {split!r}")

        self.world = AppWorld(
            task_id=task_id,
            experiment_name=self.experiment_name,
            **self.appworld_kwargs,
        )
        self.world.__enter__()
        self._closed = False
        self.split = split
        self._finished_without_completion = False
        initial_observation = self._inspect_task()
        self.trajectory = TrajectoryRecord(
            task_id=task_id,
            split=split,
            experiment_name=self.experiment_name,
            initial_observation=deepcopy(initial_observation),
        )
        return initial_observation

    def validate_tool_call(self, tool_call: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Statically validate one agent-selected action without executing environment code."""

        violations: list[dict[str, Any]] = []
        try:
            name, args = parse_tool_call(tool_call)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return [
                make_violation(
                    "invalid_tool_arguments",
                    type(exc).__name__ + ": " + str(exc),
                    component="protocol",
                    always_enforced=True,
                )
            ]

        schema = next(
            (item["function"] for item in AGENT_TOOL_SCHEMAS if item["function"]["name"] == name),
            None,
        )
        if schema is None:
            return [
                make_violation(
                    "unknown_tool",
                    f"Unknown agent tool: {name}",
                    component="protocol",
                    always_enforced=True,
                )
            ]
        parameters = schema["parameters"]
        properties = parameters.get("properties") or {}
        required = set(parameters.get("required") or [])
        missing = sorted(required - set(args))
        extra = sorted(set(args) - set(properties))
        if missing:
            violations.append(
                make_violation(
                    "missing_tool_parameters",
                    f"{name} is missing required parameters: {missing}",
                    component="protocol",
                    always_enforced=True,
                )
            )
        if extra and parameters.get("additionalProperties") is False:
            violations.append(
                make_violation(
                    "extra_tool_parameters",
                    f"{name} has unsupported parameters: {extra}",
                    component="protocol",
                    always_enforced=True,
                )
            )

        if name not in {"query_state", "run_code", "run_write_code"}:
            return violations
        code = args.get("code")
        if not isinstance(code, str):
            return violations

        for message in _runtime_safety_errors(code):
            violations.append(
                make_violation(
                    "unsafe_runtime_access",
                    message,
                    component="core_safety",
                    always_enforced=True,
                )
            )
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return violations

        docs = {(app, api): doc for app, api, doc in self._iter_api_docs()}
        allowed_apps = set(getattr(self.world.task, "allowed_apps", [])) if self.world else set()
        allowed_apps.add("supervisor")
        api_calls: list[tuple[str, str, ast.Call]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            path = _attribute_path(node.func)
            if path is None or path[0] != "apis":
                continue
            if len(path) != 3:
                violations.append(
                    make_violation(
                        "invalid_api_path",
                        "AppWorld APIs must use exactly apis.<app>.<api>(...).",
                        component="appworld_guard",
                    )
                )
                continue
            app, api = path[1], path[2]
            api_calls.append((app, api, node))
            if app not in allowed_apps:
                violations.append(
                    make_violation(
                        "app_not_allowed",
                        f"App {app!r} is outside the task's allowed apps.",
                        component="appworld_guard",
                    )
                )
            doc = docs.get((app, api))
            if doc is None:
                violations.append(
                    make_violation(
                        "api_not_found",
                        f"API {app}.{api} does not exist in this task's documentation.",
                        component="appworld_guard",
                    )
                )
                continue
            if node.args or any(keyword.arg is None for keyword in node.keywords):
                violations.append(
                    make_violation(
                        "unverifiable_api_arguments",
                        f"API {app}.{api} must use explicit keyword arguments.",
                        component="appworld_guard",
                    )
                )
                continue
            passed = {str(keyword.arg) for keyword in node.keywords if keyword.arg is not None}
            required_api, allowed_api = _documented_parameters(doc)
            missing_api = sorted(required_api - passed)
            extra_api = sorted(passed - allowed_api) if allowed_api else []
            if missing_api:
                violations.append(
                    make_violation(
                        "missing_api_parameters",
                        f"API {app}.{api} is missing required parameters: {missing_api}",
                        component="appworld_guard",
                    )
                )
            if extra_api:
                violations.append(
                    make_violation(
                        "extra_api_parameters",
                        f"API {app}.{api} has undocumented parameters: {extra_api}",
                        component="appworld_guard",
                    )
                )

        mutating = any(
            _documented_api_is_mutating(api, docs[(app, api)])
            for app, api, _ in api_calls
            if (app, api) in docs
        )
        if name in {"query_state", "run_code"} and mutating:
            violations.append(
                make_violation(
                    "write_api_in_read_tool",
                    f"{name} cannot execute a non-GET AppWorld API.",
                    component="appworld_guard",
                )
            )
        if name == "run_write_code" and not mutating:
            violations.append(
                make_violation(
                    "write_tool_without_write_api",
                    "run_write_code must contain at least one documented non-GET API call.",
                    component="appworld_guard",
                )
            )
        return violations

    def _write_provenance(self, code: str) -> list[dict[str, Any]]:
        """Build provenance for a write before it executes.

        Only prior observations and successful variable bindings are evidence;
        the current write cannot establish its own provenance. Literal values
        are compared internally but never emitted into trajectory metadata.
        """

        if self.world is None or self.trajectory is None:
            return []
        observed_values: set[str] = set()
        initial = self.trajectory.initial_observation
        _collect_provenance_values(initial, observed_values)
        initial_text = json.dumps(_jsonable(initial), ensure_ascii=False, sort_keys=True)
        bindings: dict[str, Mapping[str, Any]] = {}
        for step in self.trajectory.steps:
            _collect_provenance_values(step.observation, observed_values)
            metadata = step.metadata
            raw_bindings = metadata.get("variable_bindings") if isinstance(metadata, Mapping) else None
            if isinstance(raw_bindings, list):
                for raw in raw_bindings:
                    if isinstance(raw, Mapping) and raw.get("name"):
                        bindings[str(raw["name"])] = raw
        docs = {(app, api): doc for app, api, doc in self._iter_api_docs()}
        return extract_api_call_provenance(
            code,
            docs=docs,
            observed_values=observed_values,
            initial_text=initial_text,
            bindings=bindings,
        )

    def step(self, tool_call: Mapping[str, Any]) -> dict[str, Any]:
        if self.world is None or self.trajectory is None:
            raise RuntimeError("reset() must be called before step().")
        if len(self.trajectory.steps) >= self.max_steps:
            observation = {"error": f"maximum step count {self.max_steps} reached"}
            self._append_step("finish_without_completion", {}, observation, "max_steps", False, True)
            return observation

        name = "invalid_tool_call"
        args: dict[str, Any] = {}
        error: str | None = None
        state_change = False
        terminated = False
        request_start = self._request_count()
        candidate_bindings: list[dict[str, Any]] = []
        write_provenance: list[dict[str, Any]] = []

        try:
            name, args = parse_tool_call(tool_call)
            if name in {"query_state", "run_code", "run_write_code"}:
                code = args.get("code")
                if isinstance(code, str):
                    candidate_bindings = extract_variable_bindings(code)
            write_provenance = (
                self._write_provenance(str(args.get("code")))
                if name == "run_write_code" and isinstance(args.get("code"), str)
                else []
            )
            observation = self._dispatch(name, args)
            error = execution_error_from_observation(observation)
            if name == "run_write_code":
                state_change = True
            elif name == "complete_task" and error is None:
                state_change = True
                terminated = True
            elif name == "finish_without_completion" and error is None:
                terminated = True
                self._finished_without_completion = True
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
            observation = {"error": error}

        metadata: dict[str, Any] = {}
        if candidate_bindings:
            # A binding is usable only when the complete code event succeeded.
            # Failed or guard-rejected snippets must not teach the model that a
            # variable exists (the common source of post-compaction NameError).
            metadata["variable_bindings"] = [
                {**binding, "available": error is None, "source_event": len(self.trajectory.steps)}
                for binding in candidate_bindings
            ]
        if name == "run_write_code" and isinstance(args.get("code"), str):
            metadata["write_provenance"] = write_provenance
            metadata["provenance_version"] = "parameter-v1"
        api_events = self._api_events_since(request_start)
        if api_events:
            metadata["api_events"] = api_events
        self._append_step(
            name, args, observation, error, state_change, terminated, metadata=metadata
        )
        return {
            "tool_name": name,
            "observation": observation,
            "error": error,
            "state_change": state_change,
            "terminated": terminated,
            "metadata": metadata,
        }

    def evaluate(self, *, suppress_errors: bool = True) -> dict[str, Any]:
        if self.world is None or self.trajectory is None:
            raise RuntimeError("reset() must be called before evaluate().")
        try:
            if getattr(self.world, "remote_environment_url", None):
                tracker = self.world.evaluate(suppress_errors=suppress_errors)
                evaluation = self._serialize_evaluation(tracker)
            else:
                # AppWorld executes against in-memory DBs but persists the end state
                # for the official evaluator. Flush explicitly before the child reads it.
                # ``world.execute`` normally disables the runtime guard after each
                # interaction, but an interrupted/empty trajectory may leave it on.
                # Saving state performs filesystem operations and must run outside it.
                safety_guard = getattr(self.world, "safety_guard", None)
                if safety_guard is not None:
                    safety_guard.disable()
                self.world.save()
                evaluation = _evaluate_appworld_subprocess(
                    task_id=self.trajectory.task_id,
                    experiment_name=self.experiment_name,
                    appworld_root=self.appworld_root,
                    suppress_errors=suppress_errors,
                )
        except Exception as exc:
            evaluation = {"error": type(exc).__name__ + ": " + str(exc)}

        had_error = any(step.error for step in self.trajectory.steps)
        reward = compute_terminal_reward(
            evaluation,
            had_execution_error=had_error,
            finished_without_completion=self._finished_without_completion,
        )
        self.trajectory.evaluation = evaluation
        self.trajectory.terminal_reward = reward
        return {"evaluation": evaluation, "terminal_reward": reward}

    def export_trajectory(self) -> dict[str, Any]:
        if self.trajectory is None:
            raise RuntimeError("No trajectory is available. Call reset() first.")
        return self.trajectory.to_dict()

    def save_trajectory(self, path: str | Path) -> None:
        payload = self.export_trajectory()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def close(self) -> None:
        if self.world is None or self._closed:
            self.world = None
            return
        try:
            safety_guard = getattr(self.world, "safety_guard", None)
            if safety_guard is not None:
                safety_guard.disable()
            self.world.__exit__(None, None, None)
        finally:
            self.world = None
            self._closed = True

    def __enter__(self) -> "AppWorldAdapter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _append_step(
        self,
        name: str,
        args: dict[str, Any],
        observation: Any,
        error: str | None,
        state_change: bool,
        terminated: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        assert self.trajectory is not None
        self.trajectory.append(
            TrajectoryStep(
                step_index=len(self.trajectory.steps),
                tool_name=name,
                arguments=_jsonable(args),
                observation=_jsonable(observation),
                error=error,
                state_change=state_change,
                terminated=terminated,
                metadata=metadata or {},
            )
        )

    def _request_count(self) -> int:
        if self.world is None:
            return 0
        requester = getattr(self.world, "requester", None)
        requests = getattr(requester, "requests", None)
        return len(requests) if isinstance(requests, list) else 0

    def _api_events_since(self, start: int) -> list[dict[str, str]]:
        if self.world is None:
            return []
        requester = getattr(self.world, "requester", None)
        requests = getattr(requester, "requests", None)
        if not isinstance(requests, list):
            return []
        events: list[dict[str, str]] = []
        for request in requests[start:]:
            if not isinstance(request, Mapping):
                continue
            path = urlsplit(str(request.get("url") or "")).path
            parts = [part for part in path.split("/") if part]
            event = {
                "method": str(request.get("method") or "").upper(),
                "route": "/" + "/".join(parts),
            }
            if parts:
                event["app"] = parts[0]
                event["endpoint"] = "/".join(parts[1:])
            events.append(event)
        return events

    def _dispatch(self, name: str, args: dict[str, Any]) -> Any:
        if name == "inspect_task":
            return self._inspect_task()
        if name == "search_api_docs":
            return self._search_api_docs(str(args["query"]), int(args.get("limit", 10)))
        if name == "inspect_api_schema":
            return self._inspect_api_schema(str(args["app"]), str(args["api"]))
        if name == "query_state":
            return self._execute_guarded(str(args["code"]), require_read_only=True)
        if name == "run_code":
            return self._execute_guarded(
                str(args["code"]),
                require_read_only=not self.allow_state_change_in_run_code,
            )
        if name == "run_write_code":
            if not str(args.get("intended_change", "")).strip():
                raise ValueError("run_write_code requires intended_change")
            return self._execute_guarded(str(args["code"]), require_read_only=False)
        if name == "complete_task":
            status = str(args.get("status", "success"))
            if status not in {"success", "fail"}:
                raise ValueError("complete_task status must be 'success' or 'fail'")
            call_args = [f"status={status!r}"]
            if "answer" in args and args["answer"] is not None:
                answer = args["answer"]
                if not isinstance(answer, (str, int, float, bool)):
                    raise TypeError("complete_task answer must be a JSON scalar")
                call_args.insert(0, f"answer={answer!r}")
            code = "apis.supervisor.complete_task(" + ", ".join(call_args) + ")"
            return self._execute_guarded(code, require_read_only=False)
        if name == "finish_without_completion":
            return {"finished_without_completion": True, "reason": str(args["reason"])}
        raise ValueError(f"Unknown agent tool: {name}")

    def _execute_guarded(self, code: str, *, require_read_only: bool) -> str:
        if self.world is None:
            raise RuntimeError("reset() must be called before code execution.")
        safety_errors = _runtime_safety_errors(code)
        if safety_errors:
            raise ValueError("agent code rejected: " + "; ".join(safety_errors))
        if require_read_only and looks_state_changing(code):
            raise ValueError("read-only tool rejected code that appears to mutate AppWorld state")
        if require_read_only and QUERY_EXPLORATION_PATTERN.search(code):
            raise ValueError(
                "agent code rejected Python/runtime introspection; use AppWorld APIs, "
                "API docs, or query_state on task data instead"
            )
        return str(self.world.execute(code))

    def _inspect_task(self) -> dict[str, Any]:
        if self.world is None:
            raise RuntimeError("reset() must be called first.")
        task = self.world.task
        supervisor = getattr(task, "supervisor", None)
        return {
            "task_id": task.id,
            "split": self.split,
            "instruction": task.instruction,
            "current_datetime": task.datetime.isoformat(),
            "supervisor": _jsonable(supervisor),
            "allowed_apps": list(task.allowed_apps),
            "available_tools": [tool["function"]["name"] for tool in AGENT_TOOL_SCHEMAS],
        }

    def _iter_api_docs(self) -> list[tuple[str, str, Mapping[str, Any]]]:
        if self.world is None:
            raise RuntimeError("reset() must be called first.")
        api_docs = getattr(self.world.task, "api_docs", None)
        docs = []
        for app, apis in dict(api_docs).items():
            if app in {"admin", "api_docs"} or not isinstance(apis, Mapping):
                continue
            for api, doc in dict(apis).items():
                if isinstance(doc, Mapping):
                    docs.append((str(app), str(api), doc))
        return docs

    def _search_api_docs(self, query: str, limit: int) -> dict[str, Any]:
        tokens = [token.casefold() for token in re.findall(r"[A-Za-z0-9_]+", query)]
        if not tokens:
            raise ValueError("search_api_docs query must contain searchable terms")
        scored = []
        for app, api, doc in self._iter_api_docs():
            haystack = json.dumps(_jsonable({"app": app, "api": api, "doc": doc}), ensure_ascii=False).casefold()
            score = sum(1 for token in tokens if token in haystack)
            if score:
                scored.append((score, app, api, doc))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return {
            "query": query,
            "results": [_short_doc(app, api, doc) for _, app, api, doc in scored[:limit]],
        }

    def _inspect_api_schema(self, app: str, api: str) -> dict[str, Any]:
        for app_name, api_name, doc in self._iter_api_docs():
            if app_name == app and api_name == api:
                payload = _short_doc(app_name, api_name, doc)
                payload["raw_doc"] = _jsonable(doc)
                return payload
        raise KeyError(f"API schema not found for {app}.{api}")

    @staticmethod
    def _serialize_evaluation(tracker: Any) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if hasattr(tracker, "to_dict"):
            try:
                data["data"] = _jsonable(tracker.to_dict())
            except TypeError:
                pass
        if hasattr(tracker, "report"):
            try:
                data["report"] = str(tracker.report())
            except TypeError:
                pass
        if not data:
            data["data"] = _jsonable(tracker)
        return data
