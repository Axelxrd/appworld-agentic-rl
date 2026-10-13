"""Lightweight CLI utilities for the AppWorld Agentic RL adapter."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from appworld_agentic_rl.adapter import AGENT_TOOL_SCHEMAS, AppWorldAdapter
from appworld_agentic_rl.frozen_evaluation import evaluate_frozen_policy
from appworld_agentic_rl.online_rl import build_online_group, write_group
from appworld_agentic_rl.scripted_solvers import run_scripted_smoke
from appworld_agentic_rl.sft_dataset import build_action_sft_dataset
from appworld_agentic_rl.splits import create_split_manifests
from appworld_agentic_rl.task_generation import generate_original_train_tasks
from appworld_agentic_rl.teacher_client import OpenAICompatibleTeacherClient
from appworld_agentic_rl.teacher_collector import collect_teacher_rollouts
from appworld_agentic_rl.harness import TeacherHarnessConfig
from appworld_agentic_rl.context_policy import (
    DEFAULT_CONTEXT_POLICY,
    TEACHER_CONTEXT_TARGET_CHARS,
    TEACHER_CONTEXT_TRIGGER_CHARS,
)
from appworld_agentic_rl.teacher_recovery import (
    recover_teacher_provider_errors,
    teacher_recovery_settings,
)
from appworld_agentic_rl.training_pools import create_training_pool_manifests
from appworld_agentic_rl.trajectory_replay import (
    REPLAY_SELECTIONS,
    replay_teacher_trajectories,
)


def _smoke(args: argparse.Namespace) -> None:
    adapter = AppWorldAdapter(
        experiment_name=args.experiment_name,
        appworld_root=args.appworld_root,
        max_steps=args.max_steps,
    )
    try:
        initial = adapter.reset(args.task_id, split=args.split)
        adapter.step({"name": "inspect_task", "arguments": {}})
        adapter.step(
            {
                "name": "search_api_docs",
                "arguments": {"query": args.api_query, "limit": 5},
            }
        )
        if args.complete:
            adapter.step({"name": "complete_task", "arguments": {"summary": "smoke completion"}})
        evaluation = adapter.evaluate()
        if args.output:
            adapter.save_trajectory(args.output)
        payload = {
            "task_id": args.task_id,
            "instruction": initial["instruction"],
            "step_count": len(adapter.export_trajectory()["steps"]),
            "terminal_reward": evaluation["terminal_reward"],
            "trajectory_path": str(args.output) if args.output else None,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        adapter.close()


def _tools(_: argparse.Namespace) -> None:
    print(json.dumps(AGENT_TOOL_SCHEMAS, ensure_ascii=False, indent=2))


def _scripted_smoke(args: argparse.Namespace) -> None:
    result = run_scripted_smoke(
        task_id=args.task_id,
        split=args.split,
        experiment_name=args.experiment_name,
        output=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def _collect_teacher(args: argparse.Namespace) -> None:
    if args.appworld_root is not None:
        from appworld import update_root

        update_root(str(args.appworld_root.resolve()))
    client = OpenAICompatibleTeacherClient(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_retries=args.request_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        disable_provider_parallel_tool_calls=args.provider_disable_parallel_tool_calls,
        disable_thinking=getattr(args, "disable_thinking", False),
    )
    harness_config = TeacherHarnessConfig(
        bc_guard_mode=args.bc_guard_mode,
        max_action_attempts=args.max_action_attempts,
        invalid_action_limit=args.invalid_action_limit,
        max_identical_action_repeats=args.max_identical_action_repeats,
        max_context_chars=args.max_context_chars,
        max_observation_chars=args.max_observation_chars,
        context_compaction_enable=args.context_compaction_enable,
        context_compaction_trigger_chars=args.context_compaction_trigger_chars,
        context_compaction_target_chars=args.context_compaction_target_chars,
        context_preserve_recent_groups=args.context_preserve_recent_groups,
    )

    def adapter_factory() -> AppWorldAdapter:
        return AppWorldAdapter(
            experiment_name=args.experiment_name,
            appworld_root=args.appworld_root,
            max_steps=args.max_steps,
        )

    summary = collect_teacher_rollouts(
        client=client,
        adapter_factory=adapter_factory,
        output_dir=args.output_dir,
        model_name=args.model,
        task_ids=args.task_id,
        limit=args.limit,
        split=args.split,
        max_model_turns=args.max_model_turns,
        attempts_per_task=args.attempts_per_task,
        max_curated_per_task=args.max_curated_per_task,
        require_clean_for_sft=not args.allow_execution_errors_in_sft,
        harness_config=harness_config,
        resume=args.resume,
        run_config={
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
            "request_retries": args.request_retries,
            "retry_backoff_seconds": args.retry_backoff_seconds,
            "provider_disable_parallel_tool_calls": args.provider_disable_parallel_tool_calls,
            "experiment_name": args.experiment_name,
            "attempts_per_task": args.attempts_per_task,
            "max_curated_per_task": args.max_curated_per_task,
            "require_clean_for_sft": not args.allow_execution_errors_in_sft,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _replay_teacher(args: argparse.Namespace) -> None:
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")

    def adapter_factory() -> AppWorldAdapter:
        return AppWorldAdapter(
            experiment_name=args.experiment_name,
            appworld_root=args.appworld_root,
            max_steps=args.max_steps,
        )

    summary = replay_teacher_trajectories(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        adapter_factory=adapter_factory,
        selection=args.selection,
        task_ids=args.task_id,
        rollout_ids=args.rollout_id,
        limit=args.limit,
        resume=args.resume,
        replay_config={
            "experiment_name": args.experiment_name,
            "appworld_root": str(args.appworld_root.resolve()),
            "max_steps": args.max_steps,
            "evaluator_process": "isolated",
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _recover_teacher_provider_errors(args: argparse.Namespace) -> None:
    if args.appworld_root is not None:
        from appworld import update_root

        update_root(str(args.appworld_root.resolve()))
    settings = teacher_recovery_settings(args.source_dir)
    client = OpenAICompatibleTeacherClient(**settings)

    def adapter_factory() -> AppWorldAdapter:
        return AppWorldAdapter(
            experiment_name=args.experiment_name,
            appworld_root=args.appworld_root,
            max_steps=args.max_steps,
        )

    summary = recover_teacher_provider_errors(
        client=client,
        adapter_factory=adapter_factory,
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _evaluate_frozen(args: argparse.Namespace) -> None:
    from appworld import update_root

    update_root(str(args.appworld_root.resolve()))
    client = OpenAICompatibleTeacherClient(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_retries=args.request_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        disable_provider_parallel_tool_calls=args.provider_disable_parallel_tool_calls,
        disable_thinking=getattr(args, "disable_thinking", False),
    )
    context_token_counter = None
    if args.context_budget_mode == "tokens":
        if args.tokenizer_path is None:
            raise ValueError("--tokenizer-path is required for token-exact evaluation")
        from appworld_agentic_rl.token_counting import HuggingFaceChatTokenCounter

        context_token_counter = HuggingFaceChatTokenCounter(
            args.tokenizer_path,
            tools=AGENT_TOOL_SCHEMAS,
            enable_thinking=not args.disable_thinking,
        )
    harness = TeacherHarnessConfig(
        bc_guard_mode=args.bc_guard_mode,
        max_action_attempts=args.max_action_attempts,
        invalid_action_limit=args.invalid_action_limit,
        max_identical_action_repeats=args.max_identical_action_repeats,
        context_budget_mode=args.context_budget_mode,
        max_context_chars=args.max_context_chars,
        max_observation_chars=args.max_observation_chars,
        context_compaction_enable=args.context_compaction_enable,
        context_compaction_trigger_chars=args.context_compaction_trigger_chars,
        context_compaction_target_chars=args.context_compaction_target_chars,
        max_context_tokens=args.max_context_tokens,
        context_compaction_trigger_tokens=args.context_compaction_trigger_tokens,
        context_compaction_target_tokens=args.context_compaction_target_tokens,
        context_preserve_recent_groups=args.context_preserve_recent_groups,
    )

    def adapter_factory() -> AppWorldAdapter:
        return AppWorldAdapter(
            experiment_name=args.experiment_name,
            appworld_root=args.appworld_root,
            max_steps=args.max_steps,
        )

    summary = evaluate_frozen_policy(
        client=client,
        adapter_factory=adapter_factory,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        model_name=args.model,
        condition=args.condition,
        split=args.split,
        max_model_turns=args.max_model_turns,
        harness_config=harness,
        resume=args.resume,
        context_token_counter=context_token_counter,
        run_config={
            "model_revision": args.model_revision,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
            "request_retries": args.request_retries,
            "retry_backoff_seconds": args.retry_backoff_seconds,
            "provider_disable_parallel_tool_calls": args.provider_disable_parallel_tool_calls,
            "experiment_name": args.experiment_name,
            "max_steps": args.max_steps,
            "context_policy": harness.to_dict()["context_policy"],
            "context_policy_evaluation_mode": args.context_budget_mode,
            "tokenizer_path": str(args.tokenizer_path) if args.tokenizer_path else None,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _split_data(args: argparse.Namespace) -> None:
    metadata = create_split_manifests(
        appworld_root=args.appworld_root,
        output_dir=args.output_dir,
        validation_scenarios=args.validation_scenarios,
        final_scenarios=args.final_scenarios,
        seed=args.seed,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


def _generate_train_tasks(args: argparse.Namespace) -> None:
    result = generate_original_train_tasks(
        source_root=args.source_root,
        output_root=args.output_root,
        appworld_repo=args.appworld_repo,
        manifest_dir=args.manifest_dir,
        scenario_ids=args.scenario_id,
        max_tasks_per_scenario=args.max_tasks_per_scenario,
        random_seed=args.seed,
        validate_with_environment=not args.direct_validation,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def _build_training_pools(args: argparse.Namespace) -> None:
    result = create_training_pool_manifests(
        appworld_root=args.appworld_root,
        manifest_dir=args.manifest_dir,
        output_dir=args.output_dir,
        rl_novel_scenarios=args.rl_novel_scenarios,
        rl_active_cap=args.rl_active_cap,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def _build_sft_data(args: argparse.Namespace) -> None:
    result = build_action_sft_dataset(
        source_dirs=args.source_dir,
        output_dir=args.output_dir,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def _online_rl_smoke(args: argparse.Namespace) -> None:
    from appworld import update_root

    update_root(str(args.appworld_root.resolve()))
    client = OpenAICompatibleTeacherClient(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_retries=args.request_retries,
        disable_provider_parallel_tool_calls=True,
        disable_thinking=True,
    )
    harness = TeacherHarnessConfig(
        bc_guard_mode="enforce",
        max_action_attempts=args.max_action_attempts,
        invalid_action_limit=args.invalid_action_limit,
        max_identical_action_repeats=1,
        max_context_chars=args.max_context_chars,
        max_observation_chars=args.max_observation_chars,
        context_compaction_enable=args.context_compaction_enable,
        context_compaction_trigger_chars=args.context_compaction_trigger_chars,
        context_compaction_target_chars=args.context_compaction_target_chars,
        context_preserve_recent_groups=args.context_preserve_recent_groups,
    )

    def adapter_factory() -> AppWorldAdapter:
        return AppWorldAdapter(
            experiment_name=args.experiment_name,
            appworld_root=args.appworld_root,
            max_steps=args.max_steps,
        )

    group = build_online_group(
        task_id=args.task_id,
        client=client,
        adapter_factory=adapter_factory,
        split=args.split,
        group_size=args.group_size,
        max_model_turns=args.max_model_turns,
        harness_config=harness,
        process_lambda=args.process_lambda,
    )
    write_group(args.output, group)
    summary = {
        "task_id": args.task_id,
        "group_size": args.group_size,
        "eligible_count": group["eligible_count"],
        "excluded_count": group["excluded_count"],
        "trainable": group["trainable"],
        "output": str(args.output),
        "records": [
            {
                "terminal_reward": row["terminal_reward"],
                "process_score": row["process_score"],
                "total_reward": row["total_reward"],
                "group_advantage": row["group_advantage"],
                "training_eligible": row["training_eligible"],
                "exclusion_reason": row["exclusion_reason"],
                "action_turns": len(row["event_credit"]["turn_advantages"]),
                "action_tokens": len(row["event_credit"]["token_advantages"]),
                "end_reason": row["rollout"]["rollout"]["end_reason"],
                "infrastructure_invalid": row["rollout"]["infrastructure_invalid"],
            }
            for row in group["records"]
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="appworld-agentic-rl")
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("tools", help="print agent-selectable tool schemas")
    command.set_defaults(handler=_tools)

    command = commands.add_parser(
        "split-data",
        help="create reproducible scenario-level AppWorld-RL train/validation/final manifests",
    )
    command.add_argument("--appworld-root", type=Path, default=Path.cwd())
    command.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/manifests/appworld_rl_v1"),
    )
    command.add_argument("--validation-scenarios", type=int, default=20)
    command.add_argument("--final-scenarios", type=int, default=60)
    command.add_argument("--seed", type=int, default=20260814)
    command.set_defaults(handler=_split_data)

    command = commands.add_parser(
        "generate-train-tasks",
        help="expand official train scenarios in an isolated AppWorld data root",
    )
    command.add_argument("--source-root", type=Path, default=Path.cwd())
    command.add_argument("--output-root", type=Path, required=True)
    command.add_argument("--appworld-repo", type=Path, default=Path.cwd() / "appworld")
    command.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/manifests/appworld_rl_v1"),
    )
    command.add_argument("--scenario-id", action="append")
    command.add_argument("--max-tasks-per-scenario", type=int)
    command.add_argument("--seed", type=int, default=1)
    command.add_argument(
        "--direct-validation",
        action="store_true",
        help="validate through direct APIs instead of AppWorld.execute",
    )
    command.set_defaults(handler=_generate_train_tasks)

    command = commands.add_parser(
        "build-training-pools",
        help="partition expanded train tasks into scenario-balanced SFT and RL pools",
    )
    command.add_argument(
        "--appworld-root",
        type=Path,
        default=Path("artifacts/appworld_rl_v1_root"),
    )
    command.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/manifests/appworld_rl_v1"),
    )
    command.add_argument(
        "--output-dir",
        type=Path,
        help="defaults to --manifest-dir inside the isolated AppWorld root",
    )
    command.add_argument("--rl-novel-scenarios", type=int, default=32)
    command.add_argument("--rl-active-cap", type=int, default=16)
    command.add_argument("--seed", type=int, default=20260815)
    command.set_defaults(handler=_build_training_pools)

    command = commands.add_parser(
        "build-sft-data",
        help="overlay teacher recoveries and build clean action-only SFT JSONL",
    )
    command.add_argument(
        "--source-dir",
        action="append",
        type=Path,
        required=True,
        help="teacher run directory in base-to-latest recovery order; repeatable",
    )
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument(
        "--validation-ratio",
        type=float,
        default=0.0,
        help="optional internal holdout; default uses every teacher row for training",
    )
    command.add_argument("--seed", type=int, default=20260817)
    command.set_defaults(handler=_build_sft_data)

    command = commands.add_parser("smoke", help="run one adapter smoke trajectory")
    command.add_argument("--task-id", required=True)
    command.add_argument("--split", default="train")
    command.add_argument("--experiment-name", default="appworld_agentic_rl_mvp")
    command.add_argument("--appworld-root", type=Path)
    command.add_argument("--api-query", default="supervisor complete task")
    command.add_argument("--max-steps", type=int, default=10)
    command.add_argument("--complete", action="store_true")
    command.add_argument("--output", type=Path)
    command.set_defaults(handler=_smoke)

    command = commands.add_parser(
        "online-rl-smoke",
        help="sample an AppWorld policy group and build lightweight event-credit signals",
    )
    command.add_argument("--model", required=True)
    command.add_argument("--base-url", required=True)
    command.add_argument("--task-id", required=True)
    command.add_argument("--split", default="train")
    command.add_argument("--appworld-root", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--experiment-name", default="appworld_online_rl_smoke")
    command.add_argument("--group-size", type=int, default=2)
    command.add_argument("--process-lambda", type=float, default=0.3)
    command.add_argument("--max-steps", type=int, default=12)
    command.add_argument("--max-model-turns", type=int, default=10)
    command.add_argument("--max-action-attempts", type=int, default=20)
    command.add_argument("--invalid-action-limit", type=int, default=4)
    command.add_argument("--max-context-chars", type=int, default=TEACHER_CONTEXT_TRIGGER_CHARS)
    command.add_argument("--max-observation-chars", type=int, default=DEFAULT_CONTEXT_POLICY.max_observation_chars)
    command.add_argument(
        "--context-compaction-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable structured-memory compaction during online smoke rollout",
    )
    command.add_argument(
        "--context-compaction-trigger-chars",
        type=int,
        default=TEACHER_CONTEXT_TRIGGER_CHARS,
    )
    command.add_argument(
        "--context-compaction-target-chars",
        type=int,
        default=TEACHER_CONTEXT_TARGET_CHARS,
    )
    command.add_argument(
        "--context-preserve-recent-groups",
        type=int,
        default=DEFAULT_CONTEXT_POLICY.preserve_recent_groups,
    )
    command.add_argument("--temperature", type=float, default=0.7)
    command.add_argument("--max-tokens", type=int, default=1024)
    command.add_argument("--timeout", type=float, default=180.0)
    command.add_argument("--request-retries", type=int, default=1)
    command.set_defaults(handler=_online_rl_smoke)

    command = commands.add_parser(
        "scripted-smoke",
        help="run a registered no-LLM scripted AppWorld success-path check",
    )
    command.add_argument("--task-id", default="82e2fac_1")
    command.add_argument("--split", default="train")
    command.add_argument("--experiment-name", default="appworld_scripted_smoke")
    command.add_argument("--output", type=Path)
    command.set_defaults(handler=_scripted_smoke)

    command = commands.add_parser(
        "collect-teacher",
        help="collect train-only teacher trajectories and verifier-filter them",
    )
    command.add_argument("--model", required=True, help="OpenAI-compatible teacher model name")
    command.add_argument("--base-url", help="defaults to OPENAI_BASE_URL")
    command.add_argument("--task-id", action="append", help="one train task ID; repeatable")
    command.add_argument("--limit", type=int, help="use at most this many train tasks")
    command.add_argument(
        "--split",
        default="train",
        choices=["train", "appworld_rl_v1_teacher_sft"],
    )
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--experiment-name", default="appworld_teacher_rollout")
    command.add_argument("--appworld-root", type=Path)
    command.add_argument("--max-steps", type=int, default=50, help="maximum AppWorld tool steps")
    command.add_argument(
        "--max-model-turns",
        type=int,
        default=25,
        help="maximum teacher decisions; independent of --max-steps",
    )
    command.add_argument("--temperature", type=float, default=0.0)
    command.add_argument("--max-tokens", type=int, default=1024)
    command.add_argument("--timeout", type=float, default=60.0)
    command.add_argument(
        "--request-retries", type=int, default=3,
        help="retries for 429/5xx, connection errors, and timeouts",
    )
    command.add_argument("--retry-backoff-seconds", type=float, default=1.0)
    command.add_argument(
        "--provider-disable-parallel-tool-calls",
        action="store_true",
        help="send parallel_tool_calls=false only to providers known to support it",
    )
    command.add_argument(
        "--bc-guard-mode",
        choices=["shadow", "enforce"],
        default="enforce",
        help="observe or enforce behavior/AppWorld guard decisions",
    )
    command.add_argument("--max-action-attempts", type=int, default=80)
    command.add_argument("--invalid-action-limit", type=int, default=6)
    command.add_argument("--max-identical-action-repeats", type=int, default=1)
    command.add_argument("--max-context-chars", type=int, default=TEACHER_CONTEXT_TRIGGER_CHARS)
    command.add_argument("--max-observation-chars", type=int, default=DEFAULT_CONTEXT_POLICY.max_observation_chars)
    command.add_argument(
        "--context-compaction-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable structured-memory compaction during teacher collection",
    )
    command.add_argument(
        "--context-compaction-trigger-chars",
        type=int,
        default=TEACHER_CONTEXT_TRIGGER_CHARS,
    )
    command.add_argument(
        "--context-compaction-target-chars",
        type=int,
        default=TEACHER_CONTEXT_TARGET_CHARS,
    )
    command.add_argument(
        "--context-preserve-recent-groups",
        type=int,
        default=DEFAULT_CONTEXT_POLICY.preserve_recent_groups,
    )
    command.add_argument(
        "--attempts-per-task",
        type=int,
        default=1,
        help="independent fresh-environment rollouts per task",
    )
    command.add_argument(
        "--max-curated-per-task",
        type=int,
        default=2,
        help="maximum diverse verifier-passing trajectories retained for SFT",
    )
    command.add_argument(
        "--allow-execution-errors-in-sft",
        action="store_true",
        help="allow recovered-error successes into curated_sft.jsonl",
    )
    command.add_argument(
        "--resume",
        action="store_true",
        help="resume a compatible interrupted collection from raw.jsonl",
    )
    command.set_defaults(handler=_collect_teacher)

    command = commands.add_parser(
        "recover-teacher-provider-errors",
        help="recollect exact provider-error attempt slots from a completed teacher run",
    )
    command.add_argument("--source-dir", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument(
        "--appworld-root",
        type=Path,
        default=Path("artifacts/appworld_rl_v1_root"),
    )
    command.add_argument(
        "--experiment-name",
        default="appworld_teacher_provider_recovery_v1",
    )
    command.add_argument("--max-steps", type=int, default=50)
    command.add_argument(
        "--resume",
        action="store_true",
        help="resume a compatible interrupted exact-slot recovery",
    )
    command.set_defaults(handler=_recover_teacher_provider_errors)

    command = commands.add_parser(
        "evaluate-frozen",
        help="evaluate one frozen policy condition on an immutable AppWorld manifest",
    )
    command.add_argument("--model", required=True)
    command.add_argument("--model-revision", required=True)
    command.add_argument("--base-url", required=True)
    command.add_argument("--condition", required=True)
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument(
        "--appworld-root",
        type=Path,
        default=Path("artifacts/appworld_rl_v1_root"),
    )
    command.add_argument("--split", default="appworld_rl_v1_final")
    command.add_argument("--experiment-name", required=True)
    command.add_argument("--bc-guard-mode", choices=["shadow", "enforce"], required=True)
    command.add_argument("--max-steps", type=int, default=50)
    command.add_argument("--max-model-turns", type=int, default=40)
    command.add_argument("--temperature", type=float, default=0.0)
    command.add_argument("--max-tokens", type=int, default=2048)
    command.add_argument("--timeout", type=float, default=180.0)
    command.add_argument("--request-retries", type=int, default=2)
    command.add_argument("--retry-backoff-seconds", type=float, default=1.0)
    command.add_argument("--provider-disable-parallel-tool-calls", action="store_true")
    command.add_argument(
        "--disable-thinking",
        action="store_true",
        help="disable Qwen thinking mode so XML tool calls are parsed as tool_calls",
    )
    command.add_argument("--max-action-attempts", type=int, default=80)
    command.add_argument("--invalid-action-limit", type=int, default=6)
    command.add_argument("--max-identical-action-repeats", type=int, default=1)
    command.add_argument(
        "--context-budget-mode",
        choices=["tokens", "chars"],
        default="tokens",
        help="use tokenizer-exact budgets (default) or the teacher-compatible character fallback",
    )
    command.add_argument(
        "--tokenizer-path",
        type=Path,
        help="local tokenizer/model directory; required when --context-budget-mode=tokens",
    )
    command.add_argument("--max-context-tokens", type=int, default=155904)
    command.add_argument(
        "--context-compaction-trigger-tokens",
        type=int,
        default=DEFAULT_CONTEXT_POLICY.trigger_tokens,
    )
    command.add_argument(
        "--context-compaction-target-tokens",
        type=int,
        default=DEFAULT_CONTEXT_POLICY.target_tokens,
    )
    command.add_argument("--max-context-chars", type=int, default=TEACHER_CONTEXT_TRIGGER_CHARS)
    command.add_argument("--max-observation-chars", type=int, default=DEFAULT_CONTEXT_POLICY.max_observation_chars)
    command.add_argument(
        "--context-compaction-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable structured-memory compaction for this evaluation",
    )
    command.add_argument(
        "--context-compaction-trigger-chars",
        type=int,
        default=TEACHER_CONTEXT_TRIGGER_CHARS,
    )
    command.add_argument(
        "--context-compaction-target-chars",
        type=int,
        default=TEACHER_CONTEXT_TARGET_CHARS,
    )
    command.add_argument(
        "--context-preserve-recent-groups",
        type=int,
        default=DEFAULT_CONTEXT_POLICY.preserve_recent_groups,
    )
    command.add_argument("--resume", action="store_true")
    command.set_defaults(handler=_evaluate_frozen)

    command = commands.add_parser(
        "replay-teacher",
        help="replay stored teacher trajectories locally without model calls",
    )
    command.add_argument("--source-dir", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument(
        "--appworld-root",
        type=Path,
        default=Path("artifacts/appworld_rl_v1_root"),
    )
    command.add_argument("--experiment-name", default="appworld_teacher_replay_v2")
    command.add_argument("--max-steps", type=int, default=50)
    command.add_argument(
        "--selection",
        choices=REPLAY_SELECTIONS,
        default="terminal-evaluator-errors",
        help="default recovers evaluator errors that reached a terminal agent tool",
    )
    command.add_argument("--task-id", action="append", help="optional task filter; repeatable")
    command.add_argument(
        "--rollout-id",
        action="append",
        help="optional exact rollout slot filter; repeatable",
    )
    command.add_argument("--limit", type=int)
    command.add_argument(
        "--resume",
        action="store_true",
        help="resume a compatible interrupted replay from replay.jsonl",
    )
    command.set_defaults(handler=_replay_teacher)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
