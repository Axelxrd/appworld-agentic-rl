from pathlib import Path

from appworld_agentic_rl.frozen_evaluation import _summary


def _row(
    task_id: str,
    *,
    success: bool,
    infrastructure_invalid: bool = False,
    execution_errors: int = 0,
    collateral_damage: bool = False,
    compacted: int = 0,
    post_compaction_name_errors: int = 0,
) -> dict:
    context_events = [
        {"turn_index": index + 1, "compacted": True} for index in range(compacted)
    ]
    observations = [
        {
            "turn_index": index + 1,
            "raw": {"error": "NameError: missing variable"},
        }
        for index in range(post_compaction_name_errors)
    ]
    return {
        "task_id": task_id,
        "infrastructure_invalid": infrastructure_invalid,
        "evaluation": {
            "success": success,
            "data": {
                "passes": ["p"] if success else [],
                "failures": [] if success else ["f"],
                "num_tests": 1,
                "state_integrity_failed": collateral_damage,
            },
        },
        "rollout": {
            "end_reason": "evaluator_success" if success else "max_model_turns",
            "model_turns": 3,
            "action_attempts": 4,
            "environment_steps": 4,
        },
        "process_metrics": {"execution_errors": execution_errors, "step_count": 4},
        "harness": {
            "context_events": context_events,
        },
        "observations": observations,
    }


def test_summary_reports_complete_and_expected_denominators() -> None:
    manifest = [
        {"task_id": "a_1", "scenario_id": "a"},
        {"task_id": "a_2", "scenario_id": "a"},
        {"task_id": "b_1", "scenario_id": "b"},
    ]
    rows = [
        _row("a_1", success=True, compacted=2, post_compaction_name_errors=1),
        _row("a_2", success=False, execution_errors=2, collateral_damage=True),
        _row("b_1", success=False, infrastructure_invalid=True),
    ]
    result = _summary(
        attempt_rows=rows,
        results=rows,
        manifest_rows=manifest,
        output_dir=Path("out"),
        collection_stop=None,
    )

    assert result["completed_tasks"] == 2
    assert result["successful_tasks"] == 1
    assert result["task_success_rate"] == 0.5
    assert result["task_success_rate_over_expected"] == 1 / 3
    assert result["infrastructure_error_tasks"] == 1
    assert result["infrastructure_error_rate"] == 1 / 3
    assert result["execution_error_total"] == 2
    assert result["tasks_with_execution_errors"] == 1
    assert result["collateral_damage_tasks"] == 1
    assert result["tasks_with_compaction"] == 1
    assert result["total_compactions"] == 2
    assert result["post_compaction_name_error_total"] == 1
    assert result["tasks_with_post_compaction_name_error"] == 1
    assert result["requirement_pass_rate"] == 0.5
    assert result["expected_scenarios"] == 2
    assert result["scenario_success_rate"] == 0.0
    assert result["scenario_success_rate_over_expected"] == 0.0


def test_summary_handles_empty_evaluation_without_nan() -> None:
    result = _summary(
        attempt_rows=[],
        results=[],
        manifest_rows=[],
        output_dir=Path("out"),
        collection_stop=None,
    )

    assert result["task_success_rate"] == 0.0
    assert result["infrastructure_error_rate"] == 0.0
    assert result["mean_model_turns"] is None
    assert result["requirement_pass_rate"] is None


def test_summary_counts_name_errors_only_after_first_compaction() -> None:
    row = _row("a_1", success=False)
    row["harness"]["context_events"] = [
        {"turn_index": 0, "compacted": False},
        {"turn_index": 3, "compacted": True},
    ]
    row["observations"] = [
        {"turn_index": 2, "raw": {"error": "NameError: before"}},
        {"turn_index": 3, "raw": {"error": "NameError: after"}},
        {"turn_index": 4, "raw": {"error": "ValueError: after"}},
    ]
    result = _summary(
        attempt_rows=[row],
        results=[row],
        manifest_rows=[{"task_id": "a_1", "scenario_id": "a"}],
        output_dir=Path("out"),
        collection_stop=None,
    )

    assert result["post_compaction_name_error_total"] == 1
    assert result["tasks_with_post_compaction_name_error"] == 1
