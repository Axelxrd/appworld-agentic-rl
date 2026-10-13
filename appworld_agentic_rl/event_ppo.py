"""Strict event-level PPO packing for AppWorld multi-turn rollouts.

The standard veRL agent output represents one trajectory as one prompt and one
response.  That representation cannot retain actions generated before a later
context compaction.  This module expands the immutable event snapshots emitted
by :mod:`verl_agent_loop` into ordinary veRL rows, so the existing actor loss
can compute a PPO ratio for each action under its original context.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

if TYPE_CHECKING:
    import torch


def _group_normalize(values: Mapping[str, float], groups: Mapping[str, str]) -> dict[str, float]:
    by_group: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for key, value in values.items():
        by_group[str(groups.get(key, key))].append((key, float(value)))
    normalized: dict[str, float] = {}
    for rows in by_group.values():
        values = [score for _, score in rows]
        if len(rows) < 2:
            z = [0.0] * len(rows)
        else:
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            std = math.sqrt(variance)
            z = [0.0] * len(rows) if std <= 1e-6 else [(value - mean) / (std + 1e-6) for value in values]
        normalized.update({key: float(value) for (key, _), value in zip(rows, z, strict=True)})
    return normalized


def _normalize_group_values(values: Sequence[float], epsilon: float = 1e-6) -> list[float]:
    """Normalize a group, returning zeros when it has no meaningful variance."""

    if len(values) < 2:
        return [0.0] * len(values)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std <= epsilon:
        return [0.0] * len(values)
    return [(value - mean) / (std + epsilon) for value in values]


def _strict_event_records(output: Any) -> list[Mapping[str, Any]]:
    """Validate immutable event snapshots before materializing PPO rows."""

    raw = output.extra_fields.get("appworld_event_records") or []
    if not isinstance(raw, Sequence):
        raise ValueError("strict event PPO requires appworld_event_records to be a sequence")
    records: list[Mapping[str, Any]] = []
    previous_step = -1
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("strict event PPO event record must be a mapping")
        records.append(item)
        record = item
        step = int(record.get("step_index", -1))
        if step <= previous_step:
            raise ValueError("strict event PPO requires unique increasing event step_index")
        previous_step = step
        context = record.get("context_ids") or []
        actions = record.get("action_ids") or []
        logprobs = record.get("action_log_probs") or []
        if not context:
            raise ValueError(f"strict event PPO event {step} has empty context_ids")
        if not actions:
            raise ValueError(f"strict event PPO event {step} has empty action_ids")
        if len(logprobs) != len(actions):
            raise ValueError(
                "strict event PPO requires one rollout log-prob per action token; "
                f"got {len(logprobs)} for {len(actions)} tokens"
            )
    return records


def _event_process_credits(reward: Mapping[str, Any]) -> dict[int, float]:
    events = {
        int(item.get("step_index", -1)): float(item.get("delta") or 0.0)
        for item in (reward.get("events") or [])
        if isinstance(item, Mapping)
    }
    target = float(reward.get("process_score") or 0.0)
    if not events:
        return {}
    total = sum(events.values())
    if abs(total) > 1e-8:
        scale = target / total
        return {step: value * scale for step, value in events.items()}
    last = next(reversed(events))
    return {step: target if step == last else 0.0 for step in events}


def _pad_left(values: Sequence[int], length: int, pad_id: int) -> tuple[list[int], list[int]]:
    if len(values) > length:
        raise ValueError(f"event context has {len(values)} tokens, exceeds prompt_length={length}")
    padding = length - len(values)
    return [pad_id] * padding + list(values), [0] * padding + [1] * len(values)


def _pad_right(values: Sequence[int], length: int, pad_id: int) -> tuple[list[int], list[int]]:
    if len(values) > length:
        raise ValueError(f"event action has {len(values)} tokens, exceeds response_length={length}")
    return list(values) + [pad_id] * (length - len(values)), [1] * len(values) + [0] * (length - len(values))


def _event_advantages(
    outputs: Sequence[Any],
    *,
    process_reward_lambda: float,
    turn_discount: float = 1.05,
) -> dict[tuple[int, int], float]:
    """Compute one scalar advantage per event, before token expansion."""

    scores: dict[str, float] = {}
    groups: dict[str, str] = {}
    for row, output in enumerate(outputs):
        uid = str(output.extra_fields.get("appworld_rollout_uid") or row)
        scores[uid] = float(output.reward_score or 0.0)
        groups[uid] = str(output.extra_fields.get("appworld_group_uid") or uid)
    process_totals: dict[str, float] = defaultdict(float)
    for row, output in enumerate(outputs):
        uid = str(output.extra_fields.get("appworld_rollout_uid") or row)
        reward = output.extra_fields.get("appworld_reward") or {}
        process = _event_process_credits(reward if isinstance(reward, Mapping) else {})
        process_totals[uid] = sum(process.values())

    normalized: dict[str, float] = {}
    grouped_uids: dict[str, list[str]] = defaultdict(list)
    for uid, group in groups.items():
        grouped_uids[group].append(uid)
    for group, uids in grouped_uids.items():
        terminal_values = [scores[uid] for uid in uids]
        terminal_normalized = _normalize_group_values(terminal_values)
        if any(abs(value) > 1e-8 for value in terminal_normalized):
            normalized.update(dict(zip(uids, terminal_normalized, strict=True)))
        else:
            # If all terminal outcomes are equal, retain a learning signal only
            # when process quality differs across the sampled trajectories.
            process_normalized = _normalize_group_values([process_totals[uid] for uid in uids])
            normalized.update(dict(zip(uids, process_normalized, strict=True)))
    advantages: dict[tuple[int, int], float] = {}
    for row, output in enumerate(outputs):
        uid = str(output.extra_fields.get("appworld_rollout_uid") or row)
        records = _strict_event_records(output)
        if not records:
            continue
        active_tokens = sum(len(item.get("action_ids") or []) for item in records)
        if active_tokens <= 0:
            continue
        weights = [turn_discount ** (len(records) - 1 - i) for i in range(len(records))]
        weight_total = sum(weights) or 1.0
        outcome_total = normalized.get(uid, 0.0) * math.sqrt(active_tokens)
        reward = output.extra_fields.get("appworld_reward") or {}
        process = _event_process_credits(reward if isinstance(reward, Mapping) else {})
        for event_index, record in enumerate(records):
            length = max(1, len(record.get("action_ids") or []))
            local = process.get(int(record.get("step_index", -1)), 0.0) * process_reward_lambda
            advantages[(row, event_index)] = (outcome_total * weights[event_index] / weight_total + local) / length
    return advantages


def build_event_dataproto(inputs: Sequence[Any], config: Any, *, tokenizer: Any | None = None) -> Any:
    """Expand internal agent outputs into one fixed-shape DataProto per event."""

    from verl import DataProto
    try:
        from verl.protocol import compute_position_id_with_mask
    except ImportError:
        from verl.utils.model import compute_position_id_with_mask
    from tensordict import TensorDict
    import numpy as np
    import torch

    rollout = config.actor_rollout_ref.rollout
    prompt_length = int(rollout.prompt_length)
    response_length = int(rollout.response_length)
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    process_lambda = float(getattr(rollout.agent, "process_reward_lambda", 0.3))
    alpha = float((config.algorithm.turn_discount or {}).get("alpha", 1.05))
    advantages = _event_advantages(inputs, process_reward_lambda=process_lambda, turn_discount=alpha)

    rows: list[dict[str, Any]] = []
    for input_index, output in enumerate(inputs):
        records = _strict_event_records(output)
        event_index = 0
        while event_index < len(records):
            record = records[event_index]
            action_ids = [int(x) for x in (record.get("action_ids") or [])]
            context_ids = [int(x) for x in (record.get("context_ids") or [])]
            if not action_ids:
                event_index += 1
                continue
            if len(context_ids) > prompt_length:
                raise ValueError(
                    "strict event PPO requires every saved event context to fit prompt_length; "
                    f"event has {len(context_ids)} tokens and limit is {prompt_length}"
                )
            prompt, prompt_attn = _pad_left(context_ids, prompt_length, pad_id)
            response, response_attn = _pad_right(action_ids, response_length, pad_id)
            response_mask = [1] * len(action_ids) + [0] * (response_length - len(action_ids))
            attention = prompt_attn + response_attn
            input_ids = prompt + response
            position_ids = compute_position_id_with_mask(torch.tensor(attention, dtype=torch.long)).tolist()
            logprobs = [float(x) for x in (record.get("action_log_probs") or [])]
            if len(logprobs) != len(action_ids):
                raise ValueError(
                    "strict event PPO requires one rollout log-prob per action token; "
                    f"got {len(logprobs)} for {len(action_ids)} tokens"
                )
            rollout_log_probs = logprobs + [0.0] * (response_length - len(logprobs))
            event_advantage = float(advantages.get((input_index, event_index), 0.0))
            rm_scores = [event_advantage] * len(action_ids) + [0.0] * (response_length - len(action_ids))
            rows.append(
                {
                    "prompts": prompt,
                    "responses": response,
                    "response_mask": response_mask,
                    "attention_mask": attention,
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "rollout_log_probs": rollout_log_probs,
                    "rm_scores": rm_scores,
                    "uid": str(output.extra_fields.get("appworld_group_uid") or input_index),
                    "event_trajectory_uid": str(output.extra_fields.get("appworld_rollout_uid") or input_index),
                    "event_step_index": int(record.get("step_index", -1)),
                    "event_index": event_index,
                    "event_context_tokens": len(context_ids),
                    "event_action_tokens": len(action_ids),
                }
            )
            event_index += 1
    if not rows:
        raise RuntimeError("strict event PPO produced no trainable action events")

    batch = TensorDict(
        {
            key: torch.tensor([row[key] for row in rows], dtype=torch.long if key not in {"rollout_log_probs", "rm_scores"} else torch.float32)
            for key in ("prompts", "responses", "response_mask", "attention_mask", "input_ids", "position_ids", "rollout_log_probs", "rm_scores")
        },
        batch_size=[len(rows)],
    )
    non_tensor = {
        "uid": np.asarray([row["uid"] for row in rows], dtype=object),
        "event_trajectory_uid": np.asarray([row["event_trajectory_uid"] for row in rows], dtype=object),
        "event_step_index": np.asarray([row["event_step_index"] for row in rows], dtype=np.int64),
        "event_index": np.asarray([row["event_index"] for row in rows], dtype=np.int64),
        "event_context_tokens": np.asarray([row["event_context_tokens"] for row in rows], dtype=np.int64),
        "event_action_tokens": np.asarray([row["event_action_tokens"] for row in rows], dtype=np.int64),
        "__num_turns__": np.ones(len(rows), dtype=np.int32),
    }
    return DataProto(batch=batch, non_tensor_batch=non_tensor, meta_info={"event_ppo": True})


def build_event_tq_fields(output: Any, config: Any, *, tokenizer: Any | None = None) -> list[dict[str, Any]]:
    """Pack one agent output into fixed-shape fields for TransferQueue.

    Unlike :func:`build_event_dataproto`, this returns ordinary field
    dictionaries because the v1 TQ worker writes rows directly to its queue.
    The first active reward token stores the trajectory outcome; the second
    stores the event-local process contribution for ``event_replay``.
    """

    import torch

    proto = build_event_dataproto([output], config, tokenizer=tokenizer)
    records = _strict_event_records(output)
    reward = output.extra_fields.get("appworld_reward") or {}
    process = _event_process_credits(reward if isinstance(reward, Mapping) else {})
    rollout = config.actor_rollout_ref.rollout
    process_lambda = float(getattr(rollout.agent, "process_reward_lambda", 0.3))
    group_uid = str(output.extra_fields.get("appworld_group_uid") or "unknown")
    trajectory_uid = str(output.extra_fields.get("appworld_rollout_uid") or group_uid)
    fields: list[dict[str, Any]] = []
    for event_index, record in enumerate(records):
        response_mask = proto.batch["response_mask"][event_index].clone()
        active = torch.nonzero(response_mask > 0, as_tuple=False).flatten()
        if active.numel() == 0:
            continue
        rm_scores = torch.zeros_like(proto.batch["rm_scores"][event_index], dtype=torch.float32)
        rm_scores[active[0]] = float(output.reward_score or 0.0)
        local = process.get(int(record.get("step_index", -1)), 0.0) * process_lambda
        if active.numel() > 1:
            rm_scores[active[1]] = float(local)
        else:
            rm_scores[active[0]] += float(local)
        fields.append(
            {
                "prompts": proto.batch["prompts"][event_index].clone(),
                "responses": proto.batch["responses"][event_index].clone(),
                "response_mask": response_mask,
                "loss_mask": response_mask.clone(),
                "attention_mask": proto.batch["attention_mask"][event_index].clone(),
                "input_ids": proto.batch["input_ids"][event_index].clone(),
                "position_ids": proto.batch["position_ids"][event_index].clone(),
                "rollout_log_probs": proto.batch["rollout_log_probs"][event_index].clone(),
                "rm_scores": rm_scores,
                "uid": f"{group_uid}|{trajectory_uid}|{event_index}",
                "event_group_uid": group_uid,
                "event_trajectory_uid": trajectory_uid,
                "event_step_index": int(record.get("step_index", -1)),
                "event_index": event_index,
                "event_context_tokens": len(record.get("context_ids") or []),
                "event_action_tokens": len(record.get("action_ids") or []),
                "event_logprob_tokens": len(record.get("action_log_probs") or []),
            }
        )
    return fields


def event_replay_advantage(
    token_level_rewards: Any,
    response_mask: Any,
    index: Iterable[Any] | None = None,
    config: Any = None,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute group-relative outcome advantages for packed event rows.

    TQ collects rows from all replicas before veRL calls the estimator.  Each
    row's ``uid`` is encoded as ``group|trajectory|event_index`` by the TQ
    adapter, so this estimator can normalize one outcome per trajectory while
    retaining one loss row per action event.
    """

    import torch

    rewards = token_level_rewards.to(dtype=torch.float32)
    mask = response_mask.to(dtype=torch.float32)
    if rewards.ndim != 2 or mask.ndim != 2:
        raise ValueError("event_replay expects [batch, response_length] tensors")
    ids = list(index) if index is not None else [str(i) for i in range(rewards.shape[0])]
    if len(ids) != rewards.shape[0]:
        raise ValueError("event_replay uid count does not match reward batch")

    def parts(value: Any, row: int) -> tuple[str, str, int]:
        fields = str(value).rsplit("|", 2)
        if len(fields) != 3:
            return str(value), str(value), row
        try:
            event_index = int(fields[2])
        except ValueError:
            event_index = row
        return fields[0], fields[1], event_index

    parsed = [parts(value, row) for row, value in enumerate(ids)]
    by_group: dict[str, list[int]] = defaultdict(list)
    by_trajectory: dict[str, list[int]] = defaultdict(list)
    for row, (group, trajectory, _) in enumerate(parsed):
        by_group[group].append(row)
        by_trajectory[trajectory].append(row)

    # The first active token carries the trajectory outcome.  The optional
    # second token carries the event process contribution (see the TQ packer).
    trajectory_scores: dict[str, float] = {}
    for trajectory, rows in by_trajectory.items():
        first = rows[0]
        active = torch.nonzero(mask[first] > 0, as_tuple=False).flatten()
        trajectory_scores[trajectory] = float(rewards[first, active[0]]) if active.numel() else 0.0

    normalized: dict[str, float] = {}
    for group, rows in by_group.items():
        trajectories = list(dict.fromkeys(parsed[row][1] for row in rows))
        terminal_values = [trajectory_scores.get(trajectory, 0.0) for trajectory in trajectories]
        terminal_z = _normalize_group_values(terminal_values)
        if any(abs(value) > 1e-8 for value in terminal_z):
            z = terminal_z
        else:
            # A group where every trajectory has the same terminal score still
            # contains useful information when its event-local process scores
            # differ.  Aggregate the second active reward token per trajectory
            # and use it only as a fallback normalization signal.
            process_values: list[float] = []
            for trajectory in trajectories:
                process_total = 0.0
                for event_row in by_trajectory[trajectory]:
                    active = torch.nonzero(mask[event_row] > 0, as_tuple=False).flatten()
                    if active.numel() > 1:
                        process_total += float(rewards[event_row, active[1]])
                process_values.append(process_total)
            z = _normalize_group_values(process_values)
        normalized.update(dict(zip(trajectories, z, strict=True)))

    advantages = torch.zeros_like(rewards)
    for row, (_, trajectory, event_index) in enumerate(parsed):
        active = torch.nonzero(mask[row] > 0, as_tuple=False).flatten()
        if active.numel() == 0:
            continue
        trajectory_rows = by_trajectory[trajectory]
        ordered = sorted(trajectory_rows, key=lambda item: parsed[item][2])
        rank = ordered.index(row)
        weights = [float(getattr(config, "turn_discount", {}).get("alpha", 1.05)) ** (len(ordered) - 1 - i)
                   for i in range(len(ordered))] if config is not None else [1.0] * len(ordered)
        weight_total = sum(weights) or 1.0
        total_tokens = sum(int(mask[item].sum().item()) for item in trajectory_rows)
        outcome = normalized.get(trajectory, 0.0) * math.sqrt(max(1, total_tokens))
        local = float(rewards[row, active[1]]) if active.numel() > 1 else 0.0
        value = outcome * weights[rank] / weight_total / max(1, int(active.numel())) + local
        advantages[row, active] = value
    return advantages, advantages.clone()


__all__ = ["build_event_dataproto", "build_event_tq_fields", "event_replay_advantage"]
