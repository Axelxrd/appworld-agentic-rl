"""TransferQueue-compatible manager for AppWorld AgentLoop."""

import asyncio
from collections import defaultdict
import logging
import math
import os

import ray
import torch
import transfer_queue as tq
from tensordict import NonTensorData, NonTensorStack

from appworld_agentic_rl.event_ppo import build_event_tq_fields
from appworld_agentic_rl.verl_compat import install_appworld_advantage_estimators
from verl.experimental.agent_loop import get_trajectory_info
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ, AgentLoopWorkerTQ
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

install_appworld_advantage_estimators()
_AgentLoopWorkerTQBase = AgentLoopWorkerTQ.__ray_metadata__.modified_class.__mro__[1]
logger = logging.getLogger(__name__)


def _strict_event_mode(config) -> bool:
    value = config.actor_rollout_ref.rollout.agent.get("event_ppo_mode", "trajectory")
    return str(value).strip().lower() in {"strict", "event", "event_replay"}


@ray.remote
class AppWorldAgentLoopWorkerTQ(_AgentLoopWorkerTQBase):
    """Run one AppWorld replica per worker and close its GRPO group safely."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._appworld_run_lock = asyncio.Lock()

    async def _run_agent_loop(self, *args, **kwargs):
        async with self._appworld_run_lock:
            return await super()._run_agent_loop(*args, **kwargs)

    async def generate_sequences(self, batch) -> None:
        """Schedule one-sample batches, including TensorDict LinkedList fields."""
        validate = batch["validate"] if "validate" in batch else False
        batch.pop("validate", None)
        config = self.config.actor_rollout_ref.rollout
        sampling_params = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repetition_penalty": 1.0,
            "logprobs": config.calculate_log_probs,
        }
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature
        if "agent_name" not in batch:
            batch["agent_name"] = NonTensorData(config.agent.default_agent_loop)
        trajectory_info = await get_trajectory_info(batch["global_steps"], batch["index"], validate)
        for i in range(len(batch)):
            prompt = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    prompt[key] = value[i]
                elif isinstance(value, NonTensorStack):
                    prompt[key] = value[i].data
                elif isinstance(value, NonTensorData):
                    prompt[key] = value.data
                elif type(value).__name__ == "LinkedList":
                    prompt[key] = value[i]
                elif isinstance(value, (bool, int, float, str)):
                    prompt[key] = value
                else:
                    logger.warning("Unsupported AppWorld prompt field %s (%s)", key, type(value).__name__)
            task = asyncio.create_task(
                self._run_prompt(prompt, sampling_params, trajectory=trajectory_info[i], trace=False)
            )
            self.background_tasks.add(task)
            task.add_done_callback(self._on_background_task_done)

    def _on_background_task_done(self, task: asyncio.Task) -> None:
        """Remove a rollout task and surface unexpected callback failures.

        ``generate_sequences`` intentionally returns before the agent loop
        finishes.  Without inspecting the task here, an exception raised
        outside the normal failure path becomes an unobserved asyncio error
        and the corresponding GRPO group can wait forever.
        """

        self.background_tasks.discard(task)
        if task.cancelled():
            logger.warning("AppWorld rollout task was cancelled")
            return
        try:
            error = task.exception()
        except asyncio.CancelledError:
            logger.warning("AppWorld rollout task was cancelled while collecting its result")
            return
        if error is not None:
            logger.error(
                "Unhandled AppWorld rollout task failure: %s",
                type(error).__name__,
                exc_info=(type(error), error, error.__traceback__),
            )

    def _rollout_timeout_seconds(self) -> float:
        """Read and validate the per-replica timeout from the agent config."""

        agent_config = getattr(self.config.actor_rollout_ref.rollout, "agent", None)
        configured = agent_config.get("rollout_timeout") if hasattr(agent_config, "get") else None
        value = configured if configured is not None else os.environ.get("APPWORLD_ROLLOUT_TIMEOUT", "1200")
        try:
            timeout = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"rollout_timeout must be a finite positive number, got {value!r}") from exc
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(f"rollout_timeout must be a finite positive number, got {value!r}")
        return timeout

    async def _record_replica_failure(
        self,
        *,
        uid: str,
        partition_id: str,
        replica_index: int,
        is_leader: bool,
        reason: str,
    ) -> None:
        """Publish a terminal replica failure so its group cannot hang."""

        await tq.async_kv_put(
            key=uid,
            partition_id=partition_id,
            tag={
                "status": "failure",
                "seq_len": 0,
                "session_id": replica_index,
                "failure_reason": reason[:512],
            },
        )
        # If the leader itself timed out, no later _wait_for_group call will
        # clean markers left by sibling replicas.  Clean them here; sibling
        # requests have their own timeout and cannot remain live indefinitely.
        if is_leader:
            await self._clear_group_meta(uid, partition_id)

    async def _agent_loop_postprocess(self, output, validate, **kwargs) -> None:
        """Write either normal trajectories or strict event rows to TQ."""
        if not _strict_event_mode(self.config):
            return await super()._agent_loop_postprocess(output, validate, **kwargs)

        uid, session_id = kwargs["uid"], kwargs["session_id"]
        outputs = output if isinstance(output, list) else [output]
        if not outputs:
            logger.warning("Empty output for prompt %s_%s", uid, session_id)
            return

        await self._compute_score(outputs, kwargs=kwargs)
        final_output = outputs[-1]
        await self._compute_teacher_logprobs(
            final_output,
            prompt_ids=final_output.prompt_ids,
            response_ids=final_output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )
        if final_output.reward_score is not None:
            for item in outputs[:-1]:
                item.reward_score = final_output.reward_score
                item.extra_fields["reward_extra_info"] = final_output.extra_fields["reward_extra_info"]

        keys, fields, tags = [], [], []
        for output_index, item in enumerate(outputs):
            # Build event rows before as_dict(), which consumes teacher fields
            # from extra_fields as part of veRL's normal serialization.
            event_rows = build_event_tq_fields(item, self.config, tokenizer=getattr(self, "tokenizer", None))
            base = item.as_dict()
            tensor_keys = {
                "prompts", "responses", "response_mask", "loss_mask", "attention_mask",
                "input_ids", "position_ids", "rollout_log_probs", "rm_scores",
            }
            base_fields = {key: value for key, value in base.items() if key not in tensor_keys}
            for event_index, event in enumerate(event_rows):
                field = dict(base_fields)
                # Event rows already carry their own immutable context/action.
                # Do not duplicate the complete trajectory snapshots in every
                # TQ row; those can otherwise multiply queue memory by the
                # number of events in a long AppWorld episode.
                extra_fields = dict(base_fields.get("extra_fields") or {})
                for heavy_key in (
                    "appworld_event_records",
                    "appworld_context_snapshots",
                    "appworld_action_spans",
                ):
                    extra_fields.pop(heavy_key, None)
                field["extra_fields"] = extra_fields
                field.update(kwargs)
                field.update(event)
                field["uid"] = str(event["uid"])
                field["loss_mask"] = event["response_mask"]
                field["multi_modal_inputs"] = self._compute_multi_modal_inputs(item, event["input_ids"])
                key = f"{uid}_{session_id}_{output_index}_{event_index}"
                keys.append(key)
                fields.append(field)
                prompt_len = int(event["prompts"].size(0))
                response_len = int(event["responses"].size(0))
                tags.append(
                    {
                        "status": "success",
                        "group_uid": str(event.get("event_group_uid") or ""),
                        "trajectory_uid": str(event.get("event_trajectory_uid") or ""),
                        "event_index": int(event.get("event_index", event_index)),
                        "event_context_tokens": int(event.get("event_context_tokens", prompt_len)),
                        "event_action_tokens": int(event.get("event_action_tokens", response_len)),
                        "event_logprob_tokens": int(event.get("event_logprob_tokens", response_len)),
                        "expected_group_size": int(
                            self.config.actor_rollout_ref.rollout.val_kwargs.n
                            if validate
                            else self.config.actor_rollout_ref.rollout.n
                        ),
                        "prompt_len": prompt_len,
                        "response_len": response_len,
                        "seq_len": prompt_len + response_len,
                        "global_steps": kwargs["global_steps"],
                        "min_global_steps": item.extra_fields.get("min_global_steps"),
                        "max_global_steps": item.extra_fields.get("max_global_steps"),
                    }
                )
        if not fields:
            raise RuntimeError(f"strict event PPO produced no event rows for {uid}_{session_id}")
        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id="train" if not validate else "val",
        )

    @staticmethod
    def _group_meta_partition(partition_id: str) -> str:
        """Return a queue partition reserved for synchronization metadata.

        ReplayBuffer scans every non-prompt key in the data partition as a
        materializable trajectory.  Completion markers therefore must live in
        a separate partition, even though they are never fetched as fields.
        """

        return f"{partition_id}__appworld_meta"

    async def _clear_group_meta(self, uid: str, partition_id: str) -> None:
        meta_partition_id = self._group_meta_partition(partition_id)
        metadata = tq.kv_list(partition_id=meta_partition_id) or {}
        partition = metadata.get(meta_partition_id, {})
        keys = [key for key in partition if key.startswith(f"{uid}__session_done__")]
        if keys:
            await tq.async_kv_clear(keys=keys, partition_id=meta_partition_id)

    async def _wait_for_group(self, uid: str, partition_id: str, group_size: int) -> bool:
        """Wait for sibling replicas, returning false on failure or timeout."""
        agent_config = self.config.actor_rollout_ref.rollout.agent
        configured = agent_config.get("group_wait_timeout") if hasattr(agent_config, "get") else None
        timeout = float(
            configured
            if configured is not None
            else os.environ.get("APPWORLD_GROUP_WAIT_TIMEOUT", "1500")
        )
        deadline = asyncio.get_running_loop().time() + timeout
        prefix = f"{uid}_"
        meta_partition_id = self._group_meta_partition(partition_id)
        while asyncio.get_running_loop().time() < deadline:
            data_metadata = tq.kv_list(partition_id=partition_id) or {}
            partition = data_metadata.get(partition_id, {})
            prompt_tag = partition.get(uid, {})
            if prompt_tag.get("status") == "failure":
                await self._clear_group_meta(uid, partition_id)
                return False
            # Event rows are written before the agent-loop task returns.  Do not
            # validate as soon as the first event from every session appears:
            # long trajectories would otherwise be accepted with only a prefix
            # of their events.  Each replica publishes an explicit completion
            # marker after its event rows have been flushed.
            done_prefix = f"{uid}__session_done__"
            meta_metadata = tq.kv_list(partition_id=meta_partition_id) or {}
            meta_partition = meta_metadata.get(meta_partition_id, {})
            finished_sessions = set()
            for key, tag in meta_partition.items():
                if not key.startswith(done_prefix) or tag.get("status") != "session_finished":
                    continue
                session_id = tag.get("session_id")
                if session_id is None:
                    session_id = key[len(done_prefix) :]
                finished_sessions.add(str(session_id))
            if len(finished_sessions) < group_size:
                await asyncio.sleep(0.05)
                continue
            successful_sessions = {
                key[len(prefix) :].split("_", 1)[0]
                for key, tag in partition.items()
                if key.startswith(prefix) and tag.get("status") == "success"
            }
            if len(successful_sessions) >= group_size:
                valid = self._validate_strict_group(partition, uid, group_size)
                await self._clear_group_meta(uid, partition_id)
                if not valid:
                    await tq.async_kv_put(
                        key=uid,
                        partition_id=partition_id,
                        tag={"status": "failure", "seq_len": 0},
                    )
                    return False
                return True
            await asyncio.sleep(0.05)
        logger.error("Timed out waiting for AppWorld GRPO group uid=%s size=%s", uid, group_size)
        await self._clear_group_meta(uid, partition_id)
        return False

    @staticmethod
    def _validate_strict_group(partition: dict, uid: str, group_size: int) -> bool:
        """Validate the event cardinality and token dimensions of a GRPO group."""

        prefix = f"{uid}_"
        by_session: dict[str, list[dict]] = defaultdict(list)
        for key, tag in partition.items():
            if not key.startswith(prefix) or tag.get("status") != "success":
                continue
            session = key[len(prefix) :].split("_", 1)[0]
            # Keep the queue key alongside the tag.  Some TransferQueue
            # backends drop custom metadata fields, while the key remains
            # available and contains the event ordinal.
            by_session[session].append({**tag, "__tq_key": str(key)})
        if len(by_session) != group_size:
            logger.error(
                "Strict event group %s has %d sessions, expected %d",
                uid,
                len(by_session),
                group_size,
            )
            return False

        trajectories: set[str] = set()
        group_uids: set[str] = set()
        event_counts: dict[str, int] = {}
        for session, tags in by_session.items():
            indexed_tags = []
            for tag in tags:
                tag_index = tag.get("event_index")
                try:
                    event_index = int(tag_index)
                except (TypeError, ValueError):
                    event_index = -1

                # TransferQueue versions used on the server may omit custom
                # tag fields while retaining the key.  Keys are generated as
                # {uid}_{session}_{output_index}_{event_index}, so recover the
                # final index from the key when the tag value is absent.
                key = str(tag.get("__tq_key") or "")
                key_index = None
                if key.startswith(prefix):
                    suffix = key[len(prefix) :]
                    parts = suffix.rsplit("_", 2)
                    if len(parts) == 3:
                        try:
                            key_index = int(parts[-1])
                        except ValueError:
                            key_index = None
                if event_index < 0 and key_index is not None:
                    event_index = key_index
                elif event_index < 0:
                    # Older TransferQueue backends can expose neither custom
                    # tag fields nor the full event suffix in the metadata
                    # view.  The batch writer preserves insertion order for a
                    # session, so use that ordinal as a last-resort index and
                    # keep the strict duplicate/contiguity checks below.
                    event_index = len(indexed_tags)
                    logger.warning(
                        "Strict event group %s session %s missing event index metadata; "
                        "using queue order fallback index=%d key=%r",
                        uid,
                        session,
                        event_index,
                        key,
                    )
                elif key_index is not None and event_index != key_index:
                    logger.error(
                        "Strict event group %s session %s has mismatched event index tag=%s key=%s",
                        uid,
                        session,
                        event_index,
                        key_index,
                    )
                    return False
                indexed_tags.append((event_index, tag))

            ordered = [tag for _, tag in sorted(indexed_tags, key=lambda item: item[0])]
            indices = [event_index for event_index, _ in sorted(indexed_tags, key=lambda item: item[0])]
            if indices != list(range(len(indices))):
                logger.error("Strict event group %s session %s has non-contiguous event indices %s", uid, session, indices)
                return False
            trajectory_ids = {str(tag.get("trajectory_uid") or "") for tag in ordered}
            trajectory_ids.discard("")
            # Custom metadata is not preserved by every TransferQueue backend.
            # A replica/session is already a unique trajectory within a GRPO
            # group, so use that stable identity when trajectory_uid is absent.
            if not trajectory_ids:
                trajectory_ids = {f"{uid}:{session}"}
            if len(trajectory_ids) != 1:
                logger.error("Strict event group %s session %s has invalid trajectory UID set %s", uid, session, trajectory_ids)
                return False
            trajectory = next(iter(trajectory_ids))
            if trajectory in trajectories:
                logger.error("Strict event group %s reuses trajectory UID %s", uid, trajectory)
                return False
            trajectories.add(trajectory)
            event_counts[trajectory] = len(ordered)
            for tag in ordered:
                context_tokens = int(tag.get("event_context_tokens", 0))
                action_tokens = int(tag.get("event_action_tokens", 0))
                logprob_tokens = int(tag.get("event_logprob_tokens", 0))
                # event_* dimensions are audit metadata.  The actual tensors
                # were already validated by build_event_tq_fields; if a TQ
                # backend dropped these optional fields, do not reject an
                # otherwise valid row.  When present, retain strict checks.
                if (
                    (context_tokens > 0 and action_tokens > 0 and context_tokens < action_tokens)
                    or (action_tokens > 0 and logprob_tokens > 0 and logprob_tokens != action_tokens)
                ):
                    logger.error(
                        "Strict event group %s session %s invalid dimensions context=%d action=%d logprob=%d",
                        uid,
                        session,
                        context_tokens,
                        action_tokens,
                        logprob_tokens,
                    )
                    return False
                group_uid = str(tag.get("group_uid") or uid)
                if group_uid:
                    group_uids.add(group_uid)
        if len(trajectories) != group_size or len(group_uids) > 1:
            logger.error("Strict event group %s has trajectories=%s group_uids=%s", uid, trajectories, group_uids)
            return False
        logger.info(
            "Strict event group accepted uid=%s size=%d trajectories=%d event_counts=%s",
            uid,
            group_size,
            len(trajectories),
            event_counts,
        )
        return True

    async def _run_prompt(self, prompt, sampling_params, trajectory, trace=False):
        """Run exactly one replica, using its explicit GRPO session id."""
        uid = prompt["uid"]
        partition_id = "val" if trajectory["validate"] else "train"
        replica_index = int(prompt.pop("__session_id__", 0))
        group_size = int(prompt.pop("__group_size__", 1))
        is_leader = bool(prompt.pop("__group_leader__", False))
        prompt.pop("__rollout_n__", None)
        if is_leader:
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})

        task = asyncio.create_task(
            self._run_agent_loop(
                sampling_params,
                trajectory=trajectory,
                trace=trace,
                session_id=replica_index,
                **prompt,
            )
        )
        timeout = self._rollout_timeout_seconds()
        errors = []
        try:
            # Shield keeps cancellation under our control.  This lets us
            # explicitly await task cleanup before publishing group failure,
            # including for HTTP clients that need one event-loop turn to
            # release their pending request.
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(
                "AppWorld rollout timed out uid=%s session=%s after %.1fs; cancelling task",
                uid,
                replica_index,
                timeout,
            )
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await self._record_replica_failure(
                uid=uid,
                partition_id=partition_id,
                replica_index=replica_index,
                is_leader=is_leader,
                reason="rollout_timeout",
            )
            return
        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        except BaseException as error:
            errors.append(error)
        if errors:
            for error in errors:
                logger.error(
                    "AppWorld replica failed for uid=%s session=%s",
                    uid,
                    replica_index,
                    exc_info=(type(error), error, error.__traceback__),
                )
            await self._record_replica_failure(
                uid=uid,
                partition_id=partition_id,
                replica_index=replica_index,
                is_leader=is_leader,
                reason=f"{type(errors[0]).__name__}: {errors[0]}",
            )
            return

        # Publish completion only after _agent_loop_postprocess has written all
        # event rows for this replica.  The group leader waits for these markers
        # before validating cardinality and event ordering.
        meta_partition_id = self._group_meta_partition(partition_id)
        await tq.async_kv_put(
            key=f"{uid}__session_done__{replica_index}",
            partition_id=meta_partition_id,
            tag={"status": "session_finished", "session_id": replica_index},
        )

        if is_leader:
            finished = await self._wait_for_group(uid, partition_id, group_size)
            status = "finished" if finished else "failure"
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": status})


class AppWorldAgentLoopManager(AgentLoopManagerTQ):
    """Dispatch every GRPO replica to a distinct AppWorld worker process."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_loop_workers_class = AppWorldAgentLoopWorkerTQ

    def generate_sequences(self, prompts) -> None:
        """Expand each prompt into replicas before round-robin dispatch."""
        workers = self.agent_loop_workers
        if not workers:
            return
        worker_index = 0
        requests = []
        for sample_index in range(len(prompts)):
            sample = prompts[sample_index : sample_index + 1]
            validate = False
            if "validate" in sample:
                value = sample["validate"]
                validate = bool(value.item()) if hasattr(value, "item") else bool(value)
            rollout = self.config.actor_rollout_ref.rollout
            replica_count = int(rollout.val_kwargs.n if validate else rollout.n)
            for replica_index in range(replica_count):
                replica = sample.clone()
                replica["__rollout_n__"] = NonTensorData(1)
                replica["__session_id__"] = NonTensorData(replica_index)
                replica["__group_size__"] = NonTensorData(replica_count)
                replica["__group_leader__"] = NonTensorData(replica_index == 0)
                worker = workers[worker_index % len(workers)]
                worker_index += 1
                requests.append(worker.generate_sequences.remote(replica))
        ray.get(requests)


__all__ = ["AppWorldAgentLoopManager", "AppWorldAgentLoopWorkerTQ"]
