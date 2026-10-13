from appworld_agentic_rl.context_memory import (
    MemoryState,
    context_snapshot,
    serialize_memory,
    update_memory,
)


def test_memory_extracts_actor_visible_facts_deterministically():
    memory = MemoryState()
    update_memory(
        memory,
        {
            "step_index": 3,
            "tool_name": "query_state",
            "arguments": {"app": "venmo"},
            "observation": {
                "target_user_id": 42,
                "name": "Alice",
                "nested": {"payment_request_id": "pr-7"},
            },
        },
    )

    rendered = serialize_memory(memory)
    assert "target_user_id = 42" in rendered
    assert "nested.payment_request_id = pr-7" in rendered
    assert "name = Alice" in rendered
    assert "event 3" in rendered


def test_memory_redacts_secret_observation_values():
    memory = MemoryState()
    update_memory(
        memory,
        {
            "step_index": 1,
            "tool_name": "run_code",
            "arguments": {"code": "token = login_result['access_token']"},
            "observation": {"access_token": "must-not-leak", "user_email": "user@example.com"},
        },
    )

    rendered = serialize_memory(memory)
    assert "must-not-leak" not in rendered
    assert "access_token = available" in rendered


def test_memory_deduplicates_actions_and_keeps_recent_errors_bounded():
    memory = MemoryState()
    step = {"tool_name": "query_state", "arguments": {"code": "read()"}, "observation": {"status": "ok"}}
    update_memory(memory, step)
    update_memory(memory, step)
    for i in range(8):
        update_memory(memory, {"tool_name": "run_code", "arguments": {}, "observation": {}, "error": f"e{i}"})

    assert len(memory.completed) == 1
    assert len(memory.errors) == 4
    assert "e7" in serialize_memory(memory)
    assert "e0" not in serialize_memory(memory)


def test_context_snapshot_records_preserved_fields_and_truncation():
    memory = MemoryState(auth={"venmo_token": "available"}, entities={"payment_request_id": "pr-7"}, version=5)
    record = context_snapshot(
        turn=17,
        context_tokens_before=42150,
        context_token_limit=40960,
        tokens_removed=1190,
        memory=memory,
        recent_groups=4,
    )

    assert record["truncated"] is True
    assert record["tokens_removed"] == 1190
    assert record["context_tokens_after"] == 40960
    assert record["preserved"]["system_prompt"] is True
    assert record["preserved"]["task_instruction"] is True
    assert record["preserved"]["authentication"] is True
    assert record["preserved"]["entity_ids"] == ["payment_request_id"]


def test_memory_preserves_variable_bindings_without_secret_values():
    memory = MemoryState()
    update_memory(
        memory,
        {
            "step_index": 4,
            "tool_name": "run_code",
            "arguments": {"code": "venmo_token = auth['access_token']"},
            "observation": {"status": "ok"},
            "metadata": {
                "variable_bindings": [
                    {
                        "name": "venmo_token",
                        "kind": "derived_field",
                        "semantic": "authentication",
                        "derived_from": "auth[access_token]",
                        "source_event": 4,
                        "available": True,
                    }
                ]
            },
        },
    )

    rendered = serialize_memory(memory)
    assert "venmo_token" in rendered
    assert "derived_field" in rendered
    assert "auth[access_token]" in rendered
    assert "available" in rendered
    assert "must-not-leak" not in rendered


def test_memory_marks_failed_binding_unavailable():
    memory = MemoryState()
    update_memory(
        memory,
        {
            "step_index": 2,
            "tool_name": "query_state",
            "arguments": {"code": "venmo_token = login()"},
            "observation": {"error": "read-only tool rejected"},
            "error": "read-only tool rejected",
            "metadata": {
                "variable_bindings": [
                    {
                        "name": "venmo_token",
                        "kind": "api_result",
                        "semantic": "authentication",
                        "app": "venmo",
                        "source_event": 2,
                        "available": False,
                    }
                ]
            },
        },
    )

    assert memory.variable_bindings["venmo_token"]["available"] is False
    assert "unavailable" in serialize_memory(memory)


def test_memory_deduplicates_search_results_and_keeps_compact_fields():
    memory = MemoryState()
    search_step = {
        "step_index": 5,
        "tool_name": "search_api_docs",
        "arguments": {"query": "Venmo payment request"},
        "observation": {
            "query": "Venmo payment request",
            "results": [
                {
                    "app": "venmo",
                    "api": "create_payment_request",
                    "description": "  Send a payment request. ",
                    "parameters": [
                        {"name": "user_email", "required": True},
                        {"name": "amount", "required": True},
                        {"name": "description", "required": True},
                    ],
                    "returns": {"very": "large", "schema": "omitted"},
                },
                {
                    "app": "venmo",
                    "api": "search_friends",
                    "description": "Search users.",
                    "parameters": [],
                },
            ],
        },
    }
    update_memory(memory, search_step)
    update_memory(
        memory,
        {
            **search_step,
            "step_index": 8,
            "arguments": {"query": "Authenticate and request money"},
        },
    )

    assert set(memory.api_discoveries) == {
        "venmo.create_payment_request",
        "venmo.search_friends",
    }
    discovery = memory.api_discoveries["venmo.create_payment_request"]
    assert discovery["search_count"] == 2
    assert discovery["required_parameters"] == ["user_email", "amount", "description"]
    rendered = serialize_memory(memory)
    assert rendered.count("venmo.create_payment_request") == 1
    assert "Send a payment request." in rendered
    assert "very'" not in rendered


def test_inspected_schema_updates_existing_api_discovery():
    memory = MemoryState()
    update_memory(
        memory,
        {
            "step_index": 1,
            "tool_name": "search_api_docs",
            "arguments": {"query": "login"},
            "observation": {
                "results": [
                    {"app": "venmo", "api": "login", "description": "Login."}
                ]
            },
        },
    )
    update_memory(
        memory,
        {
            "step_index": 2,
            "tool_name": "inspect_api_schema",
            "arguments": {"app": "venmo", "api": "login"},
            "observation": {
                "required_parameters": ["username", "password"]
            },
        },
    )

    assert memory.api_discoveries["venmo.login"]["status"] == "inspected"
    assert memory.api_discoveries["venmo.login"]["required_parameters"] == [
        "username",
        "password",
    ]
