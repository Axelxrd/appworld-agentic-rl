"""Small scripted solvers used only to validate real AppWorld success paths."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from appworld_agentic_rl.adapter import AppWorldAdapter


ANSWER_MARKER = "APPWORLD_SCRIPTED_ANSWER="


def _require_success(result: dict[str, Any], step_name: str) -> str:
    error = result.get("error")
    if error:
        raise RuntimeError(f"{step_name} failed: {error}")
    observation = result.get("observation")
    if not isinstance(observation, str):
        raise RuntimeError(f"{step_name} returned a non-text observation")
    return observation


def solve_spotify_most_liked_playlist_song(adapter: AppWorldAdapter) -> str:
    """Solve task family 82e2fac using only agent-visible AppWorld APIs."""

    adapter.step({"name": "inspect_api_schema", "arguments": {"app": "spotify", "api": "login"}})
    adapter.step(
        {
            "name": "inspect_api_schema",
            "arguments": {"app": "spotify", "api": "show_playlist_library"},
        }
    )
    adapter.step({"name": "inspect_api_schema", "arguments": {"app": "spotify", "api": "show_song"}})

    auth_code = """
profile = apis.supervisor.show_profile()
passwords = apis.supervisor.show_account_passwords()
spotify_password = next(item["password"] for item in passwords if item["account_name"] == "spotify")
spotify_auth = apis.spotify.login(username=profile["email"], password=spotify_password)
spotify_access_token = spotify_auth["access_token"]
print("Spotify authentication successful")
"""
    _require_success(
        adapter.step({"name": "query_state", "arguments": {"code": auth_code}}),
        "spotify authentication",
    )

    solve_code = f"""
import json

playlists = []
page_index = 0
while True:
    page = apis.spotify.show_playlist_library(
        access_token=spotify_access_token,
        page_index=page_index,
        page_limit=20,
    )
    if not page:
        break
    playlists.extend(page)
    if len(page) < 20:
        break
    page_index += 1

song_ids = sorted({{
    song_id
    for playlist in playlists
    for song_id in playlist["song_ids"]
}})
if not song_ids:
    raise RuntimeError("No songs found in Spotify playlist library")

songs = [apis.spotify.show_song(song_id=song_id) for song_id in song_ids]
most_liked_song = max(songs, key=lambda song: (song["like_count"], -song["song_id"]))
scripted_answer = most_liked_song["title"]
print({ANSWER_MARKER!r} + json.dumps(scripted_answer))
"""
    observation = _require_success(
        adapter.step({"name": "query_state", "arguments": {"code": solve_code}}),
        "playlist and song query",
    )
    match = re.search(rf"{re.escape(ANSWER_MARKER)}(.+)", observation)
    if not match:
        raise RuntimeError("Scripted solver output did not contain an answer marker")
    answer = json.loads(match.group(1).strip())
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Scripted solver produced an invalid answer")

    _require_success(
        adapter.step(
            {
                "name": "complete_task",
                "arguments": {"answer": answer, "status": "success"},
            }
        ),
        "task completion",
    )
    return answer


def run_scripted_smoke(
    *,
    task_id: str,
    split: str = "train",
    experiment_name: str = "appworld_scripted_smoke",
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Run a registered scripted success-path check and return a compact result."""

    if not task_id.startswith("82e2fac_"):
        raise ValueError(f"No scripted solver is registered for task_id {task_id!r}")

    adapter = AppWorldAdapter(experiment_name=experiment_name, max_steps=20)
    try:
        initial = adapter.reset(task_id, split=split)
        answer = solve_spotify_most_liked_playlist_song(adapter)
        terminal = adapter.evaluate()
        if output is not None:
            adapter.save_trajectory(output)
        return {
            "task_id": task_id,
            "instruction": initial["instruction"],
            "answer": answer,
            "step_count": len(adapter.export_trajectory()["steps"]),
            "evaluation_success": terminal["evaluation"].get("data", {}).get("success"),
            "terminal_reward": terminal["terminal_reward"],
            "trajectory_path": str(output) if output is not None else None,
        }
    finally:
        adapter.close()
