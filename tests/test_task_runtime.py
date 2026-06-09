"""Unit tests for the per-challenge task launcher loader (``tasks/runtime.py``).

Pure filesystem + env coverage — no robot, no model, no server. Each test
points ``WALKIE_TASK_DIR`` at a temp dir and asserts how config and prompts are
resolved. Env is restored after every test so the suite stays order-independent.
"""

from __future__ import annotations

import os

import pytest

from tasks import runtime


@pytest.fixture(autouse=True)
def _clean_task_env():
    """Snapshot/restore the task env vars around each test."""
    saved = {k: os.environ.get(k) for k in ("WALKIE_TASK_DIR", "WALKIE_TASK")}
    for k in saved:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _activate(path):
    os.environ["WALKIE_TASK_DIR"] = str(path)


# --- active task resolution ----------------------------------------------


def test_no_task_active_is_a_noop():
    assert runtime.active_task_dir() is None
    assert runtime.active_task_name() is None
    assert runtime.load_task_config() == 0
    assert runtime.task_prompt_addendum("walkie_agent") == ""
    assert runtime.apply_task_prompt("walkie_agent", "BASE") == "BASE"


def test_blank_or_missing_dir_resolves_to_none(tmp_path):
    os.environ["WALKIE_TASK_DIR"] = "   "
    assert runtime.active_task_dir() is None
    os.environ["WALKIE_TASK_DIR"] = str(tmp_path / "does-not-exist")
    assert runtime.active_task_dir() is None


def test_active_task_name_defaults_to_dir_name(tmp_path):
    task = tmp_path / "GPSR"
    task.mkdir()
    _activate(task)
    assert runtime.active_task_dir() == task
    assert runtime.active_task_name() == "GPSR"


def test_walkie_task_overrides_dir_name(tmp_path):
    task = tmp_path / "GPSR"
    task.mkdir()
    _activate(task)
    os.environ["WALKIE_TASK"] = "Speech-and-Person-Recognition"
    assert runtime.active_task_name() == "Speech-and-Person-Recognition"


# --- config loading -------------------------------------------------------


def test_load_task_config_fills_only_unset_keys(tmp_path):
    task = tmp_path / "T"
    task.mkdir()
    (task / "config.toml").write_text(
        '[model]\nWALKIE_MODEL = "task/model"\nWALKIE_TEMPERATURE = "0.3"\n',
        encoding="utf-8",
    )
    _activate(task)
    # Pretend WALKIE_MODEL is already set (as a base config / .env value would be).
    os.environ["WALKIE_MODEL"] = "preset/model"
    try:
        filled = runtime.load_task_config()
        # Only TEMPERATURE was unset, so exactly one key filled; MODEL untouched.
        assert filled == 1
        assert os.environ["WALKIE_MODEL"] == "preset/model"
        assert os.environ["WALKIE_TEMPERATURE"] == "0.3"
    finally:
        os.environ.pop("WALKIE_MODEL", None)
        os.environ.pop("WALKIE_TEMPERATURE", None)


def test_load_task_config_no_file_is_zero(tmp_path):
    task = tmp_path / "T"
    task.mkdir()
    _activate(task)
    assert runtime.load_task_config() == 0


# --- prompt addenda -------------------------------------------------------


def test_main_agent_prompt_shorthand(tmp_path):
    task = tmp_path / "T"
    task.mkdir()
    (task / "prompt.md").write_text("Do the GPSR thing.", encoding="utf-8")
    _activate(task)
    assert runtime.task_prompt_addendum("walkie_agent") == "Do the GPSR thing."
    out = runtime.apply_task_prompt("walkie_agent", "BASE PROMPT")
    assert out.startswith("BASE PROMPT")
    assert "# Current task: T" in out
    assert out.rstrip().endswith("Do the GPSR thing.")


def test_per_subagent_prompt_file(tmp_path):
    task = tmp_path / "T"
    (task / "prompts").mkdir(parents=True)
    (task / "prompts" / "vision_agent.md").write_text("See sharply.", encoding="utf-8")
    _activate(task)
    assert runtime.task_prompt_addendum("vision_agent") == "See sharply."
    # An agent with no file gets nothing and its prompt is unchanged.
    assert runtime.task_prompt_addendum("database_agent") == ""
    assert runtime.apply_task_prompt("database_agent", "BASE") == "BASE"


def test_prompts_dir_takes_precedence_over_shorthand_for_main(tmp_path):
    task = tmp_path / "T"
    (task / "prompts").mkdir(parents=True)
    (task / "prompts" / "walkie_agent.md").write_text("explicit", encoding="utf-8")
    (task / "prompt.md").write_text("shorthand", encoding="utf-8")
    _activate(task)
    # prompts/walkie_agent.md is listed first, so it wins over prompt.md.
    assert runtime.task_prompt_addendum("walkie_agent") == "explicit"


def test_empty_prompt_file_is_ignored(tmp_path):
    task = tmp_path / "T"
    task.mkdir()
    (task / "prompt.md").write_text("   \n", encoding="utf-8")
    _activate(task)
    assert runtime.task_prompt_addendum("walkie_agent") == ""
    assert runtime.apply_task_prompt("walkie_agent", "BASE") == "BASE"
