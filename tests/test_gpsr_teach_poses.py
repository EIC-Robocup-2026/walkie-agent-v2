"""Offline tests for the teach-poses TOML writer (tools.teach_poses.set_pose).

Only the pure writer is exercised here — it must rewrite exactly one inline-table's
pose literal and leave the whole rest of the file (comments, aliases, other fields,
other tables) byte-for-byte intact. Reading the robot pose / driving are robot-side.
"""

from __future__ import annotations

from tasks.GPSR.tools.teach_poses import set_pose

_TOML = """\
# arena world model
[rooms]
kitchen     = { pose = [0.0, 0.0, 0.0] }
living_room = { pose = [0.0, 0.0, 0.0], aliases = ["living room", "lounge"] }

[locations]
kitchen_table = { room = "kitchen", placement = true, category = "dishes", pose = [0,0,0], aliases = ["kitchen table"] }
desk          = { room = "office", placement = true, pose = [0,0,0] }
"""


def test_sets_room_pose_preserving_aliases_and_other_lines():
    out, ok = set_pose(_TOML, "rooms", "living_room", (1.5, -2.0, 0.7853))
    assert ok
    assert 'living_room = { pose = [1.500, -2.000, 0.7853], aliases = ["living room", "lounge"] }' in out
    assert "kitchen     = { pose = [0.0, 0.0, 0.0] }" in out  # sibling untouched
    assert out.startswith("# arena world model\n")            # comment preserved


def test_sets_location_pose_preserving_fields():
    out, ok = set_pose(_TOML, "locations", "kitchen_table", (3.0, 4.0, 0.0))
    assert ok
    assert ('kitchen_table = { room = "kitchen", placement = true, category = "dishes", '
            'pose = [3.000, 4.000, 0.0000], aliases = ["kitchen table"] }') in out


def test_missing_key_leaves_text_unchanged():
    out, ok = set_pose(_TOML, "rooms", "garage", (1.0, 2.0, 3.0))
    assert not ok
    assert out == _TOML


def test_is_section_scoped():
    # kitchen_table lives in [locations]; asking to set it under [rooms] must no-op.
    out, ok = set_pose(_TOML, "rooms", "kitchen_table", (1.0, 2.0, 3.0))
    assert not ok
    assert out == _TOML


def test_only_the_named_key_changes():
    out, _ = set_pose(_TOML, "locations", "desk", (9.0, 8.0, 1.0))
    assert "desk          = { room = \"office\", placement = true, pose = [9.000, 8.000, 1.0000] }" in out
    assert "pose = [0,0,0], aliases = [\"kitchen table\"]" in out  # the other location untouched
