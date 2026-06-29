"""Env-knob readers for the walkie_world model (scene store + relations).

The app reads tuning via ``os.getenv`` (populated from config.toml by
``walkie_config.load_config``). These helpers centralize the model's knobs so the
:class:`~walkie_world.world.WalkieWorld` facade and the producer agree on defaults.
The env-var names use the ``WALKIE_EXPLORE_*`` prefix (renamed from the legacy
``WALKIE_GRAPHS_*``); the values live in ``walkie_world/config.toml``.
"""

from __future__ import annotations

import os


def _envf(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _envi(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _envb(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in ("1", "true", "yes")


def scene_store_kwargs() -> dict:
    """Keyword args for :class:`~walkie_world.scene.store.SceneStore` from the env."""
    return dict(
        store_dir=os.getenv("WALKIE_EXPLORE_STORE_DIR", "graph_scene"),
        min_obs_confirm=_envi("WALKIE_EXPLORE_MIN_OBS_CONFIRM", 2),
        require_confirmation=_envb("WALKIE_EXPLORE_REQUIRE_CONFIRMATION", True),
        prune_max_records=_envi("WALKIE_EXPLORE_PRUNE_MAX_RECORDS", 500),
        merge_dist=_envf("WALKIE_EXPLORE_ASSOC_MAX_DIST_M", 0.5),
    )


def relation_kwargs() -> dict:
    """Keyword args for :func:`~walkie_world.scene.relations.derive_relations`."""
    return dict(
        relation_max_dist=_envf("WALKIE_EXPLORE_RELATION_MAX_DIST", 1.0),
        near_m=_envf("WALKIE_EXPLORE_NEAR_M", 0.6),
        xy_overlap_min=_envf("WALKIE_EXPLORE_XY_OVERLAP_MIN", 0.15),
        z_tol=_envf("WALKIE_EXPLORE_Z_TOL", 0.05),
        on_gap=_envf("WALKIE_EXPLORE_ON_GAP", 0.08),
        contain_tol=_envf("WALKIE_EXPLORE_CONTAIN_TOL", 0.02),
    )
