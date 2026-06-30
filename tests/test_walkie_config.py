"""walkie_config.load_config: root + module-local files, setdefault precedence."""

from __future__ import annotations

import os

from walkie_config import load_config


def _tree(tmp_path, root_toml, module_toml):
    root = tmp_path / "config.toml"
    root.write_text(root_toml)
    mod_dir = tmp_path / "services" / "walkie_graphs"
    mod_dir.mkdir(parents=True)
    (mod_dir / "config.toml").write_text(module_toml)
    return root


def test_module_config_loads_after_root(tmp_path, monkeypatch):
    root = _tree(
        tmp_path,
        '[a]\nTEST_WCFG_ROOT_ONLY = "root"\nTEST_WCFG_SHARED = "from-root"\n',
        '[b]\nTEST_WCFG_MODULE_ONLY = "module"\nTEST_WCFG_SHARED = "from-module"\n',
    )
    for k in ("TEST_WCFG_ROOT_ONLY", "TEST_WCFG_MODULE_ONLY", "TEST_WCFG_SHARED"):
        monkeypatch.delenv(k, raising=False)
    assert load_config(root) == 3
    assert os.environ["TEST_WCFG_ROOT_ONLY"] == "root"
    assert os.environ["TEST_WCFG_MODULE_ONLY"] == "module"
    # Root loads first → it overrides the module file's knob.
    assert os.environ["TEST_WCFG_SHARED"] == "from-root"


def test_env_wins_over_both_files(tmp_path, monkeypatch):
    root = _tree(
        tmp_path,
        '[a]\nTEST_WCFG_ENV = "from-root"\n',
        '[b]\nTEST_WCFG_ENV2 = "from-module"\n',
    )
    monkeypatch.setenv("TEST_WCFG_ENV", "from-env")
    monkeypatch.setenv("TEST_WCFG_ENV2", "from-env")
    assert load_config(root) == 0
    assert os.environ["TEST_WCFG_ENV"] == "from-env"
    assert os.environ["TEST_WCFG_ENV2"] == "from-env"


def test_missing_files_are_fine(tmp_path):
    assert load_config(tmp_path / "absent.toml") == 0


def test_real_config_loads_graphs_keys():
    # The repo's actual root + services/walkie_graphs/config.toml resolve together;
    # every key load_config fills is removed again so other tests keep seeing the
    # code defaults.
    before = set(os.environ)
    try:
        load_config()
        assert "WALKIE_EXPLORE_INTERVAL_SEC" in os.environ  # from the module file
        assert "WALKIE_MODEL" in os.environ  # from the root file
    finally:
        for key in set(os.environ) - before:
            del os.environ[key]
