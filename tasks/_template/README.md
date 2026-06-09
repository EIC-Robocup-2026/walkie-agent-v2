# Task template

Copy this directory to create a new RoboCup challenge launcher:

```bash
cp -r tasks/_template tasks/MyChallenge
$EDITOR tasks/MyChallenge/{config.toml,prompt.md,README.md}
chmod +x tasks/MyChallenge/run.sh
./tasks/MyChallenge/run.sh          # start the robot configured for this task
```

## What each file does

| File | Purpose |
|------|---------|
| `run.sh` | Thin wrapper → `tasks/_run.sh` → repo-root `run.sh`. Exports `WALKIE_TASK_DIR` so the app loads this task. Supports every base subcommand (`start`, `fresh`, `reset`, `viewer`, `doctor`). |
| `config.toml` | Env-var overrides (model, temperature, perception tuning). Loaded **above** the base `config.toml`. |
| `prompt.md` | Appended to the **Walkie main agent's** system prompt under `# Current task: <NAME>`. |
| `prompts/<agent>.md` | *Optional.* Appended to a specific sub-agent's prompt. Names: `walkie_agent`, `vision_agent`, `actuator_agent`, `database_agent`, `human_agent`. |

## Precedence

```
shell env  >  .env  >  task config.toml  >  base config.toml  >  code default
```

Set a one-off override at launch without editing files:

```bash
WALKIE_MODEL=anthropic/claude-opus-4.5 ./tasks/MyChallenge/run.sh
```

See `tasks/runtime.py` for the loader and `tasks/README.md` for the overview.
