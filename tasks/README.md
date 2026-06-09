# tasks/ — per-challenge launchers

Each RoboCup challenge gets its own directory under `tasks/` so it can run with
its **own prompt, its own model, and its own tuning** without touching `main.py`
or the base config. Launch one with its `run.sh`:

```bash
./tasks/GPSR/run.sh                  # General Purpose Service Robot
./tasks/Receptionist/run.sh          # Receptionist
./tasks/GPSR/run.sh fresh            # wipe DBs first, then start
```

## Layout

```
tasks/
├── _run.sh              # shared launcher: exports WALKIE_TASK_DIR, hands off to repo run.sh
├── _template/           # copy this to add a challenge
├── runtime.py           # the loader (config + prompt) — imported by main.py & the agent factory
├── GPSR/
│   ├── run.sh           # ./tasks/GPSR/run.sh  → 3-line wrapper over _run.sh
│   ├── config.toml      # model + tuning overrides for this task
│   ├── prompt.md        # appended to the Walkie main agent's prompt
│   └── README.md
└── Receptionist/
    ├── run.sh
    ├── config.toml
    ├── prompt.md
    ├── prompts/
    │   └── human_agent.md   # optional per-sub-agent prompt addendum
    └── README.md
```

## How it works (no monkey-patching)

1. `tasks/<NAME>/run.sh` → `tasks/_run.sh` exports `WALKIE_TASK_DIR=<that dir>`,
   then `exec`s the repo-root `run.sh` (so `start` / `fresh` / `reset` / `viewer`
   / `doctor` and the stale-port cleanup all still work).
2. `main.py` calls `load_task_config()` right after `load_dotenv()` and **before**
   the base `load_config()`. Both use `setdefault`, so the task's `config.toml`
   wins over the base one. Then it prints the active task.
3. The shared agent factory (`agents/core/agent.py`) calls
   `apply_task_prompt(name, base_prompt)` for **every** agent — so each agent
   automatically gets its `tasks/<NAME>/prompts/<name>.md` addendum (and the main
   agent also accepts the shorthand `tasks/<NAME>/prompt.md`), appended under a
   `# Current task: <NAME>` heading.

When no task is active (`WALKIE_TASK_DIR` unset — i.e. plain `./run.sh` or
`uv run python main.py`), every hook is a no-op and the robot boots exactly as
before.

## Precedence

```
shell env  >  .env  >  task config.toml  >  base config.toml  >  code default
```

So a task pins its model in `config.toml`, but you can still override per launch:

```bash
WALKIE_MODEL=anthropic/claude-opus-4.5 ./tasks/Receptionist/run.sh
```

## Add a challenge

```bash
cp -r tasks/_template tasks/StoringGroceries
$EDITOR tasks/StoringGroceries/{config.toml,prompt.md,README.md}
./tasks/StoringGroceries/run.sh
```

See `tasks/runtime.py` for the loader API.
