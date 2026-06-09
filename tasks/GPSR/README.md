# GPSR — General Purpose Service Robot

Launch:

```bash
./tasks/GPSR/run.sh            # start
./tasks/GPSR/run.sh fresh      # wipe scene+object DBs, then start
```

## What this task wires

- **Model:** `anthropic/claude-opus-4.5` (stronger decomposition for compound
  commands) — see `config.toml`.
- **Prompt:** `prompt.md` is appended to the Walkie main agent, teaching it to
  decompose a command with `write_todos`, resolve references via the Database /
  Vision agents, and confirm completion by `speak`.
- **Perception:** 3 s live snapshot interval; scene memory on.

## Overriding at launch

```bash
WALKIE_MODEL=anthropic/claude-sonnet-4.5 ./tasks/GPSR/run.sh   # cheaper run
DISABLE_LISTENING=1 ./tasks/GPSR/run.sh                        # type commands at a TTY
```
