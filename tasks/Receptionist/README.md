# Receptionist

Launch:

```bash
./tasks/Receptionist/run.sh          # start
./tasks/Receptionist/run.sh fresh    # wipe DBs, then start
```

## What this task wires

- **Model:** `anthropic/claude-sonnet-4.5` (fast, conversational) — `config.toml`.
- **Human perception on:** `HUMAN_PERCEPTION_ENABLED=1` so the robot re-identifies
  returning guests.
- **Main-agent prompt:** `prompt.md` — greet → learn name + drink → seat →
  introduce loop.
- **Sub-agent prompt override:** `prompts/human_agent.md` — demonstrates per-
  sub-agent prompts. The Human agent is told to enrol each guest's face with
  their name and favourite drink. (Drop a `prompts/<agent>.md` for any of
  `walkie_agent`, `vision_agent`, `actuator_agent`, `database_agent`,
  `human_agent`.)

This task pairs with the `feat/human-recognition` work (face re-ID + people store).
