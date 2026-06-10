# Doing Laundry Challenge (§5.4)

**Status: 🔴 placeholder (not ready)** — deformable-object + bimanual
manipulation (retrieve, carry basket, fold T-shirts) is unsupported.

> Retrieve clothes from the washing machine / basket, place them on a table, and
> fold them neatly. Bonus for opening the machine, using the basket, and folding
> + stacking multiple pieces. Max time 7:00.

```bash
./tasks/DoingLaundry/run.sh        # runnable scaffold (navigation/perception only)
```

## What this task wires
- **Model:** `anthropic/claude-opus-4.5` (task planning).
- **Prompt** (`prompt.md`): faithful to §5.4; until manipulation lands the robot
  navigates, identifies the items/appliances, announces the plan, and requests
  assistance for grasp/open/fold.

## To make it ready
Needs deformable-cloth grasping, bimanual basket carry, and a folding routine.
When that exists, drop the "Manipulation not yet supported" section and flip the
status.
