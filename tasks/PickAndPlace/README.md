# Pick and Place Challenge (§5.2)

**Status: 🔴 placeholder (not ready)** — manipulation-heavy; current arm stack
cannot load a dishwasher, pour, or do precise grasp/place.

> Clean and organise the kitchen — dirty tableware/cutlery to the dishwasher,
> trash to the bin, other objects grouped in the cabinet — then set up a simple
> breakfast (bowl, spoon, cereal, milk). Max time 7:00.

```bash
./tasks/PickAndPlace/run.sh        # runnable scaffold (perception/planning only)
```

## What this task wires
- **Model:** `anthropic/claude-opus-4.5` (task planning).
- **Prompt** (`prompt.md`): faithful to §5.2; until manipulation lands the robot
  recognises + announces objects, navigates, and requests referee assistance for
  grasps (penalised, but scores recognition/planning/navigation).

## To make it ready
Needs a real manipulation pipeline (grasp/place, dishwasher door + rack, pouring).
When that exists, drop the "Manipulation not yet supported" section from
`prompt.md` and flip this status.
