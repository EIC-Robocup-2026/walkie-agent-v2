# Finals (§6.2)

**Status: 🔴 placeholder** — composes household chores + assisting people; depends
on the manipulation-heavy tasks (Pick-and-Place / Laundry) being ready.

> Maintain the household: bin floor trash, return misplaced objects to their
> default locations, answer raised-hand requests, close the dishwasher, and
> welcome a guest waiting behind the exit door (opening it unaided). Plus any
> arena-specific / custom tasks approved for the final.

```bash
./tasks/Finals/run.sh        # runnable scaffold (nav/perception/HRI parts)
```

## What this task wires
- **Model:** `anthropic/claude-opus-4.5`; human + scene perception on.
- **Prompt** (`prompt.md`): faithful to §6.2; until manipulation lands the robot
  does navigation/perception/HRI fully and requests assistance for grasp/place/
  close/open actions.

## To make it ready
Largely unblocked by the same manipulation pipeline as Pick-and-Place and Doing
Laundry, plus unaided door opening. Flip status once those are in place.
