# Restaurant Challenge (§5.5)

**Status: 🟡 partial** — gesture detection + online navigation + verbal
order-taking are feasible; physically serving the order needs manipulation.

> In a real, undisclosed restaurant: detect a calling/waving customer, navigate
> to their table, take their order (2 items), place it at the Kitchen-bar, and
> serve it. Optional tray transport. Assume **no wifi / external compute / power**
> at the venue.

```bash
./tasks/Restaurant/run.sh        # start
```

## What this task wires
- **Model:** `anthropic/claude-sonnet-4.5`.
- **Human perception on** — for calling/waving detection (builds on the C7
  gesture work).
- **Prompt** (`prompt.md`): faithful to §5.5; serving falls back to the
  Professional Barman placing items in a basket/tray when grasping isn't possible.

## Readiness notes
- ✅ Detect waving/calling customer, online navigation, take + confirm order verbally.
- 🟡 Grasp items / carry tray → manipulation partial; lean on the Barman per §5.5.
- ⚠️ No external compute/power guaranteed at the venue — keep the on-robot model
  path workable for a real deployment.
