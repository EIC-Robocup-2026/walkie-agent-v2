# HRI — Human Robot Interaction Challenge (§5.1)

**Status: 🟢 ready-leaning** — the HRI core is supported; bag handover is partial.

> The robot works as a **receptionist** at a party: welcome two guests (arriving
> separately), learn each one's name + favourite drink, escort them to the living
> room, offer a seat, and introduce them to each other while maintaining correct
> gaze. The second guest brings a **bag** to carry to the host. Max time 6:00.

```bash
./tasks/HRI/run.sh            # start
./tasks/HRI/run.sh fresh      # wipe DBs, then start
```

## What this task wires
- **Model:** `anthropic/claude-sonnet-4.5` (fast, conversational).
- **Human perception on** (`HUMAN_PERCEPTION_ENABLED=1`) — face re-ID so guests
  are recognised after they switch seats (§5.1 "Switching Places").
- **Main prompt** (`prompt.md`): full §5.1 procedure incl. doorbell, gaze rules,
  introductions, bag handover + follow-the-host, and graceful degradation when
  manipulation isn't available.
- **Sub-agent prompt** (`prompts/human_agent.md`): enrol each guest's face with
  name + favourite drink.

## Readiness notes
- ✅ Greet / learn details / escort / gaze / seat / introduce → current agents.
- 🟡 Bag handover + follow host → needs manipulation; prompt falls back to the
  HRI-scoring parts (worth the most) and the navigation/following portion.

Pairs with the `feat/human-recognition` work (face re-ID + people store).
