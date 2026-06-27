# LLM backend comparison

Same corpus, temperature=0, run offline against each backend.

## Scorecard

| metric | claude-sonnet-4.5 |
|---|---|
| structured output | ✅ |
| tool calling | ✅ |
| GPSR coverage | **100%** (56/56) |
| render defects | 0 |
| parse latency/cmd | 4.1s |
| agent routing | 100% (5/5) |

## claude-sonnet-4.5 — detail

**Agent routing:**

- ✓ 'Move forward one meter.' expect=delegate_to_actuator got=['delegate_to_actuator', 'speak']
- ✓ 'Go to the kitchen.' expect=delegate_to_actuator got=['delegate_to_actuator', 'speak']
- ✓ 'What do you see in front of you right now?' expect=delegate_to_vision got=['delegate_to_vision', 'speak']
- ✓ 'Where did you last see the cola?' expect=delegate_to_database got=['delegate_to_database', 'speak']
- ✓ 'Say hello to everyone.' expect=speak got=['speak']
