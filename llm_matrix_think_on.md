# LLM backend comparison

Same corpus, temperature=0, run offline against each backend.

## Scorecard

| metric | local-qwen3.5-9b |
|---|---|
| structured output | ✅ |
| tool calling | ✅ |
| GPSR coverage | **90%** (9/10) |
| render defects | 0 |
| parse latency/cmd | 17.8s |

## local-qwen3.5-9b — detail

**GPSR parse misses:**

- 'tell me the name of the person at the bed' steps=['navigate', 'find_person', 'get_person_info', 'say'] gaps=[('which', ''), ('info', '')]
