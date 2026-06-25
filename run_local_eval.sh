#!/usr/bin/env bash
# Wait for the remote vLLM box, then run the latency probe + a small speed/coverage
# matrix (thinking on vs off). Throwaway runner — safe to delete after.
set -u
BASE="${LOCAL_BASE_URL:-http://localhost:8000/v1}"
export LOCAL_BASE_URL="$BASE"

echo "### waiting for $BASE/models (up to 8 min) ..."
for i in $(seq 1 96); do
  if curl -s -m 4 "$BASE/models" | grep -q '"id"'; then
    echo "### up after ~$((i*5))s"; break
  fi
  sleep 5
done
if ! curl -s -m 4 "$BASE/models" | grep -q '"id"'; then
  echo "### TIMED OUT — box never came up"; exit 1
fi

echo; echo "########## PROBE: thinking ON ##########"
uv run python -m manual_tests.probe_local_latency

echo; echo "########## PROBE: thinking OFF + max_tokens 512 ##########"
uv run python -m manual_tests.probe_local_latency --thinking off --max-tokens 512

echo; echo "########## MATRIX A: local, limit 10, thinking ON ##########"
uv run python -m manual_tests.eval_llm_compare --models local --limit 10 --no-agent --out llm_matrix_think_on.md

echo; echo "########## MATRIX B: local, limit 10, thinking OFF + max_tokens 512 ##########"
uv run python -m manual_tests.eval_llm_compare --models local --limit 10 --no-agent --no-think --max-tokens 512 --out llm_matrix_think_off.md

echo; echo "### DONE"
