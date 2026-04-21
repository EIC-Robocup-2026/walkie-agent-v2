#!/bin/bash
# scripts/serve_llm.sh

MODE=${1:-vllm}
echo "🧠 Starting Model Host with Tool Support: Qwen/Qwen3.5-9B"

if [ "$MODE" == "vllm" ]; then
    # เพิ่ม --enable-auto-tool-choice และ --tool-call-parser ให้กับ vLLM
    # สำหรับ Qwen แนะนำให้ใช้ parser 'hermes' หรือ 'tool_calling' (ขึ้นอยู่กับเวอร์ชัน vLLM)
    uv run python -m vllm.entrypoints.openai.api_server \
        --model "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2" \
        --tokenizer "Qwen/Qwen3.5-9B" \
        --served-model-name "qwen3.5-9b" \
        --quantization bitsandbytes \
        --load-format bitsandbytes \
        --port 8000 \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        --enable-auto-tool-choice \
        --enable-prefix-caching \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.8 \
        --trust-remote-code

elif [ "$MODE" == "ollama" ]; then
    export OLLAMA_HOST="0.0.0.0:8000"
    ollama run qwen3.5:9b
fi
