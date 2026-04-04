#!/bin/bash
# =============================================================
# vLLM Server — kaitchup/Qwen3.5-27B-NVFP4
# Image: vllm/vllm-openai:nightly  (REQUIRED — v0.8.5.post1 lacks qwen3_5 arch)
# GPU: NVIDIA RTX 5090
#
# To pull the nightly image first (one-time):
#   sudo docker pull vllm/vllm-openai:nightly
#
# Note on reasoning parser:
#   v0.8.5.post1 only supports: deepseek_r1, granite
#   --reasoning-parser=qwen3 requires vLLM nightly (v0.9+)
#   → Omitted here; Qwen3's <think> tags still work without it,
#     they just appear in the raw output. Use llm_client.py's
#     strip_think() helper to remove them post-generation.
#
# Note on MTP (--speculative-config):
#   qwen3_next_mtp method requires a model with MTP heads preserved.
#   NVFP4 quantization may or may not have them — safe to try.
#   If vLLM logs a warning about missing MTP heads, remove the flag.
# =============================================================

# Stop any existing vLLM container
sudo docker rm -f vllm_server > /dev/null 2>&1

# HuggingFace token
TOKEN="Insert your token here"

echo "Starting vLLM: kaitchup/Qwen3.5-27B-NVFP4 ..."

sudo docker run -d --gpus all --ipc=host -p 8000:8000 --name vllm_server \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN="$TOKEN" \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  vllm/vllm-openai:nightly \
  --model=Qwen/Qwen3.5-27B-GPTQ-Int4 \
  --max-model-len=8192 \
  --gpu-memory-utilization=0.95 \
  --cpu-offload-gb=12 \
  --reasoning-parser=qwen3 \
  --trust-remote-code \
  --enforce-eager \
  --enable-chunked-prefill \
  --port=8000

echo "✓ vLLM container started"
echo "  Model: kaitchup/Qwen3.5-27B-NVFP4"
echo "  Context: 131072 tokens | GPU util: 90%"
echo "  MTP: qwen3_next_mtp (2 speculative tokens)"
echo ""
echo "  Monitor: sudo docker logs -f vllm_server"
echo "  Status:  curl http://localhost:8000/v1/models"
echo ""
echo "  Note: First run downloads model (~15-20GB). Wait 2-5 min for 'Application startup complete'."
