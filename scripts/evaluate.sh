#!/usr/bin/env bash
# Starts the exported GGUF model with llama-server, runs the evaluation
# suite against it, then shuts the server down. Requires LLAMA_CPP_DIR set
# to a built llama.cpp checkout (llama-server must exist under build/bin/).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${LLAMA_CPP_DIR:?set LLAMA_CPP_DIR to a built llama.cpp checkout}"

MODEL_PATH=$(python -c "import yaml; print(yaml.safe_load(open('configs/serving.yaml'))['export']['output_path'])")
HOST=$(python -c "import yaml; print(yaml.safe_load(open('configs/serving.yaml'))['server']['host'])")
PORT=$(python -c "import yaml; print(yaml.safe_load(open('configs/serving.yaml'))['server']['port'])")
CTX=$(python -c "import yaml; print(yaml.safe_load(open('configs/serving.yaml'))['server']['context_length'])")

"$LLAMA_CPP_DIR/build/bin/llama-server" \
  --model "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --ctx-size "$CTX" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null' EXIT

echo "waiting for llama-server (pid $SERVER_PID) on $HOST:$PORT..."
for _ in $(seq 1 30); do
  if curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

echo "== accuracy =="
python evaluation/accuracy.py

echo "== hallucination =="
python evaluation/hallucination_test.py

echo "== benchmark =="
python evaluation/benchmark.py
