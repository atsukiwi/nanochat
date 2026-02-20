#!/bin/bash

# Single-GPU training on GPU 1, pinned to CPUs 30-59
# Designed for remote sessions (runs inside tmux for persistence)
#
# Usage:
#   # Start training (launches tmux session "nanochat-train"):
#   bash runs/train_gpu1.sh
#
#   # Attach to running session later:
#   tmux attach -t nanochat-train
#
#   # Detach from tmux without stopping: Ctrl+b, then d
#
# wandb: always enabled, project "nanochat", run name "6000ada-YYYYMMDD"

set -euo pipefail

TMUX_SESSION="nanochat-train"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# If not already inside the tmux session, create one and re-run this script inside it
if [ -z "${TMUX:-}" ] || [ "$(tmux display-message -p '#S' 2>/dev/null)" != "$TMUX_SESSION" ]; then
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "tmux session '$TMUX_SESSION' already exists."
        echo "  Attach:  tmux attach -t $TMUX_SESSION"
        echo "  Kill:    tmux kill-session -t $TMUX_SESSION"
        exit 1
    fi
    echo "Starting tmux session '$TMUX_SESSION' ..."
    echo "  Detach (keep running):  Ctrl+b, then d"
    echo "  Re-attach later:        tmux attach -t $TMUX_SESSION"
    # Forward environment variables into the tmux session
    tmux new-session -d -s "$TMUX_SESSION" -c "$SCRIPT_DIR" \
        "bash runs/train_gpu1.sh"
    tmux attach -t "$TMUX_SESSION"
    exit 0
fi

# ---- From here we are running inside tmux ----

cd "$SCRIPT_DIR"

# GPU: use only GPU 0
export CUDA_VISIBLE_DEVICES=0
# CPU: pin all child processes to cores 30-59
TASKSET="taskset -c 30-59"

export OMP_NUM_THREADS=4
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"

# ---- Python venv setup ----
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate
# Flash Attention 2 (ソースビルド、初回のみ時間がかかる)
uv pip install flash-attn --no-build-isolation

# ---- wandb (always enabled, name: 6000ada-YYYYMMDD) ----
WANDB_RUN="6000ada-$(date +%Y%m%d)"

# ---- Report header ----
$TASKSET python -m nanochat.report reset

# ---- Tokenizer ----
# Download initial data shards
$TASKSET python -m nanochat.dataset -n 8
# Download remaining shards in background
$TASKSET python -m nanochat.dataset -n 370 &
DATASET_DOWNLOAD_PID=$!
# Train tokenizer
$TASKSET python -m scripts.tok_train
# Evaluate tokenizer
$TASKSET python -m scripts.tok_eval

# ---- Base model (pretraining) ----
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Single GPU training (nproc_per_node=1, no --fp8 since RTX 6000 Ada)
$TASKSET torchrun --standalone --nproc_per_node=1 \
    -m scripts.base_train -- \
    --depth=26 \
    --target-param-data-ratio=8.25 \
    --device-batch-size=8 \
    --save-every=50 \
    --run="$WANDB_RUN"

# Evaluate
$TASKSET torchrun --standalone --nproc_per_node=1 \
    -m scripts.base_eval -- --device-batch-size=32

# ---- SFT ----
curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

$TASKSET torchrun --standalone --nproc_per_node=1 \
    -m scripts.chat_sft -- \
    --device-batch-size=32 \
    --run="$WANDB_RUN"

$TASKSET torchrun --standalone --nproc_per_node=1 \
    -m scripts.chat_eval -- -i sft

# ---- Report ----
$TASKSET python -m nanochat.report generate

echo ""
echo "========================================"
echo "  Training complete!"
echo "========================================"
