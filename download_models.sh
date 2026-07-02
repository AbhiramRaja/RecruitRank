#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# download_models.sh — Download GGUF model for Stage 8 LLM Reasoning
#
# Risk 2 fix: addresses the missing-model gap identified in project_report_audit.md.
# Without a GGUF model in models/, Stage 8 uses the deterministic fallback
# template (always works, just less nuanced). Running this script enables
# the full Phi-3-mini LLM reasoning path.
#
# Usage:
#   bash download_models.sh              # downloads Phi-3-mini (default)
#   bash download_models.sh --tinyllama  # downloads TinyLlama instead (smaller)
#
# Requirements:
#   - wget or curl (any standard Linux/macOS/WSL environment)
#   - ~2.5 GB free disk space for Phi-3-mini (or ~0.7 GB for TinyLlama)
# ---------------------------------------------------------------------------

set -euo pipefail

MODELS_DIR="$(cd "$(dirname "$0")" && pwd)/models"
mkdir -p "$MODELS_DIR"

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

# Phi-3-mini-4k-instruct Q4_K_M (preferred per architecture.md Stage 8)
PHI3_URL="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
PHI3_FILE="Phi-3-mini-4k-instruct-q4.gguf"
PHI3_SIZE="~2.2 GB"

# TinyLlama-1.1B-Chat Q4_K_M (fallback — smaller, faster, less capable)
TINYLLAMA_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
TINYLLAMA_FILE="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
TINYLLAMA_SIZE="~0.7 GB"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
USE_TINYLLAMA=0
for arg in "$@"; do
    case "$arg" in
        --tinyllama) USE_TINYLLAMA=1 ;;
        --help|-h)
            echo "Usage: bash download_models.sh [--tinyllama]"
            echo ""
            echo "  (default)    Download Phi-3-mini-4k-instruct Q4_K_M ($PHI3_SIZE)"
            echo "  --tinyllama  Download TinyLlama-1.1B-Chat Q4_K_M ($TINYLLAMA_SIZE)"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg (use --help for usage)" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
if [ "$USE_TINYLLAMA" -eq 1 ]; then
    MODEL_URL="$TINYLLAMA_URL"
    MODEL_FILE="$TINYLLAMA_FILE"
    MODEL_SIZE="$TINYLLAMA_SIZE"
    MODEL_NAME="TinyLlama-1.1B-Chat"
else
    MODEL_URL="$PHI3_URL"
    MODEL_FILE="$PHI3_FILE"
    MODEL_SIZE="$PHI3_SIZE"
    MODEL_NAME="Phi-3-mini-4k-instruct"
fi

DEST="$MODELS_DIR/$MODEL_FILE"

echo "============================================================"
echo " RecruitRank — Stage 8 Model Downloader"
echo "============================================================"
echo " Model : $MODEL_NAME"
echo " Size  : $MODEL_SIZE"
echo " Dest  : $DEST"
echo "============================================================"

if [ -f "$DEST" ]; then
    echo "Model already exists at $DEST — skipping download."
    echo "Use 'rm $DEST' then re-run to force a fresh download."
    exit 0
fi

# Try wget first, then curl
if command -v wget &>/dev/null; then
    echo "Downloading with wget..."
    wget --progress=bar:force -O "$DEST" "$MODEL_URL"
elif command -v curl &>/dev/null; then
    echo "Downloading with curl..."
    curl -L --progress-bar -o "$DEST" "$MODEL_URL"
else
    echo "ERROR: Neither wget nor curl is available. Install one and retry." >&2
    exit 1
fi

echo ""
echo "Download complete: $DEST"
echo ""
echo "Next steps:"
echo "  1. pip install llama-cpp-python   # if not already installed"
echo "  2. python precompute.py           # Stage 8 will now use $MODEL_NAME"
