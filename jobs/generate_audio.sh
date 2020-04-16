#!/bin/bash
#
#SBATCH --job-name=compile_nv_wavenet_extension
#SBATCH --partition=common
#SBATCH --qos=gsn
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --export=ALL

echo "=== $(date) === debug ==="
which python
python --version

echo "=== $(date) === setup ==="
cd "/scidatasm/$USER/reformer_tts" || exit
pip intsall -e .

echo "=== $(date) === running inference ==="
MEL_DIR="/scidatasm/$USER/reformer_tts/data/raw/spectrogram"
OUT_DIR="/results/$USER/generated-audio"
mkdir "$OUT_DIR" || echo "skipped directory creation: wavenet-infer"

export WAVEGLOW_DOWNLOAD_DIR="/scidatasm/$USER/data/waveglow"
python reformer_tts/cli.py generate-audio "$MEL_DIR" -o "$OUT_DIR"

echo "=== $(date) === done ==="
