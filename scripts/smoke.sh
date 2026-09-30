#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python -m labs.environment
python -m labs.linear
python -m labs.classical --quick
for model in softmax mlp lenet residual; do
  python -m labs.vision --model "$model" --smoke --device cuda
done
python -m labs.sequence --model rnn --smoke --device cuda
python -m labs.sequence --model gru --smoke --device cuda
python -m labs.attention --smoke --device cuda
