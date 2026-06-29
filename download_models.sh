#!/usr/bin/env bash
set -euo pipefail

BEEKEEPER_HOST="http://lab.local:5000"
PROJECT="q-homebot-route-planner"
RUN_ID="${1:-latest}"
DEST="$(dirname "$0")/checkpoints"

mkdir -p "$DEST"

echo "Downloading checkpoints for run '$RUN_ID' from $BEEKEEPER_HOST..."
curl -fsSL "${BEEKEEPER_HOST}/api/v1/projects/${PROJECT}/runs/${RUN_ID}/files/checkpoints/q_model.pt" -o "$DEST/q_model.pt"
curl -fsSL "${BEEKEEPER_HOST}/api/v1/projects/${PROJECT}/runs/${RUN_ID}/files/checkpoints/q_model_best.pt" -o "$DEST/q_model_best.pt"

echo "Done. Models are in $DEST"
ls -lh "$DEST"
