#!/usr/bin/env bash
set -euo pipefail

BEEKEEPER_HOST="http://lab.local:5000"
PROJECT="q-homebot-route-planner"
RUN_ID="${1:-latest}"
DEST="$(dirname "$0")/checkpoints"

mkdir -p "$DEST"

echo "Downloading checkpoints for run '$RUN_ID' from $BEEKEEPER_HOST..."
curl -fsSL "${BEEKEEPER_HOST}/api/v1/projects/${PROJECT}/runs/${RUN_ID}/files/checkpoints?zip=1" \
    -o /tmp/beekeeper_checkpoints.zip

echo "Extracting to $DEST..."
unzip -o /tmp/beekeeper_checkpoints.zip -d "$DEST"
rm /tmp/beekeeper_checkpoints.zip

echo "Done. Models are in $DEST"
ls -lh "$DEST"
