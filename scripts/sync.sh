#!/usr/bin/env bash
# Push this working tree to the machine that runs training.
#
# The dev box is the only source of truth for code; nothing is edited in place
# on the target. CLAUDE.md is deliberately untracked by git but IS synced here --
# rsync does not read .gitignore, and it is how a session on the target box
# gets project context.
#
# Usage:
#   scripts/sync.sh omarchy
#   W2V_TARGET=omarchy scripts/sync.sh
set -euo pipefail

TARGET="${1:-${W2V_TARGET:-}}"
DEST="${W2V_REMOTE_DIR:-~/word2vec}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "$TARGET" ]; then
    echo "usage: $0 <ssh-target>   (or set W2V_TARGET)" >&2
    exit 2
fi

# results/ is produced on the target and comes back via pull-results.sh.
# Excluding it also shields it from --delete, so a sync never wipes a finished
# run. *.egg-info is left alone so the target's editable install survives.
exec rsync -az --delete --info=stats1 \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    --exclude '*.egg-info/' \
    --exclude 'results/' \
    "$REPO_ROOT/" "$TARGET:$DEST/"
