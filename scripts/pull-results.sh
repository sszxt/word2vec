#!/usr/bin/env bash
# Pull experiment results back from the target box.
#
# results/ is the only thing that flows target -> dev. No --delete, so a pull
# never removes anything local. Checkpoints stay on the target, where anything
# that loads them has to run anyway; set W2V_PULL_CHECKPOINTS=1 to fetch them.
#
# Usage:
#   scripts/pull-results.sh omarchy
#   W2V_PULL_CHECKPOINTS=1 scripts/pull-results.sh omarchy
set -euo pipefail

TARGET="${1:-${W2V_TARGET:-}}"
DEST="${W2V_REMOTE_DIR:-~/word2vec}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "$TARGET" ]; then
    echo "usage: $0 <ssh-target>   (or set W2V_TARGET)" >&2
    exit 2
fi

excludes=()
[ -z "${W2V_PULL_CHECKPOINTS:-}" ] && excludes=(--exclude '*.pt' --exclude 'checkpoints/')

exec rsync -az --info=stats1 "${excludes[@]}" \
    "$TARGET:$DEST/results/" "$REPO_ROOT/results/"
