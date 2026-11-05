#!/usr/bin/env bash
# Run once from inside the project folder. Makes the repo root == this folder and installs the guard.
set -euo pipefail
HERE=$(pwd -P)
if git rev-parse --show-toplevel >/dev/null 2>&1; then
  TOP=$(git rev-parse --show-toplevel)
  if [ "$TOP" != "$HERE" ]; then
    echo "!! Your repo root is $TOP, NOT this project folder ($HERE)."
    echo "   That is the root cause: 'git add .' from a parent sweeps in every sibling project."
    echo "   Fix history first: bash tools/fix_stray_history.sh   (then re-run this script)"; exit 1
  fi
else
  git init
fi
cp tools/git_guard.sh .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
git config status.showUntrackedFiles normal
echo "guard installed in $HERE/.git/hooks/pre-commit"
