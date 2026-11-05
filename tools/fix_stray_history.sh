#!/usr/bin/env bash
# Permanently remove foreign projects (e.g. Downloads/software_proj + its nested memoria-iteration1 repo) from history.
# Usage (inside the project folder):
#   bash tools/fix_stray_history.sh                      # auto-detect
#   bash tools/fix_stray_history.sh Downloads/software_proj other/stray/path   # purge explicit paths
# Rewrites history -> commit SHAs change. Back up first (git clone --mirror . ../backup.git). Force-push needed afterwards.
set -euo pipefail
command -v git-filter-repo >/dev/null || pip install --user git-filter-repo
TOP=$(git rev-parse --show-toplevel); HERE=$(pwd -P)
git clone --mirror "$TOP" "$TOP/../repo_backup_$(date +%s).git" >/dev/null 2>&1 && echo "backup mirror created next to the repo"
if [ "$#" -gt 0 ]; then
  args=(); for p in "$@"; do args+=(--path "$p"); done
  git filter-repo "${args[@]}" --invert-paths --force
elif [ "$TOP" != "$HERE" ]; then
  REL=${HERE#"$TOP"/}
  echo "Repo root ($TOP) is above this project; re-rooting history to '$REL' (drops EVERYTHING else, incl. stray projects)."
  git filter-repo --subdirectory-filter "$REL" --force
else
  echo "Repo root == project folder. Pass the stray paths to purge, e.g.:"
  echo "  bash tools/fix_stray_history.sh Downloads/software_proj"; exit 1
fi
rm -f .gitmodules
git reflog expire --expire=now --all && git gc --prune=now --aggressive
echo "Done. Now: bash tools/install_guard.sh ; then re-add your remote and: git push --force --all"
