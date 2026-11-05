#!/usr/bin/env bash
# pre-commit guard: refuse commits that contain anything outside this project.
set -euo pipefail
TOP=$(git rev-parse --show-toplevel)
ALLOWED_RE='^(nucseg/|tools/|tests/|README\.md$|requirements\.txt$|\.gitignore$|[A-Za-z0-9_]+\.py$|[A-Za-z0-9_ -]+\.ipynb$)'
bad=0
# 1) nested repos / submodules (gitlinks, mode 160000) — this is what dragged in memoria-iteration1
if git diff --cached --raw | awk '$2=="160000" || $1==":160000"{f=1} END{exit !f}'; then
  echo "guard: staged a nested git repo / submodule. Refusing."; bad=1
fi
# 2) paths outside the allowlist
while IFS= read -r f; do
  [[ "$f" =~ $ALLOWED_RE ]] || { echo "guard: not part of this project -> $f"; bad=1; }
done < <(git diff --cached --name-only --diff-filter=ACMR)
# 3) big / binary artefacts (data, weights, caches)
while IFS= read -r f; do
  case "$f" in *.pt|*.pth|*.npz|*.tif|*.tiff|*.svs|*.zip|*.h5|*.ckpt) echo "guard: artefact not allowed -> $f"; bad=1;; esac
  [ -f "$f" ] && [ "$(wc -c <"$f")" -gt 5000000 ] && { echo "guard: >5 MB file -> $f"; bad=1; }
done < <(git diff --cached --name-only --diff-filter=ACMR)
if [ "$bad" = 1 ]; then
  echo "guard: toplevel is $TOP. Unstage with: git restore --staged <path>   (bypass only with --no-verify)"; exit 1
fi
