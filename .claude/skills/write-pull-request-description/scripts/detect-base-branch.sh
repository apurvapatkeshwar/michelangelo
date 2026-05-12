#!/usr/bin/env bash
# Prefers the tracked upstream branch; falls back to "main".
current=$(git rev-parse --abbrev-ref HEAD)
upstream=$(git rev-parse --abbrev-ref @{u} 2>/dev/null | sed 's|^origin/||')
if [ -n "$upstream" ] && [ "$upstream" != "$current" ]; then
  echo "$upstream"
else
  echo "main"
fi
