#!/usr/bin/env bash
set -euo pipefail

upstream_url="https://github.com/OSU-MLB/ViT_PEFT_Vision.git"
upstream_commit="cea7e21b2ddecef8172f5d177da5a29d531c4279"
target_dir="${1:-code}"

if [[ -e "$target_dir" && ! -d "$target_dir/.git" ]]; then
  printf 'Target exists and is not a Git checkout: %s\n' "$target_dir" >&2
  exit 1
fi

if [[ -d "$target_dir/.git" ]]; then
  git -C "$target_dir" fetch --depth 1 origin "$upstream_commit"
  git -C "$target_dir" checkout --detach FETCH_HEAD
else
  git clone --no-checkout "$upstream_url" "$target_dir"
  git -C "$target_dir" fetch --depth 1 origin "$upstream_commit"
  git -C "$target_dir" checkout --detach FETCH_HEAD
fi

actual_commit="$(git -C "$target_dir" rev-parse HEAD)"
if [[ "$actual_commit" != "$upstream_commit" ]]; then
  printf 'Expected commit %s but checked out %s\n' "$upstream_commit" "$actual_commit" >&2
  exit 1
fi

printf 'Checked out %s at %s\n' "$upstream_url" "$actual_commit"
