#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="$REPO_ROOT/data/raw/spotify_tracks"
DEST_FILE="$DEST_DIR/dataset.csv"
KAGGLE_SLUG="maharshipandya/spotify-tracks-dataset"
KAGGLE_CONFIG_FILE="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json"
MIRROR_URL="https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv?download=true"

mkdir -p "$DEST_DIR"

if [[ -f "$DEST_FILE" ]]; then
  echo "Dataset already present at $DEST_FILE"
  exit 0
fi

if command -v kaggle >/dev/null 2>&1 && [[ -f "$KAGGLE_CONFIG_FILE" ]]; then
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "$tmp_dir"' EXIT

  echo "Downloading from Kaggle: $KAGGLE_SLUG"
  kaggle datasets download --dataset "$KAGGLE_SLUG" --path "$tmp_dir" --unzip

  if [[ ! -f "$tmp_dir/dataset.csv" ]]; then
    echo "Expected dataset.csv was not found in the Kaggle download." >&2
    exit 1
  fi

  mv "$tmp_dir/dataset.csv" "$DEST_FILE"
  echo "Saved $DEST_FILE"
  exit 0
fi

echo "Kaggle CLI credentials were not found; downloading the same dataset.csv from the public mirror."
curl -L --fail --output "$DEST_FILE" "$MIRROR_URL"
echo "Saved $DEST_FILE"
