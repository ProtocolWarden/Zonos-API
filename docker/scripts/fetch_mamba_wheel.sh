#!/usr/bin/env bash
set -euo pipefail

URL="${1:?wheel URL required}"
OUT="${2:?output path required}"
SHA_EXPECTED="${3:-}"

mkdir -p "$(dirname "${OUT}")"

curl \
  --retry 7 \
  --retry-all-errors \
  --location \
  --fail \
  --output "${OUT}.part" \
  "${URL}"

mv "${OUT}.part" "${OUT}"

if [[ -n "${SHA_EXPECTED}" ]]; then
  echo "${SHA_EXPECTED}  ${OUT}" | sha256sum --check --status
fi

echo "Fetched: ${OUT}"
