#!/usr/bin/env bash
# One-command wrapper for the full reproduction pipeline.
# Equivalent to `make all`. See README.md and DATA.md for prerequisites.
set -euo pipefail
cd "$(dirname "$0")"
make all
