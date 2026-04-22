#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${WAN2GP_REPO_URL:-https://github.com/deepbeepmeep/Wan2GP.git}"
INSTALL_DIR="${WAN2GP_INSTALL_DIR:-$HOME/Wan2GP}"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="$HOME/.local/bin:$PATH"

rm -rf "$INSTALL_DIR"
git clone "$REPO_URL" "$INSTALL_DIR"
cd "$INSTALL_DIR"

python3 - <<'PY'
import json
from pathlib import Path

path = Path("setup_config.json")
cfg = json.loads(path.read_text())

cfg["gpu_profiles"]["RTX_50"]["flash"] = None

sage_linux_cmd = (
    'pip install "setuptools<=75.8.2" && '
    'if [ -d SageAttention/.git ]; then git -C SageAttention pull; '
    'else git clone https://github.com/thu-ml/SageAttention; fi && '
    'pip install --no-build-isolation -e SageAttention'
)

for key in ("v211", "v220", "v220_cu13"):
    cfg["components"]["sage"][key]["cmd"]["linux"] = sage_linux_cmd

path.write_text(json.dumps(cfg, indent=4) + "\n")
PY

printf '%s\n' '--listen' > scripts/args.txt

python3 setup.py install --env uv --auto
