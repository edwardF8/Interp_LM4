#!/usr/bin/env bash
# Generate CLT attribution feature dashboards on PSC, with a PREFLIGHT that runs
# BEFORE the (GPU) job is submitted — so a bad path / missing dep / unloadable
# CLT fails in seconds instead of wasting a GPU allocation.
#
# Runs `clts/gen_feature_dashboards.py`, which needs only your TRAINING env
# (torch + transformer-lens + transformers + safetensors + numpy + pyyaml).
# It does NOT import circuit-tracer, so use the same Python you run trainCLT.py
# with — point $PYTHON at it (e.g. PYTHON=/path/to/venv/bin/python).
#
# Usage (run from anywhere; it cd's to the repo root):
#   scripts/gen_dashboards_psc.sh preflight     # checks only — submits nothing
#   scripts/gen_dashboards_psc.sh validate      # preflight + small run (--n-people 100)
#   scripts/gen_dashboards_psc.sh moderate      # preflight + --n-people 1000 (recommended)
#   scripts/gen_dashboards_psc.sh full          # preflight + ALL people (heaviest)
#   scripts/gen_dashboards_psc.sh moderate --top-k 30   # extra args pass through to the generator
#
# Override any path/setting via env vars, e.g.:
#   CLT_DIR=/path/to/clt/final PYTHON=~/envs/train/bin/python \
#       scripts/gen_dashboards_psc.sh validate
#
# SLURM: if you submit via sbatch, put `scripts/gen_dashboards_psc.sh full` as the
# job body (after your module load / env activation). Preflight is safe to run on
# the login node first — it WARNS (not fails) if no GPU is visible there.
set -euo pipefail

# ---- config (override via env vars) ----------------------------------------
REMOTE_BASE="${REMOTE_BASE:-/jet/home/friedmae/data_storage/LM4_Results}"
MODEL_DIR="${MODEL_DIR:-$REMOTE_BASE/runResults/bioS_N-Bd_final_grid/20260520-134455/grid/grid-L4-H6/final}"
CLT_DIR="${CLT_DIR:-$REMOTE_BASE/clt_runs/grid-L4-H6/mult16_l02_lr0.0001_ep50_n10000/final}"
DATA_DIR="${DATA_DIR:-$REMOTE_BASE/Data/bioS_N-Bd_final_grid}"
SCAN_NAME="${SCAN_NAME:-grid-L4-H6}"
FEATURES_ROOT="${FEATURES_ROOT:-$REMOTE_BASE/clt_features}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
N_PER_PERSON="${N_PER_PERSON:-2}"
CONTEXT_SIZE="${CONTEXT_SIZE:-64}"

# ---- locate repo root ------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
GEN="clts/gen_feature_dashboards.py"

red()  { printf '\033[31m%s\033[0m\n' "$*" >&2; }
grn()  { printf '\033[32m%s\033[0m\n' "$*"; }

# ---- preflight: paths + env + CLT load -------------------------------------
preflight() {
    echo "=== Preflight (repo: $REPO_ROOT) ==="
    local ok=1

    [[ -f "$GEN" ]] || { red "MISSING $GEN — run from a repo that has the attribution code (did you pull?)"; ok=0; }

    # model dir: config.json + a weights file
    if [[ -d "$MODEL_DIR" ]]; then
        [[ -f "$MODEL_DIR/config.json" ]] || { red "MODEL_DIR has no config.json: $MODEL_DIR"; ok=0; }
        if ! ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1 && [[ ! -f "$MODEL_DIR/pytorch_model.bin" ]]; then
            red "MODEL_DIR has no weights (*.safetensors / pytorch_model.bin): $MODEL_DIR"; ok=0
        fi
        echo "  model:    $MODEL_DIR"
    else
        red "MISSING MODEL_DIR: $MODEL_DIR"; ok=0
    fi

    # clt dir: config.yaml + at least layer-0 enc/dec
    if [[ -d "$CLT_DIR" ]]; then
        for f in config.yaml W_enc_0.safetensors W_dec_0.safetensors; do
            [[ -f "$CLT_DIR/$f" ]] || { red "CLT_DIR missing $f: $CLT_DIR"; ok=0; }
        done
        echo "  clt:      $CLT_DIR"
    else
        red "MISSING CLT_DIR: $CLT_DIR"; ok=0
    fi

    # data dir: the two files the generator reads
    if [[ -d "$DATA_DIR" ]]; then
        for f in old_to_new.json people.json; do
            [[ -f "$DATA_DIR/$f" ]] || { red "DATA_DIR missing $f: $DATA_DIR"; ok=0; }
        done
        echo "  data:     $DATA_DIR"
    else
        red "MISSING DATA_DIR: $DATA_DIR"; ok=0
    fi

    # features-root writable
    if mkdir -p "$FEATURES_ROOT/$SCAN_NAME" 2>/dev/null && touch "$FEATURES_ROOT/$SCAN_NAME/.write_probe" 2>/dev/null; then
        rm -f "$FEATURES_ROOT/$SCAN_NAME/.write_probe"
        echo "  features: $FEATURES_ROOT/$SCAN_NAME  (writable)"
    else
        red "FEATURES_ROOT not writable: $FEATURES_ROOT/$SCAN_NAME"; ok=0
    fi

    # python: deps + cuda + project import + CLT actually loads + model config sane
    if ! "$PYTHON" - "$MODEL_DIR" "$CLT_DIR" "$DEVICE" <<'PY'
import sys
model_dir, clt_dir, device = sys.argv[1], sys.argv[2], sys.argv[3]
missing = []
for m in ("torch", "transformer_lens", "transformers", "safetensors", "numpy", "yaml"):
    try:
        __import__(m)
    except Exception as e:  # noqa: BLE001
        missing.append(f"{m} ({e})")
if missing:
    print("  PYTHON DEPS MISSING:", *missing, sep="\n    ")
    sys.exit(1)
import torch
print(f"  python:   {sys.version.split()[0]}   torch {torch.__version__}")
if device == "cuda":
    if torch.cuda.is_available():
        print(f"  cuda:     available ({torch.cuda.get_device_name(0)})")
    else:
        print("  cuda:     NOT visible here -- ok if you preflight on a login node "
              "and run the job on a GPU node; otherwise pass DEVICE=cpu")
sys.path.insert(0, ".")
try:
    from clts.gen_feature_dashboards import generate_dashboards  # noqa: F401
    from clts.clt import CrossLayerTranscoder
except Exception as e:  # noqa: BLE001
    print(f"  PROJECT IMPORT FAILED: {e}")
    sys.exit(1)
try:
    clt = CrossLayerTranscoder.load_from_dir(clt_dir)
    print(f"  clt dims: n_layers={clt.n_layers} d_model={clt.d_model} d_transcoder={clt.d_transcoder}")
except Exception as e:  # noqa: BLE001
    print(f"  CLT LOAD FAILED ({clt_dir}): {e}")
    sys.exit(1)
try:
    import json
    cfg = json.load(open(f"{model_dir}/config.json"))
    print(f"  model cfg: n_layers={cfg.get('num_hidden_layers')} d_model={cfg.get('hidden_size')} "
          f"vocab={cfg.get('vocab_size')} rms_norm_eps={cfg.get('rms_norm_eps')}")
    if cfg.get("num_hidden_layers") != clt.n_layers:
        print(f"  WARNING: model layers ({cfg.get('num_hidden_layers')}) != CLT layers ({clt.n_layers}) "
              "-- model_dir and clt_dir may be mismatched")
except Exception as e:  # noqa: BLE001
    print(f"  model config read failed: {e}")
    sys.exit(1)
PY
    then
        red "Python/CLT preflight FAILED"; ok=0
    fi

    if [[ "$ok" == "1" ]]; then
        grn "Preflight PASSED"
        return 0
    fi
    red "Preflight FAILED -- fix the above before submitting (override paths via env vars)."
    return 1
}

# ---- run -------------------------------------------------------------------
run() {
    local mode="$1"; shift
    local people_args=()
    case "$mode" in
        validate) people_args=(--n-people 100) ;;
        moderate) people_args=(--n-people 1000) ;;
        full)     people_args=() ;;   # all people
    esac
    echo "=== Generating dashboards: mode=$mode scan=$SCAN_NAME device=$DEVICE ==="
    set -x
    "$PYTHON" "$GEN" \
        --model-dir "$MODEL_DIR" \
        --clt-dir   "$CLT_DIR" \
        --data-dir  "$DATA_DIR" \
        --scan-name "$SCAN_NAME" \
        --features-root "$FEATURES_ROOT" \
        --device "$DEVICE" \
        --n-per-person "$N_PER_PERSON" \
        --context-size "$CONTEXT_SIZE" \
        "${people_args[@]}" "$@"
    set +x
    local n
    n="$( { ls "$FEATURES_ROOT/$SCAN_NAME"/*.json 2>/dev/null || true; } | wc -l | tr -d ' ')"
    grn "Done. Wrote $n dashboard files to $FEATURES_ROOT/$SCAN_NAME"
    echo "Next (on the Mac): ./scripts/sync_from_psc.sh   then build + serve (see clts/README_attribution.md)"
}

MODE="${1:-preflight}"
shift || true
case "$MODE" in
    preflight)              preflight ;;
    validate|moderate|full) preflight && run "$MODE" "$@" ;;
    *) echo "Usage: $0 {preflight|validate|moderate|full} [extra gen_feature_dashboards.py args]" >&2; exit 1 ;;
esac
