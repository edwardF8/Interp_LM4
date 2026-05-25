"""Evaluate every SAE checkpoint under a directory.

Recursively finds all `final/` subdirectories under the given root, loads each
SAE, and runs the held-out eval defined in evalSAE.sae_eval. Useful for
spot-checking the SAEs produced by a sweep without going through wandb.

Imports the heavy model/tokenizer/eval-token setup from trainSAE.py so the
model only loads once.

Usage:
    python evalSweep.py <directory>

    # Examples
    python evalSweep.py sae/sweep-u7h4c0x6      # all trials of one sweep
    python evalSweep.py sae/                    # everything under sae/
    python evalSweep.py sae/mult8_l05_lr3e-05_ep50_n10000   # single SAE
"""
from __future__ import annotations

import sys
from pathlib import Path

from saes.trainSAE import model, eval_tokens, device
from saes.evalSAE import load_sae, sae_eval, print_report


HOOK = "blocks.1.hook_mlp_out"  # must match the hook used during training


def find_saes(root: Path) -> list[Path]:
    """Return every `final/` directory under root, sorted.

    sae_lens writes each trained SAE to `<trial_dir>/final/`, so finding all
    `final/` directories enumerates every saved SAE.
    """
    if root.name == "final" and root.is_dir():
        return [root]
    return sorted(p for p in root.rglob("final") if p.is_dir())


def eval_directory(root: Path) -> None:
    saes = find_saes(root)
    if not saes:
        print(f"No SAE checkpoints found under {root}")
        return

    print(f"Found {len(saes)} SAE checkpoint(s) under {root}\n")

    for sae_path in saes:
        label = sae_path.parent.relative_to(root) if sae_path != root else sae_path.parent.name
        print("=" * 64)
        print(f"  {label}")
        print(f"  ({sae_path})")
        print("=" * 64)
        sae = load_sae(sae_path, device)
        metrics = sae_eval(model, sae, eval_tokens, HOOK)
        print_report(metrics)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evalSweep.py <directory>")
        print("       e.g. python evalSweep.py sae/sweep-u7h4c0x6")
        sys.exit(1)
    eval_directory(Path(sys.argv[1]))
