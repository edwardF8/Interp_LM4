"""Evaluate every SAE checkpoint under a directory.

Recursively finds all `final/` subdirectories under the given root, loads each
SAE, and runs the held-out eval defined in evalSAE.sae_eval. Useful for
spot-checking the SAEs produced by a sweep without going through wandb.

Usage:
    python saes/evalSweep.py <directory> \\
        --model-dir <path> --data-dir <path> [--model-name <name>]

Examples:
    # All trials of one sweep
    python saes/evalSweep.py saes/sae_runs/grid-L8-H6/sweep-u7h4c0x6 \\
        --model-dir /jet/.../grid-L8-H6/final --data-dir /jet/.../bioS_N-Bd_final_grid

    # Everything under sae_runs/ for a model
    python saes/evalSweep.py saes/sae_runs/grid-L8-H6 \\
        --model-dir /jet/.../grid-L8-H6/final --data-dir /jet/.../bioS_N-Bd_final_grid

The hook is read from each SAE's saved cfg (no need to pass it explicitly).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import saes.trainSAE as ts
from saes.evalSAE import load_sae, print_report, sae_eval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("sweep_dir", type=Path,
                   help="Directory to scan recursively for final/ subdirs.")
    p.add_argument("--model-dir", type=Path, required=True,
                   help="HF Llama checkpoint dir (must match the SAEs' base model).")
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Dataset dir containing people.json + old_to_new.json.")
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--context-size", type=int, default=512,
                   help="Eval context size (should match training).")
    return p.parse_args()


def find_saes(root: Path) -> list[Path]:
    """Return every `final/` directory under root, sorted."""
    if root.name == "final" and root.is_dir():
        return [root]
    return sorted(p for p in root.rglob("final") if p.is_dir())


def main() -> None:
    args = parse_args()

    # Reuse trainSAE.setup() to load model + eval tokens once.
    setup_ns = argparse.Namespace(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        model_name=args.model_name,
        context_size=args.context_size,
        # The rest of trainSAE's setup namespace is unused for eval-only:
        hook="blocks.0.hook_mlp_out",
        layers=None,
        hook_template="blocks.{layer}.hook_mlp_out",
        sweep=False,
        n_examples=10_000,
        epochs=30,
        sae_mult=8,
        l0_coefficient=5.0,
        lr=5e-5,
    )
    ts.setup(setup_ns)

    saes = find_saes(args.sweep_dir)
    if not saes:
        print(f"No SAE checkpoints found under {args.sweep_dir}")
        return

    print(f"\nFound {len(saes)} SAE checkpoint(s) under {args.sweep_dir}\n")

    for sae_path in saes:
        try:
            label = sae_path.parent.relative_to(args.sweep_dir)
        except ValueError:
            label = sae_path.parent.name
        print("=" * 64)
        print(f"  {label}")
        print(f"  ({sae_path})")
        print("=" * 64)
        sae = load_sae(sae_path, ts.device)
        # SAE checkpoints carry the hook they were trained at.
        hook = getattr(sae.cfg, "hook_name", "blocks.1.hook_mlp_out")
        print_report(sae_eval(ts.model, sae, ts.eval_tokens, hook))


if __name__ == "__main__":
    main()
