"""Post-sweep inference: feature_stats + buckets + per-feature HTML dashboards
for every trained SAE under a sweep directory.

Layout produced (identical on PSC and Mac after rsync):
    saes/sae_inference/<model-name>/
        ├── index_corpus.pt                       # shared across SAEs of this model
        ├── <trial-name>/                         # name matches saes/sae_runs/.../<trial-name>/
        │   ├── feature_stats.pt
        │   ├── buckets.pt
        │   ├── dashboard_feature_*.html
        │   └── index.html
        └── ...

The hook for each SAE is read from the SAE's saved cfg, so different-layer
trials in the same sweep dir Just Work.

Usage:
    python saes/runInference.py \\
        --sweep-dir saes/sae_runs/grid-L8-H6/sweep-<id> \\
        --model-dir /jet/.../grid-L8-H6/final \\
        --data-dir /jet/.../bioS_N-Bd_final_grid

Submit on PSC:
    sbatch submit_job_psc.sh saes/runInference.py --sweep-dir ... --model-dir ... --data-dir ...

Then sync to your laptop:
    rsync -av friedmae@bridges2:Interp_LM4/saes/{sae_runs,sae_inference}/ \\
        ~/Code/Project\\ Code/CRL-Interp/Interp_LM4/saes/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

import saes.trainSAE as ts
from saes.trainSAE import STORAGE_ROOT
from saes.evalSAE import load_sae
from saes.sae_explorer import (
    build_index_corpus,
    by_input_token,
    compute_bucketed_stats,
    feature_activation_stats,
    make_dashboard,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sweep-dir", type=Path, required=True,
                   help="Directory containing trained SAEs (each as a <trial>/final/ subdir).")
    p.add_argument("--model-dir", type=Path, required=True,
                   help="HF Llama checkpoint dir. Must match the model the SAEs were trained on.")
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Dataset dir containing people.json + old_to_new.json.")
    p.add_argument("--model-name", type=str, default=None,
                   help="Identifier for this base model. Defaults to parent dir of --model-dir.")

    # Index corpus shape — 2 templates * 50k people = 100k rows; 64 tokens
    # covers every bio with margin. Match the dashboard's defaults.
    p.add_argument("--n-per-person", type=int, default=2,
                   help="Templates per person in the index corpus.")
    p.add_argument("--context-size", type=int, default=64,
                   help="Token budget per bio in the index corpus.")
    p.add_argument("--seed", type=int, default=0)

    # Toggle the heavy steps.
    p.add_argument("--no-dashboards", action="store_true",
                   help="Skip the per-feature HTML dashboards (10-30 min/SAE on A100).")
    p.add_argument("--no-stats", action="store_true",
                   help="Skip feature_stats.pt.")
    p.add_argument("--no-buckets", action="store_true",
                   help="Skip buckets.pt.")

    # sae_dashboard internals — tweak if you OOM.
    p.add_argument("--minibatch-tokens", type=int, default=128)
    p.add_argument("--minibatch-features", type=int, default=256)

    return p.parse_args()


def find_saes(root: Path) -> list[Path]:
    """Every `final/` directory under root, sorted."""
    if root.name == "final" and root.is_dir():
        return [root]
    return sorted(p for p in root.rglob("final") if p.is_dir())


def main() -> None:
    args = parse_args()

    # Reuse trainSAE.setup() to load the model + tokenizer + sampler once.
    # We don't need eval_tokens here, but setup() builds them anyway — cheap.
    setup_ns = argparse.Namespace(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        model_name=args.model_name,
        context_size=512,  # for the held-out eval set, unused here
        # The rest is unused for inference but setup() expects the namespace
        # to have these attrs (it never reads them).
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
    model_name = ts.ARGS.model_name  # filled in by setup() if user omitted it

    saes = find_saes(args.sweep_dir)
    if not saes:
        print(f"No SAE checkpoints found under {args.sweep_dir}")
        return
    print(f"\nFound {len(saes)} SAE checkpoint(s) under {args.sweep_dir}\n")

    # Shared corpus — same tokens drive every per-SAE pass, so feature numbers
    # line up between dashboards and bucket queries across all trials.
    # Writes go directly to STORAGE_ROOT (Ocean) — inference outputs are
    # mostly large HTML dashboards we don't want to stage and copy.
    inference_root = STORAGE_ROOT / "sae_inference" / model_name
    inference_root.mkdir(parents=True, exist_ok=True)
    print(f"[corpus]  n_per_person={args.n_per_person}, context_size={args.context_size}")
    tokens = build_index_corpus(
        ts.sampler, ts.tokenizer,
        n_per_person=args.n_per_person,
        context_size=args.context_size,
        seed=args.seed,
        cache_path=inference_root / "index_corpus.pt",
    )
    tokens_on_device = tokens.to(ts.device)
    print(f"          tokens shape: {tuple(tokens.shape)}\n")

    for i, sae_path in enumerate(saes, 1):
        trial_name = sae_path.parent.name
        out_dir = inference_root / trial_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 64)
        print(f"  [{i}/{len(saes)}] {trial_name}")
        print(f"  ({sae_path})")
        print("=" * 64)

        sae = load_sae(sae_path, ts.device)
        hook_name = getattr(sae.cfg, "hook_name", None)
        if hook_name is None:
            print(f"  ⚠ SAE has no cfg.hook_name; skipping.")
            continue
        print(f"  d_sae={sae.cfg.d_sae}, hook={hook_name}")

        # 1) feature_activation_stats — small file (~50 KB), powers the
        #    notebook's overview histogram + activations-per-latent bar plot.
        if not args.no_stats:
            stats_path = out_dir / "feature_stats.pt"
            if stats_path.exists():
                print(f"  [stats]   {stats_path}  (exists, skipping)")
            else:
                print(f"  [stats]   computing over {tokens.shape[0]} bios")
                stats = feature_activation_stats(
                    ts.model, sae, tokens_on_device, hook_name,
                )
                torch.save(stats, stats_path)
                n_dead = int((stats["activation_count"] == 0).sum())
                print(f"            dead features: {n_dead} / {sae.cfg.d_sae}  ->  {stats_path}")

        # 2) bucket stats — by_input_token only by default. Add more bucketers
        #    here (by_next_token, by_prev_token, by_position) to capture more
        #    axes in the same pass.
        if not args.no_buckets:
            buckets_path = out_dir / "buckets.pt"
            if buckets_path.exists():
                print(f"  [buckets] {buckets_path}  (exists, skipping)")
            else:
                print(f"  [buckets] sweeping by_input_token over {tokens.shape[0]} bios")
                bucket_stats = compute_bucketed_stats(
                    ts.model, sae,
                    tokens=tokens_on_device,
                    hook_name=hook_name,
                    bucketers=[by_input_token(ts.tokenizer.vocab_size)],
                    ignore_token_ids={ts.tokenizer.pad_token_id},
                )
                torch.save(bucket_stats, buckets_path)
                ib = bucket_stats["buckets"]["input_token"]
                n_active = int((ib["count"] > 0).sum())
                print(f"            {n_active}/{ib['count'].numel()} non-empty token buckets  ->  {buckets_path}")

        # 3) per-feature HTML dashboards — the expensive step. Skip with
        #    --no-dashboards while iterating on bucket / stats logic.
        if not args.no_dashboards:
            index_html = out_dir / "index.html"
            if index_html.exists():
                print(f"  [dash]    {index_html}  (exists, skipping)")
            else:
                print(f"  [dash]    rendering {sae.cfg.d_sae} features -> {out_dir}")
                out_html = make_dashboard(
                    ts.model, sae, tokens_on_device, ts.tokenizer,
                    out_dir=out_dir,
                    hook_name=hook_name,
                    minibatch_size_tokens=args.minibatch_tokens,
                    minibatch_size_features=args.minibatch_features,
                )
                print(f"            {out_html}")

        print()

    print("=" * 64)
    print(f"DONE. Outputs under: {inference_root}")
    print()
    print("To view on your laptop:")
    print(f"  rsync -av friedmae@bridges2:Interp_LM4/saes/{{sae_runs,sae_inference}}/ \\")
    print(f"      <local-repo>/saes/")


if __name__ == "__main__":
    main()
