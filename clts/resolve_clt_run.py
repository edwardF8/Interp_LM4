"""Resolve a wandb run (by its display name) to a field — by default the CLT
checkpoint dir that trainCLT.py logged as `storage_path`.

trainCLT.py writes `storage_path` (the `.../final` dir) into each run's wandb
summary, so this turns a human run name like 'devout-morning-219' into the exact
--clt-dir the writeFeatures stage needs.

Usage:
    python clts/resolve_clt_run.py <run_name>                 # prints storage_path
    python clts/resolve_clt_run.py <run_name> --field expansion
    python clts/resolve_clt_run.py <run_name> --project interpLM4 --entity <ent>

Prints the value to stdout (for shell capture); errors go to stderr with a
non-zero exit so a launcher can `|| continue`.
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_name", help="wandb run display name, e.g. devout-morning-219")
    ap.add_argument("--project", default="interpLM4")
    ap.add_argument("--entity", default=None, help="defaults to your wandb default entity")
    ap.add_argument("--field", default="storage_path",
                    help="summary key (default) or config key to print")
    args = ap.parse_args()

    import wandb
    api = wandb.Api()
    path = f"{args.entity + '/' if args.entity else ''}{args.project}"

    # Prefer a server-side filter; fall back to scanning (filter key varies by wandb ver).
    runs = list(api.runs(path, filters={"display_name": args.run_name}))
    if not runs:
        runs = [r for r in api.runs(path)
                if getattr(r, "display_name", None) == args.run_name or r.name == args.run_name]
    if not runs:
        sys.exit(f"ERROR: no run named {args.run_name!r} in {path}")
    if len(runs) > 1:
        print(f"WARNING: {len(runs)} runs named {args.run_name!r}; using the most recent",
              file=sys.stderr)
        runs.sort(key=lambda r: r.created_at, reverse=True)
    run = runs[0]

    val = run.summary.get(args.field)
    if val is None:
        val = run.config.get(args.field)
    if val is None:
        sys.exit(f"ERROR: run {args.run_name!r} ({run.id}) has no {args.field!r} "
                 f"in summary or config")
    print(val)


if __name__ == "__main__":
    main()
