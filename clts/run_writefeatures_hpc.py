"""HPC worker: run the writeFeatures feature-in-node rank analysis over EVERY person.

Storage-lean by design. For each month it takes all in-vocab people born that month,
sweeps the bio templates, and records where each of that month's target feature(s)
ranks among the direct input features of that month's output-token node.

Key properties (so an HPC quota survives a ~2.3M-graph run):
  * NO attribution cache is written to disk (each graph is built in memory, the
    metrics are extracted, and the graph is discarded). The interactive notebook
    cache would be ~270 GB / ~2.1M files at this scale — pointless for a one-shot run.
  * Both of a month's features are evaluated from the SAME in-memory graph (one
    `attribute()` call per person x template, not one per feature).
  * Each shard writes a small AGGREGATE per (month, feature) (bucket counts,
    multi-position counts, co-influencer counters) — not raw per-person records.
    `merge_writefeatures_hpc.py` sums the aggregates into final reports.

Designed for a SLURM array: each task processes `work[shard_index::num_shards]`.

    python clts/run_writefeatures_hpc.py --dry-run                  # size the job
    python clts/run_writefeatures_hpc.py --num-shards 200 --shard-index 0
    python clts/run_writefeatures_hpc.py --months August --limit-people 2   # smoke
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import clts.writefeatures as wf  # noqa: E402

# Local (Mac) defaults; on PSC pass --model-dir/--clt-dir/--data-dir/--scan-name
# (the writefeatures_psc.sbatch script supplies the $REMOTE_BASE paths).
DEFAULT_MODEL_DIR = REPO / "model/grid-L4-H6"
DEFAULT_CLT_DIR   = REPO / "clts/clt_runs/grid-L4-H6/mult16_l02_lr0.0001_ep50_n10000/final"
DEFAULT_DATA_DIR  = REPO / "data/bioS_N-Bd_final_grid"
DEFAULT_SCAN_NAME = "grid-L4-H6"

MONTH_STRINGS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# Per-month target features (all layer 3). 1 or 2 features per month; both evaluated.
MONTH_FEATURES = {
    "January":   [(3, 5934), (3, 1975)],
    "February":  [(3, 4282), (3, 4721)],
    "March":     [(3, 1438), (3, 4565)],
    "April":     [(3, 0)],
    "May":       [(3, 5686), (3, 1728)],
    "June":      [(3, 1682)],
    "July":      [(3, 4782)],
    "August":    [(3, 4768)],
    "September": [(3, 240)],
    "October":   [(3, 4380)],
    "November":  [(3, 3384)],
    "December":  [(3, 5328)],
}

# Per-shard co-influencer cap: keep the top-N labels per (month, feature) so partials
# stay small. A globally-common label appears in (nearly) every shard's top-N, so its
# merged count is exact; only labels rare in a shard are dropped (never globally top-K).
COINF_CAP = 500


def parse_templates(s):
    if s == "all":
        return "all"
    return [int(x) for x in s.split(",") if x.strip() != ""]


def new_agg():
    return {
        "n_records": 0, "n_skipped": 0, "n_people": 0,
        "bucket": Counter(),
        "n_ge2_pos": 0, "span": Counter(), "flagged": [],
        "meaningful": 0, "n_meaningful": Counter(),
        "coinf_count": Counter(), "coinf_sumedge": defaultdict(float), "coinf_kind": {},
    }


def update_agg(agg, *, bucket, n_positions, span, is_meaningful, n_meaningful,
               unified, pos_span_flag, person, t_key):
    agg["n_records"] += 1
    agg["bucket"][bucket] += 1
    agg["span"][span] += 1
    if n_positions >= 2:
        agg["n_ge2_pos"] += 1
        if span >= pos_span_flag and len(agg["flagged"]) < 500:
            agg["flagged"].append({"id": person["id"],
                                   "name": f"{person['first_name']} {person['last_name']}",
                                   "t_key": t_key, "span": span})
    if is_meaningful:
        agg["meaningful"] += 1
    agg["n_meaningful"][n_meaningful] += 1
    for row in unified:
        lbl = row["label"]
        agg["coinf_count"][lbl] += 1
        agg["coinf_sumedge"][lbl] += row["edge"]
        agg["coinf_kind"][lbl] = row["kind"]


def finalize_agg(agg):
    """Counters -> plain dicts; cap the co-influencer table to the top COINF_CAP labels."""
    top = [lbl for lbl, _ in agg["coinf_count"].most_common(COINF_CAP)]
    return {
        "n_records": agg["n_records"], "n_skipped": agg["n_skipped"],
        "n_people": agg["n_people"],
        "bucket": dict(agg["bucket"]),
        "n_ge2_pos": agg["n_ge2_pos"], "span": dict(agg["span"]),
        "flagged": agg["flagged"],
        "meaningful": agg["meaningful"], "n_meaningful": dict(agg["n_meaningful"]),
        "coinf": {lbl: {"count": agg["coinf_count"][lbl],
                        "sum_edge": round(agg["coinf_sumedge"][lbl], 6),
                        "kind": agg["coinf_kind"][lbl]} for lbl in top},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--clt-dir", default=str(DEFAULT_CLT_DIR))
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--scan-name", default=DEFAULT_SCAN_NAME)
    ap.add_argument("--months", default="all", help="comma list, or 'all'")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--templates", default="all", help="'all' or comma list of indices")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="(ignored) retained for compat; the fast logit-only path runs "
                         "at batch width 1, so attribution batch size no longer applies")
    ap.add_argument("--top-k", type=int, default=10, help="unified co-influencer depth per graph")
    ap.add_argument("--multi-tok-top-k", type=int, default=5)
    ap.add_argument("--pos-span-flag", type=int, default=3)
    ap.add_argument("--rank-by-abs", action="store_true")
    ap.add_argument("--no-token-nodes", action="store_true")
    ap.add_argument("--limit-people", type=int, default=None,
                    help="cap people/month: a SEEDED RANDOM sample (not first-N)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for --limit-people sampling")
    ap.add_argument("--progress-every", type=int, default=200)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    MODEL_DIR, CLT_DIR = Path(args.model_dir), Path(args.clt_dir)
    DATA_DIR, SCAN_NAME = Path(args.data_dir), args.scan_name

    from clts.storage import storage_root
    out_dir = Path(args.out_dir) if args.out_dir else \
        storage_root() / "clt_feature_explorer" / SCAN_NAME / "hpc"
    partials_dir = out_dir / "partials"

    months = MONTH_STRINGS if args.months == "all" else \
        [m.strip() for m in args.months.split(",")]
    for m in months:
        if m not in MONTH_FEATURES:
            raise SystemExit(f"unknown month {m!r}; valid: {MONTH_STRINGS}")
    templates = parse_templates(args.templates)
    n_templates = 46 if templates == "all" else len(templates)

    from clts.export_tokenizer import ensure_hf_tokenizer
    from util.bio_sampler import BioSampler
    from util.condensed_tokenizer import CondensedTokenizer

    ct = CondensedTokenizer.from_remap_path(DATA_DIR / "old_to_new.json")
    sampler = BioSampler(DATA_DIR / "people.json", fields=("birthday",))

    # Per-month pool (all people, or a seeded random subset if --limit-people),
    # then the global work list, then this shard's stride slice. sample_people uses
    # the same seed on every shard, so all shards agree on the subset before striding;
    # it returns the full pool unchanged when limit is None or >= the pool size.
    work, pools = [], {}
    for m in months:
        pool = wf.sample_people(wf.people_in_month(sampler, ct, m),
                                args.limit_people, args.seed)
        pools[m] = pool
        work += [(m, ds_idx, person) for ds_idx, person in pool]
    shard = work[args.shard_index::args.num_shards]
    shard_people = defaultdict(list)
    for m, ds_idx, person in shard:
        shard_people[m].append((ds_idx, person))

    total_people = sum(len(pools[m]) for m in months)
    print(f"[plan] months={months}  templates={n_templates}")
    print(f"[plan] people(all)={total_people:,}  graphs(all)≈{total_people*n_templates:,}")
    print(f"[plan] shard {args.shard_index}/{args.num_shards}: {len(shard):,} people "
          f"≈{len(shard)*n_templates:,} graphs")
    for m in months:
        print(f"[plan]   {m:>10}: pool={len(pools[m]):>5} shard={len(shard_people[m]):>5} "
              f"features={MONTH_FEATURES[m]}")
    print(f"[plan] out-dir={out_dir}  (NO attribution cache written)")
    if args.dry_run:
        print("[plan] --dry-run: exiting before model load / compute.")
        return

    partials_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"shard{args.shard_index:04d}of{args.num_shards:04d}"

    # Skip months already done by this shard (idempotent re-runs).
    todo_months = []
    for m in months:
        if not shard_people[m]:
            continue
        done = all((partials_dir / f"{m}_L{l}F{f}_{suffix}.pt").exists()
                   for (l, f) in MONTH_FEATURES[m])
        if args.skip_existing and done:
            print(f"[skip] {m}: all feature partials exist for this shard.")
            continue
        todo_months.append(m)
    if not todo_months:
        print("[shard] nothing to do (all partials exist).")
        return

    from clts.load_replacement_model import load_replacement_model
    t0 = time.time()
    model = load_replacement_model(MODEL_DIR, CLT_DIR, ensure_hf_tokenizer(DATA_DIR),
                                   SCAN_NAME, device=args.device)
    print(f"[load] model ready in {time.time()-t0:.1f}s device={args.device}", flush=True)

    all_t = wf.birthday_templates()
    tmpl_list = wf.resolve_templates(templates, all_t)
    twl = frozenset({"born", "birth", "day", "date"})
    include_tokens = not args.no_token_nodes

    for m in todo_months:
        feats = MONTH_FEATURES[m]
        aggs = {f: new_agg() for f in feats}
        for a in aggs.values():
            a["n_people"] = len(shard_people[m])
        t1 = time.time()
        for n_done, (ds_idx, person) in enumerate(shard_people[m], 1):
            target_token = wf.resolve_target("month", person)
            single = (target_token is not None and
                      len(model.tokenizer.encode(target_token, add_special_tokens=False)) == 1)
            for t_key, t_val in tmpl_list:
                if not single:
                    for f in feats:
                        aggs[f]["n_skipped"] += 1
                    continue
                try:
                    prompt = wf.template_prompt(person, t_val, sampler)
                    # Fast path: only the target logit node's incoming edges (features +
                    # MLP-error + token nodes), skipping the discarded feature->feature
                    # attribution. ~50-75x faster/graph than building the full graph.
                    node_all = wf.attribute_node_inputs(model, prompt, target_token)
                except Exception as exc:
                    for f in feats:
                        aggs[f]["n_skipped"] += 1
                    if n_done <= 2:
                        print(f"[warn] {m} {person['id']} t{t_key}: "
                              f"{type(exc).__name__}: {exc}", flush=True)
                    continue
                if node_all is None:        # target token not a logit node -> absent
                    for f in feats:
                        update_agg(aggs[f], bucket="absent", n_positions=0, span=0,
                                   is_meaningful=False, n_meaningful=0, unified=[],
                                   pos_span_flag=args.pos_span_flag, person=person, t_key=t_key)
                    continue
                feat_block = node_all["features"]
                fagg = wf.aggregate_by_feature(feat_block)
                roles = wf.token_roles(prompt, person, model.tokenizer, template_word_labels=twl)
                unified = wf.unified_top_labels(node_all, roles, top_k=args.top_k,
                                                rank_by_abs=args.rank_by_abs,
                                                include_tokens=include_tokens)
                for f in feats:
                    fr = wf.feature_rank(fagg, f, rank_by_abs=args.rank_by_abs)
                    mt = wf.feature_multitoken(feat_block, f,
                                               multi_tok_top_k=args.multi_tok_top_k,
                                               rank_by_abs=args.rank_by_abs)
                    update_agg(aggs[f], bucket=wf.rank_bucket(fr["rank"]),
                               n_positions=mt["n_positions"], span=fr["span"],
                               is_meaningful=mt["is_meaningful"], n_meaningful=mt["n_meaningful"],
                               unified=unified, pos_span_flag=args.pos_span_flag,
                               person=person, t_key=t_key)
            if args.progress_every and n_done % args.progress_every == 0:
                rate = n_done / (time.time() - t1)
                print(f"[{m}] {n_done}/{len(shard_people[m])} people "
                      f"({rate:.1f}/s)", flush=True)

        for (l, f) in feats:
            payload = {"month": m, "feature": [l, f], "templates": templates,
                       "rank_by_abs": args.rank_by_abs, "include_token_nodes": include_tokens,
                       "shard_index": args.shard_index, "num_shards": args.num_shards,
                       "agg": finalize_agg(aggs[(l, f)])}
            part = partials_dir / f"{m}_L{l}F{f}_{suffix}.pt"
            torch.save(payload, part)
            a = aggs[(l, f)]
            print(f"[done] {m} L{l}F{f}: {a['n_records']} records "
                  f"({a['n_skipped']} skipped) in {time.time()-t1:.1f}s -> {part.name}",
                  flush=True)

    print(f"[shard {args.shard_index}/{args.num_shards}] done in {time.time()-t0:.1f}s.",
          flush=True)


if __name__ == "__main__":
    main()
