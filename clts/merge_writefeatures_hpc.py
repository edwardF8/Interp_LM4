"""HPC merge: sum the sharded AGGREGATES into per-(month, feature) reports.

Run AFTER the SLURM array (`run_writefeatures_hpc.py`) finishes. For each
(month, feature) it sums all `<month>_L<l>F<f>_shard*of*.pt` aggregate partials into a
report with the same shape `writefeatures.build_report` produces (rank histogram,
loose/meaningful multi-position, unified co-influencers), then writes JSON per
(month, feature), a per-month combined JSON, and a top-level `summary.csv`.

    python clts/merge_writefeatures_hpc.py            # default out-dir
    python clts/merge_writefeatures_hpc.py --out-dir <dir> --top-k 15
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import clts.writefeatures as wf  # noqa: E402

SCAN_NAME = "grid-L4-H6"
_PART_RE = re.compile(r"^(?P<month>[A-Za-z]+)_L(?P<layer>\d+)F(?P<fidx>\d+)_shard\d+of\d+\.pt$")


def sum_aggs(aggs):
    """Sum a list of finalize_agg() dicts into one combined aggregate."""
    bucket, span, nmean = Counter(), Counter(), Counter()
    coinf_count, coinf_sum, coinf_kind = Counter(), defaultdict(float), {}
    n_records = n_skipped = n_people = n_ge2 = meaningful = 0
    flagged = []
    for a in aggs:
        n_records += a["n_records"]; n_skipped += a["n_skipped"]; n_people += a["n_people"]
        n_ge2 += a["n_ge2_pos"]; meaningful += a["meaningful"]
        bucket.update({k: v for k, v in a["bucket"].items()})
        span.update({int(k): v for k, v in a["span"].items()})
        nmean.update({int(k): v for k, v in a["n_meaningful"].items()})
        if len(flagged) < 500:
            flagged.extend(a["flagged"][: 500 - len(flagged)])
        for lbl, info in a["coinf"].items():
            coinf_count[lbl] += info["count"]
            coinf_sum[lbl] += info["sum_edge"]
            coinf_kind[lbl] = info["kind"]
    return dict(n_records=n_records, n_skipped=n_skipped, n_people=n_people,
                n_ge2=n_ge2, meaningful=meaningful, bucket=bucket, span=span,
                nmean=nmean, flagged=flagged, coinf_count=coinf_count,
                coinf_sum=coinf_sum, coinf_kind=coinf_kind)


def format_report(s, *, top_k, config):
    """Build a build_report-shaped dict from a summed aggregate `s`."""
    n = s["n_records"]
    counts = {b: s["bucket"].get(b, 0) for b in wf._RANK_BUCKETS}
    pct = {b: (counts[b] / n if n else 0.0) for b in wf._RANK_BUCKETS}
    co = [{"label": lbl, "kind": s["coinf_kind"][lbl], "count": s["coinf_count"][lbl],
           "frac": (s["coinf_count"][lbl] / n if n else 0.0),
           "mean_edge": round(s["coinf_sum"][lbl] / s["coinf_count"][lbl], 6)}
          for lbl in s["coinf_count"]]
    co.sort(key=lambda d: (-d["count"], -abs(d["mean_edge"]), d["label"]))
    return {
        "config": config,
        "n_records": n, "n_skipped": s["n_skipped"], "n_sampled": s["n_people"],
        "rank_histogram": {"counts": counts, "pct": pct},
        "loose_multipos": {"n_ge2_positions": s["n_ge2"],
                           "frac": (s["n_ge2"] / n if n else 0.0),
                           "span_distribution": dict(s["span"]), "flagged": s["flagged"]},
        "meaningful_crosstoken": {"n_meaningful": s["meaningful"],
                                  "frac": (s["meaningful"] / n if n else 0.0),
                                  "n_meaningful_distribution": dict(s["nmean"]),
                                  "vs_loose": {"loose_ge2": s["n_ge2"],
                                               "meaningful_ge2": s["meaningful"],
                                               "gap": s["n_ge2"] - s["meaningful"]}},
        "co_influencers": co[:top_k],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    from clts.storage import storage_root
    out_dir = Path(args.out_dir) if args.out_dir else \
        storage_root() / "clt_feature_explorer" / SCAN_NAME / "hpc"
    partials_dir = out_dir / "partials"
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for p in sorted(partials_dir.glob("*.pt")):
        mobj = _PART_RE.match(p.name)
        if not mobj:
            print(f"[warn] unrecognized partial, skipping: {p.name}")
            continue
        groups[(mobj["month"], int(mobj["layer"]), int(mobj["fidx"]))].append(p)
    if not groups:
        raise SystemExit(f"no partials found in {partials_dir}")

    summary_rows, per_month = [], defaultdict(dict)
    for (month, layer, fidx), parts in sorted(groups.items()):
        aggs, templates, rank_by_abs, incl_tok = [], "all", False, True
        for p in parts:
            blob = torch.load(p, weights_only=False)
            aggs.append(blob["agg"])
            templates = blob.get("templates", templates)
            rank_by_abs = blob.get("rank_by_abs", rank_by_abs)
            incl_tok = blob.get("include_token_nodes", incl_tok)
        s = sum_aggs(aggs)
        config = {"target_feature": [layer, fidx], "target": "month", "month": month,
                  "templates": templates, "rank_by_abs": rank_by_abs,
                  "include_token_nodes": incl_tok, "scan": SCAN_NAME,
                  "n_shards_merged": len(parts), "every_person": True}
        report = format_report(s, top_k=args.top_k, config=config)
        stem = f"report_L{layer}F{fidx}_{month.lower()}-all"
        (reports_dir / f"{stem}.json").write_text(json.dumps(report, indent=2))

        pct = report["rank_histogram"]["pct"]
        top_err = next((c for c in report["co_influencers"] if c["kind"] == "error"), None)
        summary_rows.append({
            "month": month, "feature": f"L{layer}F{fidx}",
            "n_records": report["n_records"], "n_skipped": report["n_skipped"],
            "n_people": report["n_sampled"], "n_shards": len(parts),
            "top1_pct": round(pct["top1"] * 100, 1),
            "top1_3_pct": round((pct["top1"] + pct["top2"] + pct["top3"]) * 100, 1),
            "absent_pct": round(pct["absent"] * 100, 1),
            "meaningful_pct": round(report["meaningful_crosstoken"]["frac"] * 100, 1),
            "top_error_coinfluencer": (top_err["label"] if top_err else ""),
            "top_error_mean_edge": (top_err["mean_edge"] if top_err else ""),
        })
        per_month[month][f"L{layer}F{fidx}"] = {
            "report_json": str(reports_dir / f"{stem}.json"),
            "rank_histogram": report["rank_histogram"],
            "meaningful_crosstoken": report["meaningful_crosstoken"],
            "co_influencers": report["co_influencers"],
            "n_records": report["n_records"]}
        print(f"[merge] {month:>10} L{layer}F{fidx}: {report['n_records']:,} records "
              f"from {len(parts)} shards | top1 {summary_rows[-1]['top1_pct']}% "
              f"top1-3 {summary_rows[-1]['top1_3_pct']}% "
              f"meaningful {summary_rows[-1]['meaningful_pct']}% "
              f"topErr {summary_rows[-1]['top_error_coinfluencer']}")

    for month, feats in per_month.items():
        (reports_dir / f"{month.lower()}-all_combined.json").write_text(
            json.dumps({"month": month, "features": feats}, indent=2))

    cols = ["month", "feature", "n_records", "n_skipped", "n_people", "n_shards",
            "top1_pct", "top1_3_pct", "absent_pct", "meaningful_pct",
            "top_error_coinfluencer", "top_error_mean_edge"]
    with (reports_dir / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in sorted(summary_rows, key=lambda r: (r["month"], r["feature"])):
            w.writerow(row)
    print(f"\n[merge] wrote {len(summary_rows)} report(s) + summary.csv to {reports_dir}")


if __name__ == "__main__":
    main()
