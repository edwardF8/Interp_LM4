"""Graph-comparison helpers for Notebook 2. The build wrapper imports
circuit-tracer lazily so the pure table/diff helpers import in any env."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from clts.edit_clt import drift

_REPORT_COLS = ["replacement_score", "completeness_score", "error_influence_share",
                "target_logit_prob", "top_logit_token", "n_feature_nodes_after_pruning"]


@dataclass
class GraphConfig:
    key: str
    model_dir: str
    clt_dir: str
    scan_name: str


def configs_from_manifest(manifest: dict) -> list:
    edited = manifest["edited_model_dir"]
    orig = manifest["orig_model_dir"]
    base = manifest["base_clt_dir"]
    gcs = [
        GraphConfig("baseline_orig", orig, base, "baseline_orig"),
        GraphConfig("m3_stale", edited, base, "m3_stale"),
    ]
    for key, m in manifest.get("methods", {}).items():
        clt = m.get("expected_clt_dir")
        if clt:
            gcs.append(GraphConfig(key, edited, clt, key))
    return gcs


def build_or_load_graph(gc, data_dir, graph_root, slug, prompt, target=None,
                        device="cpu") -> dict:
    from clts.build_attribution_graph import build_graph
    graph_dir = Path(graph_root) / gc.scan_name / slug
    return build_graph(
        model_dir=gc.model_dir, clt_dir=gc.clt_dir, data_dir=data_dir,
        scan_name=gc.scan_name, graph_dir=str(graph_dir), slug=slug,
        prompt=prompt, target=target, device=device,
    )


def comparison_table(reports: dict) -> pd.DataFrame:
    rows = {k: {c: r.get(c) for c in _REPORT_COLS} for k, r in reports.items()}
    return pd.DataFrame.from_dict(rows, orient="index")[_REPORT_COLS]


def feature_diff_table(graphs: dict, baseline_key="baseline_orig") -> pd.DataFrame:
    base = graphs[baseline_key]
    rows = {}
    for key, g in graphs.items():
        o = drift.active_feature_overlap(base, g)
        rows[key] = {"jaccard": o["jaccard"], "n_features": o["n_b"],
                     "appeared": o["appeared"], "disappeared": o["disappeared"]}
    return pd.DataFrame.from_dict(rows, orient="index")
