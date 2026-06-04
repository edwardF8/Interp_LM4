"""Build a CLT attribution graph for a prompt and write viewer files.

Uses circuit-tracer's attribute() UNMODIFIED (the canonical algorithm). Adds a
per-graph fidelity report: target-logit probability, the error-node influence
share, and the pruned feature-node count, so an under-explained graph (large
error nodes — a weak-CLT symptom) is surfaced rather than shipped silently.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clts.export_tokenizer import ensure_hf_tokenizer  # noqa: E402
from clts.load_replacement_model import load_replacement_model  # noqa: E402
from clts.storage import storage_root  # noqa: E402
from util.bio_sampler import BioSampler  # noqa: E402


def default_birthday_prompt(data_dir):
    """A '<Name> was born on' prompt for a real person in people.json.

    Scans people in order and returns the first whose first+last name
    tokenizes entirely within the model's condensed vocab (no OOV subwords).
    Falls back to 'Gage Clay was born on' if none found in the first 500.
    """
    from util.condensed_tokenizer import CondensedTokenizer

    data_dir = Path(data_dir)
    try:
        ct = CondensedTokenizer.from_remap_path(data_dir / "old_to_new.json")
    except FileNotFoundError:
        return "Gage Clay was born on"

    sampler = BioSampler(data_dir / "people.json", fields=("birthday",), seed=0)
    for p in sampler.people[:500]:
        name = f"{p['first_name']} {p['last_name']}"
        prompt = f"{name} was born on"
        try:
            ct.encode(prompt)
            return prompt
        except KeyError:
            continue
    return "Gage Clay was born on"


def _graph_quality_metrics(graph) -> dict:
    """Per-graph fidelity metrics using circuit-tracer's OWN scoring (verbatim).

    compute_graph_scores returns the standard replacement/completeness scores
    (no bespoke clamping or denominator choices). We surface both, plus the
    spec's error-influence share (1 - replacement_score, the library-consistent
    error fraction of token->logit influence) and the feature-node count that
    SURVIVES pruning (the spec's "feature-node count after pruning").
    """
    from circuit_tracer.graph import compute_graph_scores, prune_graph

    replacement_score, completeness_score = compute_graph_scores(graph)

    n_features = len(graph.selected_features)
    node_mask, _edge_mask, _cum = (el.cpu() for el in prune_graph(graph))
    n_kept = int(node_mask[:n_features].sum().item())   # feature nodes AFTER pruning

    return {
        "replacement_score": float(replacement_score),
        "completeness_score": float(completeness_score),
        "error_influence_share": float(1.0 - replacement_score),
        "n_feature_nodes_after_pruning": n_kept,
    }


def build_graph(model_dir, clt_dir, data_dir, scan_name, graph_dir, slug,
                prompt=None, target=None, device="cpu",
                max_n_logits=10, desired_logit_prob=0.95,
                max_feature_nodes=4096, batch_size=256, verbose=True):
    from circuit_tracer import attribute
    from circuit_tracer.utils.create_graph_files import create_graph_files

    hf_tok = ensure_hf_tokenizer(data_dir)
    model = load_replacement_model(model_dir, clt_dir, hf_tok, scan_name, device=device)
    if prompt is None:
        prompt = default_birthday_prompt(data_dir)

    graph = attribute(
        prompt=prompt,
        model=model,
        attribution_targets=([target] if target else None),
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
        batch_size=batch_size,
        max_feature_nodes=max_feature_nodes,
        offload=None,
        verbose=verbose,
    )

    graph_dir = Path(graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    pt_path = graph_dir / f"{slug}.pt"
    graph.to_pt(str(pt_path))

    # Use a local-path scan so the bundled viewer loads feature dashboards from
    # the local server.  init-feature-examples.js:85 routes scan strings that
    # start with './' to util.getFile(`${scan}/${featureIndex}.json`), which the
    # local_server.py /data/ handler serves from data_dir.  Scans that do NOT
    # start with './' fall through to the CDN (Anthropic CloudFront), which
    # won't have our custom model's features.  scan_name stays clean for the
    # report and for naming the dashboard subfolder.
    # create_graph_files OVERWRITES <slug>.json, wiping any viewer state (pins,
    # supernodes, positions, renames) a prior Save wrote there. Preserve that
    # layout across a rebuild — but ONLY when the rebuild is provably identical,
    # so a stale layout is never re-attached to a changed graph (which would
    # misalign sg_pos / mislabel nodes). Identity is checked against a per-slug
    # `_buildInputs` stamp embedded in the graph file (run-meta.json is shared
    # across slugs in this dir and overwritten each build, so it can't be used).
    # If the inputs differ, the layout is backed up instead of applied.
    # graph_status is surfaced so the notebook can report whether you're
    # reopening an arranged graph or starting fresh.
    slug_json = graph_dir / f"{slug}.json"
    build_inputs = {
        "prompt": prompt, "target": target,
        "model_dir": str(model_dir), "clt_dir": str(clt_dir),
        "data_dir": str(data_dir), "max_feature_nodes": max_feature_nodes,
        "max_n_logits": max_n_logits, "desired_logit_prob": desired_logit_prob,
    }
    saved_qparams = None
    graph_status = "new"                        # no prior <slug>.json
    layout_counts = {"pins": 0, "supernodes": 0, "renames": 0}
    if slug_json.exists():
        graph_status = "rebuilt"                # existed, but no saved layout yet
        try:
            prev = json.loads(slug_json.read_text())
        except Exception:
            prev = {}
        prev_q = prev.get("qParams") or {}

        def _count(key):
            v = prev_q.get(key)
            if isinstance(v, list):
                return len(v)
            if isinstance(v, str) and v not in ("", "null"):
                if key == "pinnedIds":
                    return len(v.split(","))
                try:
                    return len(json.loads(v))
                except Exception:
                    return 0
            return 0

        layout_counts = {"pins": _count("pinnedIds"),
                         "supernodes": _count("supernodes"),
                         "renames": _count("clerps")}
        has_layout = any(layout_counts.values()) or bool(prev_q.get("sg_pos"))
        if has_layout:
            if prev.get("_buildInputs") == build_inputs:
                saved_qparams = prev_q          # identical graph -> safe to keep
                graph_status = "reused"
            else:
                # Inputs differ (or a pre-stamp graph): don't risk a stale layout.
                backup = graph_dir / f"{slug}.qparams-backup.json"
                backup.write_text(json.dumps(prev_q, indent=2))
                graph_status = "inputs-changed"
                if verbose:
                    print(f"[build] inputs differ for '{slug}'; saved layout "
                          f"NOT applied — backed up to {backup.name}")

    local_scan = f"./data/{scan_name}"
    create_graph_files(
        graph_or_path=graph,
        slug=slug,
        output_path=str(graph_dir),
        scan=local_scan,
        node_threshold=0.8,
        edge_threshold=0.98,
    )

    # Stamp this build's inputs (for the next rebuild's identity check) and
    # re-attach the preserved viewer layout, if any.
    g_json = json.loads(slug_json.read_text())
    g_json["_buildInputs"] = build_inputs
    if saved_qparams is not None:
        g_json["qParams"] = saved_qparams
        if verbose:
            print(f"[build] preserved saved viewer layout (qParams) for '{slug}'")
    slug_json.write_text(json.dumps(g_json, indent=2))

    # Use logit_token_ids (logit_tokens is deprecated in circuit-tracer 0.4.1)
    top_token = model.tokenizer.decode(graph.logit_token_ids[0].item())
    report = {
        "prompt": prompt,
        "scan_name": scan_name,
        "top_logit_token": top_token,
        "target_logit_prob": float(graph.logit_probabilities[0].item()),
        **_graph_quality_metrics(graph),
    }
    (graph_dir / f"{slug}.report.json").write_text(json.dumps(report, indent=2))

    # Reproducibility sidecar (spec Section 6/8): record inputs + pinned library
    # versions next to the graph. Attribution is deterministic, so identical
    # inputs + versions reproduce the graph.
    import importlib.metadata as _im

    def _ver(pkg):
        try:
            return _im.version(pkg)
        except Exception:
            return "unknown"

    run_meta = {
        "scan_name": scan_name,
        "slug": slug,
        "prompt": prompt,
        "target": target,
        "model_dir": str(model_dir),
        "clt_dir": str(clt_dir),
        "data_dir": str(data_dir),
        "max_feature_nodes": max_feature_nodes,
        "max_n_logits": max_n_logits,
        "desired_logit_prob": desired_logit_prob,
        "versions": {p: _ver(p) for p in
                     ("circuit-tracer", "transformers", "transformer-lens", "torch")},
    }
    (graph_dir / "run-meta.json").write_text(json.dumps(run_meta, indent=2))

    if verbose:
        print(json.dumps(report, indent=2))
    return {"graph": graph, "pt_path": str(pt_path), "report": report,
            "run_meta": run_meta, "status": graph_status,
            "layout_counts": layout_counts}


def main():
    parser = argparse.ArgumentParser(
        description="Build a CLT attribution graph for a prompt."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--clt-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--scan-name", default=None,
                        help="Scan id (default: model_name from CLT config.yaml)")
    parser.add_argument("--prompt", default=None,
                        help="Prompt string (default: birthday recall prompt)")
    parser.add_argument("--target", default=None,
                        help="Attribution target token string")
    parser.add_argument("--slug", default="graph",
                        help="Output file slug (default: 'graph')")
    parser.add_argument("--graph-dir", default=None,
                        help="Output directory (default: storage_root()/clt_graphs/<scan>/<slug>)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-feature-nodes", type=int, default=4096)
    args = parser.parse_args()

    # Resolve scan_name from CLT config.yaml if not provided
    scan = args.scan_name
    if scan is None:
        import yaml
        cfg_path = Path(args.clt_dir) / "config.yaml"
        if cfg_path.exists():
            scan = yaml.safe_load(cfg_path.read_text()).get("model_name", "unknown")
        else:
            scan = "unknown"

    graph_dir = args.graph_dir
    if graph_dir is None:
        slug = args.slug
        graph_dir = str(storage_root() / "clt_graphs" / scan / slug)

    build_graph(
        model_dir=args.model_dir,
        clt_dir=args.clt_dir,
        data_dir=args.data_dir,
        scan_name=scan,
        graph_dir=graph_dir,
        slug=args.slug,
        prompt=args.prompt,
        target=args.target,
        device=args.device,
        max_feature_nodes=args.max_feature_nodes,
    )


if __name__ == "__main__":
    main()
