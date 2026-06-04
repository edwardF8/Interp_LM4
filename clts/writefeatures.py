# clts/writefeatures.py
"""Feature-in-node rank tester.

Given a people subset and a target CLT feature (layer, fidx), measure where that
feature ranks among the DIRECT input features of an output-token node, across many
bio templates. Pure helpers + dependency-injected model glue, so the notebook
`clts/writeFeatures.ipynb` stays a thin wrapper. See
docs/superpowers/specs/2026-06-04-writefeatures-feature-node-rank-design.md.
"""
from __future__ import annotations

import csv as _csv
import hashlib as _hashlib
import json as _json
import random
import re as _re
from collections import defaultdict
from pathlib import Path

_RANK_BUCKETS = ["top1", "top2", "top3", "top4", "top5", "6-10", ">10", "absent"]


def rank_bucket(rank):
    """Map a 1-based rank (or None) to a fixed histogram bucket."""
    if rank is None:
        return "absent"
    if 1 <= rank <= 5:
        return f"top{rank}"
    if 6 <= rank <= 10:
        return "6-10"
    return ">10"


def sample_people(pool, n, seed):
    """Random (seeded) subset of `pool`. Returns all of `pool` when `n` is None or
    `n >= len(pool)`. Never the first N — choice is randomized but reproducible."""
    pool = list(pool)
    if n is None or n >= len(pool):
        return pool
    return random.Random(seed).sample(pool, n)


def find_logit_row(logit_token_ids, target_token_id, *, n_features, n_error, n_tokens):
    """Row index of the logit node whose vocab id == target_token_id.

    Node layout: [features | error | tokens | logits]; logit k is at
    n_features + n_error + n_tokens + k. Returns (row, k), or (None, None) if the
    target token is not among the graph's logit nodes."""
    base = n_features + n_error + n_tokens
    for k, tid in enumerate(logit_token_ids):
        if int(tid) == int(target_token_id):
            return base + k, k
    return None, None


def incoming_feature_edges(adjacency, selected_features, active_features, logit_row):
    """Direct edges INTO `logit_row` from every feature node (reads a ROW, since
    adjacency_matrix is [target, source]). Returns one edge dict per feature node."""
    n_features = len(selected_features)
    e = adjacency[logit_row, :n_features]
    out = []
    for i in range(n_features):
        layer, pos, fidx = active_features[int(selected_features[i])].tolist()
        out.append({"layer": int(layer), "pos": int(pos), "fidx": int(fidx),
                    "edge": round(float(e[i]), 6)})
    return out


def aggregate_by_feature(edges):
    """Group edge dicts by (layer, fidx): sum `edge` across positions, collect
    sorted positions."""
    agg = {}
    for e in edges:
        key = (e["layer"], e["fidx"])
        slot = agg.setdefault(key, {"edge": 0.0, "positions": []})
        slot["edge"] = round(slot["edge"] + e["edge"], 6)
        slot["positions"].append(e["pos"])
    for slot in agg.values():
        slot["positions"] = sorted(set(slot["positions"]))
    return agg


def _sorted_features(agg, rank_by_abs):
    """(layer, fidx) keys sorted by summed edge, descending."""
    keyfn = (lambda k: abs(agg[k]["edge"])) if rank_by_abs else (lambda k: agg[k]["edge"])
    return sorted(agg.keys(), key=lambda k: (keyfn(k), k), reverse=True)


def feature_rank(agg, target_feature, rank_by_abs=False):
    """1-based rank of `target_feature` (layer, fidx) among aggregated features by
    summed edge (signed desc, or |edge| desc). rank=None when the feature is absent
    from the node's inputs."""
    order = _sorted_features(agg, rank_by_abs)
    if target_feature not in agg:
        return {"rank": None, "positions": [], "span": 0, "edge": 0.0,
                "n_features_in_node": len(agg)}
    rank = order.index(target_feature) + 1
    positions = agg[target_feature]["positions"]
    span = (max(positions) - min(positions)) if positions else 0
    return {"rank": rank, "positions": positions, "span": span,
            "edge": agg[target_feature]["edge"], "n_features_in_node": len(agg)}


def feature_multitoken(edges, target_feature, multi_tok_top_k=5, rank_by_abs=False):
    """Strict cross-token metric. Rank every feature NODE by edge (signed or abs);
    the target feature is a 'meaningful' multi-token contributor iff >=2 distinct
    positions each have a node whose node-level rank <= multi_tok_top_k. A
    one-big-one-tiny pair does NOT qualify (only the big one clears the threshold)."""
    layer, fidx = target_feature
    keyfn = (lambda e: abs(e["edge"])) if rank_by_abs else (lambda e: e["edge"])
    ordered = sorted(edges, key=lambda e: (keyfn(e), e["layer"], e["pos"], e["fidx"]), reverse=True)
    per_pos = []
    for node_rank, e in enumerate(ordered, start=1):
        if e["layer"] == layer and e["fidx"] == fidx:
            per_pos.append({"pos": e["pos"], "node_rank": node_rank, "edge": e["edge"]})
    n_positions = len({p["pos"] for p in per_pos})
    meaningful_positions = {p["pos"] for p in per_pos if p["node_rank"] <= multi_tok_top_k}
    n_meaningful = len(meaningful_positions)
    return {"n_positions": n_positions, "n_meaningful": n_meaningful,
            "is_meaningful": n_meaningful >= 2, "per_pos": per_pos}


def resolve_target(target, person):
    """Map the TARGET knob to a token string. "month" -> the person's own birth
    month as a leading-space token; None -> None; anything else -> literal."""
    if target is None:
        return None
    if target == "month":
        return " " + str(person["birthmonth"])
    return target


def birthday_templates():
    """The 46 dataset bio templates for the birthday field (training-identical)."""
    import util.bio_sampler  # noqa: F401  (sets the Training_On_LM4 import path)
    from data.bio_text import FIELD_SPECS
    return list(FIELD_SPECS["birthday"]["templates"])


def resolve_templates(templates, all_templates):
    """Normalise the TEMPLATES knob to a list of (t_key, t_val):
      "all"            -> [(i, i) for all 46]              (t_val int -> render via sampler)
      [int, ...]       -> [(i, i) for those indices]
      [str, ...]       -> [("str{j}", str)]               (t_val str -> format directly)
    """
    if templates == "all":
        return [(i, i) for i in range(len(all_templates))]
    if templates and isinstance(templates[0], int):
        return [(i, i) for i in templates]
    return [(f"str{j}", s) for j, s in enumerate(templates)]


def _full_name(person):
    return f"{person['first_name']} {person['middle_name']} {person['last_name']}"


def template_prompt(person, t_val, sampler):
    """Build the recall prompt that ends right before the birth date (so the next
    token is the month). int t_val -> render the trained template via `sampler` and
    truncate; str t_val -> substitute {name} and cut at {birthday}."""
    if isinstance(t_val, int):
        bio = sampler.render(person, t_val)
        date = f"{person['birthmonth']} {person['birthday']}, {person['birthyear']}"
        return bio[:bio.index(date)].rstrip()
    head = t_val.split("{birthday}")[0].replace("{name}", _full_name(person))
    if head and not head[0].isspace():
        head = " " + head
    return head.rstrip()


def build_report(result, *, top_k, pos_span_flag, multi_tok_top_k, config):
    """Aggregate per-(person, template) records into the four outputs."""
    records = result["records"]
    n = len(records)

    hist = {b: 0 for b in _RANK_BUCKETS}
    for r in records:
        hist[r["bucket"]] += 1
    hist_pct = {b: (hist[b] / n if n else 0.0) for b in _RANK_BUCKETS}

    loose_ge2 = sum(1 for r in records if r["n_positions"] >= 2)
    span_dist, flagged = {}, []
    for r in records:
        span_dist[r["span"]] = span_dist.get(r["span"], 0) + 1
        if r["n_positions"] >= 2 and r["span"] >= pos_span_flag:
            flagged.append({"id": r["id"], "name": r["name"], "t_key": r["t_key"],
                            "span": r["span"]})

    meaningful_ge2 = sum(1 for r in records if r["is_meaningful"])
    nmean_dist = {}
    for r in records:
        nmean_dist[r["n_meaningful"]] = nmean_dist.get(r["n_meaningful"], 0) + 1

    # Unified co-influencers: aggregate labeled rows across all records' unified_top.
    cnt, esum, kind_of = defaultdict(int), defaultdict(float), {}
    for r in records:
        for nd in r.get("unified_top", []):
            cnt[nd["label"]] += 1
            esum[nd["label"]] += nd["edge"]
            kind_of[nd["label"]] = nd["kind"]
    co = [{"label": lbl, "kind": kind_of[lbl], "count": cnt[lbl],
           "frac": (cnt[lbl] / n if n else 0.0),
           "mean_edge": round(esum[lbl] / cnt[lbl], 6)} for lbl in cnt]
    co.sort(key=lambda d: (-d["count"], -abs(d["mean_edge"]), d["label"]))

    return {
        "config": config,
        "n_records": n, "n_skipped": result.get("n_skipped", 0),
        "n_sampled": result.get("n_sampled"),
        "rank_histogram": {"counts": hist, "pct": hist_pct},
        "loose_multipos": {"n_ge2_positions": loose_ge2,
                           "frac": (loose_ge2 / n if n else 0.0),
                           "span_distribution": span_dist, "flagged": flagged,
                           "pos_span_flag": pos_span_flag},
        "meaningful_crosstoken": {"n_meaningful": meaningful_ge2,
                                  "frac": (meaningful_ge2 / n if n else 0.0),
                                  "n_meaningful_distribution": nmean_dist,
                                  "vs_loose": {"loose_ge2": loose_ge2,
                                               "meaningful_ge2": meaningful_ge2,
                                               "gap": loose_ge2 - meaningful_ge2}},
        "co_influencers": co[:top_k],
    }


def format_report(report):
    """Human-readable multi-line summary string of a report dict."""
    lines = []
    cfg = report.get("config", {})
    lines.append(f"feature {cfg.get('target_feature')}  target {cfg.get('target')!r}  "
                 f"n_records={report['n_records']}  skipped={report['n_skipped']}")
    lines.append("\nRank among direct input features:")
    h, p = report["rank_histogram"]["counts"], report["rank_histogram"]["pct"]
    for b in _RANK_BUCKETS:
        lines.append(f"  {b:>6}: {h[b]:>4}  ({p[b]*100:5.1f}%)")
    lm = report["loose_multipos"]
    lines.append(f"\nLoose multi-position (fired >=2 positions): {lm['n_ge2_positions']} "
                 f"({lm['frac']*100:.1f}%)  flagged (span>={lm['pos_span_flag']}): {len(lm['flagged'])}")
    mc = report["meaningful_crosstoken"]
    lines.append(f"Meaningful across tokens (strict): {mc['n_meaningful']} "
                 f"({mc['frac']*100:.1f}%)   gap vs loose: {mc['vs_loose']['gap']}")
    lines.append("\nUnified co-influencers (label | kind | count | mean_edge):")
    for c in report["co_influencers"]:
        lines.append(f"  {c['label']:<28} {c['kind']:<7} x{c['count']:<4} "
                     f"mean_edge={c['mean_edge']:+.4f}")
    return "\n".join(lines)


def save_report(report, records, out_dir, *, layer, fidx, subset_slug):
    """Write report JSON + a flat per-record CSV. Returns {'json','csv'} paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"report_L{layer}F{fidx}_{subset_slug}"
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(_json.dumps(report, indent=2))
    csv_path = out_dir / f"{stem}.csv"
    cols = ["id", "ds_idx", "name", "t_key", "prompt", "target_token", "rank",
            "bucket", "span", "n_positions", "n_meaningful", "is_meaningful"]
    with csv_path.open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(cols)
        for r in records:
            w.writerow([r.get(c) for c in cols])
    return {"json": str(json_path), "csv": str(csv_path)}


DEFAULT_BUILD_PARAMS = {"max_n_logits": 10, "desired_logit_prob": 0.95,
                        "max_feature_nodes": 4096, "batch_size": 256}


def edge_cache_key(prompt, target_token, build_params):
    """Stable 16-char key for the per-(prompt, target) edge list."""
    blob = _json.dumps({"prompt": prompt, "target": target_token,
                        "params": build_params}, sort_keys=True)
    return _hashlib.sha1(blob.encode()).hexdigest()[:16]


def load_or_build_edges(cache_dir, key, build_fn):
    """Return cached edges for `key`, else call build_fn(), cache, and return.
    build_fn is only invoked on a cache miss."""
    import torch
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.pt"
    if path.exists():
        # weights_only=False: cache stores plain dicts, not tensor checkpoints
        return torch.load(path, weights_only=False)
    edges = build_fn()
    torch.save(edges, path)
    return edges


def attribute_fast(model, prompt, target_token, *, max_n_logits=10,
                   desired_logit_prob=0.95, max_feature_nodes=4096, batch_size=256):
    """Run circuit-tracer attribution toward `target_token` on an ALREADY-LOADED
    model (no disk reload, no viewer files). Returns a Graph."""
    from circuit_tracer import attribute
    return attribute(prompt=prompt, model=model, attribution_targets=[target_token],
                     max_n_logits=max_n_logits, desired_logit_prob=desired_logit_prob,
                     batch_size=batch_size, max_feature_nodes=max_feature_nodes,
                     offload=None, verbose=False)


def node_input_all(graph, target_token, tokenizer):
    """Full decomposition of the edges into the `target_token` logit node:
    {'features': [...], 'errors': [...], 'tokens': [...]}. None if the token is not a
    logit node in this graph. Cached (feature-independent)."""
    ids = tokenizer.encode(target_token, add_special_tokens=False)
    if len(ids) != 1:
        return None
    n_features = len(graph.selected_features)
    n_tokens = len(graph.input_tokens)
    n_layers = graph.cfg.n_layers
    n_error = n_tokens * n_layers
    row, _k = find_logit_row(graph.logit_token_ids.tolist(), ids[0],
                             n_features=n_features, n_error=n_error, n_tokens=n_tokens)
    if row is None:
        return None
    A = graph.adjacency_matrix.cpu()
    return {
        "features": incoming_feature_edges(A, graph.selected_features, graph.active_features, row),
        "errors": decode_error_nodes(A, row, n_features, n_tokens, n_layers),
        "tokens": decode_token_nodes(A, row, n_features, n_tokens, n_layers),
    }


def _name_in_vocab(ct, name):
    try:
        ct.encode(name)
        return True
    except KeyError:
        return False


def people_in_month(sampler, ct, month, in_vocab_only=True):
    """All (dataset_idx, person) born in `month`, in dataset order, in-vocab only."""
    out = []
    for ds_idx, p in enumerate(sampler.people):
        if p["birthmonth"] != month:
            continue
        # Leading space matters: GPT-2 BPE is space-sensitive and the model only ever
        # saw names with a preceding space (" Gage ..."), so the in-vocab variant is
        # "ĠGage". Tokenizing without the space wrongly drops ~97% of people.
        if in_vocab_only and not _name_in_vocab(ct, f" {p['first_name']} {p['last_name']}"):
            continue
        out.append((ds_idx, p))
    return out


def people_by_ids(sampler, ids):
    wanted = set(ids)
    return [(i, p) for i, p in enumerate(sampler.people) if p["id"] in wanted]


def people_by_idx(sampler, idxs):
    return [(i, sampler.people[i]) for i in idxs]


def sample_in_month(sampler, ct, month, n, seed=0):
    return sample_people(people_in_month(sampler, ct, month), n, seed)


def _absent_record(ds_idx, person, t_key, prompt, target_token):
    return {"ds_idx": ds_idx, "id": person["id"],
            "name": f"{person['first_name']} {person['last_name']}", "t_key": t_key,
            "prompt": prompt, "target_token": target_token, "rank": None,
            "bucket": "absent", "span": 0, "positions": [], "n_positions": 0,
            "n_meaningful": 0, "is_meaningful": False,
            "unified_top": []}


def _name_field_spans(text, person):
    """Char (start, end) spans of first/middle/last name in `text`, found in order
    (the prompt always begins ' {first} {middle} {last} ...')."""
    spans, cursor = {}, 0
    for field in ("first_name", "middle_name", "last_name"):
        val = str(person[field])
        i = text.index(val, cursor)
        spans[field] = (i, i + len(val))
        cursor = i + len(val)
    return spans


def _lexical_role(text, a, b, spans, template_word_labels):
    for field, role in (("first_name", "first_name"), ("middle_name", "middle_name"),
                        ("last_name", "last_name")):
        s, e = spans[field]
        if a < e and b > s:          # char overlap (handles leading-space tokens)
            return role
    word = _re.sub(r"[^a-z]", "", text[a:b].lower())
    if word in template_word_labels:
        return f"template:{word}"
    return "template:other"


def assign_roles(offsets, text, person, template_word_labels):
    """Per prompt-token lexical role (NO BOS). The final 'last_name' token is ALWAYS
    relabeled `last_name[final]` (exactly one per graph, whether the last name is one
    token or several), so the bucket is stable regardless of tokenization; earlier
    subwords of a multi-token last name stay `last_name`."""
    spans = _name_field_spans(text, person)
    roles = [_lexical_role(text, a, b, spans, template_word_labels) for (a, b) in offsets]
    last_idxs = [i for i, r in enumerate(roles) if r == "last_name"]
    if last_idxs:
        roles[last_idxs[-1]] = "last_name[final]"
    return roles


def token_roles(prompt, person, tokenizer, *, template_word_labels):
    """Role per GRAPH position. The graph prepends BOS at position 0, so this returns
    ['BOS', <role per prompt token>...] (length == graph n_tokens). The final position
    also carries a '(recall)' suffix on its lexical role."""
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    roles = ["BOS"] + assign_roles(enc["offset_mapping"], prompt, person, template_word_labels)
    roles[-1] = roles[-1] + "(recall)"
    return roles


def decode_error_nodes(adjacency, logit_row, n_features, n_tokens, n_layers):
    """Edges into `logit_row` from each error node, decoded layer-major:
    error-block index j' -> (layer, pos) = divmod(j', n_tokens)."""
    e = adjacency[logit_row, n_features:n_features + n_tokens * n_layers]
    out = []
    for jp, w in enumerate(e.tolist()):
        layer, pos = divmod(jp, n_tokens)
        out.append({"layer": layer, "pos": pos, "edge": round(float(w), 6)})
    return out


def decode_token_nodes(adjacency, logit_row, n_features, n_tokens, n_layers):
    """Edges into `logit_row` from each token/embedding node (one per position)."""
    start = n_features + n_tokens * n_layers
    e = adjacency[logit_row, start:start + n_tokens]
    return [{"pos": pos, "edge": round(float(w), 6)} for pos, w in enumerate(e.tolist())]


def label_nodes(node_all, roles, *, include_tokens=True):
    """Unified labeled rows for ALL input nodes to the logit. Features keep
    'L{layer} F{fidx}' (summed across positions); error nodes -> 'err@{role}@L{layer}'
    (summed across positions sharing the role); token nodes -> 'tok@{role}'."""
    rows = []
    for (layer, fidx), slot in aggregate_by_feature(node_all["features"]).items():
        rows.append({"kind": "feature", "label": f"L{layer} F{fidx}", "edge": slot["edge"]})
    err = {}
    for nd in node_all["errors"]:
        label = f"err@{roles[nd['pos']]}@L{nd['layer']}"
        err[label] = round(err.get(label, 0.0) + nd["edge"], 6)
    for label, edge in err.items():
        rows.append({"kind": "error", "label": label, "edge": edge})
    if include_tokens:
        tok = {}
        for nd in node_all["tokens"]:
            label = f"tok@{roles[nd['pos']]}"
            tok[label] = round(tok.get(label, 0.0) + nd["edge"], 6)
        for label, edge in tok.items():
            rows.append({"kind": "token", "label": label, "edge": edge})
    return rows


def unified_top_labels(node_all, roles, *, top_k, rank_by_abs=False, include_tokens=True):
    """Top-K labeled rows across ALL node kinds, ranked by edge (signed or |edge|)."""
    rows = label_nodes(node_all, roles, include_tokens=include_tokens)
    keyfn = (lambda r: abs(r["edge"])) if rank_by_abs else (lambda r: r["edge"])
    rows.sort(key=lambda r: (keyfn(r), r["label"]), reverse=True)
    return rows[:top_k]


def run_hypothesis(model, sampler, ct, people, target_feature, target, templates, *,
                   cache_dir, n_cap=20, seed=0, top_k=10, multi_tok_top_k=5,
                   pos_span_flag=3, rank_by_abs=False,
                   template_word_labels=frozenset({"born", "birth", "day", "date"}),
                   include_token_nodes=True, build_params=None, progress=print):
    """Per (sampled person x template): build/lookup the cached node decomposition,
    compute the feature metrics (feature block only) AND the unified labeled top-K."""
    build_params = dict(build_params or DEFAULT_BUILD_PARAMS)
    key_params = {**build_params, "schema": "node_all_v1"}   # cache schema tag
    all_t = birthday_templates()
    tmpl_list = resolve_templates(templates, all_t)
    sampled = sample_people(list(people), n_cap, seed)
    records, n_skipped = [], 0

    for ds_idx, person in sampled:
        progress(f"  person id={person['id']} {person['first_name']} {person['last_name']}")
        for t_key, t_val in tmpl_list:
            try:
                prompt = template_prompt(person, t_val, sampler)
                target_token = resolve_target(target, person)
                if target_token is None:
                    n_skipped += 1
                    continue
                if len(model.tokenizer.encode(target_token, add_special_tokens=False)) != 1:
                    n_skipped += 1
                    continue
                key = edge_cache_key(prompt, target_token, key_params)

                def _build():
                    g = attribute_fast(model, prompt, target_token, **build_params)
                    return node_input_all(g, target_token, model.tokenizer)

                node_all = load_or_build_edges(cache_dir, key, _build)
                if node_all is None:
                    records.append(_absent_record(ds_idx, person, t_key, prompt, target_token))
                    continue
                feats = node_all["features"]
                agg = aggregate_by_feature(feats)
                fr = feature_rank(agg, target_feature, rank_by_abs=rank_by_abs)
                mt = feature_multitoken(feats, target_feature,
                                        multi_tok_top_k=multi_tok_top_k, rank_by_abs=rank_by_abs)
                roles = token_roles(prompt, person, model.tokenizer,
                                    template_word_labels=template_word_labels)
                unified = unified_top_labels(node_all, roles, top_k=top_k,
                                             rank_by_abs=rank_by_abs,
                                             include_tokens=include_token_nodes)
                records.append({
                    "ds_idx": ds_idx, "id": person["id"],
                    "name": f"{person['first_name']} {person['last_name']}",
                    "t_key": t_key, "prompt": prompt, "target_token": target_token,
                    "rank": fr["rank"], "bucket": rank_bucket(fr["rank"]),
                    "span": fr["span"], "positions": fr["positions"],
                    "n_positions": mt["n_positions"], "n_meaningful": mt["n_meaningful"],
                    "is_meaningful": mt["is_meaningful"],
                    "unified_top": unified,
                })
            except Exception as exc:   # one bad (person, template) must not kill the sweep
                n_skipped += 1
                progress(f"    skip (t={t_key}): {type(exc).__name__}: {exc}")
    return {"records": records, "n_skipped": n_skipped, "n_sampled": len(sampled)}
