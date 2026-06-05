# tests/test_writefeatures.py
"""Unit tests for clts/writefeatures.py (feature-in-node rank tester)."""
from __future__ import annotations

import clts.writefeatures as wf


def test_rank_bucket_boundaries():
    assert wf.rank_bucket(1) == "top1"
    assert wf.rank_bucket(2) == "top2"
    assert wf.rank_bucket(5) == "top5"
    assert wf.rank_bucket(6) == "6-10"
    assert wf.rank_bucket(10) == "6-10"
    assert wf.rank_bucket(11) == ">10"
    assert wf.rank_bucket(None) == "absent"


def test_sample_people_reproducible_and_random():
    pool = list(range(100))
    a = wf.sample_people(pool, 5, seed=0)
    b = wf.sample_people(pool, 5, seed=0)
    c = wf.sample_people(pool, 5, seed=1)
    assert a == b                       # reproducible with same seed
    assert a != c                       # different seed -> different sample
    assert a != [0, 1, 2, 3, 4]         # not just the first N
    assert len(a) == 5 and set(a) <= set(pool)


def test_sample_people_no_cap_returns_all():
    pool = [(0, "x"), (1, "y")]
    assert wf.sample_people(pool, None, seed=0) == pool
    assert wf.sample_people(pool, 5, seed=0) == pool   # n >= len -> all


import torch


def test_find_logit_row():
    # logit token ids for the 2 logit nodes, in node order
    logit_ids = [101, 202]
    # layout: n_features=3, n_error=4, n_tokens=2  -> logits start at 3+4+2=9
    row, k = wf.find_logit_row(logit_ids, 202, n_features=3, n_error=4, n_tokens=2)
    assert (row, k) == (10, 1)          # second logit -> base 9 + k(1)
    row, k = wf.find_logit_row(logit_ids, 999, n_features=3, n_error=4, n_tokens=2)
    assert row is None and k is None


def test_incoming_feature_edges_reads_target_row():
    # 3 feature nodes; build A so that row 5 (our "logit") has known edges into them.
    N = 6
    A = torch.zeros(N, N)
    A[5, 0] = 0.10   # feature node 0
    A[5, 1] = 0.50   # feature node 1
    A[5, 2] = -0.30  # feature node 2
    A[0, 5] = 9.99   # column into row 0 must be IGNORED (we read row, not column)
    selected_features = torch.tensor([0, 1, 2])
    # active_features rows are [layer, pos, fidx]
    active_features = torch.tensor([[3, 7, 4768],   # node 0
                                    [2, 7, 11],     # node 1
                                    [3, 1, 4768]])  # node 2 (same fidx, diff pos)
    edges = wf.incoming_feature_edges(A, selected_features, active_features, logit_row=5)
    assert edges == [
        {"layer": 3, "pos": 7, "fidx": 4768, "edge": 0.10},
        {"layer": 2, "pos": 7, "fidx": 11,   "edge": 0.50},
        {"layer": 3, "pos": 1, "fidx": 4768, "edge": -0.30},
    ]


EDGES = [
    {"layer": 3, "pos": 7, "fidx": 4768, "edge": 0.40},
    {"layer": 3, "pos": 1, "fidx": 4768, "edge": 0.20},   # same feature, 2nd position
    {"layer": 2, "pos": 7, "fidx": 11,   "edge": 0.50},
    {"layer": 1, "pos": 7, "fidx": 99,   "edge": -0.90},  # strong inhibitor
]


def test_aggregate_by_feature_sums_positions():
    agg = wf.aggregate_by_feature(EDGES)
    assert agg[(3, 4768)]["edge"] == 0.60
    assert agg[(3, 4768)]["positions"] == [1, 7]
    assert agg[(2, 11)]["edge"] == 0.50


def test_feature_rank_signed_vs_abs():
    agg = wf.aggregate_by_feature(EDGES)
    # signed: (3,4768)=0.60 > (2,11)=0.50 > (1,99)=-0.90  -> rank 1
    r = wf.feature_rank(agg, (3, 4768), rank_by_abs=False)
    assert r["rank"] == 1 and r["positions"] == [1, 7] and r["span"] == 6
    # abs: (1,99)=0.90 > (3,4768)=0.60 > (2,11)=0.50  -> (3,4768) rank 2
    r_abs = wf.feature_rank(agg, (3, 4768), rank_by_abs=True)
    assert r_abs["rank"] == 2


def test_feature_rank_absent():
    agg = wf.aggregate_by_feature(EDGES)
    r = wf.feature_rank(agg, (0, 12345), rank_by_abs=False)
    assert r["rank"] is None and r["positions"] == [] and r["span"] == 0


def _edges_with_target_at(ranks_and_positions, n_filler=20):
    """Build an edge list where the target feature (9, 4768) sits at given
    node-level ranks. `ranks_and_positions` = list of (desired_rank, pos). Filler
    features occupy the other ranks with descending edges."""
    # Give every node a distinct edge so node-rank == sorted order. Use edge = 1000 - rank.
    edges = []
    target_ranks = {r for r, _ in ranks_and_positions}
    pos_for_rank = {r: p for r, p in ranks_and_positions}
    total = n_filler + len(ranks_and_positions)
    for rank in range(1, total + 1):
        edge = float(total - rank + 1)   # rank 1 -> largest edge
        if rank in target_ranks:
            edges.append({"layer": 9, "pos": pos_for_rank[rank], "fidx": 4768, "edge": edge})
        else:
            edges.append({"layer": 0, "pos": rank, "fidx": 1000 + rank, "edge": edge})
    return edges


def test_multitoken_two_strong_positions_is_meaningful():
    edges = _edges_with_target_at([(1, 7), (3, 2)])   # ranks 1 and 3, distinct positions
    mt = wf.feature_multitoken(edges, (9, 4768), multi_tok_top_k=5)
    assert mt["n_positions"] == 2
    assert mt["n_meaningful"] == 2
    assert mt["is_meaningful"] is True


def test_multitoken_big_plus_tiny_is_not_meaningful():
    edges = _edges_with_target_at([(1, 7), (18, 2)])  # one strong (rank1), one weak (rank18)
    mt = wf.feature_multitoken(edges, (9, 4768), multi_tok_top_k=5)
    assert mt["n_positions"] == 2          # loose multi-position: still fired at 2 positions
    assert mt["n_meaningful"] == 1         # only rank1 clears top-5
    assert mt["is_meaningful"] is False    # the key distinction


def test_multitoken_absent_feature():
    edges = _edges_with_target_at([(1, 7)])
    mt = wf.feature_multitoken(edges, (0, 999999), multi_tok_top_k=5)
    assert mt["n_positions"] == 0 and mt["is_meaningful"] is False


def test_resolve_target():
    person = {"birthmonth": "August"}
    assert wf.resolve_target("month", person) == " August"
    assert wf.resolve_target(" August", person) == " August"   # literal pass-through
    assert wf.resolve_target(None, person) is None


def test_resolve_templates():
    all_t = ["{name} was born on {birthday}.", "{name} arrived on {birthday}."]
    assert wf.resolve_templates("all", all_t) == [(0, 0), (1, 1)]
    assert wf.resolve_templates([1], all_t) == [(1, 1)]
    out = wf.resolve_templates(["{name} popped out on {birthday}."], all_t)
    assert out == [("str0", "{name} popped out on {birthday}.")]


def test_template_prompt_string_path():
    person = {"first_name": "Gage", "middle_name": "Wyatt", "last_name": "Clay",
              "birthmonth": "August", "birthday": 24, "birthyear": 1712}
    # string template: substitute {name}, cut before {birthday}, single leading space
    p = wf.template_prompt(person, "{name} popped out on {birthday}.", sampler=None)
    assert p == " Gage Wyatt Clay popped out on"


import json


def _rec(rank, n_positions, span, is_meaningful, n_meaningful, top_feats):
    return {"ds_idx": 1, "id": 22, "name": "Gage Clay", "t_key": 0,
            "prompt": " Gage Clay was born on", "target_token": " August",
            "rank": rank, "bucket": wf.rank_bucket(rank), "span": span,
            "positions": [1, 7] if n_positions == 2 else [7],
            "n_positions": n_positions, "n_meaningful": n_meaningful,
            "is_meaningful": is_meaningful,
            "node_top_features": top_feats}


def test_build_report_aggregates():
    records = [
        _rec(1, 2, 6, True, 2, [{"layer": 3, "fidx": 4768, "edge": 0.6}]),
        _rec(3, 2, 5, False, 1, [{"layer": 3, "fidx": 4768, "edge": 0.4}]),
        _rec(None, 0, 0, False, 0, []),
    ]
    rep = wf.build_report({"records": records, "n_skipped": 1, "n_sampled": 2},
                          top_k=10, pos_span_flag=3, multi_tok_top_k=5,
                          config={"target_feature": [3, 4768]})
    h = rep["rank_histogram"]["counts"]
    assert h["top1"] == 1 and h["top3"] == 1 and h["absent"] == 1
    assert rep["loose_multipos"]["n_ge2_positions"] == 2
    assert rep["meaningful_crosstoken"]["n_meaningful"] == 1
    assert rep["meaningful_crosstoken"]["vs_loose"]["gap"] == 1   # 2 loose - 1 meaningful
    assert rep["co_influencers"] == []


def test_format_report_is_string():
    rep = wf.build_report({"records": [_rec(1, 1, 0, False, 1, [])], "n_skipped": 0,
                           "n_sampled": 1}, top_k=10, pos_span_flag=3,
                          multi_tok_top_k=5, config={"target_feature": [3, 4768]})
    s = wf.format_report(rep)
    assert isinstance(s, str) and "top1" in s


def test_save_report_writes_files(tmp_path):
    records = [_rec(1, 2, 6, True, 2, [{"layer": 3, "fidx": 4768, "edge": 0.6}])]
    rep = wf.build_report({"records": records, "n_skipped": 0, "n_sampled": 1},
                          top_k=10, pos_span_flag=3, multi_tok_top_k=5,
                          config={"target_feature": [3, 4768]})
    paths = wf.save_report(rep, records, tmp_path, layer=3, fidx=4768, subset_slug="august-n1-s0")
    assert paths["json"].endswith("report_L3F4768_august-n1-s0.json")
    loaded = json.loads(open(paths["json"]).read())
    assert loaded["rank_histogram"]["counts"]["top1"] == 1
    csv_text = open(paths["csv"]).read()
    assert "rank" in csv_text and "Gage Clay" in csv_text


def test_feature_rank_empty_agg():
    r = wf.feature_rank({}, (3, 4768), rank_by_abs=False)
    assert r["rank"] is None and r["n_features_in_node"] == 0


def test_edge_cache_key_stable_and_distinct():
    p = {"max_n_logits": 10}
    k1 = wf.edge_cache_key(" Gage was born on", " August", p)
    k2 = wf.edge_cache_key(" Gage was born on", " August", p)
    k3 = wf.edge_cache_key(" Gage was born on", " July", p)
    assert k1 == k2 and k1 != k3 and len(k1) == 16


def test_load_or_build_edges_caches(tmp_path):
    calls = {"n": 0}
    def build():
        calls["n"] += 1
        return [{"layer": 3, "pos": 7, "fidx": 4768, "edge": 0.5}]
    a = wf.load_or_build_edges(tmp_path, "abc123", build)
    b = wf.load_or_build_edges(tmp_path, "abc123", build)   # cache hit
    assert a == b and calls["n"] == 1                       # build ran exactly once


import os
import pytest

_MODEL_DIR = "model/grid-L4-H6"
_CLT_DIR = "clts/clt_runs/grid-L4-H6/mult16_l02_lr0.0001_ep50_n10000/final"
_DATA_DIR = "data/bioS_N-Bd_final_grid"
_HAS_ARTIFACTS = os.path.isdir(_MODEL_DIR) and os.path.isdir(_CLT_DIR) and os.path.isdir(_DATA_DIR)


def test_assign_roles_basic():
    person = {"first_name": "Gage", "middle_name": "Wyatt", "last_name": "Clay"}
    text = " Gage Wyatt Clay was born on"
    # offsets for tokens [' G','age',' Wyatt',' Clay',' was',' born',' on']
    offsets = [(0, 2), (2, 5), (5, 11), (11, 16), (16, 20), (20, 25), (25, 28)]
    roles = wf.assign_roles(offsets, text, person, {"born", "birth", "day", "date"})
    # single-token last name 'Clay' is still last_name[final] (always exactly one)
    assert roles == ["first_name", "first_name", "middle_name", "last_name[final]",
                     "template:other", "template:born", "template:other"]
    assert roles.count("last_name[final]") == 1


def test_assign_roles_marks_last_name_final_when_multitoken():
    person = {"first_name": "Gianna", "middle_name": "Adeline", "last_name": "Rawlings"}
    text = " Gianna Adeline Rawlings was born on"
    # 'Gianna'->' G'|'ian'|'na'  'Adeline'->' Adeline'  'Rawlings'->' Raw'|'lings'
    offsets = [(0, 2), (2, 5), (5, 7), (7, 15), (15, 19), (19, 24),
               (24, 28), (28, 33), (33, 36)]
    roles = wf.assign_roles(offsets, text, person, {"born"})
    assert roles[4] == "last_name"          # ' Raw'
    assert roles[5] == "last_name[final]"   # 'lings' (final subword of a multi-token last name)
    assert roles[7] == "template:born"


def test_decode_error_nodes_layer_major():
    # n_features=3, n_tokens=2, n_layers=2 -> error block cols [3,7), token cols [7,9)
    N = 10
    A = torch.zeros(N, N)
    logit_row = 9
    # error col for (layer=1, pos=0) is at error-block index 1*2+0=2 -> matrix col 3+2=5
    A[logit_row, 5] = 0.7
    errs = wf.decode_error_nodes(A, logit_row, n_features=3, n_tokens=2, n_layers=2)
    assert {"layer": 1, "pos": 0, "edge": 0.7} in errs
    assert len(errs) == 4                      # n_tokens*n_layers


def test_decode_token_nodes():
    N = 10
    A = torch.zeros(N, N)
    logit_row = 9
    A[logit_row, 7] = 0.2   # token block starts at 3 + 2*2 = 7 ; pos 0
    A[logit_row, 8] = 0.9   # pos 1
    toks = wf.decode_token_nodes(A, logit_row, n_features=3, n_tokens=2, n_layers=2)
    assert toks == [{"pos": 0, "edge": 0.2}, {"pos": 1, "edge": 0.9}]


def test_label_nodes_and_unified_top():
    node_all = {
        "features": [{"layer": 3, "pos": 7, "fidx": 4768, "edge": 0.6},
                     {"layer": 3, "pos": 1, "fidx": 4768, "edge": 0.2}],   # sums to 0.8
        "errors":   [{"layer": 2, "pos": 4, "edge": 0.5},
                     {"layer": 0, "pos": 5, "edge": 0.05}],
        "tokens":   [{"pos": 4, "edge": 0.01}],
    }
    # role per position (index = graph pos)
    roles = ["BOS", "first_name", "first_name", "middle_name", "last_name",
             "template:born", "template:other", "template:other(recall)"]
    rows = wf.label_nodes(node_all, roles, include_tokens=True)
    labels = {r["label"]: r["edge"] for r in rows}
    assert labels["L3 F4768"] == 0.8
    assert labels["err@last_name@L2"] == 0.5
    assert labels["err@template:born@L0"] == 0.05
    assert labels["tok@last_name"] == 0.01
    top = wf.unified_top_labels(node_all, roles, top_k=2, rank_by_abs=False)
    assert [t["label"] for t in top] == ["L3 F4768", "err@last_name@L2"]
    # include_tokens=False drops tok@* rows
    no_tok = wf.label_nodes(node_all, roles, include_tokens=False)
    assert all(not r["label"].startswith("tok@") for r in no_tok)


def _rec_unified(rank, unified_top):
    r = _rec(rank, 1, 0, False, 1, [])     # _rec from Task 7
    r["unified_top"] = unified_top
    return r


def test_build_report_unified_co_influencers():
    records = [
        _rec_unified(1, [{"kind": "feature", "label": "L3 F4768", "edge": 0.6},
                         {"kind": "error", "label": "err@last_name@L2", "edge": 0.5}]),
        _rec_unified(2, [{"kind": "error", "label": "err@last_name@L2", "edge": 0.4},
                         {"kind": "feature", "label": "L3 F4768", "edge": 0.3}]),
    ]
    rep = wf.build_report({"records": records, "n_skipped": 0, "n_sampled": 1},
                          top_k=10, pos_span_flag=3, multi_tok_top_k=5,
                          config={"target_feature": [3, 4768]})
    labels = {c["label"]: c for c in rep["co_influencers"]}
    assert labels["err@last_name@L2"]["count"] == 2
    assert labels["err@last_name@L2"]["kind"] == "error"
    assert abs(labels["err@last_name@L2"]["mean_edge"] - 0.45) < 1e-9


def test_unskew_guardrail_outputs_1_to_3_unchanged():
    # Two record sets identical except for unified_top contents; outputs 1-3 must match.
    base = [_rec_unified(1, [{"kind": "feature", "label": "L3 F4768", "edge": 0.6}]),
            _rec_unified(3, [{"kind": "feature", "label": "L1 F5", "edge": 0.2}])]
    witherr = [dict(r, unified_top=r["unified_top"] +
                    [{"kind": "error", "label": "err@last_name@L2", "edge": 9.9}]) for r in base]
    cfg = {"target_feature": [3, 4768]}
    a = wf.build_report({"records": base, "n_skipped": 0, "n_sampled": 1},
                        top_k=10, pos_span_flag=3, multi_tok_top_k=5, config=cfg)
    b = wf.build_report({"records": witherr, "n_skipped": 0, "n_sampled": 1},
                        top_k=10, pos_span_flag=3, multi_tok_top_k=5, config=cfg)
    assert a["rank_histogram"] == b["rank_histogram"]
    assert a["loose_multipos"] == b["loose_multipos"]
    assert a["meaningful_crosstoken"] == b["meaningful_crosstoken"]
    # ...but the co-influencer view DID change (error label now present)
    assert a["co_influencers"] != b["co_influencers"]


@pytest.mark.skipif(not _HAS_ARTIFACTS, reason="model/CLT/data artifacts not present")
def test_attribute_node_inputs_matches_full_graph():
    """The fast logit-only path must reproduce the full graph's edges INTO the logit
    node -- features, MLP-error nodes, and token nodes -- to floating-point noise."""
    from pathlib import Path
    from clts.export_tokenizer import ensure_hf_tokenizer
    from clts.load_replacement_model import load_replacement_model
    from util.bio_sampler import BioSampler
    from util.condensed_tokenizer import CondensedTokenizer

    model = load_replacement_model(Path(_MODEL_DIR), Path(_CLT_DIR),
                                   ensure_hf_tokenizer(Path(_DATA_DIR)), "grid-L4-H6",
                                   device="cpu")
    sampler = BioSampler(Path(_DATA_DIR) / "people.json", fields=("birthday",))
    ct = CondensedTokenizer.from_remap_path(Path(_DATA_DIR) / "old_to_new.json")
    _, person = wf.people_in_month(sampler, ct, "August")[0]
    prompt = wf.template_prompt(person, 0, sampler)
    target = wf.resolve_target("month", person)

    full = wf.node_input_all(wf.attribute_fast(model, prompt, target, **wf.DEFAULT_BUILD_PARAMS),
                             target, model.tokenizer)
    fast = wf.attribute_node_inputs(model, prompt, target)

    # features: fast keeps ALL active features; must match on every feature the full
    # (truncated) graph kept, summed per (layer, fidx) so position order is irrelevant.
    def agg(block):
        out = {}
        for e in block:
            out[(e["layer"], e["fidx"])] = round(out.get((e["layer"], e["fidx"]), 0.0)
                                                  + e["edge"], 6)
        return out
    af, ff = agg(full["features"]), agg(fast["features"])
    assert set(af).issubset(set(ff))                       # fast is a superset
    assert all(abs(af[k] - ff[k]) < 1e-4 for k in af)
    # target feature's rank bucket is unchanged
    assert (wf.feature_rank(wf.aggregate_by_feature(full["features"]), (3, 4768))["rank"]
            == wf.feature_rank(wf.aggregate_by_feature(fast["features"]), (3, 4768))["rank"])
    # MLP-error and token nodes: identical sets, matching edges
    fe = {(e["layer"], e["pos"]): e["edge"] for e in full["errors"]}
    xe = {(e["layer"], e["pos"]): e["edge"] for e in fast["errors"]}
    assert set(fe) == set(xe) and all(abs(fe[k] - xe[k]) < 1e-4 for k in fe)
    ft = {e["pos"]: e["edge"] for e in full["tokens"]}
    xt = {e["pos"]: e["edge"] for e in fast["tokens"]}
    assert set(ft) == set(xt) and all(abs(ft[k] - xt[k]) < 1e-4 for k in ft)


@pytest.mark.skipif(not _HAS_ARTIFACTS, reason="model/CLT/data artifacts not present")
def test_run_hypothesis_end_to_end(tmp_path):
    from pathlib import Path
    from clts.export_tokenizer import ensure_hf_tokenizer
    from clts.load_replacement_model import load_replacement_model
    from util.bio_sampler import BioSampler
    from util.condensed_tokenizer import CondensedTokenizer

    model = load_replacement_model(Path(_MODEL_DIR), Path(_CLT_DIR),
                                   ensure_hf_tokenizer(Path(_DATA_DIR)), "grid-L4-H6",
                                   device="cpu")
    sampler = BioSampler(Path(_DATA_DIR) / "people.json", fields=("birthday",))
    ct = CondensedTokenizer.from_remap_path(Path(_DATA_DIR) / "old_to_new.json")

    # token_roles on the canonical prompt (BOS + recall + 'born')
    person0 = wf.people_in_month(sampler, ct, "August")[0][1]
    roles = wf.token_roles(f" {person0['first_name']} {person0['middle_name']} "
                           f"{person0['last_name']} was born on", person0, model.tokenizer,
                           template_word_labels={"born", "birth", "day", "date"})
    assert roles[0] == "BOS" and roles[-1].endswith("(recall)")

    people = wf.people_in_month(sampler, ct, "August")[:1]
    result = wf.run_hypothesis(model, sampler, ct, people, target_feature=(3, 4768),
                               target="month", templates=[0, 1], cache_dir=tmp_path,
                               n_cap=1, seed=0)
    r = result["records"][0]
    assert r["target_token"] == " August"
    assert (r["rank"] is None) or isinstance(r["rank"], int)
    assert isinstance(r["unified_top"], list) and r["unified_top"]
    # at least one error label appears among the unified rows for this prompt
    assert any(t["label"].startswith("err@") for t in r["unified_top"])
