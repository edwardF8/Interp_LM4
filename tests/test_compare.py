import torch
from clts.edit_clt import compare


def test_configs_from_manifest_builds_four_when_all_present():
    man = {
        "edited_model_dir": "/m/edit",
        "orig_model_dir": "/m/orig",
        "base_clt_dir": "/c/base/final",
        "methods": {
            "m1_scratch": {"expected_clt_dir": "/c/m1/final", "status": "done"},
            "m2-v2-basic": {"expected_clt_dir": "/c/m2/final", "status": "done"},
        },
    }
    gcs = {g.key: g for g in compare.configs_from_manifest(man)}
    assert gcs["baseline_orig"].model_dir == "/m/orig"
    assert gcs["baseline_orig"].clt_dir == "/c/base/final"
    assert gcs["m3_stale"].model_dir == "/m/edit"
    assert gcs["m3_stale"].clt_dir == "/c/base/final"
    assert gcs["m1_scratch"].clt_dir == "/c/m1/final"
    assert gcs["m2-v2-basic"].model_dir == "/m/edit"


def test_comparison_table_columns_and_rows():
    reports = {
        "baseline_orig": {"replacement_score": 0.62, "completeness_score": 0.7,
                          "error_influence_share": 0.38, "top_logit_token": " February",
                          "target_logit_prob": 0.8, "n_feature_nodes_after_pruning": 120},
        "m3_stale": {"replacement_score": 0.4, "completeness_score": 0.5,
                     "error_influence_share": 0.6, "top_logit_token": " July",
                     "target_logit_prob": 0.55, "n_feature_nodes_after_pruning": 90},
    }
    df = compare.comparison_table(reports)
    assert list(df.index) == ["baseline_orig", "m3_stale"]
    assert "replacement_score" in df.columns
    assert "error_influence_share" in df.columns
    assert df.loc["m3_stale", "error_influence_share"] == 0.6


class _FakeGraph:
    def __init__(self, triples):
        self.active_features = torch.tensor(triples, dtype=torch.long)
        self.selected_features = torch.arange(len(triples))


def test_feature_diff_table_against_baseline():
    graphs = {
        "baseline_orig": _FakeGraph([[0, 1, 10], [1, 1, 20]]),
        "m3_stale": _FakeGraph([[1, 1, 20], [0, 1, 30]]),
    }
    df = compare.feature_diff_table(graphs, baseline_key="baseline_orig")
    assert df.loc["m3_stale", "appeared"] == 1
    assert df.loc["m3_stale", "disappeared"] == 1
    # baseline row vs itself is trivial
    assert df.loc["baseline_orig", "jaccard"] == 1.0
