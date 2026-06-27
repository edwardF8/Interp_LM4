import json
from pathlib import Path
from clts.edit_clt import edit_clt_config as cfg


def test_factspec_slug():
    fs = cfg.FactSpec()
    assert fs.slug() == "p0-month-jul"
    fs2 = cfg.FactSpec(person=5, fields=("month",), new_values={"month": "March"})
    assert fs2.slug() == "p5-month-mar"


def test_expected_clt_dir_matches_apricot_layout():
    root = Path("/store")
    # baseline apricot config -> the exact on-disk dir name
    d = cfg.expected_clt_dir(root, "grid-L4-H6", "standalone",
                             expansion=16, l0=2.0, lr=1e-4, epochs=50, n_examples=10000)
    assert d == root / "clt_runs/grid-L4-H6/standalone/mult16_l02_lr0.0001_ep50_n10000/final"


def test_expected_clt_dir_method2_variant():
    root = Path("/store")
    d = cfg.expected_clt_dir(root, "grid-L4-H6-edit-p0-month-jul", "method2-v2-basic",
                             expansion=16, l0=2.0, lr=2e-5, epochs=5, n_examples=10000)
    assert d.as_posix().endswith(
        "clt_runs/grid-L4-H6-edit-p0-month-jul/method2-v2-basic/mult16_l02_lr2e-05_ep5_n10000/final")


def test_edited_model_name_and_dir():
    c = cfg.default_config(Path("/repo"), Path("/store"),
                           base_clt_dir="/store/clt_runs/grid-L4-H6/standalone/mult16_l02_lr0.0001_ep50_n10000/final",
                           data_dir="/repo/data/bioS_N-Bd_final_grid")
    assert c.edited_model_name() == "grid-L4-H6-edit-p0-month-jul"
    assert c.edited_model_dir(Path("/repo")) == Path("/repo/model/grid-L4-H6-edit-p0-month-jul")


def test_default_config_has_method1_and_two_method2_variants():
    c = cfg.default_config(Path("/repo"), Path("/store"),
                           base_clt_dir="/b/final", data_dir="/d")
    keys = {m.key for m in c.methods}
    assert "m1_scratch" in keys
    assert "m2-v2-basic" in keys
    assert "m2-v2-fixed" in keys
    m1 = next(m for m in c.methods if m.key == "m1_scratch")
    assert (m1.expansion, m1.l0, m1.lr, m1.epochs, m1.n_examples) == (16, 2.0, 1e-4, 50, 10000)
    assert m1.resume_from is None
    m2 = next(m for m in c.methods if m.key == "m2-v2-basic")
    assert m2.resume_from == "/b/final" and m2.lr == 2e-5 and m2.out_tag == "method2-v2-basic"


def test_manifest_roundtrip(tmp_path):
    man = {"exp_id": "x", "fact": {"person": 0}, "target_stats": {"ce_recovered": 0.5}}
    p = tmp_path / "manifest.json"
    cfg.write_manifest(p, man)
    assert cfg.read_manifest(p) == man
