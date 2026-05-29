"""Fidelity + format tests for CLT attribution graphs."""
from __future__ import annotations

import os

import pytest


def test_cantor_roundtrip_matches_frontend():
    from clts.feature_index import cantor_pair, cantor_unpair

    # Frontend Node.feature_node: feature = (l+f)(l+f+1)//2 + f
    # Frontend cantorUnpair(z) -> [layer, feat]
    for layer in range(4):
        for feat in (0, 1, 5, 383, 6143):
            z = cantor_pair(layer, feat)
            assert cantor_unpair(z) == (layer, feat)

    # Spot-check the exact integer the frontend computes for (2, 100):
    assert cantor_pair(2, 100) == (2 + 100) * (2 + 100 + 1) // 2 + 100


MODEL_DIR = "model/grid-L4-H6"
CLT_DIR = "clts/clt_runs/grid-L4-H6/mult16_l02_lr0.0001_ep50_n10000/final"
SCAN_NAME = "grid-L4-H6"
DATA_DIR = "data/bioS_N-Bd_final_grid"

_HAS_ARTIFACTS = (
    os.path.isdir(MODEL_DIR)
    and os.path.isdir(CLT_DIR)
    and os.path.isfile(os.path.join(DATA_DIR, "old_to_new.json"))
)
_needs_artifacts = pytest.mark.skipif(
    not _HAS_ARTIFACTS, reason="model/CLT artifacts not present"
)


@pytest.mark.integration
def test_build_hooked_transformer_matches_hf():
    import torch
    from transformers import LlamaForCausalLM
    from clts.tl_model import build_hooked_transformer

    tl = build_hooked_transformer(MODEL_DIR, device="cpu")
    hf = LlamaForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32).eval()
    ids = torch.tensor([[1835, 5, 10, 20, 30, 40]])  # ids < vocab (1836)
    with torch.no_grad():
        tl_logits = tl(ids, return_type="logits")
        hf_logits = hf(ids).logits
    assert tl_logits.shape == hf_logits.shape
    # TL renders RMS-norm at eps=1e-5 (TL default, matching how trainCLT/evalCLT build
    # the model and how the CLT was trained).  The HF checkpoint uses eps=1e-6, so the
    # logits agree only to ~5e-3.  This is intentional fidelity to the training
    # environment, not the raw HF checkpoint.  atol=1e-2 comfortably covers the ~5e-3
    # eps gap while still catching a broken build (which would diverge by order 1+).
    assert torch.allclose(tl_logits, hf_logits, atol=1e-2, rtol=1e-2)
    # Sanity-check: next-token argmax at the last position must agree despite the eps gap.
    assert tl_logits[0, -1].argmax() == hf_logits[0, -1].argmax()


# --- Task 3: replacement-model adapter (circuit-tracer) fidelity checks -------
#
# A small, in-vocab token batch reused by the adapter tests below.  ids < 1836
# (vocab) and we lead with the bos/eos id (1835) so the first position is a
# special token, matching how attribution prompts are tokenized.
_ADAPTER_IDS = [[1835, 5, 10, 20, 30, 40, 50, 60]]


def _adapter_model(device="cpu"):
    import torch

    from clts.export_tokenizer import ensure_hf_tokenizer
    from clts.load_replacement_model import load_replacement_model

    tok_dir = ensure_hf_tokenizer(DATA_DIR)
    return load_replacement_model(
        MODEL_DIR, CLT_DIR, tok_dir, SCAN_NAME,
        device=device, dtype=torch.float32,
    )


@pytest.mark.integration
@_needs_artifacts
def test_adapter_clt_compute_equivalence():
    """Check A: circuit-tracer's loaded CLT must compute byte-for-byte (<=1e-4)
    the same encode activations and decode reconstruction as our own CLT.

    Our CLT (clts/clt.py) uses a list-of-N-tensors API (encode(list)->list).
    circuit-tracer's CrossLayerTranscoder.encode takes a *stacked* [N, n, D]
    tensor and returns dense [N, n, d_t]; its decode takes a *sparse* [N,n,d_t]
    feature tensor and returns [N, n, D].  We bridge the two layouts and assert
    equivalence -- the intent is zero loading / format / semantics drift.
    """
    import torch

    from clts.clt import CrossLayerTranscoder
    from clts.evalCLT import capture_activations

    model = _adapter_model()
    our_clt = CrossLayerTranscoder.load_from_dir(CLT_DIR)
    ct_clt = model.transcoders  # circuit-tracer's loaded CLT lives here

    ids = torch.tensor(_ADAPTER_IDS)
    x_list, _ = capture_activations(model, ids)  # N tensors, each [n, D]
    n_pos = x_list[0].shape[0]

    # --- encode equivalence -------------------------------------------------
    # Our CLT: list-in -> list-out.  Stack to [N, n, d_t] for comparison.
    our_enc = torch.stack(our_clt.encode(x_list))            # [N, n, d_t]
    # circuit-tracer CLT: stacked [N, n, D] -> dense [N, n, d_t].
    x_stacked = torch.stack(x_list)                          # [N, n, D]
    ct_enc = ct_clt.encode(x_stacked)                        # [N, n, d_t]
    assert ct_enc.shape == our_enc.shape
    enc_max = (ct_enc - our_enc).abs().max().item()
    assert enc_max < 1e-4, f"encode max abs diff {enc_max} >= 1e-4"

    # --- decode equivalence -------------------------------------------------
    # The two decoders compute the SAME math but in different summation orders:
    # our CLT does a dense [n, d_t] @ [d_t, D] matmul-sum over all features per
    # target layer; circuit-tracer gathers only the active (sparse) features,
    # scales their decoder vectors, and index_add_s them.  In fp32 these diverge
    # by ~1.5e-2 at the deepest layer (L=3 accumulates contributions from 4
    # source layers over d_t=6144) -- pure float accumulation order, NOT a
    # wiring bug: in float64 the same comparison collapses to ~2e-11 (verified).
    # We therefore assert decode equivalence in float64, which keeps the <=1e-4
    # equivalence intent (here ~1e-10, far tighter) while staying sensitive to
    # any genuine loading/format error (which would diverge at every precision).
    our64 = CrossLayerTranscoder.load_from_dir(CLT_DIR).double()
    x64 = [x.double() for x in x_list]
    our_dec = torch.stack(our64.decode(our64.encode(x64)))         # [N, n, D]

    ct_clt.to(torch.device("cpu"), torch.float64)
    ct_enc64 = ct_clt.encode(torch.stack(x64))                     # [N, n, d_t]
    ct_recon = ct_clt.decode(ct_enc64.to_sparse())                 # [N, n', D]
    # compute_reconstruction sizes n_pos from the max active position; with real
    # activations every position fires, but guard against silent truncation.
    assert ct_recon.shape[1] == n_pos, (
        f"ct reconstruction covers {ct_recon.shape[1]} of {n_pos} positions"
    )
    assert ct_recon.shape == our_dec.shape
    dec_max = (ct_recon - our_dec).abs().max().item()
    assert dec_max < 1e-4, f"decode max abs diff {dec_max} >= 1e-4"


@pytest.mark.integration
@_needs_artifacts
def test_adapter_sets_cfg_metadata():
    """The adapter must wire the metadata circuit-tracer's downstream graph
    steps depend on: model_name (used in `"gemma-3" in cfg.model_name`),
    tokenizer_name (AutoTokenizer.from_pretrained target), and the scan id.

    Verified against source: _configure_replacement_model stores the CLT's
    `scan` onto `model.scan` (replacement_model_transformerlens.py:178), so the
    scan id lands on `model.scan`, not `model.scan_name`.
    """
    from clts.export_tokenizer import ensure_hf_tokenizer

    model = _adapter_model()
    assert model.cfg.model_name == SCAN_NAME
    assert model.cfg.tokenizer_name == str(ensure_hf_tokenizer(DATA_DIR))
    assert model.scan == SCAN_NAME


@pytest.mark.integration
@_needs_artifacts
def test_adapter_base_matches_canonical_build():
    """The adapter's plain-forward logits must equal the canonical eps=1e-5
    build (clts/tl_model.build_hooked_transformer) within 1e-4 on a small
    batch.  Both use eps=1e-5, so this is tight and proves (a) the adapter
    assembled the SAME base model we train/eval against and (b) that
    _configure_replacement_model preserved the base forward through the
    renamed `old_mlp` / `old_unembed` wrappers.
    """
    import torch

    from clts.tl_model import build_hooked_transformer

    model = _adapter_model()
    canon = build_hooked_transformer(MODEL_DIR, device="cpu")

    ids = torch.tensor(_ADAPTER_IDS)
    with torch.no_grad():
        adapter_logits = model(ids, return_type="logits")
        canon_logits = canon(ids, return_type="logits")
    assert adapter_logits.shape == canon_logits.shape
    max_abs = (adapter_logits - canon_logits).abs().max().item()
    assert max_abs < 1e-4, f"adapter vs canonical logits max abs diff {max_abs}"


@pytest.mark.integration
@_needs_artifacts
def test_replacement_ce_matches_eval():
    """Check B (the spec's real one), realigned to evalCLT rather than HF.

    The spec's Check B asks that circuit-tracer attributes through the SAME
    model the CLT was evaluated against.  Because of the eps=1e-5 fidelity
    decision (tl_model omits eps -> 1e-5, matching trainCLT/evalCLT, NOT the HF
    checkpoint's 1e-6), the reference is evalCLT's full-MLP-replacement CE on
    the canonical build -- not the raw HF model.  We assert the adapter model
    behaves identically (<1e-3) to the canonical build under full MLP
    replacement, and that the CLT actually recovers CE (finite recovered).
    """
    import math
    import torch

    from clts.clt import CrossLayerTranscoder
    from clts.evalCLT import ce_recovered_full
    from clts.tl_model import build_hooked_transformer

    model = _adapter_model()
    canon = build_hooked_transformer(MODEL_DIR, device="cpu")
    our_clt = CrossLayerTranscoder.load_from_dir(CLT_DIR)

    tokens = torch.tensor(_ADAPTER_IDS)
    r_adapter = ce_recovered_full(model, our_clt, tokens)
    r_canon = ce_recovered_full(canon, our_clt, tokens)

    assert abs(r_adapter["ce_clt"] - r_canon["ce_clt"]) < 1e-3, (
        f"adapter ce_clt {r_adapter['ce_clt']} vs canonical {r_canon['ce_clt']}"
    )
    assert math.isfinite(r_adapter["ce_recovered"])


@pytest.mark.integration
def test_serve_ui_serves_local_feature_dashboards(tmp_path):
    import json
    import time
    import urllib.error
    import urllib.request

    from clts.serve_ui import start_server

    graph_dir = tmp_path / "graphs"
    graph_dir.mkdir()
    (graph_dir / "graph-metadata.json").write_text(json.dumps({"graphs": []}))
    feats = tmp_path / "clt_features" / SCAN_NAME
    feats.mkdir(parents=True)
    (feats / "5.json").write_text(json.dumps({"index": 5}))

    server = start_server(graph_dir=str(graph_dir),
                          features_dir=str(feats), scan_name=SCAN_NAME, port=8047)
    try:
        # loadFeature(scan="./data/<SCAN>", 5) fetches ./data/<SCAN>/5.json
        # local_server.py /data/ handler maps that to graph_dir/<SCAN>/5.json
        # which is the symlink -> feats/5.json created by start_server.
        url = f"http://localhost:8047/data/{SCAN_NAME}/5.json"
        body = None
        for _ in range(20):  # brief readiness retry
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    body = json.loads(r.read())
                break
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.1)
        assert body is not None and body["index"] == 5
    finally:
        server.stop()


@pytest.mark.integration
@_needs_artifacts
def test_build_graph_birthday_recall(tmp_path):
    from clts.build_attribution_graph import build_graph
    from clts.feature_index import cantor_unpair
    out = build_graph(
        model_dir=MODEL_DIR, clt_dir=CLT_DIR, data_dir=DATA_DIR,
        scan_name=SCAN_NAME, prompt=None, device="cpu",
        graph_dir=str(tmp_path), slug="test-bday", max_feature_nodes=512,
    )
    assert (tmp_path / "test-bday.json").exists()
    assert (tmp_path / "graph-metadata.json").exists()
    assert (tmp_path / "test-bday.report.json").exists()
    assert out["pt_path"]
    import torch
    from clts.clt import CrossLayerTranscoder
    clt = CrossLayerTranscoder.load_from_dir(CLT_DIR)
    graph = out["graph"]
    for row in graph.active_features[graph.selected_features].tolist():
        layer, _pos, feat = row
        z = (layer + feat) * (layer + feat + 1) // 2 + feat
        assert cantor_unpair(z) == (layer, feat)
        assert 0 <= layer < clt.n_layers and 0 <= feat < clt.d_transcoder
    r = out["report"]
    assert 0.0 <= r["replacement_score"] <= 1.0
    assert 0.0 <= r["completeness_score"] <= 1.0
    assert 0.0 <= r["error_influence_share"] <= 1.0
    assert r["target_logit_prob"] >= 0.0
    assert r["n_feature_nodes_after_pruning"] >= 0

    # Confirm the graph is locally viewable: metadata.scan must start with
    # "./data/" and end with SCAN_NAME so the bundled viewer's loadFeature
    # routes fetches to the local server (init-feature-examples.js:85-87).
    import json as _json
    written = _json.loads((tmp_path / "test-bday.json").read_text())
    graph_scan = written["metadata"]["scan"]
    assert graph_scan.startswith("./data/"), (
        f"graph scan {graph_scan!r} does not start with './data/' — "
        "local viewer would fall through to Anthropic CDN"
    )
    assert graph_scan.endswith(SCAN_NAME), (
        f"graph scan {graph_scan!r} does not end with {SCAN_NAME!r}"
    )
