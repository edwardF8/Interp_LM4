"""Compare two CLTs — a base CLT and a fine-tuned (RESUME_FROM) descendant —
to quantify **how similar they are**, in weight space and in behavior on a
random sample of bioS prompts.

Why index-aligned comparison is valid here
-------------------------------------------
The fine-tune was produced by ``--resume-from`` the base CLT, so feature index
``i`` in *both* CLTs is the *same feature* — it only drifted during continued
training. That is exactly the case where comparing ``W_dec[i]`` vs ``W_dec[i]``
and "feature ``i`` firing" vs "feature ``i`` firing" is meaningful. (The thing
you normally worry about — two *from-scratch* CLTs with unrelated feature
orderings needing greedy cosine matching — does NOT apply to a resumed
fine-tune.)

This module is intentionally self-contained: it imports only first-party
``clts/`` + ``util/`` code (no fact-editing dependency). Nothing here edits a
model or does any kind of knowledge editing — it only *reads* two CLTs and the
frozen base model and reports statistics.

Run (PSC, ``lm4-ct`` env, GPU) — defaults point at the apricot base vs. the
apricot-finetune-basic descendant on ``grid-L4-H6``::

    CLT_STORAGE_ROOT=/jet/home/friedmae/data_storage/LM4_Results \
    python finetuning/compare_clts.py --n-rows 256 --n-eval-rows 64 \
        --out finetuning/compare_apricot_vs_finetune.json

Everything is overridable (``--clt-a`` / ``--clt-b`` / ``--model-dir`` /
``--data-dir`` / seeds / sample sizes).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Project imports (run from repo root, or PYTHONPATH=repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clts.clt import CrossLayerTranscoder  # noqa: E402
from clts.evalCLT import (  # noqa: E402
    capture_activations, compute_layer_metrics, ce_recovered_full,
)
from util.bio_sampler import BioSampler  # noqa: E402
from util.condensed_tokenizer import CondensedTokenizer  # noqa: E402
from util.diverse_subset import DiverseBioSubset  # noqa: E402

CLT_SEED = 0  # matches trainCLT.py (train seed; +1 is the training eval slice)


# ----------------------------------------------------------------------------
# Defaults resolved off CLT_STORAGE_ROOT (the same root trainCLT.py uses).
# ----------------------------------------------------------------------------
def _default_paths() -> dict:
    root = Path(os.environ.get(
        "CLT_STORAGE_ROOT", "/jet/home/friedmae/data_storage/LM4_Results"))
    return {
        "model_dir": root / "runResults/bioS_N-Bd_final_grid/20260520-134455/"
                            "grid/grid-L4-H6/final",
        "data_dir": root / "Data/bioS_N-Bd_final_grid",
        "clt_a": root / "clt_runs/grid-L4-H6/standalone/"
                        "mult16_l02_lr0.0001_ep50_n10000/final",
        "clt_b": root / "clt_runs/grid-L4-H6/apricot-finetune-basic/"
                        "mult16_l02_lr2e-05_ep5_n10000/final",
    }


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _hook_templates(clt_dir: Path) -> tuple[str, str]:
    """Read enc/dec hook names from a CLT's config.yaml (fall back to CLT
    defaults). Returns the full ``blocks.{layer}.<hook>`` templates."""
    enc, dec = "hook_resid_mid", "hook_mlp_out"
    cfg_path = clt_dir / "config.yaml"
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        enc = cfg.get("feature_input_hook", enc)
        dec = cfg.get("feature_output_hook", dec)
    return f"blocks.{{layer}}.{enc}", f"blocks.{{layer}}.{dec}"


# ----------------------------------------------------------------------------
# 1. Standalone quality metrics for one CLT (ce_recovered / L0 / nMSE / dead).
# ----------------------------------------------------------------------------
def quality_metrics(model, clt, eval_tokens, enc_t, dec_t) -> dict:
    x, y = capture_activations(model, eval_tokens, enc_t, dec_t)
    layer = compute_layer_metrics(clt, x, y)
    ce = ce_recovered_full(model, clt, eval_tokens,
                           enc_hook_template=enc_t, dec_hook_template=dec_t)
    N = clt.n_layers
    return {
        "ce_recovered": ce["ce_recovered"],
        "ce_orig": ce["ce_orig"], "ce_clt": ce["ce_clt"], "ce_zero": ce["ce_zero"],
        "l0_mean": float(np.mean([layer[f"l0_L{L}"] for L in range(N)])),
        "l0_per_layer": [layer[f"l0_L{L}"] for L in range(N)],
        "nmse_mean": float(np.mean([layer[f"nmse_L{L}"] for L in range(N)])),
        "nmse_per_layer": [layer[f"nmse_L{L}"] for L in range(N)],
        "dead_frac_per_layer": [layer[f"dead_frac_L{L}"] for L in range(N)],
    }


# ----------------------------------------------------------------------------
# 2. Weight-space similarity (index-aligned; valid because of the resume).
# ----------------------------------------------------------------------------
def _cosine_summary(cos: torch.Tensor) -> dict:
    """Distribution summary for a [n_features] cosine tensor."""
    q = torch.tensor([0.01, 0.05, 0.25, 0.50])
    return {
        "mean": float(cos.mean()),
        "median": float(cos.median()),
        "min": float(cos.min()),
        "pctiles_low": {f"p{int(p*100)}": float(v)
                        for p, v in zip(q.tolist(), torch.quantile(cos, q).tolist())},
        "frac_moved_lt_0.9": float((cos < 0.9).float().mean()),
        "frac_moved_lt_0.99": float((cos < 0.99).float().mean()),
        "frac_identical_gt_0.999": float((cos > 0.999).float().mean()),
    }


def per_feature_weight_cosines(clt_a, clt_b) -> dict:
    """Raw per-feature (index-aligned) weight comparisons, pooled across layers.
    Returns tensors of length n_features_total = n_layers * d_transcoder:
      decoder_cos    – cosine of the summed-decoder vectors (sum over targets),
      encoder_cos    – cosine of the encoder rows (the feature *detectors*),
      decoder_norm_a – L2 norm of clt_a's summed-decoder vector (importance proxy),
      decoder_norm_b – same for clt_b,
      layer          – source-layer id of each feature,
    plus threshold/bias drift tensors (thr_rel, b_enc_abs). The notebook uses
    these raw arrays for histograms/scatter; `weight_similarity` summarises them."""
    cs = torch.nn.functional.cosine_similarity
    dec_cos, enc_cos, dnorm_a, dnorm_b, layer_id, thr_rel, benc_abs = ([] for _ in range(7))
    for L in range(clt_a.n_layers):
        da = clt_a.W_dec[L].detach().sum(dim=1)   # [d_t, D] summed over targets
        db = clt_b.W_dec[L].detach().sum(dim=1)
        dec_cos.append(cs(da, db, dim=-1))
        dnorm_a.append(da.norm(dim=-1)); dnorm_b.append(db.norm(dim=-1))
        enc_cos.append(cs(clt_a.W_enc[L].detach(), clt_b.W_enc[L].detach(), dim=-1))
        ta, tb = clt_a.threshold[L].detach(), clt_b.threshold[L].detach()
        thr_rel.append((tb - ta).abs() / ta.abs().clamp(min=1e-6))
        benc_abs.append((clt_b.b_enc[L].detach() - clt_a.b_enc[L].detach()).abs())
        layer_id.append(torch.full_like(dec_cos[-1], float(L)))
    return {k: torch.cat(v) for k, v in {
        "decoder_cos": dec_cos, "encoder_cos": enc_cos,
        "decoder_norm_a": dnorm_a, "decoder_norm_b": dnorm_b,
        "layer": layer_id, "thr_rel": thr_rel, "b_enc_abs": benc_abs,
    }.items()}


def decoder_cosine_by_importance(decoder_cos, decoder_norm,
                                 pcts=(0.10, 0.25, 0.50)) -> dict:
    """Decoder-cosine stratified by feature importance (summed-decoder norm).

    A converged CLT is very sparse: ~half its features have ~zero decoder norm
    and never meaningfully fire, so their decoder *direction* is unconstrained
    and drifts freely under continued training — dragging the RAW mean cosine
    down even when every feature-that-matters is unchanged. Restricting to
    high-norm features (or norm-weighting) is the honest read."""
    cos, norm = decoder_cos, decoder_norm
    order = torch.argsort(norm, descending=True)
    strata = {}
    for pct in pcts:
        k = max(1, int(norm.numel() * pct))
        c = cos[order[:k]]
        strata[f"top_{int(pct*100)}pct_by_norm"] = {
            "mean": float(c.mean()), "median": float(c.median()),
            "frac_gt_0.99": float((c > 0.99).float().mean()),
            "min_norm_in_stratum": float(norm[order[:k]][-1]),
        }
    w = norm / norm.sum().clamp(min=1e-12)
    moved = cos < 0.9
    return {
        "norm_weighted_mean_cosine": float((cos * w).sum()),
        "strata": strata,
        "moved_lt_0.9_median_norm": float(norm[moved].median()) if moved.any() else float("nan"),
        "stable_ge_0.9_median_norm": float(norm[~moved].median()) if (~moved).any() else float("nan"),
    }


def weight_similarity(clt_a, clt_b) -> dict:
    """Summary of per-feature encoder / summed-decoder cosine (index-aligned),
    the importance-stratified decoder read, and threshold/bias drift."""
    raw = per_feature_weight_cosines(clt_a, clt_b)
    return {
        "decoder_cosine": _cosine_summary(raw["decoder_cos"]),
        "encoder_cosine": _cosine_summary(raw["encoder_cos"]),
        "decoder_by_importance": decoder_cosine_by_importance(
            raw["decoder_cos"], raw["decoder_norm_a"]),
        "threshold_rel_change_mean": float(raw["thr_rel"].mean()),
        "b_enc_abs_change_mean": float(raw["b_enc_abs"].mean()),
        "n_features_total": int(raw["decoder_cos"].numel()),
    }


# ----------------------------------------------------------------------------
# 3. Behavioral similarity on a random prompt sample (chunked; memory-light).
# ----------------------------------------------------------------------------
def behavioral_similarity(model, clt_a, clt_b, tokens, enc_t, dec_t,
                          chunk_rows: int, return_arrays: bool = False) -> dict:
    """Stream the sample in row-chunks, accumulating per-feature sufficient
    stats so we never hold both CLTs' full activation matrices at once.

    Reports, per feature (index-aligned):
      * fire-rate in each CLT and their agreement,
      * Pearson corr of the two activation traces across sampled tokens
        (the sharpest 'same feature -> same behavior' signal),
      * appeared/disappeared (fires in exactly one CLT over the sample),
    and per token: Jaccard of the fired-feature sets (pooled across layers)."""
    device = next(clt_a.parameters()).device
    N = clt_a.n_layers
    F = clt_a.d_transcoder

    # Per-(layer,feature) accumulators for Pearson corr + fire rate + max.
    zeros = lambda: [torch.zeros(F, dtype=torch.float64, device=device) for _ in range(N)]
    s_a, s_b = zeros(), zeros()          # sum a, sum b
    s_aa, s_bb, s_ab = zeros(), zeros(), zeros()
    n_fire_a = [torch.zeros(F, dtype=torch.long, device=device) for _ in range(N)]
    n_fire_b = [torch.zeros(F, dtype=torch.long, device=device) for _ in range(N)]
    n_tokens = 0
    jac_sum, jac_n = 0.0, 0

    n_rows = tokens.shape[0]
    for start in range(0, n_rows, chunk_rows):
        chunk = tokens[start:start + chunk_rows]
        x, _ = capture_activations(model, chunk, enc_t, dec_t)
        with torch.no_grad():
            a = clt_a.encode(x)          # list of [T_chunk, F]
            b = clt_b.encode(x)
        T = a[0].shape[0]
        n_tokens += T
        fa_layers, fb_layers = [], []
        for L in range(N):
            aL, bL = a[L].double(), b[L].double()
            s_a[L] += aL.sum(0);  s_b[L] += bL.sum(0)
            s_aa[L] += (aL * aL).sum(0);  s_bb[L] += (bL * bL).sum(0)
            s_ab[L] += (aL * bL).sum(0)
            fa = a[L] > 0; fb = b[L] > 0
            n_fire_a[L] += fa.sum(0); n_fire_b[L] += fb.sum(0)
            fa_layers.append(fa); fb_layers.append(fb)
        # Per-token Jaccard, pooled across layers: [T, N*F].
        FA = torch.cat(fa_layers, dim=1)
        FB = torch.cat(fb_layers, dim=1)
        inter = (FA & FB).sum(1).double()
        union = (FA | FB).sum(1).double().clamp(min=1)
        jac_sum += float((inter / union).sum()); jac_n += T

    # Reduce to per-feature Pearson corr over the sample.
    n = float(n_tokens)
    cors, fired_both, appeared, disappeared = [], 0, 0, 0
    fr_abs_diff = []
    for L in range(N):
        ma, mb = s_a[L] / n, s_b[L] / n
        va = (s_aa[L] / n - ma * ma).clamp(min=0)
        vb = (s_bb[L] / n - mb * mb).clamp(min=0)
        cov = s_ab[L] / n - ma * mb
        denom = (va.sqrt() * vb.sqrt())
        ok = denom > 1e-12                       # both features actually vary
        c = torch.where(ok, cov / denom.clamp(min=1e-12),
                        torch.full_like(cov, float("nan")))
        cors.append(c[ok])
        fa = n_fire_a[L] > 0; fb = n_fire_b[L] > 0
        fired_both += int((fa & fb).sum())
        appeared += int((fb & ~fa).sum())        # dead in A, alive in B
        disappeared += int((fa & ~fb).sum())     # alive in A, dead in B
        fr_abs_diff.append((n_fire_a[L].double() - n_fire_b[L].double()).abs() / n)
    cors = torch.cat(cors) if cors else torch.tensor([])
    fr_abs_diff = torch.cat(fr_abs_diff)
    dead_a = int(sum(int((nf == 0).sum()) for nf in n_fire_a))
    dead_b = int(sum(int((nf == 0).sum()) for nf in n_fire_b))

    result = {
        "n_tokens": n_tokens,
        "per_token_firing_jaccard_mean": jac_sum / max(1, jac_n),
        "activation_pearson": {
            "n_features_compared": int(cors.numel()),
            "mean": float(cors.mean()) if cors.numel() else float("nan"),
            "median": float(cors.median()) if cors.numel() else float("nan"),
            "frac_gt_0.9": float((cors > 0.9).float().mean()) if cors.numel() else float("nan"),
            "frac_gt_0.99": float((cors > 0.99).float().mean()) if cors.numel() else float("nan"),
            "frac_lt_0.5": float((cors < 0.5).float().mean()) if cors.numel() else float("nan"),
        },
        "fire_rate_abs_diff_mean": float(fr_abs_diff.mean()),
        "n_fired_both": fired_both,
        "n_appeared_in_b_only": appeared,
        "n_disappeared_from_a": disappeared,
        "n_dead_a": dead_a, "n_dead_b": dead_b,
    }
    if return_arrays:                       # raw tensors for notebook plotting (not JSON-safe)
        result["arrays"] = {
            "activation_pearson": cors,
            "fire_rate_a": torch.cat([nf.double() for nf in n_fire_a]) / n,
            "fire_rate_b": torch.cat([nf.double() for nf in n_fire_b]) / n,
        }
    return result


# ----------------------------------------------------------------------------
# 4. Illustrative per-prompt view: top firing features, A vs B, index-aligned.
# ----------------------------------------------------------------------------
def per_prompt_examples(model, clt_a, clt_b, tokens, tokenizer, enc_t, dec_t,
                        n_prompts: int, top_k: int) -> list:
    out = []
    for r in range(min(n_prompts, tokens.shape[0])):
        row = tokens[r:r + 1]
        x, _ = capture_activations(model, row, enc_t, dec_t)
        with torch.no_grad():
            a = clt_a.encode(x); b = clt_b.encode(x)
        try:
            text = tokenizer.decode(row[0].tolist())
        except Exception:
            text = "<decode-failed>"
        layers = []
        for L in range(clt_a.n_layers):
            # Rank features by peak activation over the prompt in CLT-A.
            peak_a = a[L].amax(0); peak_b = b[L].amax(0)
            top = torch.topk(peak_a, min(top_k, peak_a.numel())).indices.tolist()
            top_b = set(torch.topk(peak_b, min(top_k, peak_b.numel())).indices.tolist())
            feats = [{
                "feat": int(f),
                "peak_a": round(float(peak_a[f]), 3),
                "peak_b": round(float(peak_b[f]), 3),
                "also_top_in_b": int(f) in top_b,
            } for f in top]
            layers.append({
                "layer": L,
                "top_overlap": len(set(top) & top_b),
                "top_features": feats,
            })
        out.append({"row": r, "text": text[:160], "layers": layers})
    return out


def _fmt(x, nd=4):
    return "nan" if x != x else f"{x:.{nd}f}"


# ----------------------------------------------------------------------------
# Shared setup — used by BOTH the CLI (`main`) and the notebook, so they load
# the same model + sample the same tokens + load the same CLTs.
# ----------------------------------------------------------------------------
def load_context(model_dir, data_dir, clt_a_dir, clt_b_dir, *,
                 device=None, n_rows=256, n_eval_rows=64, context_size=512,
                 sample_seed=CLT_SEED + 2, label_a="apricot", label_b="finetune") -> dict:
    """Load model + tokenizer + a random bioS token sample + both CLTs. Returns
    a dict of everything the metric functions need. `sample_seed` defaults to a
    fresh draw distinct from train (0) and the training-eval slice (1)."""
    model_dir, data_dir = Path(model_dir), Path(data_dir)
    clt_a_dir, clt_b_dir = Path(clt_a_dir), Path(clt_b_dir)
    for nm, pth in [("model", model_dir), ("data", data_dir),
                    ("clt-a", clt_a_dir), ("clt-b", clt_b_dir)]:
        assert pth.exists(), f"{nm} path missing: {pth}"
    device = device or pick_device()
    print(f"[device] {device}")

    from clts.tl_model import build_hooked_transformer
    model = build_hooked_transformer(model_dir, device, torch.float32)
    print(f"[model] {model_dir}\n        n_layers={model.cfg.n_layers} d_model={model.cfg.d_model}")

    tokenizer = CondensedTokenizer.from_remap_path(data_dir / "old_to_new.json")
    sampler = BioSampler(data_dir / "people.json", fields=("birthday",), seed=CLT_SEED)
    subset = DiverseBioSubset(sampler, tokenizer, context_size=context_size, seed=sample_seed)
    rows = subset.to_hf_dataset(max(n_rows, n_eval_rows), verbose=False)["input_ids"]
    all_tokens = torch.tensor(np.array(rows), dtype=torch.long, device=device)

    enc_t, dec_t = _hook_templates(clt_a_dir)
    assert (enc_t, dec_t) == _hook_templates(clt_b_dir), "CLTs use different hooks; not comparable"
    clt_a = CrossLayerTranscoder.load_from_dir(clt_a_dir).to(device).eval()
    clt_b = CrossLayerTranscoder.load_from_dir(clt_b_dir).to(device).eval()
    assert clt_a.n_layers == clt_b.n_layers and clt_a.d_transcoder == clt_b.d_transcoder, \
        "CLTs differ in shape; index-aligned comparison invalid"
    print(f"[data] sample rows={n_rows} eval rows={n_eval_rows} "
          f"(seed={sample_seed}, ctx={context_size})")
    print(f"[clts] a={label_a} b={label_b}  n_layers={clt_a.n_layers} "
          f"d_transcoder={clt_a.d_transcoder}\n")
    return {
        "device": device, "model": model, "tokenizer": tokenizer, "sampler": sampler,
        "clt_a": clt_a, "clt_b": clt_b, "enc_t": enc_t, "dec_t": dec_t,
        "sample_tokens": all_tokens[:n_rows], "eval_tokens": all_tokens[:n_eval_rows],
        "label_a": label_a, "label_b": label_b,
        "clt_a_dir": str(clt_a_dir), "clt_b_dir": str(clt_b_dir),
        "sample_seed": sample_seed, "n_rows": n_rows, "n_eval_rows": n_eval_rows,
        "context_size": context_size,
    }


def main():
    d = _default_paths()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, default=d["model_dir"])
    p.add_argument("--data-dir", type=Path, default=d["data_dir"])
    p.add_argument("--clt-a", type=Path, default=d["clt_a"], help="base CLT (apricot)")
    p.add_argument("--clt-b", type=Path, default=d["clt_b"], help="fine-tuned CLT")
    p.add_argument("--label-a", default="apricot")
    p.add_argument("--label-b", default="finetune")
    p.add_argument("--n-rows", type=int, default=256,
                   help="rows for behavioral firing stats (context-size each)")
    p.add_argument("--n-eval-rows", type=int, default=64,
                   help="rows for ce_recovered/L0/nMSE (single batch, like training eval)")
    p.add_argument("--chunk-rows", type=int, default=16)
    p.add_argument("--context-size", type=int, default=512)
    p.add_argument("--sample-seed", type=int, default=CLT_SEED + 2,
                   help="fresh random draw, distinct from train (0) and train-eval (1)")
    p.add_argument("--n-prompts", type=int, default=3)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--device", default=None, help="force cpu/cuda/mps (default: auto)")
    p.add_argument("--out", type=Path, default=None, help="write full report JSON here")
    args = p.parse_args()

    ctx = load_context(
        args.model_dir, args.data_dir, args.clt_a, args.clt_b, device=args.device,
        n_rows=args.n_rows, n_eval_rows=args.n_eval_rows, context_size=args.context_size,
        sample_seed=args.sample_seed, label_a=args.label_a, label_b=args.label_b)
    model, clt_a, clt_b = ctx["model"], ctx["clt_a"], ctx["clt_b"]
    tokenizer, enc_t, dec_t = ctx["tokenizer"], ctx["enc_t"], ctx["dec_t"]
    eval_tokens, sample_tokens = ctx["eval_tokens"], ctx["sample_tokens"]

    report = {
        "clt_a": str(args.clt_a), "clt_b": str(args.clt_b),
        "label_a": args.label_a, "label_b": args.label_b,
        "sample": {"n_rows": args.n_rows, "n_eval_rows": args.n_eval_rows,
                   "context_size": args.context_size, "seed": args.sample_seed},
        "quality": {
            args.label_a: quality_metrics(model, clt_a, eval_tokens, enc_t, dec_t),
            args.label_b: quality_metrics(model, clt_b, eval_tokens, enc_t, dec_t),
        },
        "weight_similarity": weight_similarity(clt_a, clt_b),
        "behavioral_similarity": behavioral_similarity(
            model, clt_a, clt_b, sample_tokens, enc_t, dec_t, args.chunk_rows),
        "per_prompt_examples": per_prompt_examples(
            model, clt_a, clt_b, sample_tokens, tokenizer, enc_t, dec_t,
            args.n_prompts, args.top_k),
    }

    # ---- human-readable summary ------------------------------------------
    qa, qb = report["quality"][args.label_a], report["quality"][args.label_b]
    ws, bs = report["weight_similarity"], report["behavioral_similarity"]
    print("=" * 72)
    print(f"CLT SIMILARITY REPORT   {args.label_a}  vs  {args.label_b}")
    print("=" * 72)
    print("\n-- Quality (higher ce_recovered better; lower nMSE better) --")
    print(f"  {'metric':<16}{args.label_a:>14}{args.label_b:>14}")
    print(f"  {'ce_recovered':<16}{_fmt(qa['ce_recovered']):>14}{_fmt(qb['ce_recovered']):>14}")
    print(f"  {'L0 (mean)':<16}{_fmt(qa['l0_mean'],2):>14}{_fmt(qb['l0_mean'],2):>14}")
    print(f"  {'nMSE (mean)':<16}{_fmt(qa['nmse_mean']):>14}{_fmt(qb['nmse_mean']):>14}")
    print("\n-- Weight-space similarity (index-aligned; cos=1 -> identical) --")
    dc, ec = ws["decoder_cosine"], ws["encoder_cosine"]
    print(f"  decoder cosine: mean={_fmt(dc['mean'])} median={_fmt(dc['median'])} "
          f"min={_fmt(dc['min'])}")
    print(f"                  moved(<.9)={_fmt(dc['frac_moved_lt_0.9'],3)} "
          f"identical(>.999)={_fmt(dc['frac_identical_gt_0.999'],3)}")
    di = ws["decoder_by_importance"]
    print(f"    ^ importance-weighted decoder cosine = "
          f"{_fmt(di['norm_weighted_mean_cosine'])}  "
          f"(top-25% by norm: {_fmt(di['strata']['top_25pct_by_norm']['mean'])}); "
          f"moved feats' median norm={_fmt(di['moved_lt_0.9_median_norm'])} "
          f"vs stable={_fmt(di['stable_ge_0.9_median_norm'])}  -> movement is dead features")
    print(f"  encoder cosine: mean={_fmt(ec['mean'])} median={_fmt(ec['median'])} "
          f"min={_fmt(ec['min'])}")
    print(f"  threshold rel-change mean={_fmt(ws['threshold_rel_change_mean'],3)}")
    print("\n-- Behavioral similarity on random bioS sample --")
    ap = bs["activation_pearson"]
    print(f"  per-token firing Jaccard (mean): {_fmt(bs['per_token_firing_jaccard_mean'],3)}")
    print(f"  per-feature activation Pearson: mean={_fmt(ap['mean'])} "
          f"median={_fmt(ap['median'])}")
    print(f"      frac>0.9={_fmt(ap['frac_gt_0.9'],3)}  frac>0.99={_fmt(ap['frac_gt_0.99'],3)}"
          f"  frac<0.5={_fmt(ap['frac_lt_0.5'],3)}")
    print(f"  fired in both={bs['n_fired_both']}  appeared(B only)={bs['n_appeared_in_b_only']}"
          f"  disappeared(A only)={bs['n_disappeared_from_a']}")
    print(f"  dead features: {args.label_a}={bs['n_dead_a']}  {args.label_b}={bs['n_dead_b']}")
    print("\n-- Illustrative prompts (top feature overlap A->B per layer) --")
    for ex in report["per_prompt_examples"]:
        ov = ", ".join(f"L{l['layer']}:{l['top_overlap']}/{args.top_k}" for l in ex["layers"])
        print(f"  row {ex['row']}: [{ov}]  \"{ex['text'][:70]}\"")
    print("=" * 72)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\n[out] full report -> {args.out}")


if __name__ == "__main__":
    main()
