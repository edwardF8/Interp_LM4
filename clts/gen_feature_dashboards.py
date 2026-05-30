"""Per-feature dashboard generator for CLT features.

Produces one <cantor_pair(layer, feat)>.json per fired feature under
    features_root/<scan_name>/

Each file validates against circuit_tracer.frontend.feature_models.Model
(pydantic) and is served by clts/serve_ui.py to the circuit-tracer viewer.

Typical usage on PSC:
    python clts/gen_feature_dashboards.py \
        --model-dir <path> --clt-dir <path> \
        --data-dir <path> --scan-name <name> \
        --device cuda

Default --features-root is STORAGE_ROOT/clt_features so the files land
exactly where serve_ui's symlink bridge expects them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Make the repo root importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================================
# Piece 1 — decoder_logit_effects (pure)
# ============================================================================

def decoder_logit_effects(clt, W_U: torch.Tensor) -> torch.Tensor:
    """[n_layers, d_transcoder, vocab] additive logit effect of each feature.

    A feature at layer L writes to layers L..N-1; its net residual direction is
    the sum of its decoder columns across those targets. Project through the
    unembed W_U ([d_model, vocab]) for its additive logit effect.
    """
    effects = []
    for L in range(clt.n_layers):
        summed_dec = clt.W_dec[L].sum(dim=1)   # [d_t, d_model]
        effects.append(summed_dec @ W_U)        # [d_t, vocab]
    return torch.stack(effects)                 # [n_layers, d_t, vocab]


# ============================================================================
# Piece 2 — build_feature_model (schema-locked)
# ============================================================================

def build_feature_model(
    layer: int,
    feat_idx: int,
    examples: list[dict],
    act_min: float,
    act_max: float,
    histogram: list[float],
    quantile_values: list[float],
    activation_frequency: float,
    top_logits: list[str],
    bottom_logits: list[str],
    n_quantiles: int = 5,
) -> dict:
    """Build a dict that validates against circuit_tracer.frontend.feature_models.Model.

    Args:
        examples: list of dicts with keys:
            - tokens: list[str]   the token strings for this context
            - acts:   list[float] per-token activation values (same len as tokens)
            - argmax: int         index of peak activation within the context
        act_min, act_max: global min/max activation for the feature
        histogram: list[float] of length n_bins, counts per bin
        quantile_values: list[float] of length n_quantiles+1 (bin edges) or n_quantiles
        activation_frequency: fraction of positions where feature fires
        top_logits, bottom_logits: token strings
        n_quantiles: number of quantile bands to bucket examples into

    Returns a plain dict suitable for json.dumps and Model.model_validate.
    """
    from clts.feature_index import cantor_pair

    # Sort examples by their max activation, descending.
    def _max_act(ex: dict) -> float:
        return max(ex["acts"]) if ex["acts"] else 0.0

    sorted_exs = sorted(examples, key=_max_act, reverse=True)

    # Bucket into n_quantiles bands (high act -> quantile 0, low -> last).
    # If fewer examples than bands, each non-empty band gets what it can.
    bands: list[list[dict]] = [[] for _ in range(n_quantiles)]
    n = len(sorted_exs)
    for i, ex in enumerate(sorted_exs):
        band_idx = min(int(i * n_quantiles / max(n, 1)), n_quantiles - 1)
        bands[band_idx].append(ex)

    # Build ExamplesQuantile dicts.
    examples_quantiles = []
    for q_idx, band in enumerate(bands):
        ex_dicts = []
        for ex in band:
            tokens = ex["tokens"]
            acts = ex["acts"]
            argmax = ex.get("argmax", int(np.argmax(acts)) if acts else 0)
            ex_dicts.append({
                "tokens_acts_list": [float(a) for a in acts],
                "train_token_ind": int(argmax),
                "is_repeated_datapoint": False,
                "tokens": list(tokens),
            })
        examples_quantiles.append({
            "quantile_name": f"quantile_{q_idx}",
            "examples": ex_dicts,
        })

    return {
        "transcoder_id": str(layer),
        "index": cantor_pair(layer, feat_idx),
        "examples_quantiles": examples_quantiles,
        "top_logits": list(top_logits),
        "bottom_logits": list(bottom_logits),
        "act_min": float(act_min),
        "act_max": float(act_max),
        "quantile_values": [float(v) for v in quantile_values],
        "histogram": [float(v) for v in histogram],
        "activation_frequency": float(activation_frequency),
    }


# ============================================================================
# Piece 3 — generate_dashboards (end-to-end)
# ============================================================================

def _build_corpus(
    data_dir: Path,
    n_per_person: int,
    context_size: int,
    n_people: int | None,
    seed: int = 0,
) -> tuple[torch.Tensor, object]:
    """Build a token tensor [n_examples, context_size] and return
    (tokens_tensor, tokenizer).

    Mirrors trainCLT.py's real corpus construction exactly:
        sampler   = BioSampler(data_dir / "people.json", fields=("birthday",), seed=0)
        tokenizer = CondensedTokenizer.from_remap_path(data_dir / "old_to_new.json")
        subset    = DiverseBioSubset(sampler, tokenizer, context_size=..., seed=0)
        rows      = subset.to_hf_dataset(n_examples, verbose=False)["input_ids"]
    """
    from util.bio_sampler import BioSampler
    from util.condensed_tokenizer import CondensedTokenizer
    from util.diverse_subset import DiverseBioSubset

    tokenizer = CondensedTokenizer.from_remap_path(data_dir / "old_to_new.json")
    sampler = BioSampler(data_dir / "people.json", fields=("birthday",), seed=seed)

    # If n_people is specified, cap the corpus at that many bios worth of tokens.
    total_people = len(sampler.people)
    effective_people = min(n_people, total_people) if n_people is not None else total_people
    # Each person contributes roughly one bio; pack into context_size rows.
    n_examples = max(1, (effective_people * n_per_person * 80) // context_size)
    n_examples = max(n_examples, 1)

    subset = DiverseBioSubset(sampler, tokenizer, context_size=context_size, seed=seed)
    rows = subset.to_hf_dataset(n_examples, verbose=False)["input_ids"]
    tokens = torch.tensor(np.array(rows), dtype=torch.long)
    return tokens, tokenizer


def generate_dashboards(
    model_dir: str | Path,
    clt_dir: str | Path,
    data_dir: str | Path,
    scan_name: str,
    features_root: str | Path,
    device: str = "cpu",
    n_per_person: int = 2,
    context_size: int = 64,
    n_people: int | None = None,
    top_k: int = 20,
    n_bins: int = 40,
    batch_rows: int = 8,
) -> dict:
    """Generate per-feature dashboards for all fired features.

    Two-pass design:
      Pass 1: scan all positions to compute per-feature act_max and fire count.
      Pass 2: collect running top-k examples per feature; build histograms.

    Writes one <cantor_pair(L,f)>.json per fired feature to
        features_root/<scan_name>/

    Returns:
        {"n_features_written": int, "out_dir": str, "total_positions": int}
    """
    from clts.clt import CrossLayerTranscoder
    from clts.evalCLT import capture_activations
    from clts.feature_index import cantor_pair
    from clts.tl_model import build_hooked_transformer

    model_dir = Path(model_dir)
    clt_dir = Path(clt_dir)
    data_dir = Path(data_dir)
    features_root = Path(features_root)

    # ---- Load model + CLT --------------------------------------------------
    model = build_hooked_transformer(str(model_dir), device=device)
    clt = CrossLayerTranscoder.load_from_dir(clt_dir)
    clt.to(device).eval()

    # ---- Unembedding matrix ------------------------------------------------
    W_U = model.W_U.detach()   # [d_model, vocab]

    # ---- Logit effects for top/bottom tokens per feature -------------------
    with torch.no_grad():
        effects = decoder_logit_effects(clt, W_U)   # [n_layers, d_t, vocab]

    N = clt.n_layers
    d_t = clt.d_transcoder

    # ---- Build tokenizer for decoding token ids -> strings -----------------
    from util.condensed_tokenizer import CondensedTokenizer
    tokenizer = CondensedTokenizer.from_remap_path(data_dir / "old_to_new.json")

    # Decode the whole (tiny) vocab ONCE into a lookup table so the hot loops
    # below never call tokenizer.decode per token -- that per-token decode in
    # nested loops was the O(features * positions * context) blow-up.
    TOP_BOTTOM = 10
    vocab_size = int(effects.shape[-1])
    id2str = [tokenizer.decode([i]) for i in range(vocab_size)]

    # Top/bottom logit tokens per feature via ONE batched topk on CPU -- no
    # per-feature GPU->CPU sync, no per-token decode.
    eff_cpu = effects.detach().to("cpu")
    k = min(TOP_BOTTOM, vocab_size)
    top_idx = eff_cpu.topk(k, dim=-1).indices                  # [N, d_t, k]
    bot_idx = eff_cpu.topk(k, dim=-1, largest=False).indices   # [N, d_t, k]
    top_logits_per = [[[id2str[int(i)] for i in top_idx[L, f]] for f in range(d_t)] for L in range(N)]
    bot_logits_per = [[[id2str[int(i)] for i in bot_idx[L, f]] for f in range(d_t)] for L in range(N)]
    print(f"[gen] logit tables ready ({N}x{d_t} features)", flush=True)

    # ---- Build corpus tokens -----------------------------------------------
    all_tokens, _ = _build_corpus(
        data_dir, n_per_person=n_per_person,
        context_size=context_size, n_people=n_people, seed=0,
    )
    all_tokens = all_tokens.to(device)
    n_rows = all_tokens.shape[0]
    total_positions = n_rows * context_size

    # ---- Pass 1: compute per-feature act_max and fire count ----------------
    # act_max_arr[L, f], fire_count[L, f]
    act_max_arr = np.zeros((N, d_t), dtype=np.float32)
    fire_count = np.zeros((N, d_t), dtype=np.int64)

    n_batches = (n_rows + batch_rows - 1) // batch_rows
    print(f"[gen] pass 1/2 over {n_rows} rows ({n_batches} batches)", flush=True)
    for start in range(0, n_rows, batch_rows):
        if (start // batch_rows) % 20 == 0:
            print(f"[gen]   pass 1 batch {start // batch_rows}/{n_batches}", flush=True)
        batch = all_tokens[start:start + batch_rows]
        with torch.no_grad():
            x_list, _ = capture_activations(model, batch)
            a_list = clt.encode(x_list)   # list of [B*T, d_t]
        for L in range(N):
            a = a_list[L].cpu().numpy()   # [B*T, d_t]
            batch_max = a.max(axis=0)
            np.maximum(act_max_arr[L], batch_max, out=act_max_arr[L])
            fire_count[L] += (a > 0).sum(axis=0)

    # ---- Pass 2: collect top-k examples per feature; build histograms -------
    # For each fired feature, keep top-k examples (by max activation in context).
    # top_k_examples[L][f] = sorted list (descending) of (max_act, tokens, acts, argmax)
    # We use a min-heap of size top_k per feature.

    import heapq

    # Only allocate heaps for features that actually fired.
    fired_mask = fire_count > 0   # [N, d_t]

    # heaps[L][f] = list of (neg_max_act, tokens, acts, argmax) — min-heap
    heaps: list[list[list | None]] = [[None] * d_t for _ in range(N)]
    for L in range(N):
        for f in range(d_t):
            if fired_mask[L, f]:
                heaps[L][f] = []

    # Accumulate histogram counts: we need act_max per feature first (done above).
    # Initialize histograms as arrays.
    hist_counts = np.zeros((N, d_t, n_bins), dtype=np.float32)

    print(f"[gen] pass 2/2 over {n_rows} rows ({n_batches} batches)", flush=True)
    for start in range(0, n_rows, batch_rows):
        if (start // batch_rows) % 20 == 0:
            print(f"[gen]   pass 2 batch {start // batch_rows}/{n_batches}", flush=True)
        batch = all_tokens[start:start + batch_rows]
        B, T = batch.shape
        # Decode each row's tokens ONCE per batch (via the vocab table) and reuse
        # across every layer/feature -- instead of decoding per (feature, row).
        row_strs = [[id2str[int(t)] for t in row] for row in batch.cpu().tolist()]
        with torch.no_grad():
            x_list, _ = capture_activations(model, batch)
            a_list = clt.encode(x_list)   # [B*T, d_t] each

        for L in range(N):
            a = a_list[L].cpu().numpy()   # [B*T, d_t]
            a_reshaped = a.reshape(B, T, d_t)

            for f in range(d_t):
                if not fired_mask[L, f]:
                    continue
                feat_max = act_max_arr[L, f]
                if feat_max <= 0:
                    continue

                # Histogram accumulation for positions.
                feat_acts_flat = a[:, f]   # [B*T]
                fired_pos = feat_acts_flat[feat_acts_flat > 0]
                if fired_pos.size > 0:
                    bins_edges = np.linspace(0.0, feat_max, n_bins + 1)
                    counts, _ = np.histogram(fired_pos, bins=bins_edges)
                    hist_counts[L, f] += counts

                # Keep the top_k examples with the HIGHEST max activation.
                # Min-heap keyed on ctx_max: heap[0] is the smallest kept, so a
                # new row is added only if it beats it. (The old code keyed on
                # -ctx_max with heappushpop, which evicted the highest and kept
                # the lowest -- the dashboards showed the least-activating text.)
                feat_ctx = a_reshaped[:, :, f]   # [B, T]
                row_max = feat_ctx.max(axis=1)   # [B]
                heap = heaps[L][f]
                for b in range(B):
                    ctx_max = float(row_max[b])
                    if ctx_max <= 0:
                        continue
                    if len(heap) >= top_k and ctx_max <= heap[0][0]:
                        continue   # can't make the top-k -- skip building the entry
                    ctx_acts = feat_ctx[b]   # [T]
                    entry = (ctx_max, row_strs[b], ctx_acts.tolist(), int(ctx_acts.argmax()))
                    if len(heap) < top_k:
                        heapq.heappush(heap, entry)
                    else:
                        heapq.heappushpop(heap, entry)

    # ---- Write dashboards --------------------------------------------------
    out_dir = features_root / scan_name
    out_dir.mkdir(parents=True, exist_ok=True)

    n_features_total = int(fired_mask.sum())
    print(f"[gen] writing dashboards for {n_features_total} fired features...", flush=True)
    n_written = 0
    for L in range(N):
        for f in range(d_t):
            if not fired_mask[L, f]:
                continue
            heap = heaps[L][f]
            if not heap:
                continue

            out_file = out_dir / f"{cantor_pair(L, f)}.json"
            if out_file.exists():        # resumable: skip features already written
                n_written += 1
                continue

            # Highest max-activation examples first.
            heap.sort(reverse=True)
            examples = []
            for _ctx_max, tok_strs, acts_list, argmax_idx in heap:
                examples.append({
                    "tokens": tok_strs,
                    "acts": acts_list,
                    "argmax": argmax_idx,
                })

            feat_max = float(act_max_arr[L, f])
            act_min_val = 0.0

            # Quantile values: n_quantiles+1 edges between 0 and feat_max.
            n_quantiles = 5
            q_vals = np.linspace(act_min_val, feat_max, n_quantiles + 1).tolist()

            hist = hist_counts[L, f].tolist()
            act_freq = float(fire_count[L, f]) / max(total_positions, 1)

            m = build_feature_model(
                layer=L,
                feat_idx=f,
                examples=examples,
                act_min=act_min_val,
                act_max=feat_max,
                histogram=hist,
                quantile_values=q_vals,
                activation_frequency=act_freq,
                top_logits=top_logits_per[L][f],
                bottom_logits=bot_logits_per[L][f],
                n_quantiles=n_quantiles,
            )

            out_file.write_text(json.dumps(m))
            n_written += 1

    return {
        "n_features_written": n_written,
        "out_dir": str(out_dir),
        "total_positions": total_positions,
    }


# ============================================================================
# CLI entry point
# ============================================================================

def main() -> None:
    from clts.storage import storage_root

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--clt-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--scan-name", required=True)
    p.add_argument(
        "--features-root", type=Path,
        default=storage_root() / "clt_features",
        help="Root dir for dashboard JSON files; defaults to STORAGE_ROOT/clt_features",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-per-person", type=int, default=2)
    p.add_argument("--context-size", type=int, default=64)
    p.add_argument("--n-people", type=int, default=None)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--n-bins", type=int, default=40)
    args = p.parse_args()

    result = generate_dashboards(
        model_dir=args.model_dir,
        clt_dir=args.clt_dir,
        data_dir=args.data_dir,
        scan_name=args.scan_name,
        features_root=args.features_root,
        device=args.device,
        n_per_person=args.n_per_person,
        context_size=args.context_size,
        n_people=args.n_people,
        top_k=args.top_k,
        n_bins=args.n_bins,
    )
    print(f"[dashboards] wrote {result['n_features_written']} features to {result['out_dir']}")
    print(f"             total_positions={result['total_positions']}")


if __name__ == "__main__":
    main()
