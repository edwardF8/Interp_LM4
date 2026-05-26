"""Local Neuronpedia-style exploration for the custom SAE.

Three public functions:

  build_index_corpus  - assemble + cache the tokenized bio corpus the
                        dashboard indexes against.
  make_dashboard      - wrap sae_dashboard.SaeVisRunner; write per-feature
                        HTML panels + an index page. (Task 4.)
  steer               - causal probe: boost a feature's decoder direction
                        at a hook, report logit deltas. (Task 5.)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch


def build_index_corpus(
    sampler,
    tokenizer,
    n_per_person: int = 2,
    context_size: int = 64,
    seed: int = 0,
    people: Sequence[dict] | None = None,
    cache_path: str | Path | None = None,
) -> torch.Tensor:
    """Build the [N, T] long tensor of token ids the dashboard indexes against.

    For each person in `people` (default: sampler.people), draw `n_per_person`
    distinct template indices, render each one, tokenize with the condensed
    tokenizer, prepend the EOS token, and pad/truncate to `context_size`.

    If `cache_path` is given and exists, load + return it without recomputing.
    On a miss, save the result there.
    """
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        return torch.load(cache_path, map_location="cpu")

    if people is None:
        people = sampler.people

    rng = torch.Generator().manual_seed(seed)
    n_templates = sampler.n_templates
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id

    rows: list[list[int]] = []
    for person in people:
        # distinct templates per person (or all of them if n_per_person >= n_templates)
        k = min(n_per_person, n_templates)
        perm = torch.randperm(n_templates, generator=rng).tolist()
        for exposure_idx in perm[:k]:
            text = sampler.render(person, exposure_idx)
            ids = [eos] + tokenizer.encode(text)
            if len(ids) >= context_size:
                ids = ids[:context_size]
            else:
                ids = ids + [pad] * (context_size - len(ids))
            rows.append(ids)

    tokens = torch.tensor(rows, dtype=torch.long)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tokens, cache_path)

    return tokens


def _attach_tokenizer(model, tokenizer) -> None:
    """sae_dashboard reaches for model.tokenizer; make sure ours is there."""
    if getattr(model, "tokenizer", None) is None:
        model.tokenizer = tokenizer


def make_dashboard(
    model,
    sae,
    tokens: torch.Tensor,
    tokenizer,
    out_dir: str | Path,
    hook_name: str,
    features: Iterable[int] | None = None,
    minibatch_size_tokens: int = 128,
    minibatch_size_features: int = 256,
    verbose: bool = True,
    clone_sae: bool = True,
) -> Path:
    """Run sae_dashboard against (model, sae, tokens) and write the HTML dashboard.

    Writes a single self-contained HTML file with a dropdown navigator over the
    requested features. Open it in a browser. Returns the path.

    `features=None` runs every feature in the SAE; pass a list/range for a subset.

    `clone_sae=True` (default) deepcopies the SAE before passing it to the runner,
    since SaeVisRunner calls fold_W_dec_norm() in-place. Set to False if you don't
    care about the SAE being mutated.
    """
    from sae_dashboard.sae_vis_runner import SaeVisRunner
    from sae_dashboard.sae_vis_data import SaeVisConfig
    from sae_dashboard.data_writing_fns import save_feature_centric_vis

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _attach_tokenizer(model, tokenizer)

    if features is None:
        features = list(range(sae.cfg.d_sae))
    else:
        features = list(features)

    sae_for_dashboard = copy.deepcopy(sae) if clone_sae else sae

    device = str(next(model.parameters()).device)
    cfg = SaeVisConfig(
        hook_point=hook_name,
        features=features,
        minibatch_size_tokens=minibatch_size_tokens,
        minibatch_size_features=minibatch_size_features,
        device=device,
        verbose=verbose,
        ignore_tokens={tokenizer.pad_token_id},
    )
    runner = SaeVisRunner(cfg)
    sae_vis_data = runner.run(encoder=sae_for_dashboard, model=model, tokens=tokens)

    # separate_files=True writes one HTML per feature (~500 KB each) instead of
    # one combined file (~hundreds of MB) that crashes the browser on a full SAE.
    save_feature_centric_vis(
        sae_vis_data=sae_vis_data,
        filename=str(out_dir / "dashboard.html"),
        separate_files=True,
    )
    _write_index_html(out_dir, features)
    return out_dir / "index.html"


def _write_index_html(out_dir: Path, requested_features: list[int]) -> None:
    """Write index.html linking each per-feature HTML that got generated."""
    import re

    pat = re.compile(r"dashboard_feature_(\d+)\.html$")
    rendered = {
        int(m.group(1))
        for f in out_dir.glob("dashboard_feature_*.html")
        if (m := pat.search(f.name))
    }

    rows = []
    for feat in requested_features:
        if feat in rendered:
            rows.append(f'<li><a href="dashboard_feature_{feat}.html">feature {feat}</a></li>')
        else:
            rows.append(f'<li class="dead">feature {feat} (no panel)</li>')

    html = (
        "<!doctype html><html><head><title>SAE features</title>"
        "<style>body{font-family:system-ui;margin:2rem;line-height:1.5}"
        "ul{columns:4;column-gap:2rem;list-style:none;padding:0}"
        "li.dead{color:#999}</style></head><body>"
        f"<h1>SAE features ({len(rendered)} of {len(requested_features)} rendered)</h1>"
        "<ul>" + "".join(rows) + "</ul>"
        "</body></html>"
    )
    (out_dir / "index.html").write_text(html)


def steer(
    model,
    sae,
    tokenizer,
    text: str,
    feature_idx: int,
    scale: float,
    hook_name: str,
) -> dict:
    """Boost feature `feature_idx`'s decoder direction by `scale` at `hook_name`.

    Encodes `text`, runs a clean forward and a steered forward, and returns
    the top-5 next-token predictions from each plus the top-5 tokens whose
    logit *gained* the most under steering.
    """
    device = next(model.parameters()).device
    ids = [tokenizer.eos_token_id] + tokenizer.encode(text)
    input_tokens = torch.tensor([ids], device=device)

    # sae.W_dec is [d_sae, d_in]; pick row, broadcast across the sequence.
    direction = sae.W_dec[feature_idx].to(device)

    def steering_hook(act, hook):
        return act + scale * direction

    with torch.no_grad():
        clean_logits = model(input_tokens)[0, -1]               # [d_vocab]
        steered_logits = model.run_with_hooks(
            input_tokens, fwd_hooks=[(hook_name, steering_hook)],
        )[0, -1]

    def topk_tokens(logits, k=5):
        vals, idxs = logits.topk(k)
        return [
            {"token_id": int(i), "text": tokenizer.decode([int(i)]), "logit": float(v)}
            for v, i in zip(vals, idxs)
        ]

    delta = steered_logits - clean_logits
    return {
        "clean_top_tokens":   topk_tokens(clean_logits),
        "steered_top_tokens": topk_tokens(steered_logits),
        "delta_logits":       topk_tokens(delta),
    }


def ablate(
    model,
    sae,
    tokenizer,
    text: str,
    feature_idx: int,
    hook_name: str,
    k: int = 5,
) -> dict:
    """Zero SAE feature `feature_idx` and see how the prediction changes.

    Causal complement of `steer`: instead of boosting a feature, removes it.
    The activation is replaced with the SAE reconstruction *minus* that one
    feature (encode -> zero the feature -> decode), and compared against the
    full SAE reconstruction. Comparing recon-vs-ablated (rather than clean-vs-
    ablated) holds the SAE's reconstruction error constant, so the logit
    difference reflects *only* that feature. `clean_top_tokens` is included as
    a reference so you can see the recon is faithful before trusting the delta.

    Ablating in feature space and re-`decode`-ing also sidesteps the
    `normalize_activations` scaling: both sides go through the same decoder, so
    they share the same space (subtracting `f * W_dec[i]` from the raw
    activation would not).

    Pass `text` with the answer removed (e.g. the bio up to "...born on") so
    the last-position prediction is the token you care about. `ablated_top_
    tokens` then tells you whether that token survives without the feature, and
    `dropped` tells you which tokens the feature was holding up.
    """
    device = next(model.parameters()).device
    ids = [tokenizer.eos_token_id] + tokenizer.encode(text)
    input_tokens = torch.tensor([ids], device=device)

    def recon_hook(act, hook):
        return sae(act)                              # full recon, all features

    def ablated_hook(act, hook):
        feats = sae.encode(act).clone()             # [1, T, d_sae]
        feats[..., feature_idx] = 0.0               # drop just this feature
        return sae.decode(feats)                     # recon without it

    with torch.no_grad():
        clean_logits   = model(input_tokens)[0, -1]                       # [d_vocab]
        recon_logits   = model.run_with_hooks(
            input_tokens, fwd_hooks=[(hook_name, recon_hook)])[0, -1]
        ablated_logits = model.run_with_hooks(
            input_tokens, fwd_hooks=[(hook_name, ablated_hook)])[0, -1]

    def topk_tokens(logits, k=k):
        vals, idxs = logits.topk(k)
        return [
            {"token_id": int(i), "text": tokenizer.decode([int(i)]), "logit": float(v)}
            for v, i in zip(vals, idxs)
        ]

    return {
        "clean_top_tokens":   topk_tokens(clean_logits),    # reference (no SAE)
        "recon_top_tokens":   topk_tokens(recon_logits),    # baseline (SAE, all feats)
        "ablated_top_tokens": topk_tokens(ablated_logits),  # SAE minus feature_idx
        # tokens the feature was promoting = biggest logit drop when removed
        "dropped":            topk_tokens(recon_logits - ablated_logits),
    }


def feature_activation_stats(
    model,
    sae,
    tokens: torch.Tensor,
    hook_name: str,
    batch_size: int = 8,
) -> dict:
    """Per-feature activation statistics across `tokens`.

    For each of the SAE's `d_sae` features, computes the max activation
    across the corpus, the mean activation, and the count of token positions
    where the feature fired (>0). Useful for plotting the distribution of
    feature activity and identifying dead features.

    Returns a dict with cpu tensors:
      - max_activation:   [d_sae] float
      - mean_activation:  [d_sae] float
      - activation_count: [d_sae] int
      - n_tokens:         int  (total positions seen)
    """
    device = next(model.parameters()).device
    d_sae = sae.cfg.d_sae
    max_acts = torch.zeros(d_sae, device=device)
    sum_acts = torch.zeros(d_sae, device=device)
    counts = torch.zeros(d_sae, dtype=torch.long, device=device)
    n_tokens = 0

    with torch.no_grad():
        for i in range(0, len(tokens), batch_size):
            batch = tokens[i:i + batch_size].to(device)
            _, cache = model.run_with_cache(batch, names_filter=hook_name)
            feats = sae.encode(cache[hook_name])      # [B, T, d_sae]
            flat = feats.reshape(-1, d_sae)           # [B*T, d_sae]
            max_acts = torch.maximum(max_acts, flat.max(dim=0).values)
            sum_acts += flat.sum(dim=0)
            counts += (flat > 0).sum(dim=0)
            n_tokens += flat.shape[0]

    return {
        "max_activation":   max_acts.cpu(),
        "mean_activation":  (sum_acts / n_tokens).cpu(),
        "activation_count": counts.cpu(),
        "n_tokens":         n_tokens,
    }


def top_features_for_text(
    model,
    sae,
    tokenizer,
    text: str,
    hook_name: str,
    k: int = 10,
) -> list[dict]:
    """Find the SAE features most active when the model processes `text`.

    Runs one forward pass caching activations at `hook_name`, encodes them
    through the SAE, and returns the top-k features ranked by their max
    activation across the sequence. Useful for picking which feature to
    feed into `dla()` or `steer()`.

    Each result dict has: feature_idx, max_activation, mean_activation,
    position_argmax (token position where the feature fires hardest), and
    token_at_argmax (the decoded token at that position).
    """
    device = next(model.parameters()).device
    ids = [tokenizer.eos_token_id] + tokenizer.encode(text)
    input_tokens = torch.tensor([ids], device=device)

    with torch.no_grad():
        _, cache = model.run_with_cache(input_tokens, names_filter=hook_name)
        acts = cache[hook_name]                   # [1, T, d_in]
        features = sae.encode(acts)[0]            # [T, d_sae]
        max_acts, max_pos = features.max(dim=0)   # both [d_sae]
        mean_acts = features.mean(dim=0)          # [d_sae]
        _, top_idxs = max_acts.topk(k)

    return [
        {
            "feature_idx":     int(idx),
            "max_activation":  float(max_acts[idx]),
            "mean_activation": float(mean_acts[idx]),
            "position_argmax": int(max_pos[idx]),
            "token_at_argmax": tokenizer.decode([int(input_tokens[0, max_pos[idx]])]),
        }
        for idx in top_idxs
    ]


def features_for_token_set(
    model,
    sae,
    tokens: torch.Tensor,
    target_token_ids: Iterable[int],
    hook_name: str,
    batch_size: int = 8,
    ignore_token_ids: Iterable[int] | None = None,
    eps: float = 1e-8,
) -> dict:
    """Find SAE features most active on a class of input tokens.

    At each token position in `tokens`, encode SAE features and L1-normalize
    the resulting [d_sae] vector so its entries sum to 1 (a distribution over
    features). Average these distributions separately over positions where
    the input token is in `target_token_ids` ("target") and positions where
    it isn't ("other"). Each entry of `mean_target` then reads as "average
    share of activation mass this feature claims when a target token is
    present".

    `specificity = mean_target / (mean_other + eps)` ranks features by how
    selectively they fire on the target class vs. everything else.

    Positions whose input token is in `ignore_token_ids` are dropped from
    both buckets (pass `{tokenizer.pad_token_id}` to skip pad / leading-EOS
    positions, matching the dashboard). Positions where the SAE produced an
    all-zero vector are also skipped — there is nothing to normalize.

    Returns:
      mean_target:         [d_sae] cpu float
      mean_other:          [d_sae] cpu float
      specificity:         [d_sae] cpu float
      n_target_positions:  int
      n_other_positions:   int
    """
    device = next(model.parameters()).device
    d_sae = sae.cfg.d_sae

    target_ids = torch.tensor(
        sorted({int(t) for t in target_token_ids}), device=device, dtype=torch.long,
    )
    ignore_ids = torch.tensor(
        sorted({int(t) for t in (ignore_token_ids or ())}), device=device, dtype=torch.long,
    )

    sum_target = torch.zeros(d_sae, device=device)
    sum_other = torch.zeros(d_sae, device=device)
    n_target = 0
    n_other = 0

    with torch.no_grad():
        for i in range(0, len(tokens), batch_size):
            batch = tokens[i:i + batch_size].to(device)                  # [B, T]
            _, cache = model.run_with_cache(batch, names_filter=hook_name)
            feats = sae.encode(cache[hook_name])                         # [B, T, d_sae]
            flat_feats = feats.reshape(-1, d_sae)                        # [B*T, d_sae]
            flat_tok = batch.reshape(-1)                                 # [B*T]

            norms = flat_feats.sum(dim=1)                                # [B*T]
            valid = norms > 0
            normed = torch.zeros_like(flat_feats)
            normed[valid] = flat_feats[valid] / norms[valid].unsqueeze(1)

            in_target = torch.isin(flat_tok, target_ids)
            in_ignore = torch.isin(flat_tok, ignore_ids) if ignore_ids.numel() else torch.zeros_like(in_target)
            target_mask = in_target & valid & ~in_ignore
            other_mask = ~in_target & valid & ~in_ignore

            sum_target += normed[target_mask].sum(dim=0)
            sum_other += normed[other_mask].sum(dim=0)
            n_target += int(target_mask.sum())
            n_other += int(other_mask.sum())

    mean_target = (sum_target / max(n_target, 1)).cpu()
    mean_other = (sum_other / max(n_other, 1)).cpu()
    specificity = mean_target / (mean_other + eps)

    return {
        "mean_target":        mean_target,
        "mean_other":         mean_other,
        "specificity":        specificity,
        "n_target_positions": n_target,
        "n_other_positions":  n_other,
    }


# =============================================================================
# Bucketing framework
#
# Per-token analyses ("which features fire on token class X?") all reduce to:
# at each position, encode SAE features, L1-normalize so the [d_sae] vector
# reads as a distribution over features, then aggregate that distribution
# under some bucket key (input token id, next token id, absolute position...).
#
# `compute_bucketed_stats` runs one forward+SAE pass over a corpus and updates
# every registered Bucketer in parallel. The result is a dict of small tensors
# you can save once and slice many ways with `query_bucket`.
# =============================================================================


@dataclass
class Bucketer:
    """Maps a [B, T] batch of token ids to [B, T] bucket ids.

    Return bucket id `-1` to skip a position (e.g. `next_token` at the last
    position has no defined key). Bucket ids must otherwise lie in
    `[0, n_buckets)`.
    """
    name: str
    n_buckets: int
    fn: Callable[[torch.Tensor], torch.Tensor]


def by_input_token(vocab_size: int) -> Bucketer:
    """Bucket each position by the token id at that position."""
    return Bucketer(name="input_token", n_buckets=vocab_size, fn=lambda t: t)


def by_next_token(vocab_size: int) -> Bucketer:
    """Bucket each position by the token id at the *next* position.

    The last position of every sequence has no next token; those positions
    are skipped (bucket id -1).
    """
    def fn(tokens: torch.Tensor) -> torch.Tensor:
        nxt = torch.full_like(tokens, -1)
        nxt[:, :-1] = tokens[:, 1:]
        return nxt
    return Bucketer(name="next_token", n_buckets=vocab_size, fn=fn)


def by_prev_token(vocab_size: int) -> Bucketer:
    """Bucket each position by the token id at the *previous* position.

    The first position of every sequence is skipped (bucket id -1).
    """
    def fn(tokens: torch.Tensor) -> torch.Tensor:
        prv = torch.full_like(tokens, -1)
        prv[:, 1:] = tokens[:, :-1]
        return prv
    return Bucketer(name="prev_token", n_buckets=vocab_size, fn=fn)


def by_position(context_size: int) -> Bucketer:
    """Bucket each position by its absolute index in the sequence."""
    def fn(tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        return torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
    return Bucketer(name="position", n_buckets=context_size, fn=fn)


def compute_bucketed_stats(
    model,
    sae,
    tokens: torch.Tensor,
    hook_name: str,
    bucketers: Sequence[Bucketer],
    ignore_token_ids: Iterable[int] | None = None,
    batch_size: int = 8,
) -> dict:
    """Single forward+SAE sweep that updates every registered bucketer.

    For each position, encodes SAE features, L1-normalizes the `[d_sae]`
    vector so its entries sum to 1, then scatter-adds it into each
    bucketer's `[n_buckets, d_sae]` sum tensor under the bucketer's key.

    Positions whose input token is in `ignore_token_ids` are dropped
    globally (regardless of bucketer). Positions with an all-zero SAE
    vector are also dropped (nothing to normalize).

    Returns:
        {
          "buckets": {
            <bucketer.name>: {
              "sum":   [n_buckets, d_sae] cpu float — Σ normalized feature mass per bucket
              "count": [n_buckets]        cpu long  — # positions per bucket
            },
            ...
          }
        }

    `torch.save` the result; later passes can `torch.load` and slice with
    `query_bucket` without rerunning the model.
    """
    device = next(model.parameters()).device
    d_sae = sae.cfg.d_sae
    bucketers = list(bucketers)

    ignore_ids = torch.tensor(
        sorted({int(t) for t in (ignore_token_ids or ())}),
        device=device, dtype=torch.long,
    )

    sums = {b.name: torch.zeros(b.n_buckets, d_sae, device=device) for b in bucketers}
    counts = {b.name: torch.zeros(b.n_buckets, device=device, dtype=torch.long) for b in bucketers}

    with torch.no_grad():
        for i in range(0, len(tokens), batch_size):
            batch = tokens[i:i + batch_size].to(device)              # [B, T]
            _, cache = model.run_with_cache(batch, names_filter=hook_name)
            feats = sae.encode(cache[hook_name])                     # [B, T, d_sae]

            flat_feats = feats.reshape(-1, d_sae)
            flat_tok = batch.reshape(-1)

            norms = flat_feats.sum(dim=1)
            valid = norms > 0
            normed = torch.zeros_like(flat_feats)
            normed[valid] = flat_feats[valid] / norms[valid].unsqueeze(1)

            in_ignore = (
                torch.isin(flat_tok, ignore_ids) if ignore_ids.numel()
                else torch.zeros_like(valid)
            )
            keep = valid & ~in_ignore                                # [B*T]

            for b in bucketers:
                keys = b.fn(batch).reshape(-1)                       # [B*T]
                mask = keep & (keys >= 0)
                idx = keys[mask]
                sums[b.name].index_add_(0, idx, normed[mask])
                counts[b.name].index_add_(0, idx, torch.ones_like(idx))

    return {
        "buckets": {
            name: {"sum": sums[name].cpu(), "count": counts[name].cpu()}
            for name in sums
        }
    }


def query_bucket(
    bucket: dict,
    target_keys: Iterable[int],
    ignore_keys: Iterable[int] | None = None,
    eps: float = 1e-8,
) -> dict:
    """Slice a precomputed bucket into target vs other.

    `bucket`       — one entry from `compute_bucketed_stats(...)["buckets"]`,
                     i.e. `{"sum": [n_buckets, d_sae], "count": [n_buckets]}`.
    `target_keys`  — bucket ids that count as "target". The remaining
                     non-ignored ids form the "other" baseline.
    `ignore_keys`  — bucket ids excluded from both target and other (e.g. pad
                     or EOS ids when using `by_input_token`).

    Returns the same shape as `features_for_token_set`:
      mean_target, mean_other, specificity (= mean_target / mean_other),
      n_target_positions, n_other_positions.
    """
    sums = bucket["sum"]        # [n_buckets, d_sae]
    counts = bucket["count"]    # [n_buckets]
    n_buckets = sums.shape[0]

    target_keys = [int(k) for k in target_keys]
    ignore_keys = [int(k) for k in (ignore_keys or [])]

    target_mask = torch.zeros(n_buckets, dtype=torch.bool)
    target_mask[target_keys] = True
    ignore_mask = torch.zeros(n_buckets, dtype=torch.bool)
    if ignore_keys:
        ignore_mask[ignore_keys] = True
    other_mask = ~target_mask & ~ignore_mask

    n_target = int(counts[target_mask].sum())
    n_other = int(counts[other_mask].sum())

    mean_target = sums[target_mask].sum(dim=0) / max(n_target, 1)
    mean_other = sums[other_mask].sum(dim=0) / max(n_other, 1)
    specificity = mean_target / (mean_other + eps)

    return {
        "mean_target":        mean_target,
        "mean_other":         mean_other,
        "specificity":        specificity,
        "n_target_positions": n_target,
        "n_other_positions":  n_other,
    }


def dla(sae, model, tokenizer, feature_idx: int, k: int = 10) -> dict:
    """Direct logit attribution for SAE feature `feature_idx`.

    Projects the feature's decoder direction through the model's unembed
    matrix to estimate which output tokens it linearly promotes (top) and
    suppresses (bottom). Ignores downstream attention/MLP nonlinearities
    and layernorm scaling — this is the same panel `sae_dashboard` shows
    in each feature's HTML, exposed here so you can scan features in code.

    Returns dict with 'top' and 'bottom' lists, each of length k.
    """
    with torch.no_grad():
        direction = sae.W_dec[feature_idx].to(model.W_U.device)
        logits = direction @ model.W_U  # [d_vocab]
        top_vals, top_idxs = logits.topk(k)
        bot_vals, bot_idxs = logits.topk(k, largest=False)

    def fmt(vals, idxs):
        return [
            {"token_id": int(i), "text": tokenizer.decode([int(i)]), "logit_delta": float(v)}
            for v, i in zip(vals, idxs)
        ]

    return {"top": fmt(top_vals, top_idxs), "bottom": fmt(bot_vals, bot_idxs)}
