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
from pathlib import Path
from typing import Iterable, Sequence

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


def collect_activations(
    model,
    sae,
    tokens: torch.Tensor,
    hook_name: str,
    batch_size: int = 8,
) -> torch.Tensor:
    """Run the SAE over `tokens` and return the full activation tensor.

    Returns a `[N*T, d_sae]` float tensor on CPU. Memory: 4 bytes per cell,
    so 200 bios x 64 tokens x 6144 features ~ 300 MB (or ~150 MB as float16).
    """
    device = next(model.parameters()).device
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(tokens), batch_size):
            batch = tokens[i:i + batch_size].to(device)
            _, cache = model.run_with_cache(batch, names_filter=hook_name)
            feats = sae.encode(cache[hook_name])              # [B, T, d_sae]
            chunks.append(feats.reshape(-1, sae.cfg.d_sae).cpu().float())
    return torch.cat(chunks, dim=0)


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
