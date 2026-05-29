"""Validate an attribution graph by intervention (the paper's standard check).

Build a graph, take the most influential CLT feature on the target logit,
ablate it via model.feature_intervention, and report the change in the
target-logit probability. A causally-relevant feature should move the logit.

Step 0 API audit (circuit-tracer 0.4.1):

get_top_features(graph, n=10)
    -> tuple[list[tuple[int,int,int]], list[float]]
    Ranks features by compute_node_influence weighted by logit_probabilities
    (multi-hop influence on all logit targets). Returns
    (features=[(layer,pos,feat_idx), ...], scores=[...]).
    Source: circuit_tracer/utils/demo_utils.py:43

feature_intervention(inputs, interventions, ...) -> tuple[Tensor, Tensor|None]
    interventions: Sequence[tuple[layer, pos, feat_idx, value]]
    Returns (logits, activation_cache).
    Source: replacement_model_transformerlens.py:739

get_activations(inputs) -> tuple[Tensor, Tensor]
    Returns (logits, activation_cache).
    Source: replacement_model_transformerlens.py:314

graph.logit_token_ids  -- property (logit_tokens is deprecated in 0.4.1)
graph.logit_probabilities, graph.active_features, graph.selected_features
graph.adjacency_matrix, graph.logit_targets
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clts.build_attribution_graph import build_graph, default_birthday_prompt  # noqa: E402
from clts.export_tokenizer import ensure_hf_tokenizer  # noqa: E402
from clts.load_replacement_model import load_replacement_model  # noqa: E402


def ablate_top_feature_effect(
    model_dir,
    clt_dir,
    data_dir,
    scan_name,
    device="cpu",
    prompt=None,
    max_feature_nodes=4096,
):
    """Build an attribution graph, ablate the most influential CLT feature,
    and report the change in the target-logit probability.

    The "top feature" is selected by get_top_features (multi-hop influence on
    all logit targets, weighted by their probabilities), so ablating it is
    expected to produce a non-trivial shift in the target logit.

    Args:
        model_dir: Local HF Llama checkpoint directory.
        clt_dir: Trained CLT directory (safetensors + config.yaml).
        data_dir: Data directory containing old_to_new.json (+ people.json).
        scan_name: Scan ID propagated through CLT -> model -> graph.
        device: Torch device string (default "cpu").
        prompt: Input prompt string.  If None, uses default_birthday_prompt.
        max_feature_nodes: Max feature nodes during attribution (default 4096).

    Returns:
        dict with keys:
            prompt, top_feature, target_token_id,
            target_prob_before, target_prob_after, delta
    """
    if prompt is None:
        prompt = default_birthday_prompt(data_dir)

    # Build the graph in a throwaway temp dir; we only need the graph object.
    with tempfile.TemporaryDirectory(prefix="_validate_graph_") as tmp:
        out = build_graph(
            model_dir=model_dir,
            clt_dir=clt_dir,
            data_dir=data_dir,
            scan_name=scan_name,
            graph_dir=tmp,
            slug="_validate",
            prompt=prompt,
            device=device,
            max_feature_nodes=max_feature_nodes,
            verbose=False,
        )
    graph = out["graph"]

    # --- pick the most influential feature on the target logit ---------------
    # get_top_features uses compute_node_influence weighted by
    # graph.logit_probabilities (multi-hop influence over all logit targets).
    # The top-1 feature is the one whose ablation is most likely to move
    # the target logit.
    from circuit_tracer.utils.demo_utils import get_top_features

    features, scores = get_top_features(graph, n=1)
    layer, pos, feat_idx = features[0]  # (layer, pos, feat_idx) tuple

    # --- baseline logits (no intervention) -----------------------------------
    hf_tok = ensure_hf_tokenizer(data_dir)
    model = load_replacement_model(
        model_dir, clt_dir, hf_tok, scan_name, device=device
    )

    # get_activations returns (logits, activation_cache)
    logits_before, _ = model.get_activations(prompt)

    # target token = the highest-probability logit target in the graph
    target_id = int(graph.logit_token_ids[0].item())
    p_before = torch.softmax(logits_before[0, -1].float(), dim=-1)[target_id].item()

    # --- intervened logits (ablate top feature to 0.0) -----------------------
    # feature_intervention(inputs, interventions) -> (logits, activation_cache)
    # interventions: list of (layer, pos, feat_idx, value)
    logits_after, _ = model.feature_intervention(
        prompt,
        [(layer, pos, feat_idx, 0.0)],
    )
    p_after = torch.softmax(logits_after[0, -1].float(), dim=-1)[target_id].item()

    return {
        "prompt": prompt,
        "top_feature": [int(layer), int(pos), int(feat_idx)],
        "top_feature_influence_score": float(scores[0]),
        "target_token_id": target_id,
        "target_prob_before": float(p_before),
        "target_prob_after": float(p_after),
        "delta": float(p_after - p_before),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate a CLT attribution graph via feature ablation."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--clt-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--scan-name", required=True)
    parser.add_argument(
        "--device", default="cpu", help="Torch device (default: cpu)"
    )
    parser.add_argument(
        "--prompt", default=None, help="Prompt string (default: birthday recall)"
    )
    parser.add_argument(
        "--max-feature-nodes",
        type=int,
        default=4096,
        help="Max feature nodes during attribution (default: 4096)",
    )
    args = parser.parse_args()

    result = ablate_top_feature_effect(
        model_dir=args.model_dir,
        clt_dir=args.clt_dir,
        data_dir=args.data_dir,
        scan_name=args.scan_name,
        device=args.device,
        prompt=args.prompt,
        max_feature_nodes=args.max_feature_nodes,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
