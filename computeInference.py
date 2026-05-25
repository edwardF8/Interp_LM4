"""Generate the Neuronpedia-style HTML dashboard for a trained SAE.

Heavy compute (model + SAE forward over ~100k bios). Meant to run on a GPU
machine where the model + bios + SAE are reachable, not on a laptop. Writes
a self-contained HTML file under `inference/<sae_name>/`. Copy that directory
to your laptop and open `dashboard.html` in a browser.

Local laptop (paths match the notebook defaults):
    python computeInference.py sae_runs/sweep-n66crzzw/mult16_l05_lr3e-05_ep50_n10000/final

Bridges-2 HPC:
    python computeInference.py \\
        /jet/home/friedmae/Interp_LM4/sae/sweep-n66crzzw/mult16_l05_lr3e-05_ep50_n10000/final \\
        --model-dir /jet/home/friedmae/data_storage/LM4_Results/runResults/bioS_N-Bd_final_grid/20260520-134455/grid/grid-L4-H6/final \\
        --data-dir  /jet/home/friedmae/data_storage/LM4_Results/Data/bioS_N-Bd_final_grid

Iterate fast on a feature subset:
    python computeInference.py <sae_path> --features 0-99
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import LlamaForCausalLM
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.loading_from_pretrained import convert_llama_weights

from bio_sampler import BioSampler
from condensed_tokenizer import CondensedTokenizer
from evalSAE import load_sae
from sae_explorer import build_index_corpus, make_dashboard


# Local laptop defaults — match the paths set in analyzingSAE.ipynb. Override
# on HPC via --model-dir / --data-dir.
DEFAULT_MODEL_DIR = Path("model/BD_llama_6heads_1epoch_4layers")
DEFAULT_DATA_DIR = Path("data/BD_llama_inital")
DEFAULT_HOOK = "blocks.1.hook_mlp_out"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_features(spec: str) -> list[int] | None:
    """'all' -> None; '0-99' -> [0..99]; '0,5,10' -> [0,5,10]."""
    if spec == "all":
        return None
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]


def derive_out_dir(sae_path: Path) -> Path:
    """inference/<sae_path.parent.name if name == 'final' else sae_path.name>."""
    label = sae_path.parent.name if sae_path.name == "final" else sae_path.name
    return Path("inference") / label


def build_model(model_dir: Path, device: str) -> HookedTransformer:
    """Load the HF Llama checkpoint and convert to HookedTransformer.

    Same wiring as trainSAE.py / analyzingSAE.ipynb — kept self-contained so
    this script doesn't import either of those.
    """
    hf_model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    hf_cfg = hf_model.config
    tl_cfg = HookedTransformerConfig(
        n_layers=hf_cfg.num_hidden_layers,
        d_model=hf_cfg.hidden_size,
        d_head=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
        n_heads=hf_cfg.num_attention_heads,
        d_mlp=hf_cfg.intermediate_size,
        d_vocab=hf_cfg.vocab_size,
        n_ctx=hf_cfg.max_position_embeddings,
        act_fn="silu",
        normalization_type="RMS",
        gated_mlp=True,
        positional_embedding_type="rotary",
        rotary_base=int(getattr(hf_cfg, "rope_theta", 10000.0)),
        rotary_dim=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
        final_rms=True,
        tie_word_embeddings=hf_cfg.tie_word_embeddings,
        initializer_range=hf_cfg.initializer_range,
        n_key_value_heads=hf_cfg.num_key_value_heads,
        device=device,
    )
    model = HookedTransformer(tl_cfg)
    model.load_state_dict(convert_llama_weights(hf_model, tl_cfg), strict=False)
    model.to(device).eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("sae_path", type=Path,
                    help="Path to SAE checkpoint dir (containing config + weights).")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                    help=f"HF model dir. Default: {DEFAULT_MODEL_DIR}")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help=f"Bios dir with people.json + old_to_new.json. Default: {DEFAULT_DATA_DIR}")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output dir. Default: inference/<sae_name>/")
    ap.add_argument("--n-per-person", type=int, default=2,
                    help="Distinct templates per person (default: 2).")
    ap.add_argument("--context-size", type=int, default=64,
                    help="Tokens per row in the index corpus (default: 64).")
    ap.add_argument("--features", default="all",
                    help="'all', '0-99', or '0,5,10' (default: all).")
    ap.add_argument("--hook", default=DEFAULT_HOOK,
                    help=f"Hook point (default: {DEFAULT_HOOK}).")
    ap.add_argument("--minibatch-tokens", type=int, default=128,
                    help="sae_dashboard token minibatch size (default: 128).")
    ap.add_argument("--minibatch-features", type=int, default=256,
                    help="sae_dashboard feature minibatch size (default: 256).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.sae_path.exists():
        ap.error(f"SAE path does not exist: {args.sae_path}")
    if not args.model_dir.exists():
        ap.error(f"Model dir does not exist: {args.model_dir} (set --model-dir)")
    if not args.data_dir.exists():
        ap.error(f"Data dir does not exist: {args.data_dir} (set --data-dir)")

    remap_path = args.data_dir / "old_to_new.json"
    people_path = args.data_dir / "people.json"
    if not remap_path.exists():
        ap.error(f"Missing {remap_path}")
    if not people_path.exists():
        ap.error(f"Missing {people_path}")

    out_dir = args.out or derive_out_dir(args.sae_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"[device]  {device}")
    print(f"[model]   {args.model_dir}")
    model = build_model(args.model_dir, device)
    print(f"          n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}, "
          f"d_vocab={model.cfg.d_vocab}")

    print(f"[data]    {args.data_dir}")
    tokenizer = CondensedTokenizer.from_remap_path(remap_path)
    sampler = BioSampler(people_path, fields=("birthday",), seed=args.seed)
    print(f"          {len(sampler.people):,} people, {sampler.n_templates} templates each")

    print(f"[sae]     {args.sae_path}")
    sae = load_sae(args.sae_path, device)
    print(f"          d_sae={sae.cfg.d_sae}, hook={args.hook}")

    print(f"[corpus]  n_per_person={args.n_per_person}, context_size={args.context_size}")
    tokens = build_index_corpus(
        sampler, tokenizer,
        n_per_person=args.n_per_person,
        context_size=args.context_size,
        seed=args.seed,
        cache_path=out_dir / "index_corpus.pt",
    )
    print(f"          tokens shape: {tuple(tokens.shape)}")

    features = parse_features(args.features)
    n_features = len(features) if features is not None else sae.cfg.d_sae
    print(f"[render]  {n_features} features -> {out_dir / 'dashboard.html'}")

    out_html = make_dashboard(
        model, sae, tokens.to(device), tokenizer,
        out_dir=out_dir,
        hook_name=args.hook,
        features=features,
        minibatch_size_tokens=args.minibatch_tokens,
        minibatch_size_features=args.minibatch_features,
    )

    print()
    print(f"DONE: {out_html}")
    print()
    print(f"To view on your laptop:")
    print(f"  scp -r <user>@<host>:{out_dir.resolve()} ~/Downloads/")
    print(f"  open ~/Downloads/{out_dir.name}/dashboard.html")


if __name__ == "__main__":
    main()
