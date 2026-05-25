"""Generate the Neuronpedia-style HTML dashboard for a trained SAE.

Heavy compute (model + SAE forward over ~100k bios). Paths below are
hardcoded to the Bridges-2 layout so this script runs under sbatch with
no CLI arguments. Edit the CONFIG block when you want a different SAE
or a feature subset.

Submit:
    sbatch submit_job_psc.sh computeInference.py

Then on your laptop:
    scp -r friedmae@bridges2:Interp_LM4/inference ~/Code/Project\\ Code/CRL-Interp/Interp_LM4/
and open the notebook's Option A cell.
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import LlamaForCausalLM
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.loading_from_pretrained import convert_llama_weights

from bio_sampler import BioSampler
from condensed_tokenizer import CondensedTokenizer
from evalSAE import load_sae
from sae_explorer import build_index_corpus, feature_activation_stats, make_dashboard


# ============================================================================
# CONFIG — edit these to point at a different SAE / data / model.
# ============================================================================

# Bridges-2 paths. The model + bios live in the data_storage area; the SAE
# checkpoint is under the project's sae/ tree.
MODEL_DIR = Path("/jet/home/friedmae/data_storage/LM4_Results/runResults/bioS_N-Bd_final_grid/20260520-134455/grid/grid-L4-H6/final")
DATA_DIR  = Path("/jet/home/friedmae/data_storage/LM4_Results/Data/bioS_N-Bd_final_grid")
SAE_PATH  = Path("/jet/home/friedmae/Interp_LM4/sae/sweep-n66crzzw/mult16_l05_lr3e-05_ep50_n10000/final")

HOOK_NAME = "blocks.1.hook_mlp_out"

# Index corpus shape. 2 templates * 50k people = 100k rows; 64 tokens covers
# every bio with margin.
N_PER_PERSON = 2
CONTEXT_SIZE = 64

# Which features to render. None = every feature in the SAE (~6144 for
# d_model=384 * sae_mult=16). For fast iteration set to e.g. list(range(100))
# or [42, 137, 999].
FEATURES: list[int] | None = None

# sae_dashboard internals — tweak if you OOM.
MINIBATCH_TOKENS = 128
MINIBATCH_FEATURES = 256

SEED = 0

# Output goes to inference/<sae_name>/. We strip the trailing "final" so the
# directory name is the SAE's hyperparameter string.
OUT_DIR = Path("inference") / (SAE_PATH.parent.name if SAE_PATH.name == "final" else SAE_PATH.name)

# ============================================================================


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(model_dir: Path, device: str) -> HookedTransformer:
    """Load HF Llama and convert to HookedTransformer. Same wiring as trainSAE.py."""
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
    for label, p in [("MODEL_DIR", MODEL_DIR), ("DATA_DIR", DATA_DIR), ("SAE_PATH", SAE_PATH)]:
        if not p.exists():
            raise FileNotFoundError(f"{label} does not exist: {p}\nEdit the CONFIG block in computeInference.py.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    remap_path = DATA_DIR / "old_to_new.json"
    people_path = DATA_DIR / "people.json"

    device = pick_device()
    print(f"[device]  {device}")

    print(f"[model]   {MODEL_DIR}")
    model = build_model(MODEL_DIR, device)
    print(f"          n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}, "
          f"d_vocab={model.cfg.d_vocab}")

    print(f"[data]    {DATA_DIR}")
    tokenizer = CondensedTokenizer.from_remap_path(remap_path)
    sampler = BioSampler(people_path, fields=("birthday",), seed=SEED)
    print(f"          {len(sampler.people):,} people, {sampler.n_templates} templates each")

    print(f"[sae]     {SAE_PATH}")
    sae = load_sae(SAE_PATH, device)
    print(f"          d_sae={sae.cfg.d_sae}, hook={HOOK_NAME}")

    print(f"[corpus]  n_per_person={N_PER_PERSON}, context_size={CONTEXT_SIZE}")
    tokens = build_index_corpus(
        sampler, tokenizer,
        n_per_person=N_PER_PERSON,
        context_size=CONTEXT_SIZE,
        seed=SEED,
        cache_path=OUT_DIR / "index_corpus.pt",
    )
    print(f"          tokens shape: {tuple(tokens.shape)}")

    n_features = len(FEATURES) if FEATURES is not None else sae.cfg.d_sae
    print(f"[render]  {n_features} features -> {OUT_DIR / 'dashboard.html'}")

    out_html = make_dashboard(
        model, sae, tokens.to(device), tokenizer,
        out_dir=OUT_DIR,
        hook_name=HOOK_NAME,
        features=FEATURES,
        minibatch_size_tokens=MINIBATCH_TOKENS,
        minibatch_size_features=MINIBATCH_FEATURES,
    )

    # Per-feature stats over the full corpus — small file (~50 KB), drives the
    # notebook's Overview histogram cell and the activations-per-latent bar plot.
    print(f"[stats]   computing per-feature stats over {tokens.shape[0]} bios")
    stats = feature_activation_stats(model, sae, tokens.to(device), HOOK_NAME)
    stats_path = OUT_DIR / "feature_stats.pt"
    torch.save(stats, stats_path)
    n_dead = int((stats["activation_count"] == 0).sum())
    print(f"          dead features: {n_dead} / {sae.cfg.d_sae}  ->  {stats_path}")

    print()
    print(f"DONE: {out_html}")
    print()
    print("To view on your laptop:")
    print(f"  scp -r friedmae@bridges2:{OUT_DIR.resolve()} <local-repo>/inference/")


if __name__ == "__main__":
    main()
