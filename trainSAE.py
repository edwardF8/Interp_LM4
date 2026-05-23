from pathlib import Path
from datasets import Dataset

from sae_lens import SAE, HookedSAETransformer, LoggingConfig
from sae_lens import LanguageModelSAERunnerConfig, LanguageModelSAETrainingRunner, JumpReLUTrainingSAEConfig
import numpy as np
import torch
from transformers import LlamaForCausalLM
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.loading_from_pretrained import convert_llama_weights # type: ignore

from bio_sampler import BioSampler
from condensed_tokenizer import CondensedTokenizer
from diverse_subset import DiverseBioSubset # type: ignore

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

### SAMPLER Hyperparameters
n_examples = 10_000      # the "portion"  -> 5.12M tokens
epochs     = 10
context_size  = 512
sae_mult=16
device = pick_device()
dtype = torch.float32

subModelPath = "runs/bioS_N-Bd_final_grid/20260520-134455/grid/grid-L4-H6/final"
MODEL_DIR  = Path(f"../Training_On_LM4/{subModelPath}")

subDataDir = "cache/bioS_N-Bd_final_grid"
DATA_DIR   = Path(f"../Training_On_LM4/{subDataDir}")

REMAP_PATH = DATA_DIR / "old_to_new.json"
TOKENS_PATH = DATA_DIR / "bios_postreduce.bin"
SAE_seed = 0

saeName = f"bioS_NM_BD_layer_2_{sae_mult}_{epochs}_{n_examples}"
SAE_RUN_DIR = f"sae/{saeName}"

modelName = f"bioS_NM_BD_4Layer_6Heads"

checkpoint_path = f"{SAE_RUN_DIR}/checkpoints"
output_path     = f"{SAE_RUN_DIR}/final"    

hf_model = LlamaForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=dtype)
hf_model.eval()

# Build the TL config from the HF config so dims match our custom
hf_cfg = hf_model.config
tl_cfg = HookedTransformerConfig(
        n_layers = hf_cfg.num_hidden_layers,
        d_model = hf_cfg.hidden_size,
        d_head = hf_cfg.hidden_size // hf_cfg.num_attention_heads,
        n_heads = hf_cfg.num_attention_heads,
        d_mlp= hf_cfg.intermediate_size,
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

# Pre-tokenize everything you feed the model; don't attach the tokenizer.
# TL only calls into the tokenizer for `model(str)` / `to_tokens` / `to_string`,
# none of which we use — we always pass token ids directly.
state_dict = convert_llama_weights(hf_model, tl_cfg)
model = HookedTransformer(tl_cfg)
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()
print(f"Loaded on {device}: n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}, "
        f"n_heads={model.cfg.n_heads}, d_vocab={model.cfg.d_vocab}")


sampler   = BioSampler(DATA_DIR / "people.json", fields=("birthday",), seed=SAE_seed)
tokenizer = CondensedTokenizer.from_remap_path(REMAP_PATH)   # <- REMAP_PATH finally used
subset    = DiverseBioSubset(sampler, tokenizer, context_size=context_size, seed=SAE_seed)
sae_dataset = subset.to_hf_dataset(n_examples)

batch_size = 4096
total_training_tokens = epochs * subset.n_tokens(n_examples)
total_training_steps  = total_training_tokens // batch_size


cfg = LanguageModelSAERunnerConfig(
        sae=JumpReLUTrainingSAEConfig(
            l0_coefficient=5.0, # Sparsity penalty coefficient
            jumprelu_sparsity_loss_mode="tanh",
            jumprelu_tanh_scale=4.0, # default value
            jumprelu_bandwidth=2.0,
            jumprelu_init_threshold=0.1,
            pre_act_loss_coefficient=3e-6,
            # Anthropic's settings assume normalized activations
            normalize_activations="expected_average_only_in",
            l0_warm_up_steps= (total_training_steps//50),
            d_in=model.cfg.d_model, # must match your hook point
            d_sae=model.cfg.d_model * sae_mult,
            ),
        # Data generation (Model + training distribiton)
        model_name=modelName, 
        hook_name="blocks.1.hook_mlp_out",
        dataset_path="bioS_Name_BD",  # tokenized language dataset.
        is_dataset_tokenized=True,
        disable_concat_sequences=True,
        prepend_bos=False,  # you should use whatever the base model was trained with
        streaming=True,  # we could pre-download the token dataset if it was small.
        train_batch_size_tokens=batch_size,
        context_size=context_size,
        #
        # Activations store
        n_batches_in_buffer=64,
        training_tokens=total_training_tokens,
        store_batch_size_prompts=16,
        #
        # Training hyperparameters (standard)
        lr=5e-5,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr_scheduler_name="constant",  # controls how the LR warmup / decay works
        lr_warm_up_steps= (total_training_steps//50),  # avoids large number of initial dead features
        lr_decay_steps=(total_training_steps//4),  # helps avoid overfitting
        # Training hyperparameters (resampling)
        feature_sampling_window=2000,  # how often we resample dead features
        dead_feature_window=1000,  # size of window to assess whether a feature is dead
        dead_feature_threshold=1e-4,  # threshold for classifying feature as dead, over window
        # Logging / evals
        logger=LoggingConfig(
            log_to_wandb=True, 
            wandb_project="interpLM4",
            wandb_log_frequency=30,
            eval_every_n_wandb_logs=20,
            ),
        # Misc.
        device=str(device),
        seed=SAE_seed,
n_checkpoints=5,
    output_path=output_path,
    checkpoint_path=checkpoint_path,
    dtype="float32",
)

runner = LanguageModelSAETrainingRunner(
        cfg,
        override_dataset=sae_dataset,   # <- the wrapped .bin
        override_model=model,           # <- your already-loaded HookedTransformer
        )

sae = runner.run()

# ---------------------------------------------------------------------------
# Post-training eval: quick L0 / explained-variance / CE-recovered report on a
# held-out sample, so every run self-reports whether the SAE is worth keeping.
# Eval logic lives in evalSAE.py; run `python evalSAE.py` to re-check any
# already-trained SAE without retraining.
# ---------------------------------------------------------------------------
from evalSAE import sae_eval, print_report, load_sae

# Load the SAE we just saved (an inference SAE) rather than the in-memory
# training object, so the eval path matches `python evalSAE.py` exactly.
eval_sae = load_sae(output_path, device)

# Held-out sample: a different subset seed -> identity/template packing the
# SAE never saw during training.
eval_subset = DiverseBioSubset(sampler, tokenizer, context_size=context_size,
                               seed=SAE_seed + 1)
eval_rows   = eval_subset.to_hf_dataset(64, verbose=False)["input_ids"]
eval_tokens = torch.tensor(np.array(eval_rows), dtype=torch.long, device=device)

print_report(sae_eval(model, eval_sae, eval_tokens, cfg.hook_name))
