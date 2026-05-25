# SAE Metrics & Hyperparameters Reference

A reference for every metric your `trainSAE.py` and `evalSAE.py` pipelines emit,
what each one means, and what counts as a "good" or "bad" value. The
final section explains the hyperparameters you can sweep over.

> **Headline triad.** The three numbers that decide whether an SAE is worth
> keeping are **L0**, **explained variance**, and **CE loss recovered**.
> Everything else is diagnostic.

---

## Part 1 — Metrics from `evalSAE.sae_eval`

These are the held-out eval metrics logged at the end of each training run
under the `final_eval/` namespace in wandb (see [trainSAE.py:225-227](../trainSAE.py#L225-L227)),
and printed by `print_report` in [evalSAE.py:101-112](../evalSAE.py#L101-L112).

### `n_tokens`
**What it is.** Total number of tokens the SAE was evaluated on
(`n_eval_sequences × context_size`).

**Good / bad.** Not a quality metric — just a sample-size check. For your setup
(`64 × 512 = 32,768` tokens) it's enough for L0 / EV / CE-recovered to be
stable, but **too small to trust the "dead features" estimate** — that needs
millions of tokens, which is why the report says "see wandb" for dead-feature %.

---

### `l0` — mean active features per token
**What it is.** The average number of SAE latents that fire (activation > 0)
per token. Computed as `(feats > 0).sum() / n_tokens`. This is the
*actual* sparsity, not the loss term.

**Good.** Roughly **20–80** for an MLP-out SAE at this model scale
(target printed in `print_report`).

**Bad.**
- **L0 < 5.** SAE is over-sparse; it's reconstructing a near-constant and
  CE-recovered will collapse.
- **L0 > 150.** Practically dense — the SAE isn't really sparse, and the
  features won't be cleanly interpretable.
- **L0 = 0 or NaN.** Training collapsed; everything is dead.

**Tradeoff.** Lower L0 ↔ harder reconstruction ↔ lower CE recovered. You're
hunting the Pareto frontier between L0 and CE recovered.

---

### `explained_variance`
**What it is.** `1 - SSE / SST`, where SSE is squared reconstruction error
and SST is total variance of the activations. Fraction of activation variance
the SAE captures at the hook (here `blocks.1.hook_mlp_out`).

**Good.** **> 0.80**, ideally > 0.90.

**Bad.**
- **< 0.5.** Reconstruction is poor; SAE is basically lossy compression.
- **Negative.** SAE is worse than predicting the mean — broken training.
- **> 0.99 with L0 > 100.** Suspiciously good — you've probably collapsed into
  dense reconstruction, not learned features.

**Note.** EV at the hook can look great while CE recovered is bad. The
activation may be mostly noise that doesn't matter for the LM's predictions,
so EV alone isn't sufficient — always read it alongside `ce_recovered`.

---

### `ce_clean`, `ce_sae`, `ce_zero`
**What they are.** Cross-entropy of next-token prediction under three
conditions, averaged across batches:
- `ce_clean` — model with no intervention (the floor).
- `ce_sae` — SAE reconstruction spliced in at the hook.
- `ce_zero` — hook activation zero-ablated (the ceiling: how bad it gets if
  the hook contributes nothing).

**Good / bad.** These are inputs, not endpoints — interpret them via
`ce_recovered` below. But two sanity checks:
- `ce_zero` should be meaningfully larger than `ce_clean`. If it isn't, the
  hook site barely matters and the SAE has nothing useful to learn.
- `ce_sae` should be between `ce_clean` and `ce_zero`. If `ce_sae > ce_zero`,
  the SAE is *worse than ablation* — almost always a sign the SAE is
  outputting something out-of-distribution.

---

### `ce_recovered` — **the headline metric**
**What it is.** `(ce_zero - ce_sae) / (ce_zero - ce_clean)`. The fraction
of the LM-performance gap (ablation vs clean) that the SAE recovers.
1.0 = SAE recon is as good as the clean activation; 0.0 = no better than
zero-ablation.

This is the wandb sweep objective ([trainSAE.py:103](../trainSAE.py#L103)).

**Good.** **> 98%** for a small model like this one
(target printed in `print_report`). > 95% is acceptable.

**Bad.**
- **< 90%.** The SAE is throwing away too much information that the rest of
  the model needs.
- **Negative.** `ce_sae > ce_zero` — reconstruction is worse than ablation
  (see above).
- **> 100%.** Possible but rare — the SAE is denoising. Treat with suspicion;
  often means `ce_clean` and `ce_zero` are too close to be informative.

---

### `frac_features_active`
**What it is.** Fraction of SAE latents (`d_sae` total) that fired at least
once across the eval set.

**Good.** **> 0.9** is healthy at this eval-set size.

**Bad.**
- **< 0.5.** Many features are dead — capacity is being wasted. Lower
  `l0_coefficient` or increase `dead_feature_window`.
- **= 1.0 with low L0.** Fine.
- **= 1.0 with high L0.** Suspicious — features may be firing on noise.

**Caveat.** "Active at least once on 32k tokens" is a weak threshold. The
real dead-feature percentage requires far more tokens, which `sae_lens`
tracks during training (see `sparsity/dead_features` below).

---

### `d_sae`
**What it is.** Total SAE feature count = `d_model × sae_mult`. Not a
quality metric — just metadata so you can compute `frac_features_active`
in absolute terms (`frac × d_sae` = number of features ever active).

---

## Part 2 — Metrics from `sae_lens` training (wandb dashboard)

`sae_lens` logs these during training every `wandb_log_frequency` steps
(set to 30 in your config). Names below are the wandb chart names.

### Loss components

#### `losses/mse_loss`
**What it is.** Mean squared reconstruction error between the SAE output
and the input activation.

**Good / bad.** Should decrease monotonically through training (modulo noise).
Plateauing early = LR too low or capacity too small. Spiking late = LR too
high, or `l0_coefficient` warm-up just kicked in.

---

#### `losses/l0_loss` (or `losses/l1_loss` in non-JumpReLU SAEs)
**What it is.** The sparsity penalty. For JumpReLU with
`jumprelu_sparsity_loss_mode="tanh"`, this is a smoothed approximation of
the L0 count, scaled by `l0_coefficient`.

**Good / bad.** Should decrease as the SAE learns to use fewer features
per token. If it stays flat while `mse_loss` drops, the SAE is solving
reconstruction by activating *everything*.

---

#### `losses/auxiliary_reconstruction_loss` / `losses/pre_act_loss`
**What it is.** The pre-activation regularizer (scaled by
`pre_act_loss_coefficient = 3e-6`). Pushes pre-threshold activations
toward zero to discourage features from sitting just above threshold.

**Good / bad.** Small contribution to total loss. If it dominates, raise
the coefficient downward.

---

#### `losses/overall_loss`
**What it is.** Weighted sum of the above. The number the optimizer
actually minimizes.

**Good / bad.** Trend matters more than absolute value. Compare across
runs only if hyperparameters that affect loss magnitude (e.g.
`l0_coefficient`, `d_sae`) are the same.

---

### Sparsity / feature usage

#### `metrics/l0`
**What it is.** Live L0 (same definition as eval-set L0 above) on training
batches.

**Good / bad.** Same thresholds as eval L0. Should rise during the
`l0_warm_up_steps` window, then settle. If it keeps climbing past warm-up,
your `l0_coefficient` is too low.

---

#### `metrics/mean_log10_feature_sparsity`
**What it is.** Mean of `log10(activation_frequency)` across features
(activation frequency = fraction of tokens the feature fires on).

**Good.** Typically **between -5 and -3** for a healthy MLP SAE (features
fire on 0.001%–0.1% of tokens). Lower = more selective features.

**Bad.**
- **> -2.** Features fire on > 1% of tokens — likely polysemantic /
  non-specific.
- **< -7.** Features fire so rarely they may be effectively dead.

---

#### `sparsity/dead_features`
**What it is.** Count of features that haven't fired within
`dead_feature_window` steps (500 in your config).

**Good.** **0** or very small fraction of `d_sae`.

**Bad.** Steadily growing dead count = capacity loss. `sae_lens` has a
resampling mechanism (`feature_sampling_window`); if it's running and dead
features still accumulate, lower `l0_coefficient` or raise `lr`.

---

#### `sparsity/below_1e-5`, `sparsity/below_1e-6`
**What they are.** Fraction of features that fire on fewer than 1e-5 /
1e-6 of tokens. The "soft dead" buckets — distinct from `dead_features`,
which uses the harder window-based threshold.

**Good / bad.** A small `below_1e-5` count is normal (rare features are
fine and often the most interpretable). Large and growing = capacity
slipping toward dead.

---

#### `sparsity/mean_passes_since_fired`
**What it is.** Average number of training steps since each feature last
fired.

**Good / bad.** Should hover well below `dead_feature_window`. Climbing =
features going dormant; precursor to feature death.

---

### CE recovered (during training)

#### `metrics/CE_loss_score`
**What it is.** sae_lens's name for CE recovered, computed periodically
(every `eval_every_n_wandb_logs × wandb_log_frequency = 600` steps in
your config) on a sample of training data. Same formula as
`ce_recovered` in eval.

**Good / bad.** Same thresholds as `ce_recovered`. Use the held-out
`final_eval/ce_recovered` for actual model selection — this training-time
version is on the training distribution.

---

#### `metrics/ce_loss_with_sae` / `ce_loss_without_sae` / `ce_loss_with_ablation`
**What they are.** Same three numbers as `ce_sae` / `ce_clean` / `ce_zero`
in eval — the components of CE recovered. Useful for debugging when
`CE_loss_score` looks weird.

---

### Weights diagnostics

#### `weights/W_dec_norms`, `weights/W_enc_norms`
**What they are.** Histograms / summary stats of decoder and encoder
column norms.

**Good.** Decoder norms close to 1 (sae_lens normalizes them); encoder
norms in a tight band.

**Bad.** Heavy-tailed distributions, or a handful of features with norms
much larger than the rest = those features are dominating reconstruction.

---

### JumpReLU-specific

#### `jumprelu/threshold` (mean / histogram)
**What it is.** The learnable per-feature threshold below which the
feature outputs zero. Initialized at `jumprelu_init_threshold = 0.1`.

**Good / bad.** A spread of thresholds across features is expected and
healthy. All thresholds collapsing to 0 = JumpReLU has degenerated into
ReLU; collapsing to a large value = features all dying.

---

### Training progress

#### `details/n_training_tokens`, `details/n_training_steps`
Bookkeeping — how far through training. Used for x-axis on charts.

#### `details/current_learning_rate`
LR after the warm-up / decay schedule. Sanity check that the scheduler
is doing what you expect.

---

## Part 3 — Hyperparameters

Defined in [trainSAE.py:31-38](../trainSAE.py#L31-L38) and swept in
[trainSAE.py:100-115](../trainSAE.py#L100-L115).

### Swept hyperparameters (the ones that matter most)

#### `l0_coefficient` *(swept: 2.0, 5.0, 10.0)*
The weight on the L0 sparsity penalty. **The most important knob.**

- **Too low.** L0 climbs, features become polysemantic, CE recovered looks
  great but interpretability collapses.
- **Too high.** Features die, L0 drops below useful, CE recovered tanks.
- **Tuning.** Sweep until L0 lands in the 20–80 target band at convergence.

---

#### `sae_mult` *(swept: 8, 16)* — expansion factor
`d_sae = d_model × sae_mult`. Determines SAE capacity.

- **Higher.** More features → easier to be sparse, more potential
  monosemanticity, more compute, more dead features.
- **Lower.** Forced to be denser, risk of polysemantic features.
- **Tuning.** Start at 8–16 for small models, scale up if dead features
  stay low and you still want finer-grained features.

---

#### `lr` *(swept: 3e-5, 1e-4)*
Adam learning rate. Affects how fast (and how stably) the SAE converges.

- **Too low.** Slow convergence, may not reach good L0 in budget.
- **Too high.** Loss spikes, features die in early training.
- **Tuning.** SAEs are typically less LR-sensitive than LMs; 1e-4 to 1e-5
  is the usual range.

---

#### `epochs` *(swept: 50; default 30)*
Number of passes over the training dataset.

- **Tuning.** Increase until `final_eval/ce_recovered` plateaus. Watch
  `sparsity/dead_features` — late-training feature death undoes gains.

---

### Fixed hyperparameters (less commonly tuned)

#### `n_examples` *(default 10,000)*
Number of bio sequences sampled per epoch. With `context_size=512`,
that's ~5M tokens per epoch.

#### `context_size` *(default 512)*
Sequence length. Should match (or fit within) the LM's training context.

#### `batch_size` *(4096 tokens)*
Tokens per optimizer step. Set in [trainSAE.py:41](../trainSAE.py#L41).
Bigger batches = smoother gradients but slower per-step.

#### `total_training_tokens` *(derived)*
`epochs × n_tokens(n_examples)`. The total budget the LR scheduler
plans around.

---

### JumpReLU regularizer hyperparameters

#### `jumprelu_sparsity_loss_mode = "tanh"`
Use a tanh-smoothed surrogate for the non-differentiable L0 count.
Alternatives include `"l0_step"` (straight-through estimator).

#### `jumprelu_tanh_scale = 4.0`
Sharpness of the tanh surrogate. Higher = closer to a true L0 count
but harder to optimize.

#### `jumprelu_bandwidth = 2.0`
Width of the straight-through-estimator window around the threshold.

#### `jumprelu_init_threshold = 0.1`
Initial value for every feature's learnable threshold.

#### `pre_act_loss_coefficient = 3e-6`
Weight on the pre-activation auxiliary loss (see `losses/pre_act_loss`).

---

### Schedule & warm-up hyperparameters

#### `l0_warm_up_steps` *(total_steps / 10)*
Linearly ramp `l0_coefficient` from 0 to its target over this many
steps. Stops early sparsity pressure from killing features before
reconstruction is reasonable.

#### `lr_warm_up_steps` *(total_steps / 50)*
LR ramp from 0 → `lr`.

#### `lr_decay_steps` *(total_steps / 20)*
LR decay window at the end of training. With
`lr_scheduler_name="constant"`, this is a cosine-ish tail-off.

#### `feature_sampling_window = 1000`
Steps between dead-feature resampling passes.

#### `dead_feature_window = 500`
A feature is considered dead if it hasn't fired in this many steps.

#### `dead_feature_threshold = 1e-4`
Activation magnitude below which a feature is treated as "not firing"
for the dead-feature counter.

---

### Optimizer hyperparameters

#### `adam_beta1 = 0.9`, `adam_beta2 = 0.999`
Standard Adam betas. Rarely worth tuning for SAEs.

---

### Activation pipeline hyperparameters

#### `normalize_activations = "expected_average_only_in"`
Pre-normalize activations going *into* the SAE by their expected
average norm (no output denormalization). Stabilizes training across
hook sites with different activation scales.

#### `n_batches_in_buffer = 64`
How many batches of LM activations to keep in the shuffle buffer.
Bigger = better shuffling, more memory.

#### `store_batch_size_prompts = 16`
Prompts processed at a time when filling the activation buffer.

#### `hook_name = "blocks.1.hook_mlp_out"`
Where in the LM you're extracting activations from. The MLP output of
layer 1 in your 4-layer model. **Not really a hyperparameter** — it's
the choice of *what to interpret*, and every SAE is specific to one hook.

---

## Part 4 — Quick interpretation flowchart

When a run finishes, scan the eval metrics in this order:

1. **`ce_recovered` ≥ 0.98?** If not — the SAE isn't useful for downstream
   interp regardless of how good the other numbers look.
2. **`l0` in [20, 80]?** If not — adjust `l0_coefficient` (higher to lower
   L0, lower to raise it).
3. **`explained_variance` ≥ 0.80?** If not — reconstruction is poor;
   consider raising `sae_mult` or training longer.
4. **`frac_features_active` ≥ 0.9?** If not — too many dead features.
   Lower `l0_coefficient`, raise `lr`, or increase `dead_feature_window`.
5. **Check wandb `sparsity/dead_features` curve** — even if eval looks fine,
   a growing dead-feature trajectory means the SAE is losing capacity.

Only after this triage should you start looking at individual features
for interpretability.
