# sae_dashboard 0.8.0 API notes

Package located at: `/Users/efmac/Code/Project Code/CRL-Interp/.venv/lib/python3.13/site-packages/sae_dashboard`

---

## Main runner

- **Class:** `sae_dashboard.sae_vis_runner.SaeVisRunner`
- **Import:** `from sae_dashboard.sae_vis_runner import SaeVisRunner`
- **`__init__` signature:**
  ```python
  def __init__(self, cfg: SaeVisConfig) -> None:
  ```
  Stores `cfg`, sets `self.device = cfg.device`, `self.dtype = DTYPES[cfg.dtype]`,
  and creates `cfg.cache_dir` if set.

- **Entry method:**
  ```python
  @torch.inference_mode()
  def run(
      self,
      encoder: SAE,
      model: HookedSAETransformer,
      tokens: Int[Tensor, "batch seq"],
  ) -> SaeVisData:
  ```
  NOTE: argument order is `encoder, model, tokens` (NOT `model, sae, tokens`).
  Returns a `SaeVisData` object.

---

## Config dataclass

- **Class:** `SaeVisConfig`
- **Import:** `from sae_dashboard.sae_vis_data import SaeVisConfig`
- **Required fields (no default):**
  - `hook_point: str` — TransformerLens hook point name (e.g. `"hook_resid_post"`)
  - `features: Iterable[int]` — feature indices to visualize

- **Optional fields we care about:**
  | field | default | notes |
  |-------|---------|-------|
  | `minibatch_size_features` | `256` | feature batch size (lower if OOM) |
  | `minibatch_size_tokens` | `64` | token batch size for fwd passes |
  | `device` | `"cpu"` | torch device string |
  | `dtype` | `"float32"` | must be a key in `sae_lens.config.DTYPE_MAP` |
  | `ignore_tokens` | `set()` | token ids to exclude (e.g. pad/BOS) |
  | `ignore_positions` | `[]` | position indices to mask out |
  | `verbose` | `False` | prints progress bars and a time-log table |
  | `seed` | `0` | random seed |
  | `cache_dir` | `None` | `Path` to cache intermediate activations |
  | `use_dfa` | `False` | attention DFA analysis — keep False for MLP SAEs |
  | `feature_centric_layout` | `SaeVisLayoutConfig.default_feature_centric_layout()` | layout config |

- **Minimal construction:**
  ```python
  cfg = SaeVisConfig(
      hook_point="blocks.0.hook_resid_post",
      features=list(range(n_features)),
      device="cpu",
  )
  ```

---

## Tokenizer methods called

The `sae_dashboard` code accesses `model.tokenizer` (i.e. the tokenizer attached to the
`HookedSAETransformer` object). The table below lists every call found by grepping the
package source, excluding the `neuronpedia/` subpackage (which we are not using).

| method / attribute | called from (file:line) | what it does | CondensedTokenizer has it? |
|--------------------|-------------------------|--------------|----------------------------|
| `.vocab` | `utils_fns.py:227` | `{v: k for k, v in tokenizer.vocab.items()}` — builds id→string reverse map | **NO — needs shim** |
| `.tokenize(prompt)` | `data_parsing_fns.py:373`, `data_writing_fns.py:140` | returns `list[str]` of token strings (used only by `save_prompt_centric_vis` and `get_prompt_data`) | **NO — needs shim** |
| `.encode(prompt, return_tensors="pt")` | `data_parsing_fns.py:374` | used only by `get_prompt_data` (prompt-centric path) | yes (but see note below) |

**Methods NOT called by sae_dashboard (confirmed by grep):**
`decode`, `batch_decode`, `convert_ids_to_tokens`, `convert_tokens_to_ids`, `get_vocab`

### Shims needed

**1. `.vocab` property (CRITICAL — called in the feature-centric path)**

`get_decode_html_safe_fn` in `utils_fns.py` does:
```python
vocab_dict = {v: k for k, v in tokenizer.vocab.items()}
```
This expects `.vocab` to be a `dict[str, int]` (token string → id).
`CondensedTokenizer` has no `.vocab` attribute. This is called by `save_feature_centric_vis`,
so it is required for ALL dashboard output.

Shim: add a property to `CondensedTokenizer` (or patch it at runtime):
```python
@property
def vocab(self) -> dict[str, int]:
    # Map: new_id -> gpt2_str, then invert for str -> new_id
    gpt2_vocab = self.gpt2.get_vocab()  # str -> gpt2_id
    return {
        tok_str: self.old_to_new[gpt2_id]
        for tok_str, gpt2_id in gpt2_vocab.items()
        if gpt2_id in self.old_to_new
    }
```

**2. `.tokenize(prompt)` (only for prompt-centric vis)**

Called by `save_prompt_centric_vis` and `get_prompt_data`. If we only use
`save_feature_centric_vis`, this is **not needed**. If we want prompt-centric vis, add:
```python
def tokenize(self, text: str) -> list[str]:
    gpt2_ids = self.gpt2(text, add_special_tokens=False)["input_ids"]
    valid = [gid for gid in gpt2_ids if gid in self.old_to_new]
    return self.gpt2.convert_ids_to_tokens(valid)
```

**3. `.encode` return-type issue**

`data_parsing_fns.py:374` calls:
```python
model.tokenizer.encode(prompt, return_tensors="pt")
```
`CondensedTokenizer.encode` does NOT accept `return_tensors` — it returns `list[int]`.
The `.to(device)` call on its return value will crash. This only affects `get_prompt_data`
(prompt-centric path).

Fix: either add `return_tensors` support to `encode`, or use `__call__` instead. Suggest
adding `return_tensors: str | None = None` to `encode` or delegating to `__call__`.

---

## HTML output

`save_feature_centric_vis` is a **module-level function** (not a method on `SaeVisData`):

```python
from sae_dashboard.data_writing_fns import save_feature_centric_vis

save_feature_centric_vis(
    sae_vis_data: SaeVisData,
    filename: str | Path,          # must have .html suffix; parent dir must exist
    feature_idx: int | None = None, # default starting feature; None = first in dict
    include_only: list[int] | None = None,  # subset of features to include
    separate_files: bool = False,   # if True, writes one file per feature
) -> None
```

- When `separate_files=False` (default): writes a single `filename.html` containing
  all features as a navigable dropdown dashboard.
- When `separate_files=True`: writes `{stem}_feature_{N}.html` for each feature N.
- The HTML file is self-contained (embeds all JS/CSS from the package's `js/` and `css/`
  directories), but it does pull D3 and Plotly from CDN.
- There is **no built-in index/browse page** — navigation is via a dropdown inside the
  single combined HTML.

---

## Minimal calling pattern

```python
from sae_dashboard.sae_vis_runner import SaeVisRunner
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.data_writing_fns import save_feature_centric_vis

cfg = SaeVisConfig(
    hook_point="blocks.0.hook_resid_post",   # must match SAE's hook point
    features=list(range(n_features)),         # or a subset
    device="cpu",
    verbose=True,
)

runner = SaeVisRunner(cfg)

# NOTE: argument order is encoder, model, tokens — not model, sae, tokens
sae_vis_data = runner.run(
    encoder=sae,          # sae_lens SAE object
    model=model,          # HookedSAETransformer
    tokens=tokens,        # Int[Tensor, "batch seq"]
)

# model.tokenizer.vocab must exist before calling this
save_feature_centric_vis(
    sae_vis_data=sae_vis_data,
    filename="out/features.html",
)
```

---

## Anything weird worth flagging

1. **`.vocab` is the blocking shim.** `get_decode_html_safe_fn` (called by
   `save_feature_centric_vis`) inverts `.vocab` to get a `{new_id: str}` lookup.
   Without it, calling `save_feature_centric_vis` raises `AttributeError`. Add the
   property to `CondensedTokenizer` before Task 4 is runnable end-to-end.

2. **`runner.run` argument order.** The task spec says `make_dashboard(model, sae, ...)`,
   but `SaeVisRunner.run` takes `(encoder, model, tokens)` — SAE first, then model.
   Task 4's wrapper must swap them.

3. **`SaeVisData.model` must not be `None` when saving.** `save_feature_centric_vis`
   asserts `sae_vis_data.model is not None`. `SaeVisRunner.run` sets this at the end
   of its loop, so this should be fine as long as the run completes normally.

4. **`encoder.fold_W_dec_norm()` is called unconditionally** (unless the encoder is a
   `CLTLayerWrapper`). This mutates the SAE in-place. If the SAE is reused after
   dashboard generation, its weights will have changed. Clone or re-load if needed.

5. **`ignore_tokens` should include your `pad_token_id`** to avoid pad positions
   distorting the activation histograms. Pass `ignore_tokens={tokenizer.pad_token_id}`.

6. **Prompt-centric vis is a separate code path** (`save_prompt_centric_vis` +
   `get_prompt_data`). It calls `.tokenize()` and needs the fixed `.encode()` return
   type. If Task 4 only targets feature-centric vis, neither shim is needed.

7. **No per-feature save method on `SaeVisData`.** The task spec mentions
   `data.save_feature_centric_vis(filename=..., feature_idx=...)` — this method does
   NOT exist on `SaeVisData`. The correct call is the module-level function
   `save_feature_centric_vis(sae_vis_data, filename, feature_idx=...)`.
