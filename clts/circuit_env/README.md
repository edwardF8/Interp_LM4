# circuit-tracer venv

Isolated environment for building CLT attribution graphs. Never installed into
the training/SAE env (circuit-tracer pins `transformers<=4.57.3`).

## Create (Mac and PSC both)

```bash
python3.11 -m venv clts/.venv-ct
clts/.venv-ct/bin/pip install -U pip
clts/.venv-ct/bin/pip install -r clts/circuit_env/requirements.txt

# circuit-tracer from the decoderesearch fork:
git clone https://github.com/decoderesearch/circuit-tracer.git /tmp/circuit-tracer
clts/.venv-ct/bin/pip install /tmp/circuit-tracer
```

## Import gate (run after install; must print OK)

```bash
clts/.venv-ct/bin/python - <<'PY'
from transformer_lens.loading_from_pretrained import convert_llama_weights
from circuit_tracer.replacement_model.replacement_model_transformerlens import TransformerLensReplacementModel
from circuit_tracer.transcoder.cross_layer_transcoder import load_clt
from circuit_tracer import attribute
from circuit_tracer.utils.create_graph_files import create_graph_files
from circuit_tracer.frontend.local_server import serve
from circuit_tracer.frontend.feature_models import Model as FeatureModel
print("OK")
PY
```
