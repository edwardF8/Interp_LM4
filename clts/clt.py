"""Cross-Layer Transcoder.

Matches Anthropic circuit-tracer's CrossLayerTranscoder packed-tensor
layout so trained weights load into their tooling without conversion.

Shapes (N = n_layers, D = d_model, d_t = d_transcoder = expansion * D):
    W_enc:     [N, d_t, D]
    b_enc:     [N, d_t]
    threshold: [N, d_t]
    W_dec:     ParameterList of length N; entry i is [d_t, N - i, D]
    b_dec:     [N, D]
"""
from __future__ import annotations

import torch
import torch.nn as nn


JUMPRELU_INIT_THRESHOLD = 0.1
JUMPRELU_BANDWIDTH = 2.0


class CrossLayerTranscoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, expansion: int = 16):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_transcoder = expansion * d_model

        N, D, d_t = n_layers, d_model, self.d_transcoder

        self.W_enc = nn.Parameter(torch.empty(N, d_t, D))
        self.b_enc = nn.Parameter(torch.zeros(N, d_t))
        self.threshold = nn.Parameter(torch.full((N, d_t), JUMPRELU_INIT_THRESHOLD))
        self.b_dec = nn.Parameter(torch.zeros(N, D))

        self.W_dec = nn.ParameterList([
            nn.Parameter(torch.empty(d_t, N - i, D)) for i in range(N)
        ])

        # Kaiming-uniform init for encoder/decoder weights (matches typical
        # sae_lens init; avoids any param staying at zero from above).
        for i in range(N):
            nn.init.kaiming_uniform_(self.W_enc[i], a=5 ** 0.5)
            nn.init.kaiming_uniform_(self.W_dec[i], a=5 ** 0.5)
