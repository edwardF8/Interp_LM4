import torch
from sae_CRL.windows import span_windows, pack_bios, build_bio_corpus


def test_span_windows_count_and_current_is_last():
    acts = torch.arange(5 * 2, dtype=torch.float32).reshape(5, 2)  # [T=5, x_dim=2]
    win = span_windows(acts, valid_len=4, tau=2)                   # window_size=3
    assert win.shape == (4, 2, 3)                                  # one window per valid token
    # current token (last col) of window i must equal acts[i]
    for i in range(4):
        assert torch.allclose(win[i, :, -1], acts[i])


def test_span_windows_zero_pads_start():
    acts = torch.ones(5, 2)
    win = span_windows(acts, valid_len=4, tau=2)
    assert torch.allclose(win[0, :, :-1], torch.zeros(2, 2))       # first token: empty lookback
    assert torch.allclose(win[0, :, -1], torch.ones(2))


def test_pack_bios_prefix_pad_truncate():
    tokens, valid = pack_bios([[1, 2, 3], [4]], max_bio_len=4, bos_id=9)
    assert tokens.tolist() == [[9, 1, 2, 3], [9, 4, 9, 9]]
    assert valid.tolist() == [4, 2]


class _FakeSampler:
    def sample(self, rng):
        return {"text": "ab" * rng.randint(1, 3)}


class _FakeTok:
    bos_token_id = 0
    def encode(self, text):
        return [ord(c) for c in text]


def test_build_bio_corpus_shapes():
    tokens, valid = build_bio_corpus(_FakeSampler(), _FakeTok(), n_bios=5, max_bio_len=10, seed=0)
    assert tokens.shape == (5, 10) and valid.shape == (5,)
    assert torch.all(tokens[:, 0] == 0)
