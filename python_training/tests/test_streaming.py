"""StreamingConv1d and StreamingConvTranspose1d (Milestone 2 / Stages 04-05):
chunked step() == offline forward(), strict causality, explicit state plumbing."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fvml.streaming import (StreamingConv1d, StreamingConvTranspose1d, StreamingModule,
                            init_state, named_streaming_convs, past_context_samples, state_layout)

COMBOS = [  # (kernel, stride, dilation)
    (7, 1, 1), (7, 1, 3), (7, 1, 9), (3, 1, 2), (1, 1, 1), (2, 1, 1),
    (4, 2, 1), (8, 4, 1), (10, 5, 1), (16, 8, 1), (5, 2, 3), (3, 2, 1),
]
TRANSPOSE_COMBOS = [  # (kernel, stride, dilation) — includes the 8*5*4*2 vocoder upsamplers
    (16, 8, 1), (10, 5, 1), (8, 4, 1), (4, 2, 1),
    (15, 5, 1), (12, 4, 1), (7, 2, 1), (4, 4, 1), (1, 1, 1), (3, 2, 2),
]


def run_chunks(conv, x, sizes):
    cache, ys, i = conv.init_cache(x.shape[0]), [], 0
    for n in sizes:
        before = cache.clone()
        y, new = conv.step(x[..., i:i + n], cache)
        assert torch.equal(cache, before), "step() must not mutate its input cache"
        assert new.shape == (x.shape[0], conv.in_channels, conv.context)
        ys.append(y)
        cache, i = new, i + n
    assert i == x.shape[-1]
    return torch.cat(ys, dim=-1)


@pytest.mark.parametrize("k,s,d", COMBOS)
def test_step_equals_forward(k, s, d):
    torch.manual_seed(k * 100 + s * 10 + d)
    for dtype, atol in ((torch.float64, 1e-12), (torch.float32, 1e-6)):
        conv = StreamingConv1d(3, 5, k, stride=s, dilation=d).to(dtype)
        assert conv.context == (k - 1) * d - (s - 1)
        x = torch.randn(2, 3, 60 * s, dtype=dtype)
        ref = conv(x)
        assert ref.shape == (2, 5, 60)
        # irregular chunking, every chunk a multiple of s (including single strides)
        sizes = [s, 3 * s, 7 * s, s, 20 * s, 2 * s, 26 * s]
        torch.testing.assert_close(run_chunks(conv, x, sizes), ref, atol=atol, rtol=0)
        torch.testing.assert_close(run_chunks(conv, x, [s] * 60), ref, atol=atol, rtol=0)


@pytest.mark.parametrize("k,s,d", COMBOS)
def test_causality(k, s, d):
    """Perturbing input j changes no output whose window closes before j (output
    n closes on n*s + s - 1), and does change the output that closes on it."""
    torch.manual_seed(1)
    conv = StreamingConv1d(2, 4, k, stride=s, dilation=d).double()
    x = torch.randn(1, 2, 40 * s, dtype=torch.float64)
    for j in (0, s - 1, 5 * s, 5 * s + s // 2, 40 * s - 1):
        x2 = x.clone()
        x2[..., j] += 1.0
        diff = (conv(x2) - conv(x)).abs().amax(dim=1)[0]
        before = torch.arange(40) * s + s - 1 < j
        if before.any():
            assert diff[before].max().item() == 0
        if d == 1 or s == 1:      # then sample j is a tap of the output closing at/after it
            assert diff[j // s].item() > 0


def test_negative_context_rejected_and_zero_context():
    with pytest.raises(ValueError, match="negative left context"):
        StreamingConv1d(1, 1, 2, stride=4)
    conv = StreamingConv1d(3, 2, 1)
    assert conv.context == 0
    x = torch.randn(1, 3, 10)
    y, cache = conv.step(x, conv.init_cache(1))
    assert cache.shape == (1, 3, 0)
    torch.testing.assert_close(y, conv(x))
    with pytest.raises(ValueError, match="multiple of stride"):
        StreamingConv1d(1, 1, 4, stride=2)(torch.zeros(1, 1, 5))


class Stack(StreamingModule):
    def __init__(self):
        super().__init__()
        self.a = StreamingConv1d(1, 4, 7)
        self.b = nn.ModuleList([StreamingConv1d(4, 4, 3, dilation=2), StreamingConv1d(4, 4, 1)])
        self.c = StreamingConv1d(4, 2, 4, stride=2)

    def forward(self, x, stream=None):
        x = self.a(x, stream)
        x = x + self.b[1](torch.relu(self.b[0](x, stream)), stream)
        return self.c(x, stream)

    def step(self, x, state):
        st = self.stream(state)
        return self(x, st), st.result()


def test_state_plumbing_through_a_module_tree():
    torch.manual_seed(0)
    m = Stack().double()
    names = [n for n, _ in named_streaming_convs(m)]
    assert names == ["a", "b.0", "b.1", "c"]                      # registration order
    assert [(n, c, p) for n, c, p in state_layout(m)] == [("a", 1, 6), ("b.0", 4, 4), ("c", 4, 2)]
    assert past_context_samples(m) == 6 + 4 + 0 + 2
    x = torch.randn(1, 1, 64, dtype=torch.float64)
    ref = m(x)
    state, ys = init_state(m, 1), []
    for i in range(0, 64, 8):
        y, state = m.step(x[..., i:i + 8], state)
        ys.append(y)
    torch.testing.assert_close(torch.cat(ys, -1), ref, atol=1e-12, rtol=0)
    with pytest.raises(KeyError, match="missing"):
        m.step(x[..., :8], {k: v for k, v in state.items() if k != "b.0"})


@pytest.mark.parametrize("k,s,d", TRANSPOSE_COMBOS)
def test_transpose_step_equals_forward_and_right_trimmed_conv(k, s, d):
    torch.manual_seed(k * 100 + s * 10 + d)
    for dtype, atol in ((torch.float64, 1e-12), (torch.float32, 1e-6)):
        conv = StreamingConvTranspose1d(3, 5, k, stride=s, dilation=d).to(dtype)
        assert conv.trim_right == (k - 1) * d - (s - 1)
        assert conv.context == (k - 1) * d // s
        x = torch.randn(2, 3, 24, dtype=dtype)
        ref = conv(x)
        assert ref.shape == (2, 5, 24 * s)
        # Must equal unpadded conv_transpose1d trimmed on the right (trim_right_ratio = 1.0)
        raw = F.conv_transpose1d(x, conv.weight, conv.bias, stride=s, dilation=d)
        torch.testing.assert_close(ref, raw[..., :24 * s], atol=atol, rtol=0)
        # Irregular chunk sizes, including single frames (1 < context when k > 2*s)
        sizes = [1, 2, 4, 1, 7, 3, 6]
        torch.testing.assert_close(run_chunks(conv, x, sizes), ref, atol=atol, rtol=0)
        torch.testing.assert_close(run_chunks(conv, x, [1] * 24), ref, atol=atol, rtol=0)


@pytest.mark.parametrize("k,s,d", TRANSPOSE_COMBOS)
def test_transpose_causality(k, s, d):
    """Perturbing input frame t changes no output before sample t * s, and does
    change output window [t * s, (t + 1) * s)."""
    torch.manual_seed(2)
    conv = StreamingConvTranspose1d(2, 4, k, stride=s, dilation=d).double()
    x = torch.randn(1, 2, 16, dtype=torch.float64)
    for t in (0, 1, 5, 15):
        x2 = x.clone()
        x2[..., t] += 1.0
        diff = (conv(x2) - conv(x)).abs().amax(dim=1)[0]
        if t > 0:
            assert diff[:t * s].max().item() == 0
        assert diff[t * s:(t + 1) * s].max().item() > 0


def test_transpose_validation_and_mixed_stack():
    with pytest.raises(ValueError, match="negative right trim"):
        StreamingConvTranspose1d(1, 1, 2, stride=4)
    with pytest.raises(ValueError, match="input length"):
        StreamingConvTranspose1d(1, 1, 4, stride=2)(torch.zeros(1, 1, 0))
    conv0 = StreamingConvTranspose1d(3, 2, 4, stride=4)
    assert conv0.context == 0 and conv0.trim_right == 0

    class UpStack(StreamingModule):
        def __init__(self):
            super().__init__()
            self.pre = StreamingConv1d(2, 4, 3)
            self.up = StreamingConvTranspose1d(4, 2, 8, stride=4)
            self.post = StreamingConv1d(2, 1, 5)

        def forward(self, x, stream=None):
            return self.post(F.leaky_relu(self.up(F.leaky_relu(self.pre(x, stream)), stream)), stream)

        def step(self, x, state):
            st = self.stream(state)
            return self(x, st), st.result()

    torch.manual_seed(7)
    m = UpStack().double()
    assert state_layout(m) == [("pre", 2, 2), ("up", 4, 1), ("post", 2, 4)]
    assert past_context_samples(m, input_stride=320) == 2 * 320 + 1 * 320 + 4 * 80
    x = torch.randn(1, 2, 12, dtype=torch.float64)
    ref = m(x)
    state, ys = m.init_state(1), []
    for n in (1, 2, 4, 1, 4):
        y, state = m.step(x[..., :n], state)
        x = x[..., n:]
        ys.append(y)
    torch.testing.assert_close(torch.cat(ys, -1), ref, atol=1e-12, rtol=0)

