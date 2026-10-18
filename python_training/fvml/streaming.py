"""Causal 1-D convolution with an explicit left-context cache (Milestone 2 /
Stage 04; Stage 06 exports exactly this module).

StreamingConv1d keeps P = (k - 1) * d - (s - 1) samples of left context
(EnCodec's causal padding_total). Output n then depends only on inputs up to
n * s + s - 1, the last sample of the chunk it closes. forward() pads P zeros on
the left; step() prepends the cache instead. Both run the same F.conv1d with the
same weights, so chunked == offline by construction (up to float reduction
order).

State is a plain tensor passed in and returned: no module buffer is mutated and
no Python-int position exists, so a traced/exported step() exposes every cache as
a graph input/output (`state_in.<name>` / `state_out.<name>`).

Composite modules thread a StreamState through their forward(x, stream): with
stream=None they run offline, with a StreamState every StreamingConv1d inside
reads its cache from it and writes the new one back. One code path serves both
modes, so a model's step() cannot drift from its forward().
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamingConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1,
                 bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0,
                         dilation=dilation, groups=groups, bias=bias)
        self.context = (kernel_size - 1) * dilation - (stride - 1)
        if self.context < 0:
            raise ValueError(f"kernel {kernel_size}, dilation {dilation} and stride {stride} give "
                             f"negative left context {self.context}; need (k-1)*d >= s-1")
        self.step_stride = stride

    def _conv(self, x):
        return F.conv1d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)

    def _check(self, x):
        if x.shape[-1] % self.step_stride:
            raise ValueError(f"input length {x.shape[-1]} is not a multiple of stride "
                             f"{self.step_stride}")

    def forward(self, x, stream=None):
        """(B, C_in, T) -> (B, C_out, T / s). With a StreamState: one streaming step."""
        if stream is not None:
            return stream.run(self, x)
        self._check(x)
        return self._conv(F.pad(x, (self.context, 0)))

    def step(self, x, cache):
        """(chunk, cache) -> (y, new_cache); chunk length must be a multiple of s."""
        self._check(x)
        buf = torch.cat([cache, x], dim=-1)
        # buf[..., -0:] would be all of buf; slice from an explicit start instead.
        return self._conv(buf), buf[..., buf.shape[-1] - self.context:]

    def init_cache(self, batch=1, device=None, dtype=None):
        return torch.zeros(batch, self.in_channels, self.context,
                           device=self.weight.device if device is None else device,
                           dtype=self.weight.dtype if dtype is None else dtype)


def named_streaming_convs(module):
    """(qualified name, conv) for every StreamingConv1d, in registration order."""
    return [(n, m) for n, m in module.named_modules() if isinstance(m, StreamingConv1d)]


def state_layout(module):
    """[(name, channels, context)] of the convs that carry state (context > 0),
    in registration order: the order of state_in.* / state_out.* in Stage 06."""
    return [(n, m.in_channels, m.context) for n, m in named_streaming_convs(module) if m.context]


def init_state(module, batch=1, device=None, dtype=None):
    return {n: m.init_cache(batch, device, dtype)
            for n, m in named_streaming_convs(module) if m.context}


def state_numel(module, batch=1):
    return sum(batch * c * p for _, c, p in state_layout(module))


def past_context_samples(module, input_stride=1):
    """How many past input samples the stateful convs reach back, summed along
    the chain (assumes the convs run in registration order, as a plain stack
    does). Each conv's context counts in units of the cumulative stride before it."""
    total, hop = 0, input_stride
    for _, m in named_streaming_convs(module):
        total += m.context * hop
        hop *= m.step_stride
    return total


class StreamState:
    """Cache plumbing for one streaming call: convs read their cache from
    `caches` and write the new one to `updated`, keyed by qualified name."""

    def __init__(self, names, caches):
        self.names = names            # {conv module: qualified name}
        self.caches = caches
        self.updated = {}

    def run(self, conv, x):
        if not conv.context:          # k=1-style convs hold no state
            conv._check(x)
            return conv._conv(x)
        name = self.names[conv]
        if name in self.updated:
            raise RuntimeError(f"{name} ran twice in one step")
        y, self.updated[name] = conv.step(x, self.caches[name])
        return y

    def result(self):
        missing = [n for n in self.caches if n not in self.updated]
        if missing:
            raise RuntimeError(f"step() did not update the cache of: {', '.join(missing)}")
        return {n: self.updated[n] for n in self.caches}


class StreamingModule(nn.Module):
    """Base for models whose step() threads StreamingConv1d caches as a dict."""

    def stream_names(self):
        names = self.__dict__.get("_stream_names")
        if names is None:
            names = {m: n for n, m in named_streaming_convs(self)}
            self.__dict__["_stream_names"] = names   # plain attribute, not a submodule
        return names

    def init_state(self, batch=1, device=None, dtype=None):
        return init_state(self, batch, device, dtype)

    def state_layout(self):
        return state_layout(self)

    def stream(self, state):
        expected = [n for n, _, _ in state_layout(self)]
        if list(state) != expected:
            missing = sorted(set(expected) - set(state))
            extra = sorted(set(state) - set(expected))
            if missing or extra:
                raise KeyError(f"state does not match the model: missing {missing}, extra {extra}")
            state = {n: state[n] for n in expected}
        return StreamState(self.stream_names(), state)
