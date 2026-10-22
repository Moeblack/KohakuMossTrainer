"""Style LoKr knobs reach the real LyCORIS constructors; meta tensors avoid GPU/RAM use."""

import inspect
from functools import wraps

import pytest
import torch
from lycoris.modules import LokrModule
from lycoris.wrapper import LycorisNetwork
from torch import nn

from kohakumoss.adapters.build.config import accepted_kwargs, config_from_knobs
from kohakumoss.adapters.build.network import build_targets

pytestmark = pytest.mark.cpu

ALGO_KWARGS = {
    "factor": 8,
    "full_matrix": True,
    "decompose_both": False,
    "use_tucker": False,
}
SHAPES = {
    "q": (3072, 3072),
    "k": (3072, 3072),
    "v": (3072, 3072),
    "o": (3072, 3072),
    "ffn_up_gate": (16384, 3072),
    "ffn_down": (3072, 8192),
}


def style_config(dim=99999, **algo_kwargs):
    return config_from_knobs(
        {
            "ALGO": "lokr",
            "DIM": dim,
            "ALPHA": 1,
            "PRESET": "attn-mlp",
            "TRAIN_NORMS": False,
            "ALGO_KWARGS": {**ALGO_KWARGS, **algo_kwargs},
            "KERNELS": {"lycoris": "torch"},
        }
    )


@pytest.fixture
def backbone():
    """The 42 joint blocks' exact projection geometry, plus excluded modules."""
    with torch.device("meta"):
        model = nn.Module()
        model.layers = nn.ModuleList()
        for _ in range(42):
            block = nn.Module()
            block.proj = nn.ModuleDict(
                {name: nn.Linear(inp, out, bias=False) for name, (out, inp) in SHAPES.items()}
            )
            block.norm1 = nn.LayerNorm(3072)
            block.norm2 = nn.LayerNorm(3072)
            model.layers.append(block)
        model.video_embedder = nn.Linear(64, 3072)
        model.final_linear_video = nn.Linear(3072, 64)
        model.text_embedder = nn.Module()
        model.text_embedder.proj = nn.Linear(3072, 3072)
    return model.requires_grad_(False)


def test_full_matrix_is_the_public_lycoris_keyword():
    config = style_config()
    assert config.validate() == []
    assert config.algo_kwargs == ALGO_KWARGS
    assert "full_matrix" in accepted_kwargs("lokr")
    assert "fullmatrix" not in accepted_kwargs("lokr")
    assert any("fullmatrix" in error for error in style_config(fullmatrix=True).validate())


@pytest.mark.parametrize("dim", [99999, 4], ids=["style-dim99999", "explicit-full-matrix-low-rank"])
def test_full_matrix_forwarding_and_exact_trainable_count(backbone, monkeypatch, dim):
    # A small rank also ensures full_matrix is effective, rather than relying on
    # LyCORIS's automatic full-matrix fallback for an oversized lora_dim.
    config = style_config(dim)
    assert (config.algo, config.dim, config.alpha) == ("lokr", dim, 1.0)
    assert config.preset == "attn-mlp"
    assert config.train_norms is False
    assert config.validate() == []

    network_calls = []
    module_calls = []
    network_init = LycorisNetwork.__init__
    module_init = LokrModule.__init__
    module_signature = inspect.signature(module_init)

    @wraps(network_init)
    def record_network(self, *args, **kwargs):
        network_calls.append(dict(kwargs))
        network_init(self, *args, **kwargs)

    @wraps(module_init)
    def record_module(self, *args, **kwargs):
        bound = module_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        module_calls.append(dict(bound.arguments))
        module_init(self, *args, **kwargs)

    monkeypatch.setattr(LycorisNetwork, "__init__", record_network)
    monkeypatch.setattr(LokrModule, "__init__", record_module)
    with torch.device("meta"):
        targets, norms = build_targets(backbone, config)

    assert len(network_calls) == 1
    forwarded = network_calls[0]
    assert forwarded["network_module"] == "lokr"
    assert forwarded["lora_dim"] == dim
    assert forwarded["alpha"] == 1.0
    assert forwarded["train_norm"] is False
    for key, value in ALGO_KWARGS.items():
        assert forwarded[key] == value
    assert len(module_calls) == len(targets) == 42 * 6
    for forwarded in module_calls:
        assert forwarded["lora_dim"] == dim
        assert forwarded["alpha"] == 1.0
        for key, value in ALGO_KWARGS.items():
            assert forwarded[key] == value

    assert norms == []
    assert {target.path for target in targets} == {
        f"layers.{layer}.proj.{name}" for layer in range(42) for name in SHAPES
    }
    total = 0
    for target in targets:
        lyco = target.lyco
        out, inp = SHAPES[target.path.rsplit(".", 1)[-1]]
        assert isinstance(lyco, LokrModule)
        assert tuple(target.linear.weight.shape) == (out, inp)
        assert lyco.full_matrix is True
        assert lyco.use_w1 and lyco.use_w2
        assert not lyco.tucker
        # Full/full LoKr normalizes its internal alpha to rank: the effective
        # scale is one even though the requested constructor alpha is one.
        assert lyco.scale == 1.0
        parameters = dict(lyco.named_parameters())
        assert set(parameters) == {"lokr_w1", "lokr_w2"}
        assert tuple(parameters["lokr_w1"].shape) == (8, 8)
        assert tuple(parameters["lokr_w2"].shape) == (out // 8, inp // 8)
        assert all(p.device.type == "meta" and p.requires_grad for p in parameters.values())
        total += sum(p.numel() for p in parameters.values())
    assert total == 42 * (4 * (64 + 384 * 384) + (64 + 2048 * 384) + (64 + 384 * 1024))
    assert total == 74_333_952
    assert not any(p.requires_grad for p in backbone.parameters())
