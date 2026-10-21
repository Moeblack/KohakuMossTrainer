"""Offline sampling uses exact cache identities; no model weights or GPU needed."""

from types import SimpleNamespace

import pytest
import torch

from kohakumoss.training.build import model
from kohakumoss.training.build.loop import sample_settings
from kohakumoss.types.plans import PrecisionConfig
from kohakumoss_data.cache.tensor_cache import TextCache
from kohakumoss_data.config import DataConfig, encoder_key_id

pytestmark = pytest.mark.cpu
DEVICE = torch.device("cpu")


@pytest.fixture
def setup_cache(tmp_path, monkeypatch):
    k = {
        "TEXT_CACHE": "offline", "LATENT_CACHE": "offline", "SAMPLE_EVERY": 10,
        "SAMPLE_PROMPTS": ["positive", "", "positive"], "SAMPLE_NEGATIVE": "negative",
        "CACHE_DIR": str(tmp_path / "cache"), "CACHE_DTYPE": "bf16",
        "TEXT_ENCODER_ID": model.QwenTextEncoder.encoder_id,
        "TEXT_DEVICE": "cpu", "TEXT_ENCODE": {"dtype": "bf16", "precision": "compute", "memory": "cpu"},
        "VAE_DTYPE": "bf16", "MEMORY_PROFILE": "auto",
    }
    monkeypatch.setattr(model, "_component_path", lambda knobs, kind: "unused-model-dir" if kind == model.TEXT else None)
    cfg = DataConfig.from_knobs(model.data_knobs(k, DEVICE))
    cache = TextCache(
        str(tmp_path / "cache" / "text.db"), "offline",
        encoder_key_id(cfg.text_encoder_id, cfg.text_encoder_spec), cfg.cache_dtype,
    )
    calls = []

    def load(path, device, plan, **kwargs):
        calls.append("load")

        def encode(texts):
            calls.append(list(texts))
            return [torch.full((2, 3), 99.0) for text in texts]

        return SimpleNamespace(plan=plan, encode=encode)

    monkeypatch.setattr(model.QwenTextEncoder, "from_pretrained", load)
    monkeypatch.setattr(model, "encoder_rule", lambda *args: "test")
    return k, cache, calls


def fill(k, cache, omit=()):
    for i, text in enumerate(dict.fromkeys(model.sample_texts(k))):
        if text not in omit:
            cache.put(cache.key(text), torch.full((2, 3), float(i + 1)))
    cache.store.flush()


def test_all_hit_skips_load_and_preserves_order(setup_cache):
    k, cache, calls = setup_cache
    fill(k, cache)
    vae, text, embeds = model.build_encoders(k, DEVICE, PrecisionConfig())
    assert vae is text is None
    assert calls == []
    assert len(embeds) == 4
    assert all(torch.equal(value, cache.get(cache.key(caption))) for value, caption in zip(embeds, model.sample_texts(k)))
    assert torch.equal(embeds[0], embeds[2])


@pytest.mark.parametrize("change", ["missing-positive", "missing-negative", "missing-empty", "no-db", "id", "spec", "storage"])
def test_cache_miss_falls_back_without_partial_reuse(setup_cache, change):
    k, cache, calls = setup_cache
    omitted = {"missing-positive": ("positive",), "missing-negative": ("negative",), "missing-empty": ("",)}
    if change != "no-db":
        fill(k, cache, omitted.get(change, ()))
    if change == "id":
        k["TEXT_ENCODER_ID"] = "different-model"
    elif change == "spec":
        k["TEXT_ENCODE"] = {**k["TEXT_ENCODE"], "dtype": "fp32"}
    elif change == "storage":
        k["CACHE_DTYPE"] = "fp32"
    _, text, embeds = model.build_encoders(k, DEVICE, PrecisionConfig())
    assert text is None
    assert calls == ["load", model.sample_texts(k)]
    assert all(torch.all(value == 99) for value in embeds)


def test_online_text_still_loads_encoder(setup_cache):
    k, cache, calls = setup_cache
    fill(k, cache)
    k["TEXT_CACHE"] = "online"
    _, text, embeds = model.build_encoders(k, DEVICE, PrecisionConfig())
    assert text is not None and embeds is not None
    assert calls == ["load", model.sample_texts(k)]


def test_sampling_off_does_not_read_or_load(setup_cache, monkeypatch):
    k, cache, calls = setup_cache
    k["SAMPLE_EVERY"] = 0
    monkeypatch.setattr(model, "cached_sample_embeds", lambda *args: pytest.fail("unexpected cache read"))
    assert model.build_encoders(k, DEVICE, PrecisionConfig()) == (None, None, None)
    assert calls == []


def test_corrupt_cache_error_is_not_swallowed(setup_cache, monkeypatch):
    k, cache, calls = setup_cache
    fill(k, cache)

    def corrupt(*args):
        raise ValueError("corrupt tensor header")

    monkeypatch.setattr(model.TextCache, "key", corrupt)
    with pytest.raises(ValueError, match="corrupt tensor header"):
        model.build_encoders(k, DEVICE, PrecisionConfig())
    assert calls == []


def test_default_cache_dir_uses_dataset_path(setup_cache, tmp_path):
    k, cache, calls = setup_cache
    fill(k, cache)
    k["CACHE_DIR"] = ""
    k["DATASETS"] = [{"path": str(tmp_path / "manifest.jsonl")}]
    assert model.cached_sample_embeds(k, DEVICE) is not None
    assert calls == []


def sample_knobs(**kwargs):
    return {
        "SAMPLE_PROMPTS": ["a"] * 7, "SAMPLE_SEEDS": (0,),
        "SAMPLE_SIZE": (64, 64), "SAMPLE_STEPS": 1, "SAMPLE_CFG": 3,
        "SAMPLE_SHIFT": 5, "SAMPLE_SKIP_LAYERS": (), **kwargs,
    }


def test_sample_batch_is_explicit_not_prompt_count():
    assert sample_settings(sample_knobs())["batch"] == 4
    assert sample_settings(sample_knobs(SAMPLE_BATCH=2))["batch"] == 2


@pytest.mark.parametrize("batch", [0, -1, True, 1.5, "5"])
def test_sample_batch_invalid(batch):
    with pytest.raises(ValueError, match="SAMPLE_BATCH"):
        sample_settings(sample_knobs(SAMPLE_BATCH=batch))
