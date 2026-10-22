"""Authored caption options stay one row and follow the per-seen planner variant."""

import json
import pickle
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from kohakumoss_data import DataConfig, MossDataset
from kohakumoss_data.cache.build import build_text, caption_set
from kohakumoss_data.caption import build_template
from kohakumoss_data.dataset import first_captions
from kohakumoss_data.types import CatalogRow

pytestmark = pytest.mark.cpu

OPTIONS = ["short", "a medium caption", "a much longer caption with more detail"]


@pytest.fixture
def options_config(tmp_path):
    Image.new("RGB", (32, 32)).save(tmp_path / "image.png")
    manifest = tmp_path / "captions.jsonl"
    manifest.write_text(
        json.dumps({"image": "image.png", "caption_options": OPTIONS}) + "\n",
        encoding="utf-8",
    )
    return DataConfig(
        datasets=[{"source": "jsonl", "path": str(manifest)}],
        caption={"template": "options"},
        caption_variants=3,
        target_area=32 * 32,
        size_step=16,
        batching="count",
        batch_size=1,
        latent_cache="off",
        text_cache="off",
        text_drop=0,
        cache_dir=str(tmp_path / "cache"),
        data_scan_workers=1,
        seed=21,
    )


def test_one_image_one_row_and_explicit_options(options_config):
    ds = MossDataset(options_config)
    assert len(ds.catalog) == 1
    assert ds.catalog[0].json["caption_options"] == OPTIONS
    assert [ds.caption(0, variant) for variant in range(3)] == OPTIONS
    assert first_captions(options_config.datasets[0], options_config.caption) == OPTIONS[:1]
    template = build_template("options")
    for variant, expected in enumerate(OPTIONS):
        for seed in (0, 7, 99):
            rng = np.random.default_rng(seed)
            assert template.render(ds.catalog[0], rng, variant=variant) == expected
            assert rng.random() == np.random.default_rng(seed).random()
    assert template.render(ds.catalog[0], np.random.default_rng(0)) == OPTIONS[0]


def test_per_seen_variants_and_deterministic_resume(options_config):
    ds = MossDataset(options_config)
    plans = [ds.planner.plan(index, 0, 1) for index in range(96)]
    variants = [plan.items[0].caption_variant for plan in plans]
    assert set(variants) == {0, 1, 2}
    assert all(plan.items[0].row_id == 0 for plan in plans)
    # A fresh planner and a serialized planner replay arbitrary resume indices.
    fresh = MossDataset(options_config)
    restored = pickle.loads(pickle.dumps(ds.planner))
    for index in (71, 2, 95, 32, 0):
        assert fresh.planner.plan(index, 0, 1) == plans[index]
        assert restored.plan(index, 0, 1) == plans[index]
    # Repeated occurrences of this same row within an epoch also draw independently.
    repeated = [
        ds.planner.item(0, 0, position, ds.planner.targets[0]).caption_variant
        for position in range(96)
    ]
    assert set(repeated) == {0, 1, 2}


def test_all_options_precached_and_exact_selected_length(options_config):
    cfg = replace(
        options_config, text_cache="offline", token_budget="exact",
        text_encoder_id="test-caption-options",
    )
    ds = MossDataset(cfg)
    assert caption_set(ds) == sorted(["", *OPTIONS])
    with pytest.raises(KeyError, match="variant 2"):
        ds.text_len(0, 2)
    encoded = []

    def encode(captions):
        encoded.extend(captions)
        return [torch.zeros(len(caption) + 1, 2) for caption in captions]

    encode.encoder_id = cfg.text_encoder_id
    encode.encoder_spec = cfg.text_encoder_spec
    assert build_text(ds, encode, batch_size=2) == 4
    assert encoded == sorted(["", *OPTIONS])
    assert build_text(ds, encode) == 0
    ds = MossDataset(cfg)
    for variant, caption in enumerate(OPTIONS):
        assert ds.text_len(0, variant) == len(caption) + 1
        cached = ds.texts.get(ds.texts.key(caption))
        assert cached.shape[0] == ds.text_len(0, variant)
    for index in range(32):
        plan = ds.planner.plan(index, 0, 1)
        item = plan.items[0]
        expected = len(OPTIONS[item.caption_variant]) + 1
        image_tokens, text_tokens = ds.planner.sizes(item)
        assert text_tokens == expected
        assert plan.token_budget == image_tokens + expected


@pytest.mark.parametrize("name", ["tags", "nl", "mix", "json"])
@pytest.mark.parametrize("caption_nl", ["", "natural language"])
def test_legacy_two_argument_render_unchanged(name, caption_nl):
    template = build_template(name)
    row = CatalogRow(
        id=0, locator="image.png", w=32, h=32,
        tags=["long_hair", "smile", "blue_eyes"], artist="artist_name",
        caption_nl=caption_nl, json={"layout": "portrait"},
    )
    for seed in range(8):
        legacy = template.render(row, np.random.default_rng(seed))
        for variant in range(3):
            assert template.render(row, np.random.default_rng(seed), variant) == legacy


def test_composed_templates_forward_variant():
    row = CatalogRow(id=0, locator="x", w=32, h=32, json={"caption_options": OPTIONS})
    for config in (
        {"template": "nl", "fallback": {"template": "options"}},
        {"template": "mix", "tags": {"template": "options"}},
    ):
        template = build_template(config)
        assert template.render(row, np.random.default_rng(0), variant=2) == OPTIONS[2]


@pytest.mark.parametrize("options", [None, [], "not a list", ["text", 12]])
def test_invalid_options_rejected(options):
    row = CatalogRow(id=0, locator="x", w=32, h=32, json={"caption_options": options})
    with pytest.raises(ValueError, match="caption_options"):
        build_template("options").render(row, np.random.default_rng(0))


@pytest.mark.parametrize("variant", [-1, 3])
def test_out_of_range_variant_is_not_silently_wrapped(variant):
    row = CatalogRow(id=0, locator="x", w=32, h=32, json={"caption_options": OPTIONS})
    with pytest.raises(IndexError, match="caption variant"):
        build_template("options").render(row, np.random.default_rng(0), variant)
