"""Model-free regressions for unique effective keys and sampling text pre-cache."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kohakumoss_data.cache import build
from kohakumoss_data.cache.tensor_cache import LatentCache, TextCache
from kohakumoss_data.config import DataConfig, encoder_key_id
from kohakumoss_data.types import CatalogRow, PlanItem

pytestmark = pytest.mark.cpu


def test_latents_deduplicate_full_keys_before_decode(tmp_path, monkeypatch):
    cfg = DataConfig(cache_dir=str(tmp_path), latent_factor=16, num_workers=1)
    rows = [
        CatalogRow(id=i, locator="same-image", w=64, h=64, extra={"hash": "same"})
        for i in range(3)
    ]
    first = PlanItem(0, (64, 64), (0, 0, 64, 64), False, 0)
    variants = [
        first,
        replace(first, flip=True),
        replace(first, target_hw=(32, 64)),
        replace(first, crop_ltrb=(1, 0, 64, 64)),
    ]
    # Three caption rows share each geometry, not just a single fixed image count.
    items = [replace(item, row_id=i) for item in variants for i in range(3)]
    cache = LatentCache(str(tmp_path / "latents.db"), "offline", "vae/bf16")
    opened, encoded, progress = [], [], []

    def open_image(row_id):
        opened.append(row_id)
        return b"image"

    def encode(images, hw):
        encoded.extend(images)
        return torch.ones(int(hw.prod(dim=1).sum()), 2)

    ds = SimpleNamespace(
        cfg=cfg, catalog=rows, latents=cache, open=open_image,
        decode=lambda *args: torch.ones(3, 64, 64),
    )
    monkeypatch.setattr(build, "latent_items", lambda dataset: iter(items))
    monkeypatch.setattr(build, "record", lambda *args: None)
    # Already-cached keys still skip, while other geometries remain distinct.
    cache.put(cache.key(rows[0], first), torch.zeros(16, 2))
    assert build.build_latents(ds, encode, batch_size=2, progress=progress.append) == 3
    assert len(opened) == len(encoded) == 3
    assert progress[-1] == 3
    assert all(cache.get(cache.key(rows[item.row_id], item)) is not None for item in items)
    assert build.build_latents(ds, encode) == 0
    assert len(encoded) == 3


def test_text_extras_empty_duplicates_and_resume(tmp_path, monkeypatch):
    cfg = DataConfig(cache_dir=str(tmp_path), text_encoder_id="fake", text_encoder_spec="bf16/compute")
    cache = TextCache(
        str(tmp_path / "text.db"), "offline",
        encoder_key_id(cfg.text_encoder_id, cfg.text_encoder_spec), cfg.cache_dtype,
    )
    ds = SimpleNamespace(
        cfg=cfg, texts=cache,
        catalog=SimpleNamespace(live_ids=lambda: np.array([0, 1])),
        planner=SimpleNamespace(caption_variants=2),
        caption=lambda row, variant: f"caption-{row}",
    )
    calls = []

    def encode(texts):
        calls.extend(texts)
        return [torch.full((2, 3), float(len(text))) for text in texts]

    encode.encoder_id = cfg.text_encoder_id
    encode.encoder_spec = cfg.text_encoder_spec
    monkeypatch.setattr(build, "record", lambda *args: None)
    extra = ["positive", "custom negative", "", "positive", "caption-0"]
    assert build.build_text(ds, encode, batch_size=2, extra_texts=iter(extra)) == 5
    assert sorted(calls) == ["", "caption-0", "caption-1", "custom negative", "positive"]
    assert cache.get(cache.key("")) is not None
    assert build.build_text(ds, encode, extra_texts=extra) == 0
    assert len(calls) == 5
    # The original positional API remains supported.
    assert build.build_text(ds, encode, 2, None) == 0
    encode.encoder_spec = "fp16/compute"
    with pytest.raises(ValueError, match="spec"):
        build.build_text(ds, encode, extra_texts=extra)
    encode.encoder_spec = cfg.text_encoder_spec
    encode.encoder_id = "wrong"
    with pytest.raises(ValueError, match="encoder id"):
        build.build_text(ds, encode)
