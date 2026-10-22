"""CPU-only preflight contracts: no models, cache construction, or training."""

import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

SPEC = importlib.util.spec_from_file_location(
    "molab_preflight", Path(__file__).parents[1] / "scripts/molab_preflight.py"
)
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


class FakePlanner:
    def __init__(self):
        self.calls = []
        self.drops = {}
        self.unbatched = {}

    def epoch(self, epoch):
        self.calls.append(epoch)
        self.drops[epoch] = epoch % 2
        self.unbatched[epoch] = 0
        sizes = (3, 1) if epoch % 2 == 0 else (2, 2, 1)
        return [[SimpleNamespace(row_id=slot, caption_variant=slot % 3,
                                 joint_tokens=100 + 10 * slot)
                 for slot in range(size)] for size in sizes]

    def tokens(self, item):
        return item.joint_tokens


def config(**overrides):
    return {"STEPS": 2, "GRAD_ACC": 4, "SAVE_EVERY": 100,
            "LR": 0.0001, "PARALLEL": "none", **overrides}


def test_twenty_real_epochs_variable_batches_tails_and_recommendations():
    planner = FakePlanner()
    knobs = config()
    original = copy.deepcopy(knobs)
    report = preflight.audit_planner(planner, knobs)
    assert knobs == original
    assert planner.calls == list(range(20))
    assert len(report["epochs"]) == 20
    assert report["total"]["seen_samples"] == 90
    assert report["total"]["microbatches"] == 50
    assert report["total"]["batch_size_distribution"] == {"1": 20, "2": 20, "3": 10}
    assert report["total"]["caption_variants"] == {"0": 50, "1": 30, "2": 10}
    assert report["total"]["actual_joint_tokens"] == 9500
    assert report["total"]["mean_batch_size"] == pytest.approx(1.8)
    assert [row["seen_samples"] for row in report["epochs"]] == [4, 5] * 10
    assert [row["microbatches"] for row in report["epochs"]] == [2, 3] * 10
    assert report["recommendations"]["steps"] == 13
    assert report["recommendations"]["lr"] == pytest.approx(
        3e-4 * 9289728 / 74333952 * math.sqrt(1.8 * 4 / 4)
    )
    assert report["recommendations"]["applied"] is False
    assert "save_every" not in report["recommendations"]
    assert report["checkpoints"]["save_every"] == 100
    assert report["configured"] == {key: knobs[key] for key in ("STEPS", "SAVE_EVERY", "LR", "GRAD_ACC")}


def test_configured_budget_stops_inside_real_epoch_not_at_mean_batch():
    report = preflight.audit_planner(FakePlanner(), config())
    budget = report["configured_budget"]
    assert budget["microbatches"] == 8
    assert budget["seen_samples"] == 15
    assert budget["complete_epochs"] == 3
    assert budget["microbatches_in_partial_epoch"] == 1
    assert budget["batch_size_distribution"] == {"1": 3, "2": 3, "3": 2}


def test_8000_steps_extend_past_twenty_epochs_and_preserve_save100():
    report = preflight.audit_planner(FakePlanner(), config(STEPS=8000))
    assert len(report["epochs"]) == 20
    assert report["configured_budget"]["microbatches"] == 32000
    assert report["configured_budget"]["seen_samples"] == 57600
    assert report["checkpoints"]["configured_periodic_count"] == 80
    assert report["checkpoints"]["recommended_steps_periodic_count"] == 0
    assert "save_every" not in report["recommendations"]


def test_actual_tokens_can_account_for_seeded_text_dropout():
    indices = []

    def actual(batch, index):
        indices.append(index)
        return sum(item.joint_tokens for item in batch) - 10, 1

    report = preflight.audit_planner(FakePlanner(), config(), actual_tokens=actual)
    assert indices == list(range(50))
    assert report["total"]["planned_joint_tokens"] == 9500
    assert report["total"]["actual_joint_tokens"] == 9000
    assert report["total"]["text_dropped_samples"] == 50


def test_empty_epoch_and_invalid_accumulation_fail():
    with pytest.raises(ValueError, match="empty epoch"):
        preflight.audit_planner(SimpleNamespace(epoch=lambda epoch: []), config())
    with pytest.raises(ValueError, match="positive integers"):
        preflight.audit_planner(FakePlanner(), config(GRAD_ACC=0))
    with pytest.raises(ValueError, match="PARALLEL"):
        preflight.audit_planner(FakePlanner(), config(PARALLEL="ddp"))


def test_zero_budget_still_audits_twenty_epochs():
    report = preflight.audit_planner(FakePlanner(), config(STEPS=0, SAVE_EVERY=0))
    assert report["configured_budget"]["seen_samples"] == 0
    assert report["total"]["microbatches"] == 50
    assert report["checkpoints"]["configured_periodic_count"] == 0


def test_cache_spec_honors_explicit_compute_and_quantization():
    assert preflight._spec_matches("bf16/int8/g0/src:int8-convrot256", {}, "text")
    assert not preflight._spec_matches("bf16/int8/g0", {"TEXT_ENCODE": {"dtype": "fp16"}}, "text")
    assert not preflight._spec_matches("bf16/int8/g0", {"TEXT_ENCODE": {"precision": "compute"}}, "text")
    assert not preflight._spec_matches("bf16/int8/g0", {"TEXT_ENCODE": {"precision": "int8", "quant_group": 64}}, "text")
    assert not preflight._spec_matches("fp16", {"VAE_DTYPE": "bf16"}, "latents")
    assert not preflight._spec_matches("bf16/int8/g0", {"TEXT_ENCODER_SPEC": "bf16/compute"}, "text")


def test_cli_writes_report_without_modifying_config(tmp_path):
    path = tmp_path / "training.json"
    output = tmp_path / "metadata" / "planner-preflight.json"
    path.write_text(json.dumps(config()) + "\n", encoding="utf-8")
    before = path.read_bytes()
    report = preflight.audit_planner(FakePlanner(), config())
    with patch.object(preflight, "audit_config", return_value=report) as audit:
        preflight.main(["--config", str(path), "--output", str(output)])
    audit.assert_called_once_with(config())
    assert path.read_bytes() == before
    assert json.loads(output.read_text())["recommendations"]["steps"] == 13
    assert json.loads(output.read_text())["checkpoints"]["save_every"] == 100


def test_cli_rejects_overwriting_config(tmp_path):
    path = tmp_path / "training.json"
    path.write_text(json.dumps(config()), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(SystemExit):
        preflight.main(["--config", str(path), "--output", str(path)])
    assert path.read_bytes() == before


@pytest.fixture
def cached_options_manifest(tmp_path):
    import torch
    from PIL import Image

    from kohakumoss_data import MossDataset
    from kohakumoss_data.cache.build import build_text
    from kohakumoss_data.cache.settings import record
    from kohakumoss_data.scripting import data_config

    options = [
        [f"<|artist|>artist name<|/artist|>\n{size}" for size in ("short", "medium", "long")],
        ["short without artist", "medium without artist", "long without artist"],
        ["<|artist|>another artist<|/artist|>\nshort", "no artist medium", "no artist long"],
        ["<|artist|>  <|/artist|>\nempty", "<|artist|>unclosed", "body <|artist|>not a prefix<|/artist|>"],
    ]
    manifest = tmp_path / "train.jsonl"
    rows = []
    for index, captions in enumerate(options):
        image = f"image-{index}.png"
        Image.new("RGB", (32, 32)).save(tmp_path / image)
        rows.append({"image": image, "caption_options": captions})
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    knobs = config(
        DATASETS=[{"source": "jsonl", "path": str(manifest)}],
        CAPTION={"template": "options"}, CAPTION_VARIANTS=3,
        TARGET_AREA=32 * 32, SIZE_STEP=16, BATCHING="tokens", MAX_TOKENS=16384,
        MAX_SAMPLES=0, LATENT_CACHE="offline", TEXT_CACHE="offline", TEXT_DROP=0,
        TOKEN_BUDGET="exact", CACHE_DIR=str(tmp_path / "cache"), DATA_SCAN_WORKERS=1,
        TEXT_ENCODER_ID="test-preflight-options", TEXT_ENCODER_SPEC="bf16/compute",
        LATENT_ENCODER_SPEC="bf16",
    )
    cfg = data_config(knobs)
    dataset = MossDataset(cfg)

    def encode(captions):
        return [torch.zeros(len(caption) + 1, 2) for caption in captions]

    encode.encoder_id = cfg.text_encoder_id
    encode.encoder_spec = cfg.text_encoder_spec
    build_text(dataset, encode, batch_size=3)
    record(cfg, "latents")
    assert all(not dataset.catalog[index].artist for index in range(len(rows)))
    return knobs, manifest, options


def test_real_jsonl_cache_artist_prefixes_not_catalog_artist(cached_options_manifest):
    knobs, manifest, options = cached_options_manifest
    original = copy.deepcopy(knobs)
    before = manifest.read_bytes()
    report = preflight.audit_config(knobs)
    assert report["catalog"]["live_rows"] == 4
    assert report["catalog"]["artist_detection"] == "rendered_caption_prefix"
    assert report["catalog"]["missing_artist_rows"] == 2
    assert report["catalog"]["partial_artist_rows"] == 1
    assert report["catalog"]["missing_artist_by_variant"] == {"0": 2, "1": 3, "2": 3}
    assert any("2 live catalog rows lack a valid artist prefix" in warning for warning in report["warnings"])
    assert any("1 live catalog rows have an artist prefix in only some" in warning for warning in report["warnings"])
    assert report["total"]["seen_samples"] == 80
    assert len(report["epochs"]) == 20
    assert knobs == original
    assert manifest.read_bytes() == before
    dataset, _ = preflight.build_dataset(knobs)
    assert [dataset.caption(row, variant) for row in range(4) for variant in range(3)] == [caption for row in options for caption in row]
