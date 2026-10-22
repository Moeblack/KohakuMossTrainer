"""CPU-only cache-stage and worker failure regressions; no models or HF calls."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from kohakumoss_data import DataConfig, MossDataset
from kohakumoss_data.cache.build import latent_items
from tests.test_molab_train import checkpoint, config, runner

pytestmark = pytest.mark.cpu
ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("order", [("latents", "text"), ("text", "latents")])
def test_cache_stages_bootstrap_exact_training_without_text_cache(tmp_path, monkeypatch, order):
    """Run the real script mains/builders with tiny CPU fake encoders, twice."""
    Image.new("RGB", (32, 32)).save(tmp_path / "image.png")
    captions = ["short", "medium caption", "long caption with details"]
    catalog = tmp_path / "captions.jsonl"
    catalog.write_text(json.dumps({"image": "image.png", "caption_options": captions}) + "\n")
    knobs = {
        "DATASETS": [{"source": "jsonl", "path": str(catalog)}],
        "CACHE_DIR": str(tmp_path / "cache"), "WEIGHTS": str(tmp_path),
        "VAE": "fake-vae", "TEXT_ENCODER": "fake-text", "VAE_DTYPE": "bf16",
        "TEXT_ENCODER_ID": "fake-text", "TEXT_ENCODE": {"dtype": "bf16"},
        "LATENT_CACHE": "offline", "TEXT_CACHE": "offline", "CACHE_DTYPE": "bf16",
        "RESOLUTION_MODE": "resize", "TARGET_AREA": 1024 * 1024, "SIZE_STEP": 16,
        "UPSCALE": "never", "CROP": "center", "FLIP": False,
        "CAPTION": {"template": "options"}, "CAPTION_VARIANTS": 3,
        "BATCHING": "tokens", "TOKEN_BUDGET": "exact", "MAX_TOKENS": 16384,
        "MAX_SAMPLES": 0, "MAX_TEXT_TOKENS": 4096, "GRAD_ACC": 4,
        "PLANNER_KWARGS": {"drop_last": False}, "OVERLONG": "drop_tokens",
        "PRESET": "attn-mlp", "SAVE_EVERY": 100, "STEPS": 8000,
        "NUM_WORKERS": 1, "DATA_SCAN_WORKERS": 1,
    }
    training = tmp_path / "training.json"
    training.write_text(json.dumps(knobs))
    original = training.read_bytes()
    c = {"PROJECT_DIR": str(ROOT), "LATENT_ENCODE_BATCH": 8, "TEXT_ENCODE_BATCH": 128}
    encoded = {"latents": [], "text": []}
    datasets = []

    def image_encode(images, hw):
        encoded["latents"].extend(images)
        return torch.ones(int(hw.prod(dim=1).sum()), 2)

    image_encode.encoder_spec = "bf16"

    def text_encode(texts):
        encoded["text"].extend(texts)
        return [torch.ones(len(text) + 1, 2) for text in texts]

    text_encode.encoder_id = "fake-text"
    text_encode.encoder_spec = "bf16/compute"
    original_namespace = runner.script_namespace

    def script_namespace(path):
        ns = original_namespace(path)
        encode = image_encode if path.name == "cache_latents.py" else text_encode
        ns["build_encoder"] = lambda *args: encode

        def dataset(cfg):
            ds = MossDataset(cfg)
            datasets.append(ds)
            assert cfg.token_budget == "estimate"
            assert cfg.batching == "tokens" and cfg.max_tokens == 16384
            return ds

        ns["MossDataset"] = dataset
        return ns

    monkeypatch.setattr(runner, "script_namespace", script_namespace)
    for name in order:
        runner.stage(c, name, tmp_path)
    assert len(encoded["latents"]) == 1
    assert sorted(encoded["text"]) == sorted(["", *captions])
    for name in order:
        runner.stage(c, name, tmp_path)
    assert len(encoded["latents"]) == 1
    assert len(encoded["text"]) == 4  # Existing cache keys are reused.
    assert training.read_bytes() == original
    train_cfg = DataConfig.from_knobs({**knobs, "LATENT_ENCODER_SPEC": "bf16", "TEXT_ENCODER_SPEC": "bf16/compute"})
    assert train_cfg.token_budget == "exact"
    trained = MossDataset(train_cfg)
    assert [trained.text_len(0, v) for v in range(3)] == [len(text) + 1 for text in captions]
    for item in latent_items(trained):
        assert trained.latents.get(trained.latents.key(trained.catalog[item.row_id], item)) is not None
    # Cached exact lengths are usable by the final token planner, not just parseable.
    assert trained.planner.plan(0, 0, 1).items
    assert json.loads(training.read_text())["GRAD_ACC"] == 4
    assert json.loads(training.read_text())["PRESET"] == "attn-mlp"
    assert json.loads(training.read_text())["SAVE_EVERY"] == 100


@pytest.mark.parametrize("phase", ["latents", "text", "planner_preflight", "train"])
def test_worker_subprocess_failure_is_terminal_and_keeps_stage(tmp_path, monkeypatch, phase):
    c = config(tmp_path)
    work = Path(c["WORK_DIR"]) / c["RUN_NAME"]
    runner.atomic_json(work / "status.json", {"phase": phase, "pid": 123})
    runner.atomic_json(work / "request.json", c)
    (work / "dataset").mkdir()
    (work / "dataset" / "retained").write_text("dataset")
    error = subprocess.CalledProcessError(7, ["hidden-secret-argument"])

    def fail(*args):
        raise error

    monkeypatch.setattr(runner, "run_worker", fail)
    monkeypatch.setattr(runner.os, "fstat", lambda fd: None)
    monkeypatch.setattr(sys, "argv", ["runner", "--config", str(work / "request.json"), "--worker", "--lock-fd", "9"])
    with pytest.raises(SystemExit) as stopped:
        runner.main()
    assert stopped.value.code == 1
    state = json.loads((work / "status.json").read_text())
    assert state["phase"] == "failed" and state["active"] is False
    assert state["failed_phase"] == phase
    assert state["returncode"] == 7
    assert state["unbacked"] is False
    assert state["checkpoint_count"] == 0
    assert "hidden-secret" not in (work / "status.json").read_text()
    assert (work / "dataset" / "retained").read_text() == "dataset"


def test_parent_failure_preserves_child_cause_and_backup_truth(tmp_path):
    c = config(tmp_path)
    work = Path(c["WORK_DIR"]) / c["RUN_NAME"]
    runner.atomic_json(work / "status.json", {"phase": "latents"})
    child = runner.record_failure(c, work, ValueError("private URL must not be persisted"))
    state = runner.record_failure(c, work, subprocess.CalledProcessError(1, ["stage"]))
    assert state["error_type"] == child["error_type"] == "ValueError"
    assert state["failed_phase"] == "latents"
    assert state["worker_error_type"] == "CalledProcessError"
    assert not state["unbacked"]
    assert "private URL" not in json.dumps(state)
    runner.atomic_json(work / "backup-status.json", {"pending": 1, "unbacked": True})
    assert runner.record_failure(c, work, RuntimeError("failed"))["unbacked"]


def test_saved_checkpoint_without_backup_is_retained_and_unbacked(tmp_path):
    c = config(tmp_path)
    work = Path(c["WORK_DIR"]) / c["RUN_NAME"]
    saved = checkpoint(Path(c["OUTPUT_DIR"]) / c["RUN_NAME"] / "checkpoints")
    runner.atomic_json(work / "status.json", {"phase": "train"})
    state = runner.record_failure(c, work, RuntimeError("stopped"))
    assert state["unbacked"] and state["checkpoint_count"] == 1
    assert all((saved / name).exists() for name in runner.REQUIRED_STATE)
    runner.atomic_json(work / "uploaded.json", {
        f"{c['OUTPUT_REPO']}/{c['RUN_NAME']}/checkpoints/{saved.name}": {"source": runner.snapshot_source(saved)}
    })
    runner.atomic_json(work / "status.json", {"phase": "train"})
    assert not runner.record_failure(c, work, RuntimeError("stopped"))["unbacked"]


def test_worker_validation_failure_does_not_stay_starting(tmp_path, monkeypatch):
    c = config(tmp_path)
    c["TRAIN"]["STEPS"] = 0
    work = Path(c["WORK_DIR"]) / c["RUN_NAME"]
    runner.atomic_json(work / "status.json", {"phase": "starting"})
    runner.atomic_json(work / "request.json", c)
    monkeypatch.setattr(sys, "argv", ["runner", "--config", str(work / "request.json"), "--worker", "--lock-fd", "9"])
    with pytest.raises(SystemExit):
        runner.main()
    assert json.loads((work / "status.json").read_text())["phase"] == "failed"
