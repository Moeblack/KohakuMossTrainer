"""CPU-only dual archive and credential routing contracts; no hub access."""

import importlib.util
import io
import json
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

SPEC = importlib.util.spec_from_file_location("molab_dual_runner", Path(__file__).parents[1] / "scripts/molab_train.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class DualZipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.work = self.root / "work"
        self.work.mkdir()
        self.c = {
            "DATASET_REPO": "owner/data", "DATASET_REVISION": "revision-one",
            "DATASET_FILES": ["images_v1.zip", "captions_v1.zip"],
            "OUTPUT_REPO": "owner/output", "RUN_NAME": "run", "TRIGGER_WORD": "unused",
            "TRIGGER_TEMPLATE": "<|artist|>{trigger}<|/artist|>\n",
            "MAX_EXTRACT_BYTES": 1000000, "UPLOAD_ATTEMPTS": 1, "UPLOAD_RETRY_SECONDS": 0,
            "TRAIN": {"ALGO": "lokr", "DIM": 16, "ALPHA": 16.0, "ALGO_KWARGS": {"factor": 8},
                      "PARALLEL": "none", "DRIVER": "native", "LATENT_CACHE": "offline",
                      "TEXT_CACHE": "offline", "CHECKPOINT_EXPORT": True, "FINAL_CHECKPOINT": True,
                      "SAVE_FORMAT": "comfy", "SAMPLE_EVERY": 0, "SAMPLE_PROMPTS": [],
                      "STEPS": 2, "SAVE_EVERY": 1},
        }
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4)).save(buffer, format="PNG")
        self.image = buffer.getvalue()
        self.options = [f"<|artist|>actual artist<|/artist|>\n{text}" for text in ("short", "medium", "long")]
        self.payload = {"caption": dict(zip(("short", "medium", "long"), self.options)),
                        "watermark": {"caption": "do not train"}, "borders": "do not train"}
        self.archives()
        self.download = Mock(side_effect=lambda **kw: str(self.root / kw["filename"]))

    def archives(self, images=None, captions=None):
        members = [images if images is not None else [("directory/001.png", self.image)],
                   captions if captions is not None else [("directory/001.json", json.dumps(self.payload))]]
        for filename, entries in zip(self.c["DATASET_FILES"], members):
            with zipfile.ZipFile(self.root / filename, "w") as archive:
                for name, value in entries:
                    archive.writestr(name, value)

    def prepare(self):
        with patch.object(runner, "credential", return_value="synthetic-dataset-token"):
            return runner.prepare_dataset(self.c, self.work, self.download)

    def test_lokr_allowed_and_validated(self):
        self.assertEqual(runner.validate_config(self.c), [])
        for key, value in (("ALGO", "full"), ("DIM", 0), ("DIM", True), ("ALPHA", float("nan")),
                           ("ALPHA", -1), ("ALGO_KWARGS", []), ("ALGO_KWARGS", {"factor": 0})):
            with self.subTest(key=key, value=value):
                c = dict(self.c, TRAIN=dict(self.c["TRAIN"], **{key: value}))
                with self.assertRaises(ValueError):
                    runner.validate_config(c)
        self.c["TRAIN"]["ALGO"] = "lora"
        self.assertEqual(runner.validate_config(self.c), [])

    def test_dual_zip_one_row_options_order_and_metadata_ignored(self):
        data, catalog, rows, images, identity = self.prepare()
        records = [json.loads(line) for line in catalog.read_text().splitlines()]
        self.assertEqual((rows, images, len(records)), (1, 1, 1))
        self.assertEqual(records[0]["caption_options"], self.options)
        self.assertEqual(set(records[0]), {"image", "caption_options"})
        self.assertTrue((data / "images/directory/001.png").is_file())
        self.assertTrue((data / "captions/directory/001.json").is_file())
        self.assertEqual(identity["repo"], self.c["DATASET_REPO"])
        self.assertEqual([a["file"] for a in identity["archives"]], self.c["DATASET_FILES"])
        self.assertEqual(self.prepare()[4], identity)
        self.assertTrue(all(call.kwargs["token"] == "synthetic-dataset-token" for call in self.download.call_args_list))

    def test_untagged_character_captions_are_preserved_verbatim(self):
        texts = ["  deepseek-tan short\n", "deepseek-tan medium", "deepseek-tan long"]
        self.payload["caption"] = dict(zip(("short", "medium", "long"), texts))
        self.archives()
        row = json.loads(self.prepare()[1].read_text())
        self.assertEqual(row["caption_options"], texts)

    def test_missing_caption_and_orphan_caption_rejected(self):
        for entries in ([], [("directory/002.json", json.dumps(self.payload))]):
            with self.subTest(entries=entries):
                self.archives(captions=entries)
                with tempfile.TemporaryDirectory(dir=self.root) as work:
                    with patch.object(runner, "credential", return_value="fake"):
                        with self.assertRaisesRegex(ValueError, "one-to-one"):
                            runner.prepare_dataset(self.c, work, self.download)

    def test_missing_empty_or_wrong_caption_variant_rejected(self):
        for value in (None, "", [], "\x00"):
            with self.subTest(value=value):
                self.payload["caption"]["medium"] = value
                self.archives()
                with tempfile.TemporaryDirectory(dir=self.root) as work:
                    with patch.object(runner, "credential", return_value="fake"):
                        with self.assertRaisesRegex(ValueError, "short/medium/long"):
                            runner.prepare_dataset(self.c, work, self.download)

    def test_unsafe_paths_and_duplicate_members_in_either_archive(self):
        for side in ("images", "captions"):
            for name in ("../escape", "/absolute", "a\\b", "C:/escape", "a/../../escape"):
                with self.subTest(side=side, name=name):
                    self.archives(**{side: [(name, b"bad")]})
                    with self.assertRaises(ValueError):
                        self.prepare()
                    self.assertFalse((self.work / "dataset").exists())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.archives(**{side: [("a/same", b"one"), ("a/same", b"two")]})
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                self.prepare()
            self.assertFalse((self.work / "dataset").exists())

    def test_duplicate_image_stems_rejected(self):
        self.archives(images=[("directory/001.png", self.image), ("directory/001.jpg", self.image)])
        with self.assertRaisesRegex(ValueError, "Duplicate images stem"):
            self.prepare()

    def test_identity_marker_rejects_repo_revision_and_archive_changes(self):
        self.prepare()
        for key in ("DATASET_REPO", "DATASET_REVISION"):
            original = self.c[key]
            self.c[key] = "different/value"
            with self.assertRaisesRegex(ValueError, "Dataset changed"):
                self.prepare()
            self.c[key] = original
        self.payload["caption"]["long"] += " changed"
        self.archives()
        with self.assertRaisesRegex(ValueError, "Dataset changed"):
            self.prepare()

    def test_missing_identity_marker_rejected(self):
        self.prepare()
        (self.work / "dataset.identity.json").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete extraction"):
            self.prepare()

    def test_old_single_zip_jsonl_and_trigger_still_supported(self):
        del self.c["DATASET_FILES"]
        self.c["DATASET_FILE"] = "legacy.zip"
        with zipfile.ZipFile(self.root / "legacy.zip", "w") as archive:
            archive.writestr("images/one.png", self.image)
            archive.writestr("train.jsonl", json.dumps({"image": "images/one.png", "caption": "legacy"}) + "\n")
        _, catalog, rows, images, _ = self.prepare()
        self.assertEqual((rows, images), (1, 1))
        self.assertEqual(json.loads(catalog.read_text())["caption"], "<|artist|>unused<|/artist|>\nlegacy")
        self.assertEqual(runner.validate_config(self.c), [])

    def test_config_zip_shape_names_and_duplicates(self):
        for files in ([], "images.zip", ["a.zip", "a.zip"], ["../a.zip"], ["a.txt"], ["a.zip", "b.zip", "c.zip"]):
            with self.subTest(files=files):
                with self.assertRaises(ValueError):
                    runner.dataset_files(dict(self.c, DATASET_FILES=files))

    def test_explicit_token_paths_override_environment_and_remain_separate(self):
        write_path = self.root / "synthetic-write-token"
        read_path = self.root / "synthetic-read-token"
        write_path.write_text("fake-write")
        read_path.write_text("fake-read")
        write_path.chmod(0o600)
        read_path.chmod(0o600)
        self.c.update(HF_TOKEN_PATH=str(write_path), DATASET_HF_TOKEN_PATH=str(read_path))
        with patch.dict(runner.os.environ, {"HF_TOKEN": "fake-environment"}):
            self.assertEqual(runner.credential(self.c), "fake-write")
            self.assertEqual(runner.credential(self.c, dataset=True), "fake-read")
            runner.prepare_dataset(self.c, self.work, self.download)
            self.assertTrue(all(call.kwargs["token"] == "fake-read" for call in self.download.call_args_list))
        read_path.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "DATASET_HF_TOKEN_PATH"):
            runner.credential(self.c, dataset=True)
        self.assertEqual(runner.credential(self.c), "fake-write")

    def test_legacy_token_file_fallback(self):
        path = self.root / "fake-legacy-token"
        path.write_text("fake-legacy")
        path.chmod(0o600)
        with patch.dict(runner.os.environ, {}, clear=True):
            self.assertEqual(runner.credential({"TOKEN_FILE": str(path)}), "fake-legacy")
            self.assertEqual(runner.credential({"TOKEN_FILE": str(path)}, dataset=True), "fake-legacy")

    def test_worker_routes_write_api_and_dataset_download_and_options(self):
        from types import SimpleNamespace

        self.c.update(PROJECT_DIR=str(self.root), OUTPUT_DIR=str(self.root / "output"))
        self.c["TRAINER_COMMIT"] = "a" * 40
        self.c["TRAIN"]["WEIGHTS"] = str(self.root / "weights")
        for key, folder in (("TRANSFORMER", "diffusion_models"), ("TEXT_ENCODER", "text_encoders"), ("VAE", "vae")):
            self.c["TRAIN"][key] = "mock.safetensors"
            path = self.root / "weights" / folder / "mock.safetensors"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"not real weights")
        api = Mock()
        api.repo_info.return_value.private = True
        hub = SimpleNamespace(HfApi=Mock(return_value=api), hf_hub_download=self.download)
        backup = Mock()
        context = Mock()
        context.__enter__ = Mock(return_value=backup)
        context.__exit__ = Mock(return_value=False)
        with patch.dict(runner.sys.modules, {"huggingface_hub": hub}), \
             patch.object(runner, "credential", side_effect=lambda c, dataset=False: "fake-read" if dataset else "fake-write"), \
             patch.object(runner, "Backups", return_value=context) as backups, \
             patch.object(runner.subprocess, "check_output", return_value=self.c["TRAINER_COMMIT"]), \
             patch.object(runner.subprocess, "run") as processes:
            runner.run_worker(self.c, self.work)
        hub.HfApi.assert_called_once_with(token="fake-write")
        api.whoami.assert_called_once_with()
        self.assertTrue(all(call.args[0] is api for call in backups.call_args_list))
        self.assertTrue(all(call.kwargs["token"] == "fake-read" for call in self.download.call_args_list))
        training = json.loads((self.work / "training.json").read_text())
        self.assertEqual(training["CAPTION"], {"template": "options"})
        self.assertEqual(training["CAPTION_VARIANTS"], 3)
        commands = [call.args[0] for call in processes.call_args_list]
        self.assertEqual(len(commands), 4)
        self.assertEqual(commands[1][-1], "text")
        self.assertTrue(commands[2][1].endswith("molab_preflight.py"))
        self.assertEqual(commands[3][-1], "train")
        provenance = json.loads((self.work / "metadata/provenance.json").read_text())
        self.assertEqual(provenance["trainer_commit"], self.c["TRAINER_COMMIT"])

    def test_trainer_pin_validation_and_legacy_default(self):
        self.assertEqual(runner.trainer_pin(self.c), runner.TRAINER_COMMIT)
        for value in ("main", "123abcd", "g" * 40, None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "40-hex"):
                    runner.validate_config(dict(self.c, TRAINER_COMMIT=value))
        self.assertEqual(runner.validate_config(dict(self.c, TRAINER_COMMIT="b" * 40)), [])

    def test_worker_rejects_mismatched_configured_pin_before_hub_use(self):
        from types import SimpleNamespace

        self.c.update(PROJECT_DIR=str(self.root), OUTPUT_DIR=str(self.root / "output"), TRAINER_COMMIT="a" * 40)
        hub = SimpleNamespace(HfApi=Mock(), hf_hub_download=Mock())
        with patch.dict(runner.sys.modules, {"huggingface_hub": hub}), \
             patch.object(runner.subprocess, "check_output", return_value="b" * 40):
            with self.assertRaisesRegex(ValueError, "supported pin"):
                runner.run_worker(self.c, self.work)
        hub.HfApi.assert_not_called()
        hub.hf_hub_download.assert_not_called()

    def test_lokr_boolean_options(self):
        for key in ("full_matrix", "decompose_both", "use_tucker"):
            self.c["TRAIN"]["ALGO_KWARGS"] = {key: True}
            self.assertEqual(runner.validate_config(self.c), [])
            self.c["TRAIN"]["ALGO_KWARGS"] = {key: 1}
            with self.assertRaisesRegex(ValueError, "boolean"):
                runner.validate_config(self.c)

    def test_legacy_marker_migrates_only_matching_original_identity(self):
        del self.c["DATASET_FILES"]
        self.c["DATASET_FILE"] = "legacy.zip"
        with zipfile.ZipFile(self.root / "legacy.zip", "w") as archive:
            archive.writestr("images/one.png", self.image)
            archive.writestr("train.jsonl", json.dumps({"image": "images/one.png", "caption": "legacy"}) + "\n")
        runner.safe_extract(self.root / "legacy.zip", self.work / "dataset", 1000000)
        digest = runner.file_digest(self.root / "legacy.zip")
        (self.work / "dataset.sha256").write_text(digest)
        old_path = self.work / "dataset.legacy-request.json"
        # A current request cannot override the captured original identity.
        runner.atomic_json(self.work / "request.json", self.c)
        for key in ("DATASET_REPO", "DATASET_REVISION", "DATASET_FILE"):
            runner.atomic_json(old_path, dict(self.c, **{key: "other"}))
            with self.assertRaisesRegex(ValueError, "Dataset changed"):
                self.prepare()
        runner.atomic_json(old_path, self.c)
        (self.work / "dataset.sha256").write_text("wrong-hash")
        with self.assertRaisesRegex(ValueError, "Dataset changed"):
            self.prepare()
        (self.work / "dataset.sha256").write_text(digest)
        identity = self.prepare()[4]
        self.assertEqual(json.loads((self.work / "dataset.identity.json").read_text()), identity)


if __name__ == "__main__":
    unittest.main()
