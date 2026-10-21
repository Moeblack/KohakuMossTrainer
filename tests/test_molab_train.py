"""No weights, GPU, network, or actual training: CELL2 lifecycle contracts."""

import importlib.util
import json
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location("molab_train", Path(__file__).parents[1] / "scripts/molab_train.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def config(root):
    return {"DATASET_REPO": "user/data", "DATASET_FILE": "dataset.zip", "OUTPUT_REPO": "user/output", "RUN_NAME": "run", "TRIGGER_WORD": "", "TRIGGER_TEMPLATE": "<|artist|>{trigger}<|/artist|>\n", "WORK_DIR": str(root / "work"), "OUTPUT_DIR": str(root / "output"), "UPLOAD_ATTEMPTS": 3, "UPLOAD_RETRY_SECONDS": 0, "RESUME": "", "TRAIN": {"PARALLEL": "none", "DRIVER": "native", "ALGO": "lora", "LATENT_CACHE": "offline", "TEXT_CACHE": "offline", "CHECKPOINT_EXPORT": True, "FINAL_CHECKPOINT": True, "SAVE_FORMAT": "comfy", "SAMPLE_EVERY": 0, "SAMPLE_PROMPTS": [], "STEPS": 2, "SAVE_EVERY": 1}}


def checkpoint(root, step=1, export=True):
    p = root / f"step_{step:08d}"
    p.mkdir(parents=True)
    for name in runner.REQUIRED_STATE:
        (p / name).write_bytes(b"mock state")
    (p / "state.json").write_text(json.dumps({"step": step, "world": 1}))
    if export:
        (p / "export").mkdir()
        (p / "export" / "lora.safetensors").write_bytes(b"mock export")
    return p


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.c = config(self.root)

    def test_unconfigured_is_noop(self):
        self.c["DATASET_REPO"] = ""
        with patch.object(runner.subprocess, "Popen") as spawn, patch.object(runner.Path, "mkdir") as mkdir:
            self.assertIsNone(runner.launch(self.c, "unused"))
        spawn.assert_not_called()
        mkdir.assert_not_called()

    def test_zip_paths_and_links_rejected(self):
        for index, name in enumerate(("../escape", "/absolute", "images/../../escape", "images\\escape", "C:/escape")):
            z = self.root / f"{index}.zip"
            with zipfile.ZipFile(z, "w") as f:
                f.writestr(name, b"bad")
            with self.assertRaises(ValueError):
                runner.safe_extract(z, self.root / f"out{index}", 100)
            self.assertFalse((self.root / f"out{index}").exists())
        z = self.root / "symlink.zip"
        with zipfile.ZipFile(z, "w") as f:
            item = zipfile.ZipInfo("images/link")
            item.external_attr = (stat.S_IFLNK | 0o777) << 16
            f.writestr(item, "../../escape")
        with self.assertRaises(ValueError):
            runner.safe_extract(z, self.root / "link", 100)

    def test_zip_size_limit_and_valid_extract(self):
        z = self.root / "dataset.zip"
        with zipfile.ZipFile(z, "w") as f:
            f.writestr("images/pic.png", b"payload")
        with self.assertRaises(ValueError):
            runner.safe_extract(z, self.root / "small", 1)
        runner.safe_extract(z, self.root / "okay", 100)
        self.assertEqual((self.root / "okay/images/pic.png").read_bytes(), b"payload")

    def test_three_captions_preserved_and_paths_checked(self):
        from PIL import Image

        data = self.root / "dataset"
        (data / "images").mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(data / "images/a.png")
        rows = [{"image": "images/a.png", "caption": text} for text in ("short", "medium\nsecond line", "long")]
        catalog = data / "train.jsonl"
        catalog.write_text("\n".join(json.dumps(row) for row in rows))
        path, count, images = runner.prepare_rows(data, self.c)
        self.assertEqual((count, images), (3, 1))
        self.assertEqual([json.loads(line)["caption"] for line in path.read_text().splitlines()], [row["caption"] for row in rows])
        catalog.write_text(json.dumps({"image": "../a.png", "caption": "x"}))
        with self.assertRaises(ValueError):
            runner.prepare_rows(data, self.c)
        catalog.write_text(json.dumps({"image": "images/a.png", "caption": " "}))
        with self.assertRaises(ValueError):
            runner.prepare_rows(data, self.c)

    def test_lock_prevents_duplicate_start(self):
        work = Path(self.c["WORK_DIR"])
        work.mkdir()
        with (work / ".training.lock").open("a+") as lock:
            runner.fcntl.flock(lock, runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB)
            with patch.object(runner.subprocess, "Popen") as spawn:
                self.assertIsNone(runner.launch(self.c, "unused"))
                spawn.assert_not_called()

    def test_launch_is_background_and_completed_request_is_noop(self):
        with patch.object(runner.subprocess, "Popen", return_value=SimpleNamespace(pid=123)) as spawn:
            self.assertEqual(runner.launch(self.c, "unused"), 123)
            self.assertTrue(spawn.call_args.kwargs["start_new_session"])
            self.assertEqual(len(spawn.call_args.kwargs["pass_fds"]), 1)
        work = Path(self.c["WORK_DIR"]) / "run"
        runner.atomic_json(work / "status.json", {"phase": "completed"})
        with patch.object(runner.subprocess, "Popen") as spawn:
            self.assertIsNone(runner.launch(self.c, "unused"))
            spawn.assert_not_called()

    def test_resume_requires_full_state_and_explicit_path(self):
        output = self.root / "run"
        with self.assertRaises(ValueError):
            runner.resolve_resume("latest", output)
        p = checkpoint(output / "checkpoints")
        self.assertEqual(runner.resolve_resume("latest", output), p.resolve())
        self.assertEqual(runner.resolve_resume(str(p), output), p.resolve())
        self.assertIsNone(runner.resolve_resume("", output))
        with self.assertRaises(ValueError):
            runner.resolve_resume("relative/step_00000001", output)
        with self.assertRaises(ValueError):
            runner.resolve_resume(str(p / "export/lora.safetensors"), output)
        (p / "rng.0.pt").unlink()
        with self.assertRaises(ValueError):
            runner.resolve_resume(str(p), output)
        with self.assertRaises(ValueError):
            runner.resolve_resume("latest", output)

    def test_checkpoint_hook_waits_for_export_and_upload_retries(self):
        p = checkpoint(self.root, export=False)
        api = Mock()
        api.upload_folder.side_effect = [OSError("network"), SimpleNamespace(oid="commit")]
        backup = runner.Backups(api, self.c, self.root)
        self.addCleanup(backup.close, drain=False)

        class Manager:
            def save(self, step, parts):
                # The real save invokes after_save/export before returning.
                (p / "export").mkdir()
                (p / "export/lora.safetensors").write_bytes(b"finished")
                return p

        original = runner.install_checkpoint_upload(Manager, backup)
        self.addCleanup(setattr, Manager, "save", original)
        self.assertEqual(Manager().save(1, {}), p)
        backup.drain()
        self.assertEqual(api.upload_folder.call_count, 2)
        stored = json.loads((self.root / "uploaded.json").read_text())
        self.assertIn("rng.0.pt", next(iter(stored.values()))["files"])
        self.assertIn("export/lora.safetensors", next(iter(stored.values()))["files"])
        backup.checkpoint(p)
        backup.drain()
        self.assertEqual(api.upload_folder.call_count, 2)

    def test_incomplete_export_or_partial_never_uploaded(self):
        api = Mock()
        backup = runner.Backups(api, self.c, self.root)
        self.addCleanup(backup.close, drain=False)
        p = checkpoint(self.root, export=False)
        with self.assertRaises(ValueError):
            backup.checkpoint(p)
        partial = p.with_name(p.name + ".partial")
        p.rename(partial)
        with self.assertRaises(ValueError):
            backup.checkpoint(partial)
        api.upload_folder.assert_not_called()

    def test_permanent_upload_failure_preserves_state_and_no_manifest(self):
        p = checkpoint(self.root)
        api = Mock()
        api.upload_folder.side_effect = OSError("offline")
        backup = runner.Backups(api, self.c, self.root)
        self.addCleanup(backup.close, drain=False)
        backup.checkpoint(p)
        with self.assertRaisesRegex(RuntimeError, "UNBACKED"):
            backup.drain()
        self.assertEqual(api.upload_folder.call_count, 3)
        self.assertTrue((p / "optim.pt").is_file())
        self.assertFalse((self.root / "uploaded.json").exists())

    def test_manifest_scoped_to_target_repo(self):
        p = checkpoint(self.root)
        api = Mock()
        with runner.Backups(api, self.c, self.root) as backup:
            backup.checkpoint(p)
        self.c["OUTPUT_REPO"] = "user/other"
        with runner.Backups(api, self.c, self.root) as backup:
            backup.checkpoint(p)
        self.assertEqual(api.upload_folder.call_count, 2)


if __name__ == "__main__":
    unittest.main()
