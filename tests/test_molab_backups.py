"""Durable asynchronous backups: CPU files and blocking fake APIs only."""

import json
import shutil
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_molab_train import checkpoint, config, runner


class BlockingAPI:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = []
        self.payloads = []

    def upload_folder(self, **kwargs):
        self.calls.append(kwargs)
        self.started.set()
        if not self.release.wait(10):
            raise TimeoutError("test did not release fake upload")
        folder = Path(kwargs["folder_path"])
        self.payloads.append({str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob("*") if p.is_file()})
        return SimpleNamespace(oid="mock-commit")


class AsyncBackupTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.c = config(self.root)
        self.work = self.root / "control"

    def backup(self, api=None):
        api = api or Mock()
        b = runner.Backups(api, self.c, self.work)
        self.addCleanup(b.close, drain=False)
        if isinstance(api, BlockingAPI):
            self.addCleanup(api.release.set)
        return b

    def test_save_returns_during_blocked_upload_and_prune_cannot_delete_snapshot(self):
        api = BlockingAPI()
        b = self.backup(api)
        root = self.root / "checkpoints"

        class Manager:
            def prune(self):
                # Even pruning the just-written path cannot beat the snapshot.
                for p in root.glob("step_*"):
                    shutil.rmtree(p)

            def save(self, step, parts):
                p = checkpoint(root, step, export=False)
                self.prune()
                (p / "export").mkdir()
                (p / "export/lora.safetensors").write_bytes(b"finished model")
                (p / "sample.json").write_text('{"seed": 42}')
                return p

        original = runner.install_checkpoint_upload(Manager, b)
        self.addCleanup(setattr, Manager, "save", original)
        returned = threading.Event()
        errors = []

        def save():
            try:
                Manager().save(1, {})
                returned.set()
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=save)
        thread.start()
        self.addCleanup(thread.join, 12)
        self.assertTrue(returned.wait(3), errors)
        self.assertTrue(api.started.wait(3))
        self.assertFalse(api.release.is_set())
        self.assertFalse(list(root.glob("step_*")))
        ledger = json.loads((self.work / "pending.json").read_text())
        self.assertEqual(len(ledger["jobs"]), 1)
        self.assertEqual(ledger["jobs"][0]["state"], "transferring")
        api.release.set()
        b.close()
        self.assertFalse(b.thread.is_alive())
        self.assertEqual(api.payloads[0]["export/lora.safetensors"], b"finished model")
        self.assertIn("rng.0.pt", api.payloads[0])
        self.assertIn("sample.json", api.payloads[0])

    def test_snapshot_is_independent_even_if_original_is_modified_in_place(self):
        api = BlockingAPI()
        b = self.backup(api)
        p = checkpoint(self.root)
        b.checkpoint(p)
        self.assertTrue(api.started.wait(3))
        snapshot = Path(api.calls[0]["folder_path"])
        self.assertNotEqual((snapshot / "optim.pt").stat().st_ino, (p / "optim.pt").stat().st_ino)
        (p / "optim.pt").write_bytes(b"new optimizer")
        shutil.rmtree(p)
        api.release.set()
        b.drain()
        self.assertEqual(api.payloads[0]["optim.pt"], b"mock state")

    def test_copy_fallback_is_independent_and_read_only(self):
        source = self.root / "source"
        dest = self.root / "dest"
        source.write_bytes(b"original")
        with patch.object(runner.fcntl, "ioctl", side_effect=OSError("no reflink")):
            runner.snapshot_file(source, dest)
        source.write_bytes(b"changed")
        self.assertEqual(dest.read_bytes(), b"original")
        self.assertEqual(dest.stat().st_mode & 0o777, 0o400)

    def test_queue_is_sequential_and_deduplicates_pending_and_completed(self):
        api = BlockingAPI()
        b = self.backup(api)
        p = checkpoint(self.root, 1)
        q = checkpoint(self.root, 2)
        b.checkpoint(p)
        self.assertTrue(api.started.wait(3))
        b.checkpoint(p)
        b.checkpoint(q)
        self.assertEqual(len(api.calls), 1)
        self.assertEqual(len(b._pending()), 2)
        api.release.set()
        b.drain()
        b.checkpoint(q)
        b.drain()
        self.assertEqual([c["path_in_repo"] for c in api.calls], ["run/checkpoints/step_00000001", "run/checkpoints/step_00000002"])
        self.assertEqual(json.loads((self.work / "pending.json").read_text())["jobs"], [])

    def test_metadata_change_requeues_but_identical_content_deduplicates(self):
        api = Mock()
        b = self.backup(api)
        metadata = self.root / "metadata"
        metadata.mkdir()
        p = metadata / "sampling.json"
        p.write_text('{"seed": 1}')
        b.upload(metadata, "run/metadata")
        b.drain()
        p.write_text('{"seed": 1}')
        b.upload(metadata, "run/metadata")
        b.drain()
        self.assertEqual(api.upload_folder.call_count, 1)
        p.write_text('{"seed": 2}')
        b.upload(metadata, "run/metadata")
        b.drain()
        self.assertEqual(api.upload_folder.call_count, 2)

    def test_failure_propagates_and_resume_uploads_pruned_snapshot(self):
        api = Mock()
        api.upload_folder.side_effect = OSError("https://signed.invalid/?token=SECRET")
        b = self.backup(api)
        p = checkpoint(self.root)
        b.checkpoint(p)
        with self.assertRaisesRegex(RuntimeError, "UNBACKED"):
            b.drain()
        with self.assertRaisesRegex(RuntimeError, "UNBACKED"):
            b.check()
        b.close(drain=False)
        self.assertEqual(api.upload_folder.call_count, 3)
        pending = (self.work / "pending.json").read_text()
        self.assertNotIn("SECRET", pending)
        self.assertEqual(json.loads(pending)["jobs"][0]["state"], "failed")
        shutil.rmtree(p)
        recovered = self.backup()
        recovered.close()
        self.assertEqual(recovered.api.upload_folder.call_count, 1)
        self.assertFalse(recovered.thread.is_alive())
        self.assertEqual(json.loads((self.work / "pending.json").read_text())["jobs"], [])

    def test_recovery_uses_write_ahead_descriptor_if_pending_ledger_is_lost(self):
        api = Mock()
        api.upload_folder.side_effect = OSError("offline")
        b = self.backup(api)
        b.checkpoint(checkpoint(self.root))
        with self.assertRaises(RuntimeError):
            b.drain()
        b.close(drain=False)
        job = next(iter(b.jobs.values()))
        job["state"] = "transferring"  # process died during an HTTP call
        runner.atomic_json(b.spool / job["id"] / "job.json", job)
        (self.work / "pending.json").unlink()
        recovered = self.backup()
        recovered.close()
        self.assertEqual(recovered.api.upload_folder.call_count, 1)

    def test_crash_after_manifest_commit_does_not_reupload(self):
        b = self.backup()
        p = checkpoint(self.root)
        with patch.object(runner.shutil, "rmtree", side_effect=OSError("crash after commit")):
            b.checkpoint(p)
            with self.assertRaises(RuntimeError):
                b.drain()
        b.close(drain=False)
        recovered = self.backup()
        recovered.close()
        recovered.api.upload_folder.assert_not_called()

    def test_backlog_limit_fails_without_waiting_for_network(self):
        self.c["BACKUP_MAX_PENDING"] = 1
        api = BlockingAPI()
        b = self.backup(api)
        b.checkpoint(checkpoint(self.root, 1))
        self.assertTrue(api.started.wait(3))
        p = checkpoint(self.root, 2)
        with self.assertRaisesRegex(RuntimeError, "backlog/disk limit"):
            b.checkpoint(p)
        self.assertTrue(p.exists())
        self.assertFalse(api.release.is_set())
        self.assertTrue(json.loads((self.work / "backup-status.json").read_text())["unbacked"])
        api.release.set()
        b.close(drain=False)

    def test_byte_limit_and_disk_reserve_fail_before_snapshot(self):
        for setting, value in (("BACKUP_MAX_BYTES", 1), ("BACKUP_MIN_FREE_BYTES", 2**63)):
            with self.subTest(setting=setting):
                c = dict(self.c, **{setting: value})
                p = checkpoint(self.root / setting)
                with runner.Backups(Mock(), c, self.root / (setting + "-work")) as b:
                    with self.assertRaisesRegex(RuntimeError, "UNBACKED"):
                        b.checkpoint(p)
                    self.assertFalse(list(b.spool.iterdir()))
                    b.close(drain=False)

    def test_abort_joins_active_worker_and_preserves_queued_jobs(self):
        api = BlockingAPI()
        b = self.backup(api)
        b.checkpoint(checkpoint(self.root, 1))
        self.assertTrue(api.started.wait(3))
        b.checkpoint(checkpoint(self.root, 2))
        closed = threading.Event()

        def close():
            b.close(drain=False)
            closed.set()

        thread = threading.Thread(target=close)
        thread.start()
        self.addCleanup(thread.join, 12)
        self.assertTrue(b.stop.wait(3))
        self.assertFalse(closed.is_set())
        api.release.set()
        self.assertTrue(closed.wait(3))
        self.assertFalse(b.thread.is_alive())
        self.assertEqual(len(api.calls), 1)
        recovered = self.backup()
        recovered.close()
        self.assertEqual(recovered.api.upload_folder.call_count, 1)

    def test_second_worker_cannot_race_manifest(self):
        b = self.backup()
        with self.assertRaisesRegex(RuntimeError, "Another backup worker"):
            runner.Backups(Mock(), self.c, self.work)
        b.close()
        self.backup().close()

    def test_symlink_snapshot_is_rejected(self):
        b = self.backup()
        p = checkpoint(self.root)
        (p / "link").symlink_to(p / "optim.pt")
        with self.assertRaisesRegex(ValueError, "symlinks"):
            b.checkpoint(p)
        b.api.upload_folder.assert_not_called()

    def test_snapshot_failure_preserves_checkpoint_and_restores_prune(self):
        b = self.backup()
        p = checkpoint(self.root)

        class Manager:
            prune = Mock()

            def save(self, *args):
                self.prune()
                return p

        original = runner.install_checkpoint_upload(Manager, b)
        self.addCleanup(setattr, Manager, "save", original)
        manager = Manager()
        with patch.object(runner, "snapshot_file", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                manager.save()
        self.assertTrue(p.exists())
        Manager.prune.assert_not_called()
        self.assertIs(manager.prune, Manager.prune)
        self.assertTrue(list(b.spool.glob("*.partial")))

    def test_interrupted_snapshot_rebuilds_on_resume(self):
        b = self.backup()
        p = checkpoint(self.root)
        with patch.object(runner, "snapshot_file", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                b.checkpoint(p)
        b.close(drain=False)
        recovered = self.backup()
        recovered.close()
        self.assertEqual(recovered.api.upload_folder.call_count, 1)
        self.assertFalse(list(recovered.spool.iterdir()))

    def test_complete_partial_is_recovered_without_original(self):
        api = Mock()
        api.upload_folder.side_effect = OSError("offline")
        b = self.backup(api)
        p = checkpoint(self.root)
        b.checkpoint(p)
        with self.assertRaises(RuntimeError):
            b.drain()
        b.close(drain=False)
        job = next(iter(b.jobs.values()))
        job["state"] = "queued"
        folder = b.spool / job["id"]
        runner.atomic_json(folder / "job.json", job)
        folder.rename(folder.with_name(folder.name + ".partial"))
        shutil.rmtree(p)
        with patch.object(runner, "snapshot_file", side_effect=AssertionError("must reuse snapshot")):
            recovered = self.backup()
            recovered.close()
        self.assertEqual(recovered.api.upload_folder.call_count, 1)

    def test_step_callback_surfaces_failure_even_without_checkpoint(self):
        backup = Mock()

        class Callbacks:
            def run(self, state, out):
                backup.check.side_effect = RuntimeError("UNBACKED")

        original = runner.install_backup_checks(Callbacks, backup)
        self.addCleanup(setattr, Callbacks, "run", original)
        with self.assertRaisesRegex(RuntimeError, "UNBACKED"):
            Callbacks().run(None, None)
        self.assertEqual(backup.check.call_count, 2)

    def fake_stage(self, api, main):
        class Manager:
            def save(self, *args):
                return checkpoint(self_root / "state")

        class Callbacks:
            def run(self, *args):
                return None

        self_root = self.root
        k = {"OUTPUT_DIR": str(self.root / "output"), "RUN_NAME": "run"}
        self.work.mkdir(exist_ok=True)
        runner.atomic_json(self.work / "training.json", k)
        self.c["PROJECT_DIR"] = str(self.root)
        namespace = dict(k, main=lambda: main(Manager, Callbacks))
        modules = {
            "huggingface_hub": SimpleNamespace(HfApi=lambda **kwargs: api),
            "kohakumoss.training.checkpoint.manager": SimpleNamespace(CheckpointManager=Manager),
            "kohakumoss.training.callbacks.runner": SimpleNamespace(CallbackRunner=Callbacks),
        }
        return Manager, Callbacks, namespace, modules

    def test_stage_enqueues_final_export_drains_and_restores_hooks(self):
        api = BlockingAPI()
        self.addCleanup(api.release.set)
        draining, returned = threading.Event(), threading.Event()
        errors = []

        def main(manager, callbacks):
            manager().save()
            callbacks().run(None, None)
            export = self.root / "output/run/export"
            export.mkdir(parents=True)
            (export / "lora.safetensors").write_bytes(b"final export")

        manager, callbacks, ns, modules = self.fake_stage(api, main)
        saved, run = manager.save, callbacks.run
        original_drain = runner.Backups.drain

        def drain(backup):
            draining.set()
            return original_drain(backup)

        def stage():
            try:
                runner.stage(self.c, "train", self.work)
            except BaseException as exc:
                errors.append(exc)
            finally:
                returned.set()

        with patch.dict(runner.sys.modules, modules), patch.object(runner, "script_namespace", return_value=ns), patch.object(runner, "credential", return_value="mock-private-credential"), patch.object(runner.Backups, "drain", drain):
            thread = threading.Thread(target=stage)
            thread.start()
            self.addCleanup(thread.join, 12)
            self.addCleanup(api.release.set)
            self.assertTrue(draining.wait(3), errors)
            self.assertFalse(returned.is_set())
            self.assertEqual(json.loads((self.work / "status.json").read_text())["phase"], "draining_backups")
            api.release.set()
            self.assertTrue(returned.wait(3), errors)
            thread.join()
        self.assertEqual(errors, [])
        self.assertIs(manager.save, saved)
        self.assertIs(callbacks.run, run)
        self.assertEqual([call["path_in_repo"] for call in api.calls], ["run/checkpoints/step_00000001", "run/export"])
        self.assertEqual(api.payloads[-1]["lora.safetensors"], b"final export")
        self.assertEqual(json.loads((self.work / "pending.json").read_text())["jobs"], [])
        self.backup().close()  # stage released process lock and joined its worker

    def test_stage_exception_cleans_up_hooks_and_worker(self):
        def main(manager, callbacks):
            raise ValueError("mock training error")

        manager, callbacks, ns, modules = self.fake_stage(Mock(), main)
        saved, run = manager.save, callbacks.run
        with patch.dict(runner.sys.modules, modules), patch.object(runner, "script_namespace", return_value=ns), patch.object(runner, "credential", return_value="mock-private-credential"):
            with self.assertRaisesRegex(ValueError, "mock training error"):
                runner.stage(self.c, "train", self.work)
        self.assertIs(manager.save, saved)
        self.assertIs(callbacks.run, run)
        self.backup().close()

    def test_final_export_failure_propagates_instead_of_success(self):
        api = Mock()
        api.upload_folder.side_effect = OSError("offline")

        def main(manager, callbacks):
            export = self.root / "output/run/export"
            export.mkdir(parents=True)
            (export / "lora.safetensors").write_bytes(b"final")

        manager, callbacks, ns, modules = self.fake_stage(api, main)
        with patch.dict(runner.sys.modules, modules), patch.object(runner, "script_namespace", return_value=ns), patch.object(runner, "credential", return_value="mock-private-credential"):
            with self.assertRaisesRegex(RuntimeError, "UNBACKED"):
                runner.stage(self.c, "train", self.work)
        self.assertEqual(api.upload_folder.call_count, 3)
        self.assertNotEqual(json.loads((self.work / "status.json").read_text())["phase"], "completed")
        self.assertEqual(json.loads((self.work / "pending.json").read_text())["jobs"][0]["remote"], "run/export")


if __name__ == "__main__":
    unittest.main()
