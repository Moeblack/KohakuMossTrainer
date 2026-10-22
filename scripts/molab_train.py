"""CELL2's secret-free launcher for the pinned, single-GPU native trainer.

No GPU/import/network work before complete explicit configuration. Checkpoint
save hooks only create protected local snapshots; a single durable-queue worker
uploads them. Training success requires draining every backup, including export.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
import zipfile
import uuid
from pathlib import Path, PurePosixPath

TRAINER_COMMIT = "05eb27299708ff553fbe53b2dbcf4cebf99e422a"
REQUIRED_STATE = ("trainable.safetensors", "optim.pt", "rng.0.pt", "state.json")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    fsync_dir(path.parent)


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def snapshot_file(source, destination):
    """Independent inode: fast Linux CoW clone, reliable copy on other filesystems.

    Never hardlink: same-step saves, metadata and sample sidecars may be rewritten.
    The caller serializes the trainer save through the completed snapshot rename.
    """
    with open(source, "rb") as src, open(destination, "xb") as dst:
        try:
            fcntl.ioctl(dst.fileno(), 0x40049409, src.fileno())  # FICLONE
        except OSError:
            dst.seek(0)
            dst.truncate()
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    os.chmod(destination, 0o400)
    return destination


def snapshot_source(folder):
    paths = sorted(folder.rglob("*"))
    if folder.is_symlink() or any(p.is_symlink() for p in paths):
        raise ValueError("Backup snapshots cannot contain symlinks")
    source = []
    for p in paths:
        s = p.stat()
        if stat.S_ISREG(s.st_mode):
            source.append([str(p.relative_to(folder)), s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns])
        elif not stat.S_ISDIR(s.st_mode):
            raise ValueError("Backup snapshots require regular files/directories")
    if not source:
        raise ValueError("Refusing an empty backup")
    return source


def missing_config(c):
    missing = [k for k in ("DATASET_REPO", "OUTPUT_REPO", "RUN_NAME") if not c.get(k)]
    if not c.get("DATASET_FILES") and not c.get("DATASET_FILE"):
        missing.append("DATASET_FILES or DATASET_FILE")
    if c.get("TRIGGER_WORD") is None:
        missing.append('TRIGGER_WORD (use "" explicitly for no trigger)')
    return missing


def validate_config(c):
    missing = missing_config(c)
    if missing:
        return missing
    trainer_pin(c)
    for key in ("DATASET_REPO", "OUTPUT_REPO"):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", c[key]):
            raise ValueError(f"{key} must be an explicit owner/repository")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", c["RUN_NAME"]):
        raise ValueError("RUN_NAME must be a safe, unique directory name")
    dataset_files(c)
    if not isinstance(c["TRIGGER_WORD"], str) or "\x00" in c["TRIGGER_WORD"]:
        raise ValueError("TRIGGER_WORD must be text")
    k = c["TRAIN"]
    if k.get("ALGO") not in ("lora", "lokr"):
        raise ValueError("This runner requires ALGO lora or lokr")
    if k["ALGO"] == "lokr":
        if type(k.get("DIM", 32)) is not int or k.get("DIM", 32) <= 0:
            raise ValueError("LoKr DIM must be a positive integer")
        alpha = k.get("ALPHA", 32.0)
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < float("inf"):
            raise ValueError("LoKr ALPHA must be finite and positive")
        kwargs = k.get("ALGO_KWARGS", {})
        if not isinstance(kwargs, dict):
            raise ValueError("LoKr ALGO_KWARGS must be an object")
        factor = kwargs.get("factor", -1)
        if type(factor) is not int or (factor != -1 and factor <= 0):
            raise ValueError("LoKr factor must be -1 or a positive integer")
        for key in ("full_matrix", "decompose_both", "use_tucker"):
            if key in kwargs and type(kwargs[key]) is not bool:
                raise ValueError(f"LoKr {key} must be a boolean")
    for key, value in {"PARALLEL": "none", "DRIVER": "native", "LATENT_CACHE": "offline", "TEXT_CACHE": "offline", "CHECKPOINT_EXPORT": True, "FINAL_CHECKPOINT": True, "SAVE_FORMAT": "comfy"}.items():
        if k.get(key) != value:
            raise ValueError(f"This runner requires TRAIN[{key!r}]={value!r}")
    if k["SAMPLE_EVERY"] and not k["SAMPLE_PROMPTS"]:
        raise ValueError("Sampling enabled without user-provided SAMPLE_PROMPTS")
    if k["STEPS"] <= 0 or k["SAVE_EVERY"] <= 0 or c["UPLOAD_ATTEMPTS"] < 1:
        raise ValueError("STEPS, SAVE_EVERY and UPLOAD_ATTEMPTS must be positive")
    for key, default in (("BACKUP_MAX_PENDING", 8), ("BACKUP_MAX_BYTES", 32 * 1024**3)):
        if not isinstance(c.get(key, default), int) or c.get(key, default) < 1:
            raise ValueError(f"{key} must be a positive integer")
    if c.get("BACKUP_MIN_FREE_BYTES", 2 * 1024**3) < 0 or c["UPLOAD_RETRY_SECONDS"] < 0:
        raise ValueError("Backup free-space reserve and retry delay cannot be negative")
    return []


def trainer_pin(c):
    pin = c.get("TRAINER_COMMIT", TRAINER_COMMIT)
    if not isinstance(pin, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", pin):
        raise ValueError("TRAINER_COMMIT must be a full 40-hex commit")
    return pin.lower()


def dataset_files(c):
    files = c.get("DATASET_FILES")
    if files is None:
        files = [c.get("DATASET_FILE")]
    elif not isinstance(files, list) or len(files) not in (1, 2):
        raise ValueError("DATASET_FILES must contain one ZIP or [images ZIP, captions ZIP]")
    for name in files:
        if not isinstance(name, str) or not name.lower().endswith(".zip"):
            raise ValueError("Dataset files must name ZIPs in the HF dataset repository")
        safe_relative(name)
    if len(set(files)) != len(files):
        raise ValueError("Dataset ZIP names must be distinct")
    return files


def safe_relative(name):
    p = PurePosixPath(name)
    if not name or "\\" in name or "\x00" in name or p.is_absolute() or ".." in p.parts or ":" in name or not p.parts:
        raise ValueError("Unsafe archive/image path")
    return p


def safe_extract(archive, destination, max_bytes):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(archive) as z:
            total = 0
            seen = set()
            for item in z.infolist():
                p = safe_relative(item.filename)
                mode = item.external_attr >> 16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                    raise ValueError("Archive contains a link or special file")
                if str(p) in seen:
                    raise ValueError("Duplicate archive member")
                seen.add(str(p))
                total += item.file_size
                if total > max_bytes:
                    raise ValueError("Archive exceeds MAX_EXTRACT_BYTES")
                target = destination.joinpath(*p.parts)
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(item) as src, target.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def prepare_dual_rows(root, c):
    from PIL import Image

    root = Path(root).resolve()
    images = {}
    captions = {}
    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif"}
    for folder, entries, extensions in (("images", images, image_extensions), ("captions", captions, {".json"})):
        base = root / folder
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            relative = path.relative_to(base)
            if len(relative.parts) != 2:
                raise ValueError(f"Expected artist/id file in {folder}: {relative}")
            key = relative.with_suffix("").as_posix()
            if key in entries:
                raise ValueError(f"Duplicate {folder} stem: {key}")
            entries[key] = path
    if not images or images.keys() != captions.keys():
        raise ValueError("Images and caption JSONs must match one-to-one by artist/id stem")
    rows = []
    for key, image in images.items():
        payload = json.loads(captions[key].read_text(encoding="utf-8"))
        values = payload.get("caption") if isinstance(payload, dict) else None
        if not isinstance(values, dict):
            raise ValueError(f"Missing caption object: {key}")
        options = [values.get(name) for name in ("short", "medium", "long")]
        if any(not isinstance(text, str) or not text.strip() or "\x00" in text for text in options):
            raise ValueError(f"Require nonempty short/medium/long captions: {key}")
        # Captions are authoritative, including untagged character captions.
        # Directory names are matching keys, not identities to inject into text.
        with Image.open(image) as im:
            im.verify()
        with Image.open(image) as im:
            im.load()
        rows.append({"image": str(image), "caption_options": options})
    normalized = root.parent / "validated_train.jsonl"
    normalized.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return normalized, len(rows), len(images)


def prepare_dataset(c, work, download):
    work = Path(work)
    files = dataset_files(c)
    token = credential(c, dataset=True)
    archives = [download(repo_id=c["DATASET_REPO"], repo_type="dataset", filename=name,
                         revision=c["DATASET_REVISION"], token=token) for name in files]
    identity = {"repo": c["DATASET_REPO"], "revision": c["DATASET_REVISION"],
                "archives": [{"file": name, "sha256": file_digest(path)} for name, path in zip(files, archives)]}
    data = work / "dataset"
    marker = work / "dataset.identity.json"
    if data.exists():
        if not marker.exists() and len(files) == 1:
            legacy = work / "dataset.sha256"
            old_request = work / "dataset.legacy-request.json"
            if not old_request.exists():
                old_request = work / "request.json"
            if legacy.is_file() and old_request.is_file():
                old = json.loads(old_request.read_text(encoding="utf-8"))
                if (old.get("DATASET_REPO") == c["DATASET_REPO"]
                        and old.get("DATASET_REVISION") == c["DATASET_REVISION"]
                        and old.get("DATASET_FILE") == files[0]
                        and not old.get("DATASET_FILES")
                        and legacy.read_text().strip() == identity["archives"][0]["sha256"]):
                    atomic_json(marker, identity)
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != identity:
            raise ValueError("Dataset changed/incomplete extraction: use a new RUN_NAME")
    else:
        if len(archives) == 1:
            safe_extract(archives[0], data, c["MAX_EXTRACT_BYTES"])
        else:
            data.mkdir(parents=True)
            try:
                for archive, folder in zip(archives, ("images", "captions")):
                    safe_extract(archive, data / folder, c["MAX_EXTRACT_BYTES"])
            except BaseException:
                shutil.rmtree(data, ignore_errors=True)
                raise
        atomic_json(marker, identity)
    prepare = prepare_dual_rows if len(files) == 2 else prepare_rows
    catalog, rows, images = prepare(data, c)
    return data, catalog, rows, images, identity


def prepare_rows(root, c):
    from PIL import Image

    root = Path(root).resolve()
    catalogs = list(root.rglob("train.jsonl"))
    if len(catalogs) != 1:
        raise ValueError("ZIP must contain exactly one train.jsonl alongside images/")
    catalog = catalogs[0]
    base = catalog.parent
    image_root = base / "images"
    if not image_root.is_dir():
        raise ValueError("Missing images/ next to train.jsonl")
    rows = []
    checked = set()
    for number, line in enumerate(catalog.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("image"), str) or not isinstance(row.get("caption"), str):
            raise ValueError(f"train.jsonl line {number}: require string image and caption")
        image = base.joinpath(*safe_relative(row["image"]).parts).resolve()
        if not image.is_relative_to(image_root.resolve()) or not image.is_file():
            raise ValueError(f"train.jsonl line {number}: image must exist under images/")
        caption = row["caption"].strip()
        if not caption or "\x00" in caption:
            raise ValueError(f"train.jsonl line {number}: empty/invalid caption")
        if image not in checked:
            with Image.open(image) as im:
                if min(im.size) <= 0:
                    raise ValueError("Invalid image dimensions")
                im.verify()
            with Image.open(image) as im:
                im.load()
            checked.add(image)
        trigger = c["TRIGGER_WORD"]
        if trigger:
            prefix = c["TRIGGER_TEMPLATE"].format(trigger=trigger)
            if not caption.startswith(prefix):
                caption = prefix + caption
        # Repeated images with short/medium/long captions stay separate rows.
        rows.append({"image": str(image), "caption": caption})
    if not rows:
        raise ValueError("Dataset has no training rows")
    normalized = root.parent / "validated_train.jsonl"
    normalized.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return normalized, len(rows), len(checked)


def checkpoint_complete(path, require_export=False):
    path = Path(path)
    if not re.fullmatch(r"step_\d{8}", path.name):
        raise ValueError("Resume requires a complete step_XXXXXXXX directory, not a LoRA export/.partial")
    for name in REQUIRED_STATE:
        if not (path / name).is_file() or not (path / name).stat().st_size:
            raise ValueError(f"Incomplete checkpoint: missing {name}")
    state = json.loads((path / "state.json").read_text())
    if int(state.get("step", -1)) != int(path.name[5:]):
        raise ValueError("Checkpoint state step does not match directory")
    if state.get("world", 1) != 1:
        raise ValueError("This runner only resumes single-rank state")
    if require_export and not list((path / "export").glob("*.safetensors")):
        raise ValueError("Checkpoint export has not finished")
    return path.resolve()


def resolve_resume(value, output):
    if not value:
        return None
    if value == "latest":
        candidates = sorted((Path(output) / "checkpoints").glob("step_*"), reverse=True)
        for candidate in candidates:
            try:
                return checkpoint_complete(candidate)
            except ValueError:
                continue
        raise ValueError("RESUME=latest requested, but no complete local checkpoint exists")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("RESUME must be an absolute checkpoint directory or latest")
    return checkpoint_complete(path)


def credential(c, dataset=False):
    key = "DATASET_HF_TOKEN_PATH" if dataset else "HF_TOKEN_PATH"
    path = c.get(key)
    if dataset and not path:
        return credential(c)
    token = None if path else os.environ.get("HF_TOKEN")
    if not token:
        p = Path(path or c["TOKEN_FILE"]).expanduser()
        if p.stat().st_mode & 0o077:
            raise ValueError(f"Private {key} must have permissions 0600")
        token = p.read_text().strip()
    if not token:
        raise ValueError("Restore a private HF credential on this machine first")
    return token


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Backups:
    """One process/worker owns manifest writes; queued snapshots survive pruning/crash.

    The queue has both byte and item limits. Exceeding either fails immediately,
    never waits for the network. Only drain/close waits for an in-flight upload.
    """

    def __init__(self, api, c, work):
        self.api, self.c = api, dict(c)
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.manifest = self.work / "uploaded.json"
        self.spool = self.work / "backup-queue"
        self.spool.mkdir(mode=0o700, exist_ok=True)
        self.lock = (self.work / ".backup.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            raise RuntimeError("Another backup worker owns this run") from None
        self.condition = threading.Condition()
        self.producer_lock = threading.RLock()
        self.stop = threading.Event()
        self.closed = False
        self.error = None
        self.jobs = {}
        self.last_source = {}
        try:
            self.done = json.loads(self.manifest.read_text()) if self.manifest.exists() else {}
            for key, entry in self.done.items():
                self.last_source[key] = entry.get("source")
            # A ready directory carries its own write-ahead descriptor: recover
            # even if the process died before pending.json could be updated.
            for folder in sorted(self.spool.iterdir()):
                job = json.loads((folder / "job.json").read_text())
                if job["id"] != folder.name.removesuffix(".partial") or job["repo"] != c["OUTPUT_REPO"]:
                    raise RuntimeError("UNBACKED: pending backup identity differs; retain this run's original OUTPUT_REPO")
                safe_relative(job["remote"])
                if folder.name.endswith(".partial"):
                    self._snapshot(job, folder)
                    folder = self.spool / job["id"]
                if job["state"] == "succeeded" or self.done.get(self._key(job), {}).get("source") == job["source"]:
                    shutil.rmtree(folder)
                    continue
                if not (folder / "files").is_dir():
                    raise RuntimeError("UNBACKED: pending snapshot is missing")
                job.update(state="queued", attempt=0)
                self.jobs[job["id"]] = job
                self.last_source[self._key(job)] = job["source"]
                atomic_json(folder / "job.json", job)
            self._persist()
        except BaseException:
            self.lock.close()
            raise
        self.thread = threading.Thread(target=self._worker, name="checkpoint-backups", daemon=False)
        self.thread.start()

    def _snapshot(self, job, partial):
        if job["state"] == "snapshotting":
            source = Path(job["source_path"])
            if snapshot_source(source) != job["source"]:
                raise RuntimeError("UNBACKED: interrupted snapshot source changed; retain backup-queue and recover complete original state")
            if (partial / "files").exists():
                shutil.rmtree(partial / "files")
            shutil.copytree(source, partial / "files", copy_function=snapshot_file)
            for directory in sorted((partial / "files").rglob("*"), reverse=True):
                if directory.is_dir():
                    fsync_dir(directory)
            fsync_dir(partial / "files")
            job["state"] = "queued"
            atomic_json(partial / "job.json", job)
        fsync_dir(partial)
        partial.rename(self.spool / job["id"])
        fsync_dir(self.spool)

    def _key(self, job):
        return job["repo"] + "/" + job["remote"]

    def _pending(self):
        return [j for j in self.jobs.values() if j["state"] != "succeeded"]

    def _persist(self):
        pending = self._pending()
        atomic_json(self.work / "pending.json", {"version": 1, "jobs": pending})
        atomic_json(self.work / "backup-status.json", {
            "pending": len(pending), "pending_bytes": sum(j["bytes"] for j in pending),
            "unbacked": bool(pending) or self.error is not None,
            "error": self.error, "jobs": [{k: j[k] for k in ("id", "remote", "state", "attempt")} for j in self.jobs.values()],
        })

    def _state(self, job, state, **extra):
        job.update(state=state, **extra)
        atomic_json(self.spool / job["id"] / "job.json", job)
        self._persist()
        self.condition.notify_all()

    def check(self):
        with self.condition:
            if self.error:
                raise RuntimeError(self.error)
            if self.closed:
                raise RuntimeError("Backup queue is closed")

    def upload(self, folder, remote_path):
        """Snapshot and enqueue only: no file hashing, HF calls or retry sleeps."""
        with self.producer_lock:
            self.check()
            safe_relative(remote_path)
            folder = Path(folder)
            source = snapshot_source(folder)
            size = sum(entry[3] for entry in source)
            key = self.c["OUTPUT_REPO"] + "/" + remote_path
            with self.condition:
                if self.last_source.get(key) == source:
                    return
                pending = self._pending()
                max_pending = self.c.get("BACKUP_MAX_PENDING", 8)
                max_bytes = self.c.get("BACKUP_MAX_BYTES", 32 * 1024**3)
                reserve = self.c.get("BACKUP_MIN_FREE_BYTES", 2 * 1024**3)
                free = shutil.disk_usage(self.spool).free
                if (len(pending) >= max_pending
                        or sum(j["bytes"] for j in pending) + size > max_bytes
                        or free - size < reserve):
                    self.error = (f"UNBACKED: backup backlog/disk limit exceeded (max_pending={max_pending}, "
                                  f"max_bytes={max_bytes}, new_bytes={size}, free_bytes={free}, reserve_bytes={reserve}); "
                                  "training stopped, source and pending snapshots retained")
                    self._persist()
                    self.condition.notify_all()
                    raise RuntimeError(self.error)
            identity = f"{time.time_ns():020d}-{uuid.uuid4().hex}"
            job = {"id": identity, "repo": self.c["OUTPUT_REPO"], "remote": remote_path,
                   "source": source, "source_path": str(folder.resolve()), "bytes": size,
                   "state": "snapshotting", "attempt": 0}
            partial = self.spool / (identity + ".partial")
            partial.mkdir(mode=0o700)
            # Keep incomplete copies on I/O failure; never claim they were queued.
            atomic_json(partial / "job.json", job)
            try:
                self._snapshot(job, partial)
                with self.condition:
                    self.jobs[identity] = job
                    self.last_source[key] = source
                    self._persist()
                    self.condition.notify_all()
            except BaseException:
                with self.condition:
                    self.error = "UNBACKED: local snapshot/ledger failed; source and snapshot files retained"
                    try:
                        self._persist()
                    except Exception:
                        pass  # Full disk: descriptor/source and main status remain.
                    self.condition.notify_all()
                raise

    def checkpoint(self, path):
        path = checkpoint_complete(path, require_export=True)
        self.upload(path, f"{self.c['RUN_NAME']}/checkpoints/{path.name}")

    def _transfer(self, job):
        folder = self.spool / job["id"] / "files"
        signature = {str(p.relative_to(folder)): file_digest(p) for p in sorted(folder.rglob("*")) if p.is_file()}
        key = self._key(job)
        commit = self.done.get(key, {}).get("commit", "")
        if self.done.get(key, {}).get("files") != signature:
            for attempt in range(1, self.c["UPLOAD_ATTEMPTS"] + 1):
                if self.stop.is_set():
                    return False
                with self.condition:
                    self._state(job, "transferring", attempt=attempt)
                try:
                    result = self.api.upload_folder(repo_id=job["repo"], repo_type="model", folder_path=str(folder), path_in_repo=job["remote"], commit_message=f"Back up {job['remote']}")
                    commit = str(getattr(result, "oid", ""))
                    break
                except Exception as exc:
                    # Exception strings may contain signed URLs/tokens. Class only.
                    print(f"Upload attempt {attempt} failed ({type(exc).__name__}); snapshot retained", flush=True)
                    with self.condition:
                        self._state(job, "retrying", error_type=type(exc).__name__)
                    if attempt == self.c["UPLOAD_ATTEMPTS"]:
                        raise RuntimeError("Upload retries exhausted") from None
                    if self.stop.wait(self.c["UPLOAD_RETRY_SECONDS"] * (2 ** (attempt - 1))):
                        return False
        self.done[key] = {"files": signature, "commit": commit, "source": job["source"]}
        atomic_json(self.manifest, self.done)
        return True

    def _worker(self):
        job = None
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.stop.is_set() or self.error or any(j["state"] == "queued" for j in self.jobs.values()))
                    if self.stop.is_set() or self.error:
                        return
                    job = next(j for j in self.jobs.values() if j["state"] == "queued")
                    self._state(job, "transferring")
                if not self._transfer(job):
                    with self.condition:
                        self._state(job, "queued")
                    return
                with self.condition:
                    self._state(job, "succeeded")
                    # Cleanup before releasing the queue slot/lock to producers.
                    shutil.rmtree(self.spool / job["id"])
                    fsync_dir(self.spool)
        except BaseException as exc:
            with self.condition:
                self.error = "UNBACKED: background backup failed; training must stop, pending snapshots retained"
                try:
                    if job is not None:
                        self._state(job, "failed", error_type=type(exc).__name__)
                    else:
                        self._persist()
                except Exception:
                    pass  # In-memory failure still propagates if the disk is full.
                self.condition.notify_all()

    def drain(self):
        with self.condition:
            self.condition.wait_for(lambda: self.error or not self._pending())
            self.check()

    def close(self, drain=True):
        if self.closed:
            return
        try:
            if drain:
                self.drain()
        finally:
            self.stop.set()
            with self.condition:
                self.condition.notify_all()
            # Do not detach a live HTTP call. On SIGKILL the durable descriptor
            # remains transferring and is retried idempotently at next startup.
            self.thread.join()
            self.closed = True
            self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close(drain=kind is None)


def install_checkpoint_upload(manager, backups):
    original = manager.save

    def save(self, *args, **kwargs):
        # Serializes save/prune and snapshot creation, not network transfers.
        with backups.producer_lock:
            backups.check()
            prune = getattr(self, "prune", None)
            if prune is not None:
                self.prune = lambda: None
            try:
                path = original(self, *args, **kwargs)
                if path is not None:
                    backups.checkpoint(path)
            finally:
                if prune is not None:
                    self.prune = prune
            if prune is not None:
                prune()  # Only after a successful snapshot; errors preserve sources.
            return path

    manager.save = save
    return original


def install_backup_checks(callback_runner, backups):
    original = callback_runner.run

    def run(self, *args, **kwargs):
        # Pinned native driver invokes CallbackRunner.run every optimizer step,
        # even when no callback is due. No trainer code or model tensors change.
        backups.check()
        result = original(self, *args, **kwargs)
        backups.check()
        return result

    callback_runner.run = run
    return original


def script_namespace(path):
    namespace = {"__name__": "_molab_script", "__file__": str(path)}
    exec(compile(Path(path).read_text(), str(path), "exec"), namespace)
    return namespace


def stage(c, name, work):
    project = Path(c["PROJECT_DIR"])
    k = json.loads((Path(work) / "training.json").read_text())
    scripts = {"latents": "scripts/data/cache_latents.py", "text": "scripts/data/cache_text.py", "train": "scripts/train/moss.py"}
    ns = script_namespace(project / scripts[name])
    if name == "train":
        unknown = set(k) - set(ns)
        if unknown:
            raise ValueError(f"Unsupported trainer knobs: {sorted(unknown)}")
    ns.update(k)
    if name == "latents":
        ns.update(IMAGE_ENCODER="kohakumoss.encoders.vae_encoder", IMAGE_ENCODER_KWARGS={"path": str(Path(k["WEIGHTS"]) / "vae" / k["VAE"]), "device": "cuda"}, ENCODE_BATCH=c["LATENT_ENCODE_BATCH"])
    elif name == "text":
        ns.update(TEXT_ENCODER="kohakumoss.encoders.text_encoder", TEXT_ENCODER_KWARGS={"path": str(Path(k["WEIGHTS"]) / "text_encoders" / k["TEXT_ENCODER"]), "device": "cuda", "plan": {}}, ENCODE_BATCH=c["TEXT_ENCODE_BATCH"])
    else:
        from huggingface_hub import HfApi

        from kohakumoss.training.callbacks.runner import CallbackRunner
        from kohakumoss.training.checkpoint.manager import CheckpointManager

        with Backups(HfApi(token=credential(c)), c, work) as backup:
            original = install_checkpoint_upload(CheckpointManager, backup)
            original_run = install_backup_checks(CallbackRunner, backup)
            try:
                ns["main"]()
                backup.upload(Path(k["OUTPUT_DIR"]) / k["RUN_NAME"] / "export", f"{c['RUN_NAME']}/export")
                atomic_json(Path(work) / "status.json", {"phase": "draining_backups", "pid": os.getpid(), "unbacked": True})
            finally:
                CheckpointManager.save = original
                CallbackRunner.run = original_run
            # Context exit drains all checkpoints AND final export, then joins
            # the non-daemon worker; parent cannot mark success before this.
        return
    ns["main"]()


def run_worker(c, work):
    from huggingface_hub import HfApi, hf_hub_download

    work = Path(work)
    output = Path(c["OUTPUT_DIR"]) / c["RUN_NAME"]
    resume = resolve_resume(c.get("RESUME", ""), output)
    if not resume and (output / "checkpoints").exists():
        raise ValueError("Existing run state: choose a new RUN_NAME or explicit RESUME; never overwrite")
    if not resume and (output / "export").exists():
        raise ValueError("Existing final export: choose a new RUN_NAME or explicit RESUME")
    commit = subprocess.check_output(["git", "-C", c["PROJECT_DIR"], "rev-parse", "HEAD"], text=True).strip()
    if commit != trainer_pin(c):
        raise ValueError("CELL1 trainer revision differs from the runner's supported pin")
    k = c["TRAIN"]
    for knob, folder in (("TRANSFORMER", "diffusion_models"), ("TEXT_ENCODER", "text_encoders"), ("VAE", "vae")):
        model = Path(k["WEIGHTS"]) / folder / k[knob]
        if not model.is_file():
            raise ValueError(f"Missing {knob} weights: finish CELL1 before training")
    if resume and k.get("EMA", 0) and not (resume / "ema.safetensors").is_file():
        raise ValueError("EMA is enabled but its resume state is missing")
    token = credential(c)
    api = HfApi(token=token)
    api.whoami()  # authentication required; never log token or response
    api.create_repo(repo_id=c["OUTPUT_REPO"], repo_type="model", private=True, exist_ok=True)
    if not api.repo_info(c["OUTPUT_REPO"], repo_type="model").private:
        raise ValueError("OUTPUT_REPO already exists publicly; refusing to upload")
    # Startup/resume may wait for network: recover pending snapshots even when
    # their original checkpoint has already been pruned. No worker overlaps the
    # training subprocess or competes for its manifest.
    with Backups(api, c, work) as backup:
        backup.drain()
        for p in sorted((output / "checkpoints").glob("step_*")):
            if re.fullmatch(r"step_\d{8}", p.name):
                backup.checkpoint(p)
                backup.drain()
    atomic_json(work / "status.json", {"phase": "downloading", "pid": os.getpid(), "unbacked": False})
    data, catalog, rows, images, identity = prepare_dataset(c, work, hf_hub_download)
    k = dict(c["TRAIN"])
    k.update(DATASETS=[{"source": "jsonl", "path": str(catalog), "root": str(data)}], CACHE_DIR=str(work / "cache"), OUTPUT_DIR=c["OUTPUT_DIR"], RUN_NAME=c["RUN_NAME"], RESUME=str(resume) if resume else "")
    if len(dataset_files(c)) == 2:
        k.update(CAPTION={"template": "options"}, CAPTION_VARIANTS=3)
    atomic_json(work / "training.json", k)
    metadata = work / "metadata"
    metadata.mkdir(exist_ok=True)
    atomic_json(metadata / "training.json", k)
    atomic_json(metadata / "provenance.json", {"trainer_commit": commit, "runner_sha256": file_digest(__file__), "dataset_repo": c["DATASET_REPO"], "dataset_revision": c["DATASET_REVISION"], "dataset_identity": identity, "rows": rows, "images": images, "resume": str(resume) if resume else ""})
    with Backups(api, c, work) as backup:
        backup.upload(metadata, f"{c['RUN_NAME']}/metadata")
    for name in ("latents", "text", "train"):
        if name == "train" and len(dataset_files(c)) == 2:
            atomic_json(work / "status.json", {"phase": "planner_preflight", "pid": os.getpid()})
            subprocess.run([sys.executable, str(Path(c["PROJECT_DIR"]) / "scripts/molab_preflight.py"),
                            "--config", str(work / "training.json"), "--output", str(metadata / "planner-preflight.json")],
                           cwd=c["PROJECT_DIR"], check=True)
            with Backups(api, c, work) as backup:
                backup.upload(metadata, f"{c['RUN_NAME']}/metadata")
        atomic_json(work / "status.json", {"phase": name, "pid": os.getpid(), "rows": rows, "images": images})
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--config", str(work / "request.json"), "--stage", name], cwd=c["PROJECT_DIR"], check=True)
    atomic_json(work / "status.json", {"phase": "completed", "unbacked": False, "message": "Training and all checkpoint/final uploads completed"})


def launch(c, config):
    missing = validate_config(c)
    if missing:
        print("WAITING FOR CONFIGURATION: " + ", ".join(missing) + ". No download/cache/training/upload started.")
        return None
    workspace = Path(c["WORK_DIR"])
    workspace.mkdir(parents=True, exist_ok=True)
    lock = (workspace / ".training.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        print("A runner is already active; no duplicate started. Read the run's status.json / runner.log.")
        return None
    work = workspace / c["RUN_NAME"]
    work.mkdir(exist_ok=True)
    try:
        request = work / "request.json"
        if request.exists():
            old = json.loads(request.read_text())
            status = work / "status.json"
            state = json.loads(status.read_text()) if status.exists() else {}
            if state.get("phase") == "completed" and old == c:
                print("This exact run already completed and was backed up; no duplicate started.")
                return None
            if (Path(c["OUTPUT_DIR"]) / c["RUN_NAME"]).exists() and not c.get("RESUME"):
                raise ValueError("Run already exists. Set explicit RESUME or choose a new RUN_NAME")
            legacy_request = work / "dataset.legacy-request.json"
            if (work / "dataset.sha256").is_file() and not (work / "dataset.identity.json").exists() and not legacy_request.exists():
                atomic_json(legacy_request, {key: old.get(key) for key in ("DATASET_REPO", "DATASET_REVISION", "DATASET_FILE", "DATASET_FILES")})
        atomic_json(request, c)
        atomic_json(work / "status.json", {"phase": "starting"})
        with (work / "runner.log").open("a") as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--config", str(request), "--worker", "--lock-fd", str(lock.fileno())], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
        print(f"Background runner PID {child.pid}. Status: {work / 'status.json'}; log: {work / 'runner.log'}")
        return child.pid
    finally:
        lock.close()  # Child retains the same flock until completion/crash.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--lock-fd", type=int)
    parser.add_argument("--stage", choices=("latents", "text", "train"))
    args = parser.parse_args()
    c = json.loads(Path(args.config).read_text())
    if missing_config(c):
        launch(c, args.config)
        return
    work = Path(c["WORK_DIR"]) / c["RUN_NAME"]
    if not args.worker and not args.stage:
        launch(c, args.config)
        return
    validate_config(c)
    try:
        if args.stage:
            stage(c, args.stage, work)
        else:
            if args.lock_fd is None:
                raise ValueError("Worker must be started through the locked launcher")
            os.fstat(args.lock_fd)
            run_worker(c, work)
    except BaseException as exc:
        atomic_json(work / "status.json", {"phase": "failed", "error_type": type(exc).__name__, "unbacked": True, "message": "Stopped; local assets retained. Inspect runner.log, restore complete state with RESUME; backup is NOT guaranteed."})
        # Deliberately avoid traceback/HTTP exception URLs containing credentials.
        print(f"FAILED ({type(exc).__name__}): local assets retained; backup NOT guaranteed.", flush=True)
        if isinstance(exc, (ValueError, RuntimeError)):
            print(str(exc), flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
