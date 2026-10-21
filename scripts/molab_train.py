"""CELL2's secret-free launcher for the pinned, single-GPU native trainer.

No GPU/import/network work before complete explicit configuration. Checkpoint
uploads are synchronous AFTER CheckpointManager.save returns (including exports),
so failures stop training rather than allowing retention to delete unbacked state.
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
import time
import zipfile
from pathlib import Path, PurePosixPath

TRAINER_COMMIT = "05eb27299708ff553fbe53b2dbcf4cebf99e422a"
REQUIRED_STATE = ("trainable.safetensors", "optim.pt", "rng.0.pt", "state.json")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def missing_config(c):
    missing = [k for k in ("DATASET_REPO", "DATASET_FILE", "OUTPUT_REPO", "RUN_NAME") if not c.get(k)]
    if c.get("TRIGGER_WORD") is None:
        missing.append('TRIGGER_WORD (use "" explicitly for no trigger)')
    return missing


def validate_config(c):
    missing = missing_config(c)
    if missing:
        return missing
    for key in ("DATASET_REPO", "OUTPUT_REPO"):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", c[key]):
            raise ValueError(f"{key} must be an explicit owner/repository")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", c["RUN_NAME"]):
        raise ValueError("RUN_NAME must be a safe, unique directory name")
    safe_relative(c["DATASET_FILE"])
    if not c["DATASET_FILE"].lower().endswith(".zip"):
        raise ValueError("DATASET_FILE must name a ZIP in the HF dataset repository")
    if not isinstance(c["TRIGGER_WORD"], str) or "\x00" in c["TRIGGER_WORD"]:
        raise ValueError("TRIGGER_WORD must be text")
    k = c["TRAIN"]
    for key, value in {"PARALLEL": "none", "DRIVER": "native", "ALGO": "lora", "LATENT_CACHE": "offline", "TEXT_CACHE": "offline", "CHECKPOINT_EXPORT": True, "FINAL_CHECKPOINT": True, "SAVE_FORMAT": "comfy"}.items():
        if k.get(key) != value:
            raise ValueError(f"This runner requires TRAIN[{key!r}]={value!r}")
    if k["SAMPLE_EVERY"] and not k["SAMPLE_PROMPTS"]:
        raise ValueError("Sampling enabled without user-provided SAMPLE_PROMPTS")
    if k["STEPS"] <= 0 or k["SAVE_EVERY"] <= 0 or c["UPLOAD_ATTEMPTS"] < 1:
        raise ValueError("STEPS, SAVE_EVERY and UPLOAD_ATTEMPTS must be positive")
    return []


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


def credential(c):
    token = os.environ.get("HF_TOKEN")
    if not token:
        p = Path(c["TOKEN_FILE"]).expanduser()
        if p.stat().st_mode & 0o077:
            raise ValueError("Private TOKEN_FILE must have permissions 0600")
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
    def __init__(self, api, c, work):
        self.api, self.c = api, c
        self.manifest = Path(work) / "uploaded.json"
        self.done = json.loads(self.manifest.read_text()) if self.manifest.exists() else {}

    def upload(self, folder, remote_path):
        folder = Path(folder)
        files = sorted(p for p in folder.rglob("*") if p.is_file())
        if not files:
            raise ValueError("Refusing an empty backup")
        signature = {str(p.relative_to(folder)): file_digest(p) for p in files}
        key = self.c["OUTPUT_REPO"] + "/" + remote_path
        if self.done.get(key, {}).get("files") == signature:
            return
        for attempt in range(self.c["UPLOAD_ATTEMPTS"]):
            try:
                commit = self.api.upload_folder(repo_id=self.c["OUTPUT_REPO"], repo_type="model", folder_path=str(folder), path_in_repo=remote_path, commit_message=f"Back up {remote_path}")
                self.done[key] = {"files": signature, "commit": str(getattr(commit, "oid", ""))}
                atomic_json(self.manifest, self.done)
                return
            except Exception as exc:
                # HTTP exception strings can contain signed URLs. Log class only.
                print(f"Upload attempt {attempt + 1} failed ({type(exc).__name__}); local assets retained", flush=True)
                if attempt + 1 < self.c["UPLOAD_ATTEMPTS"]:
                    time.sleep(self.c["UPLOAD_RETRY_SECONDS"] * (2 ** attempt))
        raise RuntimeError(f"UNBACKED: upload failed for {remote_path}; all local assets retained")

    def checkpoint(self, path):
        path = checkpoint_complete(path, require_export=True)
        self.upload(path, f"{self.c['RUN_NAME']}/checkpoints/{path.name}")


def install_checkpoint_upload(manager, backups):
    original = manager.save

    def save(self, *args, **kwargs):
        path = original(self, *args, **kwargs)
        if path is not None:
            backups.checkpoint(path)
        return path

    manager.save = save
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

        from kohakumoss.training.checkpoint.manager import CheckpointManager

        backup = Backups(HfApi(token=credential(c)), c, work)
        original = install_checkpoint_upload(CheckpointManager, backup)
        try:
            ns["main"]()
            backup.upload(Path(k["OUTPUT_DIR"]) / k["RUN_NAME"] / "export", f"{c['RUN_NAME']}/export")
        finally:
            CheckpointManager.save = original
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
    if commit != TRAINER_COMMIT:
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
    backup = Backups(api, c, work)
    # Retry older completed checkpoints before another optimizer step/prune.
    for p in sorted((output / "checkpoints").glob("step_*")):
        if re.fullmatch(r"step_\d{8}", p.name):
            backup.checkpoint(p)
    atomic_json(work / "status.json", {"phase": "downloading", "pid": os.getpid(), "unbacked": False})
    archive = hf_hub_download(repo_id=c["DATASET_REPO"], repo_type="dataset", filename=c["DATASET_FILE"], revision=c["DATASET_REVISION"], token=token)
    data = work / "dataset"
    archive_hash = file_digest(archive)
    marker = work / "dataset.sha256"
    if data.exists() and (not marker.exists() or marker.read_text() != archive_hash):
        raise ValueError("Dataset changed/incomplete extraction: use a new RUN_NAME (do not silently change resume data)")
    if not data.exists():
        safe_extract(archive, data, c["MAX_EXTRACT_BYTES"])
        marker.write_text(archive_hash)
    catalog, rows, images = prepare_rows(data, c)
    k = dict(c["TRAIN"])
    k.update(DATASETS=[{"source": "jsonl", "path": str(catalog), "root": str(data)}], CACHE_DIR=str(work / "cache"), OUTPUT_DIR=c["OUTPUT_DIR"], RUN_NAME=c["RUN_NAME"], RESUME=str(resume) if resume else "")
    atomic_json(work / "training.json", k)
    metadata = work / "metadata"
    metadata.mkdir(exist_ok=True)
    atomic_json(metadata / "training.json", k)
    atomic_json(metadata / "provenance.json", {"trainer_commit": TRAINER_COMMIT, "runner_sha256": file_digest(__file__), "dataset_repo": c["DATASET_REPO"], "dataset_revision": c["DATASET_REVISION"], "dataset_file": c["DATASET_FILE"], "dataset_sha256": archive_hash, "rows": rows, "images": images, "resume": str(resume) if resume else ""})
    backup.upload(metadata, f"{c['RUN_NAME']}/metadata")
    for name in ("latents", "text", "train"):
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
