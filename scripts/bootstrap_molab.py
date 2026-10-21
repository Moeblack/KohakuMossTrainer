"""Pinned molab setup; no dataset, training, native-environment changes or uploads.

Run with --commit <validated full fork SHA>. The notebook pins this script's SHA256
and the same commit; no branch tip or credential is embedded here.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

UPSTREAM_COMMIT = "da5dd97e15492e4dfb3caa171f023388004884ff"
FORK_URL = "https://github.com/Moeblack/KohakuMossTrainer.git"
UPSTREAM_URL = "https://github.com/KohakuBlueleaf/KohakuMossTrainer.git"
PROJECT = Path("/marimo/KohakuMossTrainer")
VENV = Path("/marimo/.venvs/kohakumoss-py312-da5dd97")
MODELS = Path("/marimo/ComfyUI/models")
TORCH = "torch==2.13.0+cu130"
TORCHVISION = "torchvision==0.28.0+cu130"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"


def child_env():
    env = os.environ.copy()
    for key in (
        "VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT",
        "UV_SYSTEM_PYTHON", "UV_PYTHON", "UV_PYTHON_ENVIRONMENT", "UV_TARGET", "UV_PREFIX",
    ):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def run(*args, capture=False):
    return subprocess.run(
        [str(arg) for arg in args], env=child_env(), check=True, text=True,
        capture_output=capture,
    ).stdout


def canonical_remote(url):
    return url.strip().replace("git@github.com:", "https://github.com/").removesuffix(".git").rstrip("/").lower()


def prepare_source(project, commit):
    """Only migrate a clean official base or reuse the exact pinned fork revision."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("--commit must be a full validated 40-character commit SHA")
    if project.exists() or project.is_symlink():
        if project.is_symlink() or not (project / ".git").is_dir():
            raise RuntimeError(f"Refusing to overwrite an unknown project: {project}")
        origin = run("git", "-C", project, "remote", "get-url", "origin", capture=True).strip()
        head = run("git", "-C", project, "rev-parse", "HEAD", capture=True).strip()
        dirty = run("git", "-C", project, "status", "--porcelain", "--untracked-files=all", capture=True).strip()
        if dirty:
            raise RuntimeError("Project has local changes/untracked files; preserved unchanged. Commit or move them yourself first.")
        if canonical_remote(origin) not in {canonical_remote(FORK_URL), canonical_remote(UPSTREAM_URL)}:
            raise RuntimeError("Unknown project origin; preserved unchanged.")
        if head not in (UPSTREAM_COMMIT, commit):
            raise RuntimeError(f"Unexpected project HEAD {head}; preserved unchanged.")
        if head == commit:
            print("Reusing exact fork commit; no fetch or checkout.", flush=True)
            return
        # Fetch by URL: preserve origin/upstream and any user-owned remote names.
        run("git", "-C", project, "fetch", "--no-tags", FORK_URL, commit)
    else:
        project.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--no-checkout", FORK_URL, project)
        run("git", "-C", project, "fetch", "--no-tags", FORK_URL, commit)
    resolved = run("git", "-C", project, "rev-parse", f"{commit}^{{commit}}", capture=True).strip()
    if resolved != commit:
        raise RuntimeError("Fetched revision is not the pinned commit")
    run("git", "-C", project, "merge-base", "--is-ancestor", UPSTREAM_COMMIT, commit)
    run("git", "-C", project, "checkout", "--detach", commit)


# Metadata-only check: do not import torch, transformers or GPU kernels. Follow
# self-referencing extras in the official pyproject instead of duplicating its list.
DEPENDENCY_CHECK = r'''
import importlib.metadata as md
import json
from pathlib import Path
import sys
import tomllib
from urllib.parse import unquote, urlparse

project = Path(sys.argv[1]).resolve()
try:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
except ImportError:
    print(json.dumps({"problems": ["packaging is missing"], "editable": False}))
    raise SystemExit(0)

spec = tomllib.loads((project / "pyproject.toml").read_text())["project"]
requirements = list(spec["dependencies"])
extras = ["cuda"]
visited = set()
while extras:
    extra = extras.pop()
    if extra in visited:
        continue
    visited.add(extra)
    for raw in spec["optional-dependencies"].get(extra, []):
        req = Requirement(raw)
        if req.marker and not req.marker.evaluate({"extra": extra}):
            continue
        if canonicalize_name(req.name) == "kohakumoss":
            extras.extend(req.extras)
        else:
            requirements.append(raw)
requirements += ["torch==2.13.0+cu130", "torchvision==0.28.0+cu130", "huggingface_hub"]
problems = []
for raw in requirements:
    req = Requirement(raw)
    if req.marker and not req.marker.evaluate():
        continue
    try:
        version = md.version(req.name)
    except md.PackageNotFoundError:
        problems.append(f"missing {raw}")
    else:
        if req.specifier and not req.specifier.contains(version, prereleases=True):
            problems.append(f"{req.name}=={version} does not satisfy {req.specifier}")
try:
    direct = json.loads(md.distribution("kohakumoss").read_text("direct_url.json") or "{}")
except md.PackageNotFoundError:
    direct = {}
url = urlparse(direct.get("url", ""))
editable = bool(direct.get("dir_info", {}).get("editable")) and url.scheme == "file" and Path(unquote(url.path)).resolve() == project
print(json.dumps({"problems": problems, "editable": editable}))
'''


def prepare_environment(project, venv, uv):
    python = venv / "bin/python"
    if venv.is_symlink() or venv.resolve() == Path(sys.prefix).resolve():
        raise RuntimeError("Training venv must not be a symlink or the notebook environment")
    fresh = not venv.exists()
    if fresh:
        venv.parent.mkdir(parents=True, exist_ok=True)
        run(uv, "--no-config", "venv", "--python", "3.12", venv)
    cfg = venv / "pyvenv.cfg"
    if not cfg.is_file() or not python.is_file():
        raise RuntimeError("Incomplete existing venv; preserved unchanged")
    settings = dict(line.split("=", 1) for line in cfg.read_text().splitlines() if "=" in line)
    settings = {key.strip(): value.strip() for key, value in settings.items()}
    if settings.get("include-system-site-packages", "").lower() != "false":
        raise RuntimeError("Refusing a venv that inherits system site-packages")
    run(python, "-I", "-c", "import sys; assert sys.version_info[:2] == (3, 12); assert sys.prefix != sys.base_prefix")
    if fresh:
        run(uv, "--no-config", "pip", "install", "--python", python,
            TORCH, TORCHVISION, "--index-url", TORCH_INDEX)
        run(uv, "--no-config", "pip", "install", "--python", python,
            "--editable", f"{project}[cuda]", TORCH, TORCHVISION, "huggingface_hub",
            "--index-url", "https://pypi.org/simple", "--extra-index-url", TORCH_INDEX,
            "--index-strategy", "unsafe-best-match")
    else:
        state = json.loads(run(python, "-I", "-c", DEPENDENCY_CHECK, project, capture=True))
        if state["problems"]:
            raise RuntimeError("Existing dependencies differ; refusing upgrades/reinstallation: " + "; ".join(state["problems"]))
        if not state["editable"]:
            run(uv, "--no-config", "pip", "install", "--python", python,
                "--no-deps", "--reinstall-package", "kohakumoss", "--editable", project)
        else:
            print("Reusing editable project and all installed dependencies; no pip install.", flush=True)
    return python


DOWNLOAD_MODELS = r'''
from pathlib import Path
import sys
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError

repo = "Comfy-Org/MOSS-Image"
revision = "ee64003102c8470ee32501cf883419235db81884"
root = Path(sys.argv[1])
files = (
    "diffusion_models/moss_image_sft_bf16.safetensors",
    "text_encoders/qwen3.5_9b_bf16.safetensors",
    "vae/flux2-vae.safetensors",
)
for filename in files:
    try:
        cached = Path(hf_hub_download(repo_id=repo, filename=filename, revision=revision, local_files_only=True))
    except LocalEntryNotFoundError:
        cached = Path(hf_hub_download(repo_id=repo, filename=filename, revision=revision))
    target = root / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.exists() or not target.samefile(cached):
            raise RuntimeError(f"Refusing to replace an existing different model/link: {target}")
        print(f"Reusing {target}", flush=True)
    else:
        target.symlink_to(cached.resolve())
        print(f"Linked {target}", flush=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    uv = shutil.which("uv")
    if not uv or not shutil.which("git"):
        raise RuntimeError("git and uv must already be available; notebook packages are not modified")
    print("[1/3] Preparing pinned fork source", flush=True)
    prepare_source(PROJECT, args.commit)
    print("[2/3] Preparing/reusing isolated Python 3.12", flush=True)
    python = prepare_environment(PROJECT, VENV, uv)
    print("[3/3] Reusing/downloading fixed-revision model cache", flush=True)
    run(python, "-I", "-c", DOWNLOAD_MODELS, MODELS)
    print(f"Ready: {PROJECT}; Python: {python}; models: {MODELS}", flush=True)
    print("No training data downloaded, no training started, no upload.", flush=True)


if __name__ == "__main__":
    main()
