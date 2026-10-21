"""Bootstrap safety/idempotence with fake commands and tiny cached model files.

No network, package installation, real weights, GPU or native-env mutation.
"""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

pytestmark = pytest.mark.cpu
SPEC = importlib.util.spec_from_file_location(
    "bootstrap_molab", Path(__file__).resolve().parents[1] / "scripts/bootstrap_molab.py"
)
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)
PATCH = "1" * 40


def source_runner(monkeypatch, project, head=None, dirty="", origin=None):
    (project / ".git").mkdir(parents=True)
    calls = []

    def run(*args, **kwargs):
        calls.append(tuple(str(arg) for arg in args))
        if args[3:6] == ("remote", "get-url", "origin"):
            return origin or bootstrap.UPSTREAM_URL
        if args[3:] == ("rev-parse", "HEAD"):
            return head or bootstrap.UPSTREAM_COMMIT
        if args[3] == "status":
            return dirty
        if args[3] == "rev-parse":
            return PATCH
        return ""

    monkeypatch.setattr(bootstrap, "run", run)
    return calls


def test_clean_official_source_migrates_without_reset(tmp_path, monkeypatch):
    calls = source_runner(monkeypatch, tmp_path)
    bootstrap.prepare_source(tmp_path, PATCH)
    assert ("git", "-C", str(tmp_path), "fetch", "--no-tags", bootstrap.FORK_URL, PATCH) in calls
    assert calls[-1][-3:] == ("checkout", "--detach", PATCH)
    assert not any("reset" in c or "clean" in c or "pull" in c for c in calls)


def test_pinned_source_has_no_network_or_checkout(tmp_path, monkeypatch):
    calls = source_runner(monkeypatch, tmp_path, head=PATCH, origin=bootstrap.FORK_URL)
    bootstrap.prepare_source(tmp_path, PATCH)
    assert len(calls) == 3


@pytest.mark.parametrize("state", ["dirty", "head", "origin"])
def test_unknown_or_dirty_source_stops_before_mutation(tmp_path, monkeypatch, state):
    kwargs = {
        "dirty": {"dirty": "?? user-config.py"},
        "head": {"head": "2" * 40},
        "origin": {"origin": "https://example.org/unknown.git"},
    }[state]
    calls = source_runner(monkeypatch, tmp_path, **kwargs)
    with pytest.raises(RuntimeError):
        bootstrap.prepare_source(tmp_path, PATCH)
    assert len(calls) == 3


def test_unpinned_revision_rejected_before_commands(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "run", lambda *a, **kw: pytest.fail("unexpected command"))
    with pytest.raises(ValueError, match="full validated"):
        bootstrap.prepare_source(tmp_path, "main")


def test_child_environment_does_not_mutate_notebook(monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/notebook-env")
    monkeypatch.setenv("PYTHONPATH", "/notebook-path")
    monkeypatch.setenv("HF_HOME", "/existing-hf-cache")
    env = bootstrap.child_env()
    assert "VIRTUAL_ENV" not in env and "PYTHONPATH" not in env
    assert env["HF_HOME"] == "/existing-hf-cache"
    assert bootstrap.os.environ["VIRTUAL_ENV"] == "/notebook-env"


def make_venv(venv):
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").touch()
    (venv / "pyvenv.cfg").write_text("include-system-site-packages = false\n")


@pytest.mark.parametrize("editable", [True, False])
def test_existing_dependencies_not_reinstalled(tmp_path, monkeypatch, editable):
    venv = tmp_path / "venv"
    make_venv(venv)
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        return json.dumps({"problems": [], "editable": editable})

    monkeypatch.setattr(bootstrap, "run", run)
    bootstrap.prepare_environment(tmp_path / "project", venv, "uv")
    installs = [c for c in calls if c[0] == "uv"]
    if editable:
        assert installs == []
    else:
        assert len(installs) == 1
        assert "--no-deps" in installs[0]
        assert "--editable" in installs[0]
        assert installs[0][installs[0].index("--reinstall-package") + 1] == "kohakumoss"


def test_existing_incompatible_environment_is_preserved(tmp_path, monkeypatch):
    venv = tmp_path / "venv"
    make_venv(venv)
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        return json.dumps({"problems": ["torch incompatible"], "editable": True})

    monkeypatch.setattr(bootstrap, "run", run)
    with pytest.raises(RuntimeError, match="refusing upgrades"):
        bootstrap.prepare_environment(tmp_path, venv, "uv")
    assert all(c[0] != "uv" for c in calls)


def test_fresh_environment_pins_both_cuda_wheels(tmp_path, monkeypatch):
    venv = tmp_path / "venv"
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        if "venv" in args:
            make_venv(venv)
        return ""

    monkeypatch.setattr(bootstrap, "run", run)
    bootstrap.prepare_environment(tmp_path / "project", venv, "uv")
    installs = [c for c in calls if "install" in c]
    assert len(installs) == 2
    assert all(bootstrap.TORCH in c and bootstrap.TORCHVISION in c for c in installs)
    assert "--editable" in installs[1]
    assert str(tmp_path / "project") + "[cuda]" in installs[1]


def fake_hub(monkeypatch, tmp_path, missing=False):
    hub = ModuleType("huggingface_hub")
    errors = ModuleType("huggingface_hub.errors")

    class LocalEntryNotFoundError(Exception):
        pass

    errors.LocalEntryNotFoundError = LocalEntryNotFoundError
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        assert kwargs["revision"] == "ee64003102c8470ee32501cf883419235db81884"
        assert kwargs["repo_id"] == "Comfy-Org/MOSS-Image"
        if missing and kwargs.get("local_files_only"):
            raise LocalEntryNotFoundError()
        cached = tmp_path / "hf" / kwargs["filename"]
        cached.parent.mkdir(parents=True, exist_ok=True)
        if not cached.exists():
            cached.write_bytes(b"tiny cached model")
        return str(cached)

    hub.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    monkeypatch.setattr(sys, "argv", ["download", str(tmp_path / "models")])
    return calls


def test_model_cache_reused_without_network(tmp_path, monkeypatch):
    calls = fake_hub(monkeypatch, tmp_path)
    exec(bootstrap.DOWNLOAD_MODELS, {})
    links = sorted((tmp_path / "models").rglob("*.safetensors"))
    assert len(links) == 3 and all(p.is_symlink() for p in links)
    before = [(p, p.lstat().st_ino, p.readlink()) for p in links]
    exec(bootstrap.DOWNLOAD_MODELS, {})
    assert all(c.get("local_files_only") for c in calls)
    assert before == [(p, p.lstat().st_ino, p.readlink()) for p in links]


def test_only_local_cache_miss_triggers_download(tmp_path, monkeypatch):
    calls = fake_hub(monkeypatch, tmp_path, missing=True)
    exec(bootstrap.DOWNLOAD_MODELS, {})
    assert len(calls) == 6
    assert [bool(c.get("local_files_only")) for c in calls] == [True, False] * 3


def test_different_model_is_not_overwritten(tmp_path, monkeypatch):
    fake_hub(monkeypatch, tmp_path)
    path = tmp_path / "models/diffusion_models/moss_image_sft_bf16.safetensors"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"user model")
    with pytest.raises(RuntimeError, match="Refusing to replace"):
        exec(bootstrap.DOWNLOAD_MODELS, {})
    assert path.read_bytes() == b"user model"


@pytest.mark.parametrize("source_match", [True, False])
def test_dependency_check_uses_module_origin_without_direct_url(tmp_path, monkeypatch, capsys, source_match):
    import importlib.metadata as md
    from types import SimpleNamespace

    (tmp_path / "pyproject.toml").write_text(
        '[project]\ndependencies = []\n[project.optional-dependencies]\ncuda = []\n'
    )
    versions = {"torch": "2.13.0+cu130", "torchvision": "0.28.0+cu130", "huggingface_hub": "1.0"}
    monkeypatch.setattr(md, "version", lambda name: versions[name])
    monkeypatch.setattr(md, "distribution", lambda name: pytest.fail("direct_url is not required"))
    original_find_spec = importlib.util.find_spec

    def find_spec(name):
        if name in ("kohakumoss", "kohakumoss_data"):
            root = tmp_path if source_match else tmp_path / "other-project"
            return SimpleNamespace(origin=str(root / "src" / name / "__init__.py"))
        return original_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(sys, "argv", ["dependency_check", str(tmp_path)])
    exec(bootstrap.DEPENDENCY_CHECK, {})
    state = json.loads(capsys.readouterr().out)
    assert state == {"problems": [], "editable": source_match}
