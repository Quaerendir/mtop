"""The single-file artifact must build and behave like the package."""

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _bundle(tmp_path):
    out = tmp_path / "mtop.py"
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "bundle.py"), "-o", str(out)],
                       capture_output=True, text=True, check=True)
    assert "1 embedded module" in r.stdout
    return out


def test_bundle_version_matches_package(tmp_path):
    out = _bundle(tmp_path)
    sys.path.insert(0, str(ROOT / "src"))
    import mtop
    r = subprocess.run([sys.executable, str(out), "--version"], capture_output=True, text=True)
    assert r.returncode == 0
    assert r.stdout.strip() == f"mtop {mtop.__version__}"


def test_bundle_json_api_mode_against_dead_port(tmp_path):
    out = _bundle(tmp_path)
    # Port 1 is never an Ollama; this exercises Collector + gpu module inside
    # the bundle without touching docker or nvidia-smi.
    r = subprocess.run([sys.executable, str(out), "--json", "--mode", "api", "--no-gpu",
                        "-u", "127.0.0.1:1"], capture_output=True, text=True, timeout=30)
    assert r.returncode == 1
    snap = json.loads(r.stdout)
    assert snap["mode"] == "api" and snap["models_ok"] is False
    assert snap["api_url"] == "http://127.0.0.1:1"


def test_bundle_rejects_triple_single_quotes(tmp_path, monkeypatch):
    # Guard for the embedding format: a ''' in a submodule would truncate it.
    pkg = tmp_path / "src" / "mtop"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text('"""doc\n"""\nfrom .bad import x\n')
    (pkg / "bad.py").write_text("x = '''oops'''\n")
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "bundle.py").write_text((ROOT / "tools" / "bundle.py").read_text())
    r = subprocess.run([sys.executable, str(tools / "bundle.py"), "-o", str(tmp_path / "o.py")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "'''" in r.stderr
