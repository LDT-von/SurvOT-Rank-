import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_dct_v38_transport_consistency.py"
SPEC = importlib.util.spec_from_file_location("dct_v38_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_child_cuda_preflight_uses_the_training_interpreter_and_environment(monkeypatch):
    captured = {}

    class Completed:
        returncode = 0
        stdout = '{"cuda_available": true, "device_name": "test-gpu"}\n'
        stderr = ""

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return Completed()

    monkeypatch.setattr(RUNNER.subprocess, "run", fake_run)
    environment = {"CUDA_VISIBLE_DEVICES": "2", "LD_LIBRARY_PATH": "/conda/lib"}

    assert RUNNER.verify_child_cuda("/conda/bin/python", environment)
    assert captured["command"][:2] == ["/conda/bin/python", "-c"]
    assert captured["kwargs"]["env"] is environment
    assert captured["kwargs"]["capture_output"] is True
    assert captured["kwargs"]["text"] is True


def test_child_cuda_preflight_rejects_cpu_fallback(monkeypatch):
    class Completed:
        returncode = 1
        stdout = '{"cuda_available": false}\n'
        stderr = ""

    monkeypatch.setattr(RUNNER.subprocess, "run", lambda *_args, **_kwargs: Completed())

    assert not RUNNER.verify_child_cuda("python", {"CUDA_VISIBLE_DEVICES": "0"})
