from types import SimpleNamespace

from scripts import smoke_verification


def test_smoke_verification_configures_utf8_stdout(monkeypatch):
    calls = []
    stream = SimpleNamespace(reconfigure=lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(smoke_verification.sys, "stdout", stream)

    smoke_verification._configure_stdout()

    assert calls == [{"encoding": "utf-8"}]
