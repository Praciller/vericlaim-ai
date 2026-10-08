from scripts.smoke_providers import model_check


def test_gemini_model_check_keeps_api_key_out_of_query_parameters(monkeypatch):
    api_key = "unit-test-only-gemini-key"
    captured = {}

    class FakeResponse:
        def json(self):
            return {"models": [{"name": "models/test-model"}]}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("scripts.smoke_providers.httpx.get", fake_get)
    provider = type("Gemini", (), {"api_key": api_key, "model": "test-model"})()

    assert model_check("gemini", provider) == "PASS"
    assert "key" not in captured.get("params", {})
    assert captured["headers"]["x-goog-api-key"] == api_key
