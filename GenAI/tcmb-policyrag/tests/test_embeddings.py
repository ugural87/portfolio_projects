from types import SimpleNamespace

import numpy as np

from tcmb_policyrag.embeddings import OpenAIEmbedder


class FakeEmbeddings:
    def create(self, model: str, input: list[str]) -> SimpleNamespace:  # noqa: A002
        data = [
            SimpleNamespace(index=index, embedding=[float(index + 1), 1.0])
            for index, _text in enumerate(input)
        ]
        return SimpleNamespace(data=data)


class FakeClient:
    embeddings = FakeEmbeddings()


def test_build_embedding_progress_and_cache_hits(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("tcmb_policyrag.embeddings.OpenAI", FakeClient)
    embedder = OpenAIEmbedder(
        "test-model",
        cache_dir=tmp_path,
        batch_size=2,
        show_progress=True,
    )

    first = embedder.embed(["a", "b", "c"])
    first_stderr = capsys.readouterr().err
    assert first.shape == (3, 2)
    assert "Embedding progress: 0/3" in first_stderr
    assert "Embedding progress: 3/3 (100.0%)" in first_stderr
    assert "cache hits: 0" in first_stderr

    second = embedder.embed(["a", "b", "c"])
    second_stderr = capsys.readouterr().err
    np.testing.assert_array_equal(second, first)
    assert "Embedding progress: 3/3 (100.0%) | cache hits: 3" in second_stderr
