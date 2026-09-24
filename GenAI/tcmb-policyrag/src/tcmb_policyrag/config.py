from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    chat_model: str = "gpt-5.6-luna"
    embedding_model: str = "text-embedding-3-small"
    index_dir: Path = Path("artifacts/index")
    top_k: int = 8
    min_score: float = 0.20

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        return cls(
            chat_model=os.getenv("OPENAI_CHAT_MODEL", cls.chat_model),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", cls.embedding_model),
            index_dir=Path(os.getenv("TCMB_RAG_INDEX_DIR", str(cls.index_dir))),
            top_k=int(os.getenv("TCMB_RAG_TOP_K", str(cls.top_k))),
            min_score=float(os.getenv("TCMB_RAG_MIN_SCORE", str(cls.min_score))),
        )
