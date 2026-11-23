from dataclasses import dataclass
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

@dataclass
class Chunk:
    text: str
    source: str = ""
    page: int | None = None

class TfidfRetriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def top_k(self, query: str, k: int = 6) -> List[Tuple[Chunk, float]]:
        qv = self.vectorizer.transform([query])
        scores = (self.matrix @ qv.T).toarray().ravel()
        if scores.size == 0:
            return []
        idx = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in idx if scores[i] > 0]
