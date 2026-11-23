from sentence_transformers import SentenceTransformer, util
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import split_uri, RDFS


class SemanticExtractor:
    def __init__(self, mpath: str):
        self.model = SentenceTransformer(mpath)

    def load_turtle(self, file_path: str) -> Graph:
        g = Graph()
        g.parse(file_path, format="turtle")
        return g

    def get_best_text(self, term, graph: Graph) -> str:
        """
        Prefer rdfs:label if present; else literal value; else local name.
        """
        if isinstance(term, Literal):
            return str(term)

        if isinstance(term, URIRef):
            label = graph.value(term, RDFS.label)
            if isinstance(label, Literal) and str(label).strip():
                return str(label)

        # fallback: local name
        return self._safe_local_name(term)

    def _safe_local_name(self, term) -> str:
        if isinstance(term, (Literal, BNode)):
            return str(term)
        if isinstance(term, URIRef):
            s = str(term)
            try:
                _, local = split_uri(s)
                return local
            except ValueError:
                return s.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
        return str(term)

    def extract_relevant_triples(
        self,
        filepath: str,
        section_title: str,
        threshold: float = 0.45,
        top_k: int = 30,
        ensure_nonempty: bool = True,
    ):
        """
        Returns a list of (subj, pred, obj) local names.
        - Scores subject + predicate + object
        - Sorts by best similarity score
        - If ensure_nonempty=True: returns top_k even if below threshold (useful for short headings)
        """
        graph = self.load_turtle(filepath)

        # Encode query once
        query = (section_title or "").strip()
        if not query:
            return []
        q_emb = self.model.encode(query, convert_to_tensor=True, normalize_embeddings=True)

        scored = []

        for (s, p, o) in graph:
            s_text = self.get_best_text(s, graph)
            p_text = self.get_best_text(p, graph)  # predicate matters a lot
            o_text = self.get_best_text(o, graph)

            # Encode once per element (still 3, but query is cached)
            s_emb = self.model.encode(s_text, convert_to_tensor=True, normalize_embeddings=True)
            p_emb = self.model.encode(p_text, convert_to_tensor=True, normalize_embeddings=True)
            o_emb = self.model.encode(o_text, convert_to_tensor=True, normalize_embeddings=True)

            s_score = util.cos_sim(q_emb, s_emb).item()
            p_score = util.cos_sim(q_emb, p_emb).item()
            o_score = util.cos_sim(q_emb, o_emb).item()

            best_score = max(s_score, p_score, o_score)

            tri0 = self._safe_local_name(s)
            tri1 = self._safe_local_name(p)
            tri2 = self._safe_local_name(o)

            scored.append((best_score, (tri0, tri1, tri2)))

        if not scored:
            return []

        scored.sort(key=lambda x: x[0], reverse=True)

        # Filter by threshold
        filtered = [tri for score, tri in scored if score >= threshold]

        if filtered:
            return filtered[:top_k]

        # If nothing passes threshold, return top_k anyway (prevents "empty KG")
        return [tri for _, tri in scored[:top_k]] if ensure_nonempty else []
