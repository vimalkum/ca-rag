# modules/extract_ontology.py

import re
from rdflib import Graph, RDF, RDFS, URIRef, Literal
from rdflib.namespace import OWL
from sentence_transformers import SentenceTransformer, util
import Levenshtein


class OntologyExtractor:
    DEFAULT_THRESHOLD = 0.45
    EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

    def __init__(self, file_path: str):
        """
        Load ontology ONCE at initialization.
        """
        self.file_path = file_path
        self.graph = Graph()
        self.graph.parse(file_path, format="turtle")

    # -------------------------
    # Utilities
    # -------------------------

    @staticmethod
    def get_local_name(uri):
        if isinstance(uri, URIRef):
            u = str(uri)
            return u.split("#")[-1] if "#" in u else u.split("/")[-1]
        return str(uri)

    @staticmethod
    def normalize(text):
        text = (text or "").lower().strip()
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def _clean_title(title: str) -> str:
        title = re.sub(r"[^\w\s/-]+", " ", title or "")
        return re.sub(r"\s+", " ", title).strip()

    @staticmethod
    def _description(g, s):
        c = g.value(s, RDFS.comment)
        if isinstance(c, Literal) and str(c).strip():
            return str(c).strip()
        return ""

    def _all_labels(self, s: URIRef) -> list[str]:
        labels = []

        for o in self.graph.objects(s, RDFS.label):
            if isinstance(o, Literal):
                labels.append(str(o))

        labels.append(self.get_local_name(s))

        seen = set()
        out = []
        for l in labels:
            if l.lower() not in seen:
                seen.add(l.lower())
                out.append(l)
        return out

    @staticmethod
    def _hybrid_score(query: str, label: str, emb_score: float) -> float:
        lex = Levenshtein.ratio(query.lower(), label.lower())
        return 0.75 * float(emb_score) + 0.25 * float(lex)

    # -------------------------
    # Properties (desc)
    # -------------------------

    def semantic_match_properties(self, title: str, top_k: int = 1):
        query = self._clean_title(title)
        if not query:
            return {}

        candidates = []
        texts = []

        for prop_type in (OWL.ObjectProperty, OWL.DatatypeProperty):
            for s in self.graph.subjects(RDF.type, prop_type):
                labels = self._all_labels(s)
                desc = self._description(self.graph, s)

                searchable = " | ".join([*labels, desc]).strip()
                if not searchable:
                    continue

                candidates.append({
                    "labels": labels,
                    "comment": desc,
                })
                texts.append(self.normalize(searchable))

        if not texts:
            return {}

        q_emb = self.EMBEDDING_MODEL.encode(self.normalize(query), convert_to_tensor=True, normalize_embeddings=True)
        t_emb = self.EMBEDDING_MODEL.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        scores = util.cos_sim(q_emb, t_emb)[0]

        ranked = []
        for i, emb in enumerate(scores.tolist()):
            score = self._hybrid_score(query, candidates[i]["labels"][0], emb)
            ranked.append((i, score))

        ranked.sort(key=lambda x: x[1], reverse=True)

        # Always return best match (even if low score)
        results = {}
        for i, score in ranked[:top_k]:
            key = candidates[i]["labels"][0]
            results[key] = {
                "comment": candidates[i]["comment"],
                "score": float(score),
                "labels": candidates[i]["labels"],
            }

        return results

    # -------------------------
    # Classes (concepts)
    # -------------------------

    def semantic_match_classes(self, title: str, top_k: int = 3):
        query = self._clean_title(title)
        if not query:
            return []

        candidates = []
        texts = []

        for s in self.graph.subjects(RDF.type, OWL.Class):
            labels = self._all_labels(s)
            desc = self._description(self.graph, s)

            searchable = " | ".join([*labels, desc]).strip()
            if not searchable:
                continue

            candidates.append({
                "label": labels[0],
                "definition": desc,
            })
            texts.append(self.normalize(searchable))

        if not texts:
            return []

        q_emb = self.EMBEDDING_MODEL.encode(self.normalize(query), convert_to_tensor=True, normalize_embeddings=True)
        t_emb = self.EMBEDDING_MODEL.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        scores = util.cos_sim(q_emb, t_emb)[0]

        ranked = []
        for i, emb in enumerate(scores.tolist()):
            score = self._hybrid_score(query, candidates[i]["label"], emb)
            ranked.append((i, score))

        ranked.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                "label": candidates[i]["label"],
                "definition": candidates[i]["definition"],
                "score": float(score),
            }
            for i, score in ranked[:top_k]
        ]

    # -------------------------
    # PUBLIC API
    # -------------------------

    def extract_info(self, titles):
        """
        Returns:
          desc[title]    -> best OWL property
          classes[title] -> top 3 OWL classes
        """
        desc = {}
        classes = {}

        for t in titles:
            title = t.rstrip()
            desc[title] = self.semantic_match_properties(title)

        for t in titles:
            title = t.rstrip()
            classes[title] = self.semantic_match_classes(title)

        return desc, classes
