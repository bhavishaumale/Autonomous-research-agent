# agent/memory.py
# Using FAISS for vector storage + Google Gemini for embeddings (free)

import os
import json
import numpy as np
import faiss
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from agent.tools import Paper

FAISS_INDEX_PATH = "./faiss_index/index.faiss"
METADATA_PATH = "./faiss_index/metadata.json"
IDS_PATH = "./faiss_index/ids.json"


class PaperMemory:

    def __init__(self):
        self.embedder = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        self.dimension = 3072  # Gemini embedding size (different from OpenAI's 1536)
        os.makedirs("./faiss_index", exist_ok=True)

        if os.path.exists(FAISS_INDEX_PATH):
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            self.ids = self._load_json(IDS_PATH, default=[])
            self.metadata = self._load_json(METADATA_PATH, default=[])
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.ids = []
            self.metadata = []

    def _load_json(self, path, default):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return default

    def _save(self):
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        with open(IDS_PATH, "w") as f:
            json.dump(self.ids, f)
        with open(METADATA_PATH, "w") as f:
            json.dump(self.metadata, f)

    def is_new(self, paper_id: str) -> bool:
        return paper_id not in self.ids

    def store(self, paper: Paper):
        embedding = self.embedder.embed_query(paper.abstract)
        vector = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vector)
        self.index.add(vector)
        self.ids.append(paper.id)
        self.metadata.append({
            "title": paper.title,
            "url": paper.url,
            "authors": ", ".join(paper.authors[:3]),
            "published": paper.published
        })
        self._save()

    def search_similar(self, query: str, n_results: int = 5) -> list:
        if self.index.ntotal == 0:
            return []
        query_embedding = self.embedder.embed_query(query)
        query_vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)
        k = min(n_results, self.index.ntotal)
        distances, indices = self.index.search(query_vector, k)
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.metadata):
                results.append(self.metadata[idx])
        return results

    def count(self) -> int:
        return self.index.ntotal