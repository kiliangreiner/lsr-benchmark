import unittest
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from sentence_transformers import SentenceTransformer
from sentence_transformers_embeddings import embedd_text_with_model
from lsr_benchmark.irds import embeddings as load_embeddings


def calculate_similarity(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> list[float]:
    similarities = []
    for a, b in zip(embeddings_a, embeddings_b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            similarities.append(0.0)
        else:
            similarities.append(float(np.dot(a, b) / (norm_a * norm_b)))
    return similarities


def reconstruct_dense_embeddings(loaded: list, n_dims: int) -> np.ndarray:
    result = []
    for doc_id, tokens, values in loaded:
        vec = np.zeros(n_dims)
        for t, v in zip(tokens, values):
            vec[int(t)] = v
        result.append(vec)
    return np.array(result)


class TestEmbeddings(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = SentenceTransformer('BAAI/bge-m3')
        cls.texts = ['brown fox', 'green cat', 'yellow rabbit']
        cls.ids = ['id-1', 'id-2', 'id-3']
        cls.n_dims = 1024

    def _save_embeddings(self, d, texts, ids, text_type="doc"):
        """Helper: save embeddings in the correct directory structure."""
        type_dir = Path(d) / text_type
        type_dir.mkdir(exist_ok=True)
        output = type_dir / f"{text_type}-embeddings.npz"
        embedd_text_with_model(self.model, texts, ids, output)
        return type_dir

    def test_01_embeddings_save_and_load(self):
        """Saving and loading embeddings should return identical vectors."""
        with TemporaryDirectory() as d:
            self._save_embeddings(d, self.texts, self.ids)
            loaded = load_embeddings("dummy", d, "doc")
            actual = reconstruct_dense_embeddings(loaded, self.n_dims)
            target = self.model.encode(self.texts, normalize_embeddings=True)

            similarities = calculate_similarity(target, actual)

            self.assertEqual(len(similarities), 3)
            self.assertAlmostEqual(similarities[0], 1.0, places=5)
            self.assertAlmostEqual(similarities[1], 1.0, places=5)
            self.assertAlmostEqual(similarities[2], 1.0, places=5)

    def test_02_ids_saved_correctly(self):
        """IDs should be saved and loaded correctly."""
        with TemporaryDirectory() as d:
            self._save_embeddings(d, self.texts, self.ids)
            loaded = load_embeddings("dummy", d, "doc")
            loaded_ids = [doc_id for doc_id, _, _ in loaded]
            self.assertEqual(loaded_ids, self.ids)

    def test_03_number_of_embeddings_correct(self):
        """Number of saved embeddings should match number of input texts."""
        with TemporaryDirectory() as d:
            self._save_embeddings(d, self.texts, self.ids)
            loaded = load_embeddings("dummy", d, "doc")
            self.assertEqual(len(loaded), len(self.texts))

    def test_04_single_text(self):
        """A single text should also be embedded correctly."""
        with TemporaryDirectory() as d:
            self._save_embeddings(d, ['single text'], ['id-1'])
            loaded = load_embeddings("dummy", d, "doc")
            actual = reconstruct_dense_embeddings(loaded, self.n_dims)
            target = self.model.encode(['single text'], normalize_embeddings=True)

            similarities = calculate_similarity(target, actual)
            self.assertAlmostEqual(similarities[0], 1.0, places=5)

    def test_05_different_texts_have_different_embeddings(self):
        """Different texts should produce different embeddings."""
        with TemporaryDirectory() as d:
            self._save_embeddings(d, self.texts, self.ids)
            loaded = load_embeddings("dummy", d, "doc")
            actual = reconstruct_dense_embeddings(loaded, self.n_dims)

            sim_01 = float(np.dot(actual[0], actual[1]))
            sim_02 = float(np.dot(actual[0], actual[2]))

            self.assertLess(sim_01, 1.0)
            self.assertLess(sim_02, 1.0)


if __name__ == "__main__":
    unittest.main()
